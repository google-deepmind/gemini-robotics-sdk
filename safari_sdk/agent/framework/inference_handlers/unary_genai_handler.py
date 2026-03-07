# Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""A handler for unary (non-streaming) GenAI API.

This provides an alternative to GeminiLiveAPIHandler that uses the core
generate_content API instead of the Live streaming API. It maintains
conversation history explicitly and processes tool calls in a request-response
loop.
"""

import asyncio
import base64
from collections.abc import Sequence
import datetime
import enum
import os

from absl import logging
from google import genai
from google.genai import types

from safari_sdk.agent.framework import config as framework_config
from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.inference_handlers import nonstreaming_handler


_DEFAULT_MODEL_NAME = "models/gemini-2.0-flash"
# Number of retries for empty candidates in a single generate_content call.
# Successful query with empty candidate retries will be treated as a single try.
_EMPTY_CANDIDATES_INNER_RETRIES = 2


@enum.unique
class GenAIOrchestratorHealth(
    nonstreaming_handler.NonstreamingOrchestratorHealth
):
  """Class for tracking the health status of the API."""

  NORMAL = "NORMAL"
  ERROR_QUOTA_EXCEEDED = "ERROR_QUOTA_EXCEEDED"
  ERROR_OTHER = "ERROR_OTHER"


class UnaryGenAIHandler(nonstreaming_handler.NonStreamingHandler):
  """Handler for unary (non-streaming) Gemini GenAI API.

  This handler uses the generate_content API instead of the Live streaming API.
  It maintains conversation history explicitly and processes requests in a
  request-response pattern rather than continuous streaming.

  Key differences from GeminiLiveAPIHandler:
  - No real-time audio/video streaming
  - Images are buffered and sent per-turn
  - Conversation history is maintained as a list of Content objects
  - Tool calls are processed in a synchronous loop until model returns text
  """

  def __init__(
      self,
      bus: event_bus.EventBus,
      config: framework_config.AgentFrameworkConfig,
      system_instruction: str | types.Content | None = None,
      tools: Sequence[types.Tool] | None = None,
      tool_config: types.ToolConfig | None = None,
      camera_names: Sequence[str] | None = None,
      stream_name_to_camera_name: dict[str, str] | None = None,
      http_options: dict[str, str] | None = None,
      ignore_image_inputs: bool = False,
      temperature: float | None = None,
      max_output_tokens: int | None = None,
      thinking_budget: int | None = None,
      media_resolution: types.MediaResolution | None = None,
  ):
    super().__init__(
        bus=bus,
        config=config,
        camera_names=camera_names,
        stream_name_to_camera_name=stream_name_to_camera_name,
        ignore_image_inputs=ignore_image_inputs,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        thinking_budget=thinking_budget,
        media_resolution=media_resolution,
    )

    api_key = self._get_api_key(config.api_key)
    self._client = genai.Client(
        api_key=api_key,
        http_options=http_options,
    )
    self._api_health = GenAIOrchestratorHealth.NORMAL
    self._model_name = config.agent_model_name or _DEFAULT_MODEL_NAME
    self._system_instruction = system_instruction
    self._tools = tools
    self._tool_config = tool_config
    self._conversation_history: list[types.Content] = []

    self._thinking_config = types.ThinkingConfig(
        thinking_level=(
            types.ThinkingLevel(config.non_streaming_thinking_level)
            if config.non_streaming_thinking_level
            else None
        ),
    )

    self._function_call_timestamp: datetime.datetime | None = None

  async def connect(self) -> None:
    """Activates the handler (no persistent session needed)."""
    logging.info("Activating UnaryGenAIHandler...")
    self._loop = asyncio.get_running_loop()
    self._is_active = True
    self._conversation_history = []
    self._function_call_timestamp = None
    self._pending_tool_results = asyncio.Queue()
    self._clear_image_state()
    self._start_image_stitching()
    await self._bootup_test()
    await self._publish_session_metadata()
    await self._publish_health_status(GenAIOrchestratorHealth.NORMAL, None)
    logging.info("UnaryGenAIHandler activated.")

  async def _bootup_test(self) -> None:
    """Sends a simple message to verify client connectivity.

    This is called during connect() to catch client failures early.
    """
    try:
      logging.info("Calling generate with BOOTUP TEST")
      bootup_response = await asyncio.wait_for(
          self._client.aio.models.generate_content(
              model=self._model_name,
              contents=types.Content(
                  role="user",
                  parts=[types.Part.from_text(text="hi")],
              ),
              config=types.GenerateContentConfig(
                  system_instruction=self._system_instruction,
                  tools=self._tools,
                  tool_config=self._tool_config,
                  temperature=self._temperature,
                  max_output_tokens=self._max_output_tokens,
                  media_resolution=self._media_resolution,
                  thinking_config=self._thinking_config,
              ),
          ),
          timeout=15.0,
      )
      bootup_response_text = str(bootup_response.text)[:250]
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.error("Error generating BOOTUP TEST response: %s", e)
      await self._maybe_publish_health_event(e)
      raise e
    logging.info("BOOTUP TEST Success: %s", bootup_response_text)

  async def disconnect(self) -> None:
    """Deactivates the handler."""
    logging.info("Deactivating UnaryGenAIHandler...")
    self._is_active = False
    self._stop_image_stitching()
    self._conversation_history = []
    self._function_call_timestamp = None
    self._clear_image_state()
    logging.info("UnaryGenAIHandler deactivated.")

  async def _generate_response(self, user_text: str) -> None:
    """Generates a response using the model.

    This method:
    1. Builds the user message with text and any buffered images
    2. Calls generate_content with the conversation history
    3. Publishes model response events
    4. If tool calls are returned, publishes them and waits for results
    5. Continues until the model returns only text

    Args:
      user_text: The user's text input.
    Raises:
      Exception: If the model returns an error.
    """
    async with self._generate_lock:
      # Build user content (text + buffered images) and add to history.
      self._build_and_append_user_content(user_text)
      # Remove images from older turns to keep context window manageable.
      self._trim_images_in_history()

      # Agentic loop: call model, handle tool calls, repeat until text-only.
      # Empty candidates are handled with a separate retry counter since this
      # is a different failure mode than transient network errors.
      remaining_empty_candidates_retries = _EMPTY_CANDIDATES_INNER_RETRIES
      while True:
        try:
          response = await self._retry_wrapper(self._call_model)
          await self._maybe_publish_health_event(None)
        except Exception as e:
          logging.error("Error generating response: %s", e, exc_info=True)
          await self._maybe_publish_health_event(e)
          raise

        if not response.candidates:
          logging.warning("No candidates in response.")
          remaining_empty_candidates_retries -= 1
          if remaining_empty_candidates_retries <= 0:
            err = RuntimeError(
                "No candidates in response after "
                f"{_EMPTY_CANDIDATES_INNER_RETRIES} (empty-candidates) retries."
            )
            await self._maybe_publish_health_event(err)
            raise err
          logging.info(
              "Retrying due to empty candidates (%d retries remaining)...",
              remaining_empty_candidates_retries,
          )
          await asyncio.sleep(nonstreaming_handler.RETRY_INTERVAL_SECONDS)
          continue

        # Add model response to history.
        # NOTE: We must keep the full response including thought_signature
        # for Gemini 3 models, as it's required for function calling.
        model_content = response.candidates[0].content
        self._conversation_history.append(model_content)

        # Publish the model turn so downstream listeners can process it.
        await self._publish_event(
            event_bus.EventType.MODEL_TURN,
            model_content,
        )

        # Check for tool calls — if present, handle them and loop again.
        tool_calls = self._extract_tool_calls(model_content)
        if tool_calls:
          if len(tool_calls) > 1:
            logging.warning(
                "!!! MULTIPLE FUNCTION CALLS DETECTED: %d calls in a single"
                " turn: %s. The system prompt specifies ONLY ONE function call"
                " at a time.",
                len(tool_calls),
                [tc.name for tc in tool_calls],
            )
          await self._handle_tool_calls(tool_calls)
          # Reset empty candidates retries after successful tool handling.
          remaining_empty_candidates_retries = _EMPTY_CANDIDATES_INNER_RETRIES
          continue

        # No tool calls — model returned final text, we're done.
        await self._publish_event(
            event_bus.EventType.MODEL_TURN_COMPLETE,
            True,
        )
        break

  def _build_and_append_user_content(self, user_text: str) -> None:
    """Builds user content from text and buffered images, appends to history."""
    # Start with the user's text.
    parts: list[types.Part] = [types.Part.from_text(text=user_text)]

    # Drain buffered images and append them as parts.
    # Two modes: stitched (grid frames) or raw (individual camera blobs).
    images_added_count = 0
    stitched_frames_count = 0
    if self._enable_image_stitching:
      # Stitched mode: grab pre-composed grid frames under the stitch lock.
      with self._stitch_lock:
        stitched_frames = list(self._stitched_frames)
        if self._discard_images_after_turn:
          self._stitched_frames = []

      if self._user_turn_latest_image_only and stitched_frames:
        stitched_frames = [stitched_frames[-1]]

      stitched_frames_count = len(stitched_frames)
      for _, frame in stitched_frames:
        parts.append(frame)
      logging.info(
          "Added %d pre-stitched frames to inference request",
          stitched_frames_count,
      )
    else:
      # Raw mode: send individual camera images with optional display names.
      images_to_add = list(self._image_buffer)
      if self._discard_images_after_turn:
        self._image_buffer = []

      if self._user_turn_latest_image_only and len(images_to_add) > 1:
        latest_per_stream: dict[
            str, tuple[datetime.datetime, str, types.Blob]
        ] = {}
        for img_ts, sname, blob in images_to_add:
          if (
              sname not in latest_per_stream
              or img_ts > latest_per_stream[sname][0]
          ):
            latest_per_stream[sname] = (img_ts, sname, blob)
        images_to_add = list(latest_per_stream.values())

      images_added_count = len(images_to_add)
      for _, _, blob in images_to_add:
        if self._include_stream_names and blob.display_name:
          parts.append(types.Part.from_text(text=blob.display_name))
        parts.append(
            types.Part.from_bytes(data=blob.data, mime_type="image/jpeg")
        )

    # Create user content and add to history.
    self._conversation_history.append(types.Content(role="user", parts=parts))

    # Log a summary of the full history contents for debugging.
    num_text_parts, num_images, num_fc, num_fr = self._count_content_parts()
    logging.info(
        "Inference request: text_parts=%d, images=%d, function_calls=%d,"
        " function_responses=%d (added %d %s)",
        num_text_parts,
        num_images,
        num_fc,
        num_fr,
        stitched_frames_count
        if self._enable_image_stitching
        else images_added_count,
        "stitched frames" if self._enable_image_stitching else "raw images",
    )

  async def _call_model(self) -> types.GenerateContentResponse:
    """Calls generate_content with the current conversation history."""
    self._log_conversation_history()
    _, num_images, _, _ = self._count_content_parts()
    logging.info(
        "Calling generate_content with model=%s, history_len=%d, tools=%s,"
        " images=%d",
        self._model_name,
        len(self._conversation_history),
        len(self._tools) if self._tools else 0,
        num_images,
    )
    inference_start = datetime.datetime.now()
    response = await asyncio.wait_for(
        self._client.aio.models.generate_content(
            model=self._model_name,
            contents=self._conversation_history,
            config=types.GenerateContentConfig(
                system_instruction=self._system_instruction,
                tools=self._tools,
                tool_config=self._tool_config,
                temperature=self._temperature,
                max_output_tokens=self._max_output_tokens,
                media_resolution=self._media_resolution,
                thinking_config=self._thinking_config,
            ),
        ),
        timeout=60.0,
    )
    inference_duration = (
        datetime.datetime.now() - inference_start
    ).total_seconds()
    metadata: dict[str, int | float] = {
        "inference_duration_seconds": round(inference_duration, 4)
    }
    if response.usage_metadata:
      metadata["prompt_token_count"] = (
          response.usage_metadata.prompt_token_count
      )
      metadata["candidates_token_count"] = (
          response.usage_metadata.candidates_token_count
      )
      metadata["total_token_count"] = response.usage_metadata.total_token_count

    if self._config.non_streaming_enable_context_snapshot_logging:
      await self._publish_event(
          event_bus.EventType.CONTEXT_SNAPSHOT,
          self._serialize_conversation_history(),
          metadata=metadata,
      )

    if response.usage_metadata:
      logging.info(
          "\n"
          "=================================================================\n"
          "generate_content response received in %.2fs\n"
          "Tokens: Prompt=%s | Candidates=%s | Total=%s\n"
          "=================================================================",
          inference_duration,
          response.usage_metadata.prompt_token_count,
          response.usage_metadata.candidates_token_count,
          response.usage_metadata.total_token_count,
      )
    else:
      logging.info(
          "\n"
          "=================================================================\n"
          "generate_content response received in %.2fs\n"
          "=================================================================",
          inference_duration,
      )
    return response

  async def _handle_tool_calls(
      self, tool_calls: list[types.FunctionCall]
  ) -> None:
    """Handles tool calls, waits for results, and publishes/updates history."""
    # Record the wall-clock time of the function call so we can later
    # identify which images arrived during tool execution.
    fc_timestamp = datetime.datetime.now(tz=datetime.timezone.utc)
    self._function_call_timestamp = fc_timestamp

    # Publish tool call event so the tool executor can start running.
    await self._publish_event(
        event_bus.EventType.TOOL_CALL,
        types.LiveServerToolCall(function_calls=tool_calls),
    )

    # Wait for all tool results to arrive from the tool executor(s).
    wait_start = datetime.datetime.now(tz=datetime.timezone.utc)
    tool_results = await self._wait_for_tool_results(len(tool_calls))
    fr_timestamp = datetime.datetime.now(tz=datetime.timezone.utc)
    logging.info(
        "_wait_for_tool_results took: %.3fs",
        (fr_timestamp - wait_start).total_seconds(),
    )

    # Collect any images that arrived during tool execution (FC→FR window)
    # and prepend them to the function response content.
    image_parts, _ = self._collect_images_for_fr(fc_timestamp, fr_timestamp)

    # Assemble the function response: images first, then tool results.
    tool_result_parts: list[types.Part] = list(image_parts)
    # create a map of tool call IDs to determine which ones timed out
    called_tool_names = {tc.name: tc for tc in tool_calls}
    for result in tool_results:
      fr_kwargs = {"name": result.name, "response": result.response}
      if result.id:
        fr_kwargs["id"] = result.id

      tool_result_parts.append(
          types.Part(function_response=types.FunctionResponse(**fr_kwargs))
      )
      if result.name in called_tool_names:
        del called_tool_names[result.name]

    # Handle timeouts by appending synthetic error responses
    for missing_tool_name, tc in called_tool_names.items():
      logging.warning(
          "Tool %s timed out, sending synthetic error", missing_tool_name
      )
      fr_kwargs = {
          "name": missing_tool_name,
          "response": {"error": "Tool execution timed out or failed silently"},
      }
      if tc.id:
        fr_kwargs["id"] = tc.id
      tool_result_parts.append(
          types.Part(function_response=types.FunctionResponse(**fr_kwargs))
      )

    # Add tool results to history as a user turn.
    self._conversation_history.append(
        types.Content(role="user", parts=tool_result_parts)
    )

    # Prune old images from earlier turns to keep context window manageable.
    if self._discard_images_after_turn:
      self._discard_old_images_in_history(len(self._conversation_history) - 1)

    self._trim_images_in_history()

  def _count_content_parts(self) -> tuple[int, int, int, int]:
    """Counts text, image, function call and response parts in history.

    Returns:
      A tuple of (num_text, num_images, num_function_calls,
      num_function_responses).
    """
    num_text = 0
    num_images = 0
    num_fc = 0
    num_fr = 0
    for content in self._conversation_history:
      if content.parts:
        for part in content.parts:
          if part.text:
            num_text += 1
          elif part.inline_data:
            num_images += 1
          elif part.function_call:
            num_fc += 1
          elif part.function_response:
            num_fr += 1
    return num_text, num_images, num_fc, num_fr

  def _serialize_conversation_history(self) -> list[dict[str, object]]:
    entries = []
    for content in self._conversation_history:
      parts_data = []
      if content.parts:
        for part in content.parts:
          part_dict: dict[str, object] = {}
          if part.text:
            part_dict["text"] = part.text
          elif part.inline_data:
            part_dict["inline_data"] = base64.b64encode(
                part.inline_data.data
            ).decode("ascii")
            part_dict["mime_type"] = part.inline_data.mime_type
          elif part.function_call:
            part_dict["function_call_name"] = part.function_call.name
            part_dict["function_call_args"] = str(part.function_call.args)
            if part.function_call.id:
              part_dict["function_call_id"] = part.function_call.id
          elif part.function_response:
            part_dict["function_response_name"] = part.function_response.name
            part_dict["function_response_response"] = str(
                part.function_response.response
            )
            if part.function_response.id:
              part_dict["function_response_id"] = part.function_response.id
          if hasattr(part, "thought") and part.thought:
            part_dict["thought"] = True
          if hasattr(part, "thought_signature") and part.thought_signature:
            part_dict["thought_signature"] = base64.b64encode(
                part.thought_signature
            ).decode("ascii")
          if part_dict:
            parts_data.append(part_dict)
      entries.append({"role": content.role, "parts": parts_data})
    return entries

  def _extract_tool_calls(
      self, content: types.Content
  ) -> list[types.FunctionCall]:
    """Extracts function calls from model content."""
    tool_calls = []
    if content.parts:
      for part in content.parts:
        if part.function_call:
          tool_calls.append(part.function_call)
    return tool_calls

  def _discard_old_images_in_history(self, latest_fr_content_idx: int) -> None:
    """Removes all images from conversation history except in the latest FR.

    Called when discard_images_after_turn is True. This aggressively strips
    images from ALL history entries except the one at latest_fr_content_idx,
    ensuring only the most recent observation (the latest function response)
    remains in context.

    In non-stitching mode, each image in the parts list is preceded by a
    camera-name text label (e.g. ["cam_left", <image>, "cam_right", <image>]).
    When removing an image, we also remove its preceding camera-name label
    to avoid leaving orphaned labels in the history. In stitching mode,
    camera names are burned into the stitched image itself, so text parts
    are never camera labels and should be preserved.

    Args:
      latest_fr_content_idx: Index into _conversation_history of the FR content
        entry whose images should be kept.
    """

    removed = 0
    for idx, content in enumerate(self._conversation_history):
      # Skip the latest FR entry — its images are kept.
      if idx == latest_fr_content_idx:
        continue
      if not content.parts:
        continue

      # Collect indices of parts to remove: images + their camera labels.
      indices_to_drop: set[int] = set()
      for part_idx, part in enumerate(content.parts):
        if part.inline_data:
          indices_to_drop.add(part_idx)
          # In non-stitching mode, check if the preceding part is a
          # camera-name text label that should also be removed.
          if not self._enable_image_stitching and part_idx > 0:
            preceding = content.parts[part_idx - 1]
            if (
                preceding.text
                and preceding.text in self._stream_name_to_camera_name.values()
                and not preceding.function_call
                and not preceding.function_response
                and not preceding.inline_data
            ):
              indices_to_drop.add(part_idx - 1)

      if indices_to_drop:
        removed += sum(
            1 for i in indices_to_drop if content.parts[i].inline_data
        )
        new_parts = [
            p for i, p in enumerate(content.parts) if i not in indices_to_drop
        ]
        if new_parts:
          self._conversation_history[idx] = types.Content(
              role=content.role, parts=new_parts
          )
        else:
          # If all parts were images/labels, replace with empty text to
          # preserve the Content entry (Gemini requires non-empty parts).
          self._conversation_history[idx] = types.Content(
              role=content.role, parts=[types.Part.from_text(text="")]
          )
    if removed:
      logging.info(
          "Discarded %d images from history (kept latest FR).", removed
      )

  def _trim_images_in_history(self) -> None:
    """Trims earliest images from history using a watermark strategy.

    Only prunes when the image count exceeds image_pruning_trigger_amount
    (high-water
    mark), then removes the oldest images down to _image_pruning_target_amount
    (low-water
    mark). This batch pruning preserves the prefill cache by keeping the
    conversation prefix stable between pruning events.

    In non-stitching mode, each image is preceded by a camera-name text part.
    Both the image and its label are removed together.
    """

    if self._image_pruning_trigger_amount <= 0:
      return

    # Build an ordered list of (content_idx, part_idx) for every image in
    # the conversation history. Oldest images appear first.
    image_locations: list[tuple[int, int]] = []
    for content_idx, content in enumerate(self._conversation_history):
      if content.parts:
        for part_idx, part in enumerate(content.parts):
          if part.inline_data:
            image_locations.append((content_idx, part_idx))

    # Only prune when we exceed the high-water mark.
    if len(image_locations) <= self._image_pruning_trigger_amount:
      return

    # Calculate how many to remove to reach the low-water mark.
    images_to_remove = len(image_locations) - self._image_pruning_target_amount
    if images_to_remove <= 0:
      return

    logging.warning(
        "\n##############################################\n"
        "# IMAGE PRUNING TRIGGERED\n"
        "# Images in context: %d (max: %d)\n"
        "# Removing %d images, keeping newest %d\n"
        "##############################################",
        len(image_locations),
        self._image_pruning_trigger_amount,
        images_to_remove,
        self._image_pruning_target_amount,
    )

    # Remove the oldest images (first N entries in the list).
    locations_to_remove = image_locations[:images_to_remove]

    # Process removals in reverse order so that earlier part indices in the
    # same Content remain valid when we remove later parts first.
    for content_idx, part_idx in reversed(locations_to_remove):
      content = self._conversation_history[content_idx]
      if content.parts:
        indices_to_drop = {part_idx}
        # In non-stitching mode, each image is preceded by a camera-name
        # text label (e.g. "cam_left"). Remove the label along with the
        # image to avoid orphaned text. In stitching mode, camera names
        # are burned into the image itself, so text parts are unrelated.
        if not self._enable_image_stitching and part_idx > 0:
          preceding = content.parts[part_idx - 1]
          if (
              preceding.text
              and preceding.text in self._stream_name_to_camera_name.values()
              and not preceding.function_call
              and not preceding.function_response
              and not preceding.inline_data
          ):
            indices_to_drop.add(part_idx - 1)
        new_parts = [
            p for i, p in enumerate(content.parts) if i not in indices_to_drop
        ]
        if new_parts:
          self._conversation_history[content_idx] = types.Content(
              role=content.role, parts=new_parts
          )
        else:
          self._conversation_history[content_idx] = types.Content(
              role=content.role, parts=[]
          )

  def _get_api_key(self, api_key: str | None) -> str:
    if api_key is None:
      api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
      raise ValueError(
          "No API key provided and GOOGLE_API_KEY environment variable not set."
      )
    return api_key

  def _get_system_instruction_text(self) -> str | None:
    if self._system_instruction:
      return (
          self._system_instruction
          if isinstance(self._system_instruction, str)
          else str(self._system_instruction)
      )
    return None

  def _format_tools_for_context(self, output_lines: list[str]) -> None:
    if self._tools:
      for tool in self._tools:
        if tool.function_declarations:
          for i, fn_decl in enumerate(tool.function_declarations):
            output_lines.append(f"  [{i}] {fn_decl.name}")
            if fn_decl.description:
              desc_preview = (
                  fn_decl.description[:80] + "..."
                  if len(fn_decl.description) > 80
                  else fn_decl.description
              )
              output_lines.append(f"      Description: {desc_preview}")
            if fn_decl.parameters and fn_decl.parameters.properties:
              params = list(fn_decl.parameters.properties.keys())
              output_lines.append(f"      Parameters: {params}")
    else:
      output_lines.append("  (No tools registered)")

  def _format_history_for_context(self, output_lines: list[str]) -> None:
    output_lines.append(f"Total entries: {len(self._conversation_history)}")
    output_lines.append("-" * 70)
    for i, content in enumerate(self._conversation_history):
      role = content.role or "NO_ROLE"
      parts_summary = []
      for part in content.parts or []:
        if part.text:
          preview = (
              part.text[:100] + "..." if len(part.text) > 100 else part.text
          )
          parts_summary.append(f"TEXT: {preview}")
        elif part.inline_data:
          size = len(part.inline_data.data) if part.inline_data.data else 0
          parts_summary.append(
              f"[IMAGE: {part.inline_data.mime_type}, {size} bytes]"
          )
        elif part.function_call:
          parts_summary.append(f"[FC: {part.function_call.name}]")
        elif part.function_response:
          parts_summary.append(f"[FR: {part.function_response.name}]")
        elif part.thought:
          parts_summary.append("[THOUGHT]")
        else:
          parts_summary.append("[UNKNOWN PART]")
      output_lines.append(
          f"  [{i}] role={role}, parts={len(content.parts or [])}"
      )
      for ps in parts_summary:
        output_lines.append(f"      {ps}")

  def clear_history(self) -> None:
    """Clears the conversation history."""
    self._conversation_history = []

  @property
  def conversation_history(self) -> tuple[types.Content, ...]:
    """Returns a copy of the conversation history."""
    return tuple(self._conversation_history)

  async def _maybe_publish_health_event(self, e: Exception | None) -> None:
    if e is None:
      await self._publish_health_status(GenAIOrchestratorHealth.NORMAL, None)
    elif "quota".lower() in str(e).lower():
      await self._publish_health_status(
          GenAIOrchestratorHealth.ERROR_QUOTA_EXCEEDED,
          e,
      )
    else:
      await self._publish_health_status(GenAIOrchestratorHealth.ERROR_OTHER, e)
