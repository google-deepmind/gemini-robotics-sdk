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

"""Abstract base class for non-streaming (turn-based) inference handlers.

This module provides the NonStreamingHandler abstract class that captures shared
infrastructure between UnaryGenAIHandler and EvergreenHandler —
both of which operate in a request-response pattern rather than continuous
streaming.
"""

import abc
import asyncio
from collections.abc import Sequence
import datetime
import enum
import threading

from absl import logging
from google.api_core import exceptions as api_exceptions
from google.genai import types
import grpc

from safari_sdk.agent.framework import config as framework_config
from safari_sdk.agent.framework import constants
from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.utils import image_processing


NUM_RETRIES = 3
RETRY_INTERVAL_SECONDS = 1.0

# Exceptions that are always retriable (no code check needed).
_ALWAYS_RETRIABLE_EXCEPTIONS = (
    asyncio.TimeoutError,
    ConnectionError,
    # GenAI/api_core exceptions that are explicitly transient.
    api_exceptions.ServiceUnavailable,  # 503: Server temporarily down.
    api_exceptions.InternalServerError,  # 500: Transient server error.
    api_exceptions.DeadlineExceeded,  # Timeout.
)

# Evergreen canonical codes that are retriable.
# - DEADLINE_EXCEEDED: Request took too long, likely transient.
# - RESOURCE_EXHAUSTED: Rate limiting (per-minute quota), retriable.
# Note: UNAVAILABLE is NOT included because Evergreen reports it for
# permanent failures like invalid model URLs or server shutdown.

# gRPC status codes that are retriable.
_RETRIABLE_GRPC_STATUS_CODES = frozenset([
    grpc.StatusCode.DEADLINE_EXCEEDED,
    grpc.StatusCode.RESOURCE_EXHAUSTED,
])


def is_retriable_exception(e: Exception) -> bool:
  """Determines if an exception is retriable.

  Args:
    e: The exception to check.

  Returns:
    True if the exception is transient and should be retried.
  """
  # Always-retriable exception types.
  if isinstance(e, _ALWAYS_RETRIABLE_EXCEPTIONS):
    return True

  # ResourceExhausted can be per-minute rate limiting (retriable) or
  # daily quota exceeded (not retriable). We retry it by default but
  # the loop will eventually exhaust retries for quota issues.
  if isinstance(e, api_exceptions.ResourceExhausted):
    return True

  # Evergreen client errors: check the canonical code.

  # Raw gRPC errors: check the status code.
  if isinstance(e, grpc.RpcError):
    try:
      return e.code() in _RETRIABLE_GRPC_STATUS_CODES  # pytype: disable=attribute-error
    except AttributeError:
      # If we can't get the code, don't retry.
      return False

  return False


class NonstreamingOrchestratorHealth(enum.Enum):
  """Abstract base class for tracking the health status of non-streaming APIs.

  Subclasses should define their own health status enum values.
  """


class NonStreamingHandler(abc.ABC):
  """Abstract base class for non-streaming (turn-based) inference handlers.

  This provides common infrastructure for handlers that use a request-response
  pattern (as opposed to continuous streaming). Both UnaryGenAIHandler
  and EvergreenHandler share this base.

  Subclasses must implement:
    - connect(): Establish connection / activate the handler.
    - disconnect(): Teardown connection / deactivate the handler.
    - _generate_response(): The handler-specific agentic loop.
    - _handle_context_command(): Handler-specific context dump formatting.
  """

  def __init__(
      self,
      bus: event_bus.EventBus,
      config: framework_config.AgentFrameworkConfig,
      camera_names: Sequence[str] | None = None,
      stream_name_to_camera_name: dict[str, str] | None = None,
      ignore_image_inputs: bool = False,
      temperature: float | None = None,
      max_output_tokens: int | None = None,
      thinking_budget: int | None = None,
      media_resolution: types.MediaResolution | None = None,
  ):
    self._bus = bus
    self._config = config
    self._api_health = None
    self._is_active = False
    self._camera_names = tuple(camera_names) if camera_names else ()
    self._stream_name_to_camera_name = (
        stream_name_to_camera_name
        if stream_name_to_camera_name is not None
        else ({camera_names[0]: ""} if camera_names else {})
    )

    if self._camera_names:
      invalid_cameras = set(self._stream_name_to_camera_name.keys()) - set(
          self._camera_names
      )
      if invalid_cameras:
        raise ValueError(
            f"Initial camera names {invalid_cameras} not found in available"
            f" cameras {self._camera_names}."
        )
    self._ignore_image_inputs = ignore_image_inputs
    self._pending_tool_results: asyncio.Queue[types.FunctionResponse] = (
        asyncio.Queue()
    )
    self._generate_lock = asyncio.Lock()
    self._temperature = temperature
    self._max_output_tokens = max_output_tokens
    self._thinking_budget = thinking_budget
    self._media_resolution = media_resolution

    self._image_buffer: list[tuple[datetime.datetime, str, types.Blob]] = []
    self._last_image_input_time: dict[str, datetime.datetime] = {}
    self._image_pruning_trigger_amount = (
        config.non_streaming_image_pruning_trigger_amount
    )
    prune_to = config.non_streaming_image_pruning_target_amount
    self._image_pruning_target_amount = (
        prune_to
        if prune_to > 0
        else max(1, self._image_pruning_trigger_amount // 2)
    )
    # Image retention policy — these three flags are independent:
    #   _discard_images_after_turn: when True, clears ALL buffered images
    #     after each inference turn (user message or FC-FR cycle). Treats
    #     visual context as ephemeral. When False, images accumulate.
    #   _fr_latest_image_only: when True, only the latest image per camera
    #     stream is attached to a Function Response (reduces tokens during
    #     long-running tool execution).
    #   _user_turn_latest_image_only: when True, only the latest image per
    #     camera stream is included in the user turn message (reduces tokens
    #     for the initial observation).
    self._discard_images_after_turn = (
        config.non_streaming_discard_images_after_turn
    )
    self._fr_latest_image_only = config.non_streaming_fr_latest_image_only
    self._user_turn_latest_image_only = (
        config.non_streaming_user_turn_latest_image_only
    )
    self._include_stream_names = config.non_streaming_include_stream_names
    self._enable_image_stitching = config.enable_image_stitching
    self._show_camera_name_in_stitched_image = (
        config.show_camera_name_in_stitched_image
    )
    self._stitch_interval_seconds = (
        config.non_streaming_image_buffering_interval_seconds
    )

    # Background stitching thread infrastructure. When image stitching is
    # enabled, a daemon thread (_stitch_thread) runs _stitch_loop() at
    # _stitch_interval_seconds. It reads per-camera blobs from
    # _latest_images, composites them into a single grid frame, and
    # appends the result to _stitched_frames. All access to these
    # structures is serialized through _stitch_lock. _loop is captured
    # from the main asyncio loop so the stitch thread can publish
    # REAL_TIME_IMAGE_SENT events back to the event bus.
    self._latest_images: dict[str, types.Blob] = {}
    self._stitched_frames: list[tuple[datetime.datetime, types.Part]] = []
    self._stitch_lock = threading.Lock()
    self._stitch_thread: threading.Thread | None = None
    self._stitch_stop_event = threading.Event()
    self._loop: asyncio.AbstractEventLoop | None = None
    self._tool_result_timeout: float = (
        config.non_streaming_tool_result_timeout_seconds
    )

  async def connect(self) -> None:
    ...

  @abc.abstractmethod
  async def disconnect(self) -> None:
    ...

  async def _retry_wrapper(
      self,
      generate_fn,
      *args,
      **kwargs,
  ):
    """Wraps a generate function with retry logic.

    Args:
      generate_fn: The async generate function to call.
      *args: Positional arguments to pass to generate_fn.
      **kwargs: Keyword arguments to pass to generate_fn.

    Returns:
      The generated response.

    Raises:
      The last exception if all retries are exhausted.
    """
    last_exception = None

    for attempt in range(NUM_RETRIES):
      try:
        return await generate_fn(*args, **kwargs)
      except Exception as e:  # pylint: disable=broad-exception-caught
        if not is_retriable_exception(e):
          logging.error("Generate failed with non-retriable error: %s", e)
          raise
        last_exception = e
        logging.warning(
            "Generate attempt %d/%d failed with retriable error: %s. "
            "Retrying in %.1fs...",
            attempt + 1,
            NUM_RETRIES,
            type(e).__name__,
            RETRY_INTERVAL_SECONDS,
        )
        if attempt < NUM_RETRIES - 1:
          await asyncio.sleep(RETRY_INTERVAL_SECONDS)

    raise last_exception

  def _clear_image_state(self) -> None:
    self._image_buffer = []
    self._last_image_input_time = {}
    with self._stitch_lock:
      self._latest_images = {}
      self._stitched_frames = []

  def _start_image_stitching(self) -> None:
    if self._enable_image_stitching:
      self._stitch_stop_event.clear()
      self._stitch_thread = threading.Thread(
          target=self._stitch_loop, daemon=True
      )
      self._stitch_thread.start()
      logging.info("Started background stitching thread.")

  def _stop_image_stitching(self) -> None:
    if self._stitch_thread and self._stitch_thread.is_alive():
      self._stitch_stop_event.set()
      self._stitch_thread.join(timeout=2.0)
      logging.info("Stopped background stitching thread.")
    self._stitch_thread = None

  async def _publish_session_metadata(self) -> None:
    """Publishes session config to the bus as a LOG_SESSION_METADATA event."""

    session_config = {
        "system_instruction": getattr(self, "_system_instruction", None),
        "model_name": getattr(self, "_model_name", None),
        "tools": (
            str(getattr(self, "_tools", None))
            if getattr(self, "_tools", None)
            else None
        ),
        "temperature": self._temperature,
        "max_output_tokens": self._max_output_tokens,
        "media_resolution": (
            str(self._media_resolution) if self._media_resolution else None
        ),
        "thinking_config": str(getattr(self, "_thinking_config", None)),
    }
    logging.info("Publishing session metadata to event bus.")
    await self._bus.publish(
        event=event_bus.Event(
            type=event_bus.EventType.LOG_SESSION_METADATA,
            source=event_bus.EventSource.MAIN_AGENT,
            data=(
                f"Non-streaming API Config (also in metadata): {session_config}"
            ),
            metadata=session_config,
        )
    )

  def register_event_subscribers(self) -> None:
    """Registers subscribers for events from the event bus."""
    self._bus.subscribe(
        event_types=[event_bus.EventType.MODEL_TEXT_INPUT],
        handler=self._handle_text_in_event,
    )
    self._bus.subscribe(
        event_types=[event_bus.EventType.TOOL_RESULT],
        handler=self._handle_tool_result_event,
    )
    self._bus.subscribe(
        event_types=[event_bus.EventType.DEBUG],
        handler=self._handle_debug_event,
    )
    if not self._ignore_image_inputs:
      self._bus.subscribe(
          event_types=[event_bus.EventType.MODEL_IMAGE_INPUT],
          handler=self._handle_image_in_event,
      )
    self._register_extra_subscribers()

  def _register_extra_subscribers(self) -> None:
    pass

  async def _handle_debug_event(self, event: event_bus.Event) -> None:
    """Handles debug events from the event bus."""
    logging.debug(
        "Debug event received: command=%s, is_active=%s",
        event.metadata.get("command"),
        self._is_active,
    )
    if not self._is_active:
      return
    try:
      if event.metadata.get("command") == "context_dump":
        await self._handle_context_command()
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.error("Error handling debug event: %s", e, exc_info=True)

  async def _handle_context_command(self) -> None:
    """Handles the /context debug command by dumping full context info."""
    output_lines = []
    output_lines.append("\n" + "=" * 70)
    output_lines.append("FULL CONTEXT WINDOW DUMP")
    output_lines.append("=" * 70)

    output_lines.append("\n📋 SYSTEM INSTRUCTION:")
    output_lines.append("-" * 70)
    si = self._get_system_instruction_text()
    if si:
      si_preview = si[:500] + "..." if len(si) > 500 else si
      output_lines.append(f"  {si_preview}")
    else:
      output_lines.append("  (None)")
    output_lines.append("-" * 70)

    output_lines.append("\n🔧 REGISTERED TOOLS:")
    output_lines.append("-" * 70)
    self._format_tools_for_context(output_lines)
    output_lines.append("-" * 70)

    output_lines.append("\n💬 CONVERSATION HISTORY:")
    self._format_history_for_context(output_lines)
    output_lines.append("-" * 70)

    output_lines.append("\n📸 BUFFERED IMAGES:")
    output_lines.append("-" * 70)
    self._format_images_for_context(output_lines)
    output_lines.append("-" * 70)

    output_lines.append("=" * 70)
    output_lines.append("END OF CONTEXT DUMP")
    output_lines.append("=" * 70 + "\n")

    context_output = "\n".join(output_lines)
    print(context_output)

    await self._publish_event(
        event_bus.EventType.MODEL_TURN,
        types.Content(
            role="model", parts=[types.Part.from_text(text=context_output)]
        ),
    )
    await self._publish_event(
        event_bus.EventType.MODEL_TURN_COMPLETE,
        True,
    )

  def _get_system_instruction_text(self) -> str | None:
    return None

  def _format_tools_for_context(self, output_lines: list[str]) -> None:
    output_lines.append("  (No tools registered)")

  @abc.abstractmethod
  def _format_history_for_context(self, output_lines: list[str]) -> None:
    ...

  def _format_images_for_context(self, output_lines: list[str]) -> None:
    if self._image_buffer:
      output_lines.append(f"  {len(self._image_buffer)} buffered image(s)")
      for ts, stream_name, blob in self._image_buffer[-5:]:
        output_lines.append(
            f"  {stream_name}: {len(blob.data) if blob.data else 0} bytes"
            f" @ {ts}"
        )
    else:
      output_lines.append("  (No buffered images)")

  def _log_conversation_history(self) -> None:
    lines: list[str] = []
    self._format_history_for_context(lines)
    logging.info("=== Conversation History ===")
    for line in lines:
      logging.info(line)
    logging.info("=== End History ===")

  async def _handle_text_in_event(self, event: event_bus.Event) -> None:
    """Handles text events from the event bus."""
    logging.info("Received text input event: %s", event.data)
    if not self._is_active:
      logging.info("Handler is not active, ignoring text input.")
      return
    try:
      text = event.data
      logging.info("Processing text input: %s", text)
      await self._generate_response(text)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.error("Error handling text input: %s", e, exc_info=True)
      await self._maybe_publish_health_event(e)

  async def _handle_image_in_event(self, event: event_bus.Event) -> None:
    """Handles image events from the event bus by buffering them."""
    if not self._is_active:
      return

    if constants.STREAM_NAME_METADATA_KEY not in event.metadata:
      logging.warning(
          "Image event metadata should contain %s",
          constants.STREAM_NAME_METADATA_KEY,
      )
      return

    stream_name = event.metadata[constants.STREAM_NAME_METADATA_KEY]
    if stream_name not in self._stream_name_to_camera_name.keys():
      return

    try:
      blob = types.Blob(
          display_name=self._stream_name_to_camera_name[stream_name],
          data=event.data,
          mime_type="image/jpeg",
      )

      if self._enable_image_stitching:
        with self._stitch_lock:
          self._latest_images[stream_name] = blob
      else:
        last_time = self._last_image_input_time.get(stream_name, None)
        delta = datetime.timedelta(
            seconds=self._config.non_streaming_image_buffering_interval_seconds
        )
        if last_time and event.timestamp - last_time < delta:
          return
        self._last_image_input_time[stream_name] = event.timestamp
        self._image_buffer.append((event.timestamp, stream_name, blob))

        await self._bus.publish(
            event=event_bus.Event(
                type=event_bus.EventType.REAL_TIME_IMAGE_SENT,
                source=event_bus.EventSource.ROBOT,
                data=event.data,
            )
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.info("Error buffering image: %s", e)

  async def _handle_tool_result_event(self, event: event_bus.Event) -> None:
    """Handles tool result events from the event bus."""
    if not self._is_active:
      logging.info("Handler is not active, ignoring tool result.")
      return
    try:
      tool_response = event.data
      if isinstance(tool_response, types.LiveClientToolResponse):
        for fn_response in tool_response.function_responses:
          await self._pending_tool_results.put(fn_response)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.error("Error handling tool result: %s", e)

  @abc.abstractmethod
  async def _generate_response(self, user_text: str) -> None:
    ...

  def _collect_images_for_fr(
      self,
      fc_timestamp: datetime.datetime,
      fr_timestamp: datetime.datetime,
  ) -> tuple[list[types.Part], int]:
    """Collects images captured during tool execution (FC to FR window).

    Drains images from the buffer that fall within [fc_timestamp, fr_timestamp].
    Handles both stitching and non-stitching modes. Updates the buffer according
    to the discard_images_after_turn policy.

    Args:
      fc_timestamp: Wall-clock time when the function call was issued.
      fr_timestamp: Wall-clock time when the function response was received.

    Returns:
      A tuple of (image_parts, count) where image_parts is a list of Part
      objects to prepend to the function response, and count is the number
      of images collected.
    """
    image_parts: list[types.Part] = []

    if self._enable_image_stitching:
      stitched_frames_for_fr: list[types.Part] = []
      remaining_stitched_frames: list[tuple[datetime.datetime, types.Part]] = []
      with self._stitch_lock:
        for ts, frame in self._stitched_frames:
          if fc_timestamp <= ts <= fr_timestamp:
            stitched_frames_for_fr.append(frame)
          else:
            remaining_stitched_frames.append((ts, frame))

        if self._discard_images_after_turn:
          self._stitched_frames = []
        else:
          self._stitched_frames = remaining_stitched_frames

      image_parts.extend(stitched_frames_for_fr)
      images_count = len(stitched_frames_for_fr)

      if self._fr_latest_image_only and images_count > 1:
        image_parts = [stitched_frames_for_fr[-1]]
        images_count = 1

      logging.info(
          "FC to FR window duration: %.3fs. Collected %d stitched frames.",
          (fr_timestamp - fc_timestamp).total_seconds(),
          images_count,
      )
    else:
      images_for_fr: list[tuple[datetime.datetime, str, types.Blob]] = []
      remaining_images: list[tuple[datetime.datetime, str, types.Blob]] = []
      for img_ts, stream_name, blob in self._image_buffer:
        if fc_timestamp <= img_ts <= fr_timestamp:
          images_for_fr.append((img_ts, stream_name, blob))
        elif img_ts > fr_timestamp or not self._discard_images_after_turn:
          remaining_images.append((img_ts, stream_name, blob))

      self._image_buffer = remaining_images

      if self._fr_latest_image_only and len(images_for_fr) > 1:
        latest_per_stream: dict[
            str, tuple[datetime.datetime, str, types.Blob]
        ] = {}
        for img_ts, sname, blob in images_for_fr:
          if (
              sname not in latest_per_stream
              or img_ts > latest_per_stream[sname][0]
          ):
            latest_per_stream[sname] = (img_ts, sname, blob)
        images_for_fr = list(latest_per_stream.values())

      for _, _, blob in images_for_fr:
        if self._include_stream_names and blob.display_name:
          image_parts.append(types.Part.from_text(text=blob.display_name))
        image_parts.append(
            types.Part.from_bytes(data=blob.data, mime_type="image/jpeg")
        )
      images_count = len(images_for_fr)

    if images_count > 0:
      logging.info(
          "Attached %d %s to function response",
          images_count,
          "stitched frames" if self._enable_image_stitching else "images",
      )

    return image_parts, images_count

  def _stitch_images(self, img_dict: dict[str, types.Blob]) -> types.Part:
    """Stitches multiple camera images into a single image Part.

    Args:
      img_dict: Dictionary mapping stream names to image Blobs.

    Returns:
      A Part containing the stitched image.
    """
    images = {name: blob.data for name, blob in img_dict.items()}
    labels = {
        name: blob.display_name if blob.display_name else name
        for name, blob in img_dict.items()
    }
    stitched_bytes = image_processing.stitch_images(
        images, labels, show_labels=self._show_camera_name_in_stitched_image
    )
    return types.Part.from_bytes(data=stitched_bytes, mime_type="image/jpeg")

  def _stitch_loop(self) -> None:
    """Background thread loop that stitches images at a fixed rate.

    Waits for all cameras to report before stitching. Runs until
    _stitch_stop_event is set.
    """
    expected_cameras = set(self._stream_name_to_camera_name.keys())
    while not self._stitch_stop_event.wait(
        timeout=self._stitch_interval_seconds
    ):
      with self._stitch_lock:
        if not self._latest_images:
          continue
        if set(self._latest_images.keys()) != expected_cameras:
          logging.debug(
              "Waiting for all cameras: have %s, need %s",
              set(self._latest_images.keys()),
              expected_cameras,
          )
          continue
        stitch_start = datetime.datetime.now()
        stitched = self._stitch_images(dict(self._latest_images))
        stitch_duration = (
            datetime.datetime.now() - stitch_start
        ).total_seconds()
        self._stitched_frames.append(
            (datetime.datetime.now(tz=datetime.timezone.utc), stitched)
        )
        stitched_data = (
            stitched.inline_data.data if stitched.inline_data else b""
        )
        asyncio.run_coroutine_threadsafe(
            self._bus.publish(
                event=event_bus.Event(
                    type=event_bus.EventType.REAL_TIME_IMAGE_SENT,
                    source=event_bus.EventSource.ROBOT,
                    data=stitched_data,
                )
            ),
            self._loop,
        )
        if len(self._stitched_frames) > self._image_pruning_trigger_amount:
          self._stitched_frames = self._stitched_frames[
              -self._image_pruning_target_amount :
          ]
      logging.debug(
          "Stitched frame from %d cameras in %.3fs, %d frames buffered",
          len(expected_cameras),
          stitch_duration,
          len(self._stitched_frames),
      )

  async def _wait_for_tool_results(
      self, expected_count: int
  ) -> list[types.FunctionResponse]:
    """Waits for the expected number of tool results."""
    results = []
    timeout = self._tool_result_timeout
    for _ in range(expected_count):
      try:
        result = await asyncio.wait_for(
            self._pending_tool_results.get(), timeout=timeout
        )
        results.append(result)
      except asyncio.TimeoutError:
        logging.error("Timeout waiting for tool result.")
        break
    return results

  async def _publish_event(
      self,
      event_type: event_bus.EventType,
      data: object,
      source: event_bus.EventSource = event_bus.EventSource.MAIN_AGENT,
      metadata: dict[str, object] | None = None,
  ) -> None:
    await self._bus.publish(
        event=event_bus.Event(
            type=event_type, source=source, data=data, metadata=metadata or {}
        )
    )

  async def _publish_health_status(
      self, status: NonstreamingOrchestratorHealth, exception: Exception | None
  ) -> None:
    if self._api_health != status:
      self._api_health = status
      await self._publish_event(
          event_bus.EventType.ORCHESTRATOR_CLIENT_HEALTH,
          {
              "health_status": status.value,
              "exception_message": str(exception) if exception else None,
          },
      )

  @abc.abstractmethod
  async def _maybe_publish_health_event(self, e: Exception | None) -> None:
    ...

  @abc.abstractmethod
  def clear_history(self) -> None:
    ...

  @property
  @abc.abstractmethod
  def conversation_history(self) -> object:
    ...
