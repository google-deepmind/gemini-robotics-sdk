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

"""A handler for Gemini Live API."""

import asyncio
from collections.abc import AsyncIterator, Sequence
import datetime
import enum
import os
import time
from typing import Any

from absl import logging
from google import genai
from google.genai import types

from safari_sdk.agent.framework import config as framework_config
from safari_sdk.agent.framework import constants
from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.utils import image_processing


_DEFAULT_MODEL_NAME = "models/gemini-2.0-flash-live-001"


@enum.unique
class LiveAPIOrchestratorHealth(enum.Enum):
  """Class for tracking the type of the external controller endpoint."""

  NORMAL = "NORMAL"
  # live API returned 1011 quota exceeded error.
  ERROR_QUOTA_EXCEEDED = "ERROR_QUOTA_EXCEEDED"
  ERROR_INTERNAL_1011 = "ERROR_INTERNAL_1011"
  # live API returned 1007 internal error.
  ERROR_INTERNAL_1007 = "ERROR_INTERNAL_1007"
  # live API returned other error.
  ERROR_OTHER = "ERROR_OTHER"


def _prepare_image_parts(
    img_dict: dict[str, types.Blob],
    enable_stitching: bool = False,
    show_labels: bool = True,
) -> list[types.Part]:
  """Prepares a list of image parts for sending to the Agent.

  Args:
    img_dict: Dictionary mapping camera names to image Blobs.
    enable_stitching: If True, stitches all camera images into a single
      composite grid image. If False, returns individual image parts.
    show_labels: If True and stitching is enabled, displays camera name labels
      on each cell. Has no effect when stitching is disabled.

  Returns:
    A list of Parts. If stitching is enabled, returns a single Part with the
    stitched image. Otherwise, returns one Part per camera image.
  """
  if not img_dict:
    return []

  if enable_stitching:
    images = {name: blob.data for name, blob in img_dict.items()}
    labels = {
        name: blob.display_name if blob.display_name else name
        for name, blob in img_dict.items()
    }
    stitched_bytes = image_processing.stitch_images(
        images, labels, show_labels=show_labels
    )
    return [types.Part.from_bytes(data=stitched_bytes, mime_type="image/jpeg")]

  return [
      types.Part.from_bytes(data=blob.data, mime_type=blob.mime_type)
      for blob in img_dict.values()
  ]


def _get_attr_or_item(obj: object, key: str) -> object | None:
  """Safely gets an attribute or item from an object or dict."""
  if isinstance(obj, dict):
    return obj.get(key)
  return getattr(obj, key, None)


class GeminiLiveAPIHandler:
  """Handler for Gemini Live API."""

  def __init__(
      self,
      bus: event_bus.EventBus,
      config: framework_config.AgentFrameworkConfig,
      live_config: types.LiveConnectConfigDict | types.LiveConnectConfig,
      camera_names: Sequence[str],
      stream_name_to_camera_name: dict[str, str] | None = None,
      http_options: dict[str, str] | None = None,
      ignore_image_inputs: bool = False,
  ):
    """Initializes the handler.

    Args:
      bus: The event bus to exchange events with.
      config: The agent framework configuration object.
      live_config: The live API configuration. This will usually be scene
        specific, and created by an "agent" module connecting handlers and
        embodiments.
      camera_names: The names of the camera streams available from the
        embodiment.
      stream_name_to_camera_name: Mapping from image stream (endpoint) names to
        camera names. It specifies which camera streams are sent to the
        orchestrator model as well as the names with which to prepend the
        images. If None, the first camera is used and an empty string will be
        prepended. Note that prepending of the camera name is only supported
        under the following conditions:
            `update_vision_after_fr=True` AND
            `turn_coverage=TURN_INCLUDES_ONLY_ACTIVITY`
      http_options: HTTP options to use for the client.
      ignore_image_inputs: Whether to ignore image inputs. In this mode, the
        handler will not send any images to the model.
    """
    api_key = self._get_api_key(config.api_key)
    self._config = config
    self._client = genai.Client(
        api_key=api_key,
        http_options=http_options,
    )
    self._live_config = live_config
    self._bus = bus
    self._session: genai.live.AsyncSession | None = None
    self._is_active = False
    self._live_api_health = LiveAPIOrchestratorHealth.NORMAL
    self._gemini_live_api_task: asyncio.Task[Any] | None = None
    self._receive_responses_task: asyncio.Task[Any] | None = None
    self._non_streaming_inputs_queue = asyncio.Queue[
        str | types.LiveClientToolResponse | None
    ]()
    self._send_non_streaming_inputs_task: asyncio.Task[Any] | None = None
    self._streaming_input_queue = asyncio.Queue[types.Blob | None](maxsize=5)
    self._send_streaming_inputs_task: asyncio.Task[Any] | None = None
    self._last_image_input_time = {}
    self._last_image_received = {}
    self._ignore_image_inputs = ignore_image_inputs
    self._camera_names = camera_names
    self._stream_name_to_camera_name = (
        stream_name_to_camera_name
        if stream_name_to_camera_name is not None
        else {camera_names[0]: ""}
    )
    invalid_cameras = (
        set(self._stream_name_to_camera_name.keys()) - set(camera_names)
    )
    if invalid_cameras:
      raise ValueError(
          f"Initial camera names {invalid_cameras} not found in available"
          f" cameras {camera_names}."
      )
    self._turn_coverage = self._get_turn_coverage()
    self._session_resumption_handle: str | None = None
    self._go_away_task: asyncio.Task[Any] | None = None

  async def connect(self):
    """Establishes connection to Gemini Live API."""
    if self._session and self._is_active:
      await self.disconnect()
    logging.info("Connecting to Gemini Live API...")
    self._gemini_live_api_task = asyncio.create_task(
        self._create_and_manage_session()
    )
    # Wait up to 3 seconds for session to be established
    timeout = 3.0
    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout:
      if self._session and self._is_active:
        break
      await asyncio.sleep(0.1)
    if self._session and self._is_active:
      logging.info(
          "Connected to Gemini Live API (session"
          " established in background task)."
      )
    else:
      logging.info(
          "Failed to connect to Gemini Live API (session"
          " not established in background task)."
      )

  async def disconnect(self):
    """Disconnects from Gemini Live API."""
    logging.info("Disconnecting from Gemini Live API.")
    if (
        not self._is_active
        and not self._gemini_live_api_task
        and not self._send_streaming_inputs_task
    ):
      logging.info("Already disconnected from Gemini Live API.")
      return
    self._is_active = False

    # Send shutdown signals to queues
    if self._non_streaming_inputs_queue:
      await self._non_streaming_inputs_queue.put(None)
    if self._streaming_input_queue:
      await self._streaming_input_queue.put(None)

    # Cancel and await the main session task
    if self._gemini_live_api_task and not self._gemini_live_api_task.done():
      self._gemini_live_api_task.cancel()
      try:
        await self._gemini_live_api_task
      except asyncio.CancelledError:
        pass
      finally:
        self._gemini_live_api_task = None

    if self._session is not None:
      self._session = None

    # Cancel any pending GO_AWAY reconnect task
    if self._go_away_task and not self._go_away_task.done():
      self._go_away_task.cancel()
      try:
        await self._go_away_task
      except asyncio.CancelledError:
        pass
      finally:
        self._go_away_task = None

    logging.info("Disconnected from Gemini Live API.")

  async def send_non_streaming_inputs(self):
    """Send non-streaming inputs to the Gemini Live API."""
    logging.info("Starting non-streaming input sender task.")
    while self._is_active and self._session:
      try:
        next_input = await self._non_streaming_inputs_queue.get()
        if next_input is None:
          break
        if isinstance(next_input, str):
          await self._send_text(next_input)
        elif isinstance(next_input, types.LiveClientToolResponse):
          await self._send_tool_response(next_input)
        self._non_streaming_inputs_queue.task_done()
      except asyncio.CancelledError:
        break
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.info("Error sending conversational input: %s", e)
        await self._maybe_publish_live_api_health_event(e)

    logging.info("Stopped non-streaming input sender task.")

  async def send_streaming_inputs(self):
    """Send streaming inputs to the Gemini Live API."""
    logging.info("Starting streaming input sender task.")
    while self._is_active and self._session:
      try:
        realtime_input = await self._streaming_input_queue.get()
        if realtime_input is None:
          break
        if isinstance(realtime_input, types.Blob):
          try:
            assert realtime_input.mime_type is not None
            if realtime_input.mime_type.startswith("image/"):
              await self._session.send_realtime_input(video=realtime_input)
              await self._bus.publish(
                  event=event_bus.Event(
                      type=event_bus.EventType.REAL_TIME_IMAGE_SENT,
                      source=event_bus.EventSource.ROBOT,
                      data=realtime_input.data,
                  )
              )
            elif realtime_input.mime_type.startswith("audio/"):
              await self._session.send_realtime_input(audio=realtime_input)
              await self._bus.publish(
                  event=event_bus.Event(
                      type=event_bus.EventType.REAL_TIME_AUDIO_SENT,
                      source=event_bus.EventSource.ROBOT,
                      data=realtime_input.data,
                  )
              )
          except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error("Failed to send streaming input because %s", e)
        else:
          logging.warning(
              "Unsupported input type in real-time queue: %s",
              type(realtime_input).__name__,
          )
        self._streaming_input_queue.task_done()
      except asyncio.CancelledError:
        break
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error("Failed to send streaming input because %s", e)
        await self._maybe_publish_live_api_health_event(e)

  def register_event_subscribers(self):
    """Registers subscribers for events from the event bus."""
    # Listen for text input from the user.
    self._bus.subscribe(
        event_types=[event_bus.EventType.MODEL_TEXT_INPUT],
        handler=self._handle_text_in_event,
    )
    # Listen for tool call responses.
    self._bus.subscribe(
        event_types=[event_bus.EventType.TOOL_RESULT],
        handler=self._handle_tool_result_event,
    )
    # Listen for image frames to be sent as real-time inputs.
    if not self._ignore_image_inputs:
      self._bus.subscribe(
          event_types=[event_bus.EventType.MODEL_IMAGE_INPUT],
          handler=self._handle_image_in_event,
      )
    # Listen for audio frames to be sent as real-time inputs.
    self._bus.subscribe(
        event_types=[event_bus.EventType.MODEL_AUDIO_INPUT],
        handler=self._handle_audio_in_event,
    )
    # Listen for GO_AWAY events from external controller (for testing).
    self._bus.subscribe(
        event_types=[event_bus.EventType.GO_AWAY],
        handler=self._handle_go_away_event,
    )

  async def _handle_text_in_event(self, event: event_bus.Event) -> None:
    """Handles text events from the event bus."""
    if not self._ensure_active_session():
      return
    try:
      await self._non_streaming_inputs_queue.put(event.data)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.error("Error queuing text: %s", e)

  async def _handle_tool_result_event(self, event: event_bus.Event) -> None:
    """Handles tool result events from the event bus."""
    if not self._ensure_active_session():
      return
    try:
      await self._non_streaming_inputs_queue.put(event.data)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.error("Error queuing function response: %s", e)

  async def _handle_image_in_event(self, event: event_bus.Event) -> None:
    """Handles image events from the event bus."""
    if not self._ensure_active_session():
      return

    if constants.STREAM_NAME_METADATA_KEY not in event.metadata:
      logging.warning(
          "image event metadata should contain %s",
          constants.STREAM_NAME_METADATA_KEY,
      )
      return

    stream_name = event.metadata[constants.STREAM_NAME_METADATA_KEY]
    if stream_name not in self._stream_name_to_camera_name.keys():
      logging.log_every_n_seconds(
          logging.DEBUG,
          "image event stream name %s not in current camera names",
          10,
          stream_name,
      )
      return

    try:
      if self._turn_coverage == types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY:
        self._last_image_received[stream_name] = types.Blob(
            display_name=self._stream_name_to_camera_name[stream_name],
            data=event.data,
            mime_type="image/jpeg",
        )
      elif self._config.enable_image_stitching:
        # When stitching is enabled, accumulate images from all cameras
        self._last_image_received[stream_name] = types.Blob(
            display_name=self._stream_name_to_camera_name[stream_name],
            data=event.data,
            mime_type="image/jpeg",
        )
        # Rate limit at the stitching level using a special key
        last_time = self._last_image_input_time.get("__stitched__", None)
        delta = datetime.timedelta(
            seconds=self._config.gemini_live_image_streaming_interval_seconds
        )
        if last_time and event.timestamp - last_time < delta:
          return
        # Only send if we have at least one image
        if not self._last_image_received:
          return
        self._last_image_input_time["__stitched__"] = event.timestamp
        # Stitch all available camera images
        stitched_bytes = image_processing.stitch_images(
            {
                name: blob.data
                for name, blob in self._last_image_received.items()
            },
            {
                name: blob.display_name or name
                for name, blob in self._last_image_received.items()
            },
            show_labels=self._config.show_camera_name_in_stitched_image,
        )
        realtime_input = types.Blob(
            data=stitched_bytes,
            mime_type="image/jpeg",
        )
        if self._streaming_input_queue.full():
          try:
            self._streaming_input_queue.get_nowait()
          except asyncio.QueueEmpty:
            pass
        await self._streaming_input_queue.put(realtime_input)
      else:
        # We check the timestamp of the current image frame compared to the
        # previous one and skip sending it if the frequency is exceeded.
        last_time = self._last_image_input_time.get(stream_name, None)
        delta = datetime.timedelta(
            seconds=self._config.gemini_live_image_streaming_interval_seconds
        )
        if last_time and event.timestamp - last_time < delta:
          return
        self._last_image_input_time[stream_name] = event.timestamp

        # Convert the image data to a Blob object for the realtime input queue.
        realtime_input = types.Blob(
            data=event.data,
            mime_type="image/jpeg",
        )
        # If the input queue is full, remove the oldest item to keep low latency
        if self._streaming_input_queue.full():
          try:
            self._streaming_input_queue.get_nowait()
          except asyncio.QueueEmpty:
            pass
        # Add the Blob to the realtime input queue.
        await self._streaming_input_queue.put(realtime_input)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.info("Error queuing image: %s", e)

  async def _handle_audio_in_event(self, event: event_bus.Event) -> None:
    """Handles audio events from the event bus."""
    if not self._ensure_active_session():
      return
    try:
      realtime_input = types.Blob(
          data=event.data,
          mime_type=f"audio/pcm;rate={constants.DEFAULT_AUDIO_INPUT_RATE}",
      )
      # If the input queue is full, remove the oldest item to keep low latency
      if self._streaming_input_queue.full():
        try:
          self._streaming_input_queue.get_nowait()
        except asyncio.QueueEmpty:
          pass
      await self._streaming_input_queue.put(realtime_input)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception("Error queuing audio: %s", e)

  async def _handle_go_away_event(self, event: event_bus.Event) -> None:
    """Handles GO_AWAY events from external sources (e.g., external controller).

    This allows testing the session resumption feature without waiting for an
    actual GO_AWAY from the Live API server.

    Args:
      event: The GO_AWAY event from the event bus.
    """

    if not self._ensure_active_session():
      return
    logging.info(
        "Received GO_AWAY event from event bus (source: %s)", event.source
    )
    if self._go_away_task is None or self._go_away_task.done():
      # Create a mock LiveServerGoAway object for the handler
      go_away = types.LiveServerGoAway(time_left="simulated")
      self._go_away_task = asyncio.create_task(
          self._handle_go_away_and_reconnect(go_away)
      )

  def _get_api_key(self, api_key: str | None) -> str:
    if api_key is None:
      api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
      raise ValueError(
          "No API key provided and GOOGLE_API_KEY environment variable not set."
      )
    return api_key

  def _ensure_active_session(self) -> bool:
    if not self._session or not self._is_active:
      logging.info(
          "Session is not connected or not active."
          " self._session: %s, self._is_active: %s",
          self._session,
          self._is_active,
      )
      return False
    return True

  async def _publish_event(
      self,
      event_type: event_bus.EventType,
      data: object,
      source: event_bus.EventSource = event_bus.EventSource.MAIN_AGENT,
  ) -> None:
    await self._bus.publish(
        event=event_bus.Event(type=event_type, source=source, data=data)
    )

  async def _publish_health_status(
      self, status: LiveAPIOrchestratorHealth, exception: Exception | None
  ) -> None:
    if self._live_api_health != status:
      self._live_api_health = status
      await self._publish_event(
          event_bus.EventType.ORCHESTRATOR_CLIENT_HEALTH,
          {
              "health_status": status.value,
              "exception_message": str(exception) if exception else None,
          },
      )

  def _clear_queue(self, queue: asyncio.Queue[Any]) -> None:
    while not queue.empty():
      try:
        queue.get_nowait()
      except asyncio.QueueEmpty:
        break

  def _get_turn_coverage(self):
    """Returns the turn coverage for the current session."""
    default_turn_coverage = types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY
    realtime_config = _get_attr_or_item(
        self._live_config, "realtime_input_config"
    )
    if not realtime_config:
      logging.warning("`realtime_input_config` not found in `live_config`")
      return default_turn_coverage
    turn_coverage = _get_attr_or_item(realtime_config, "turn_coverage")
    if not turn_coverage:
      logging.warning("`turn_coverage` not found in `realtime_input_config`")
      return default_turn_coverage
    return turn_coverage

  async def _maybe_publish_live_api_health_event(self, e: Exception | None):
    """Publishes the Live API health event to the event bus."""
    if e is None:
      await self._publish_health_status(
          LiveAPIOrchestratorHealth.NORMAL, None
      )
    elif "quota".lower() in str(e):
      await self._publish_health_status(
          LiveAPIOrchestratorHealth.ERROR_QUOTA_EXCEEDED, e
      )
    elif "1011" in str(e):
      await self._publish_health_status(
          LiveAPIOrchestratorHealth.ERROR_INTERNAL_1011, e
      )
    elif "1007" in str(e):
      await self._publish_health_status(
          LiveAPIOrchestratorHealth.ERROR_INTERNAL_1007, e
      )
    else:
      await self._publish_health_status(
          LiveAPIOrchestratorHealth.ERROR_OTHER, e
      )

  async def _publish_live_config_to_event_bus(self):
    """Publishes the live config to the event bus."""
    logging.info("Publishing live config to event bus.")

    session_config = {
        "model_name": self._config.agent_model_name,
        "is_streaming": True,
        "system_instruction": str(
            _get_attr_or_item(self._live_config, "system_instruction")
        ),
        "tools": str(_get_attr_or_item(self._live_config, "tools")),
        "live_config_repr": str(self._live_config),
    }

    await self._bus.publish(
        event=event_bus.Event(
            type=event_bus.EventType.LOG_SESSION_METADATA,
            source=event_bus.EventSource.MAIN_AGENT,
            data=f"Live API Config (also in metadata): {session_config}",
            metadata=session_config,
        )
    )

  async def _create_and_manage_session(self):
    """Create and manage a Gemini Live API session."""
    try:
      logging.info("Creating and managing Gemini Live API session.")
      await self._publish_live_config_to_event_bus()
      async with self._client.aio.live.connect(
          model=self._config.agent_model_name, config=self._live_config
      ) as session:
        logging.info("Connected to Gemini Live API session.")
        self._is_active = True
        self._session = session
        self._receive_responses_task = asyncio.create_task(
            self._receive_responses()
        )
        self._send_streaming_inputs_task = asyncio.create_task(
            self.send_streaming_inputs()
        )
        self._send_non_streaming_inputs_task = asyncio.create_task(
            self.send_non_streaming_inputs()
        )
        await self._maybe_publish_live_api_health_event(None)
        while self._is_active:
          await asyncio.sleep(1.0)
    except asyncio.CancelledError:
      logging.info("Cancelled Gemini Live API session.")
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception("Error creating and managing Gemini Live API session.")
      await self._maybe_publish_live_api_health_event(e)
    finally:
      self._session = None
      self._is_active = False

      # Cancel and await all tasks
      tasks_to_cancel = []
      if self._send_streaming_inputs_task:
        self._send_streaming_inputs_task.cancel()
        tasks_to_cancel.append(self._send_streaming_inputs_task)
      if self._send_non_streaming_inputs_task:
        self._send_non_streaming_inputs_task.cancel()
        tasks_to_cancel.append(self._send_non_streaming_inputs_task)
      if self._receive_responses_task:
        self._receive_responses_task.cancel()
        tasks_to_cancel.append(self._receive_responses_task)

      # Await all cancelled tasks
      if tasks_to_cancel:
        await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

      self._send_streaming_inputs_task = None
      self._send_non_streaming_inputs_task = None
      self._receive_responses_task = None

      if self._non_streaming_inputs_queue:
        self._clear_queue(self._non_streaming_inputs_queue)
      if self._streaming_input_queue:
        self._clear_queue(self._streaming_input_queue)
      logging.info("Gemini Live API session finally cleanup.")

  async def _receive_responses(self):
    """Task to receive responses from the session."""
    logging.info("Start receiving responses from the session.")
    while self._is_active and self._session:
      try:
        await self._process_response_stream(self._session.receive())
        await self._maybe_publish_live_api_health_event(None)
      except asyncio.CancelledError:
        logging.info("Receive responses task cancelled.")
        break
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error("Error receiving responses: %s", e)
        await self._maybe_publish_live_api_health_event(e)
        await asyncio.sleep(1)  # Avoid tight loop on connection failure

  async def _process_response_stream(
      self, response_iterator: AsyncIterator[types.LiveServerMessage]
  ):
    """Processes a stream of responses from `session.receive()`.

    Args:
      response_iterator: An async iterator of responses from the session.
    """
    async for response in response_iterator:
      ##########################################################################
      # LiveServerSetupComplete
      ##########################################################################
      if setup_complete := response.setup_complete:
        await self._publish_event(
            event_bus.EventType.SETUP_COMPLETE, setup_complete
        )
      ##########################################################################
      # LiveServerContent
      ##########################################################################
      elif server_content := response.server_content:
        # We handle the different types of server content that could be
        # returned.
        if model_turn := server_content.model_turn:
          await self._publish_event(event_bus.EventType.MODEL_TURN, model_turn)
        elif turn_complete := server_content.turn_complete:
          await self._publish_event(
              event_bus.EventType.MODEL_TURN_COMPLETE, turn_complete
          )
        elif interrupted := server_content.interrupted:
          await self._publish_event(
              event_bus.EventType.MODEL_TURN_INTERRUPTED, interrupted
          )
        elif generation_complete := server_content.generation_complete:
          await self._publish_event(
              event_bus.EventType.GENERATION_COMPLETE, generation_complete
          )
        # Other content that could be returned for completeness.
        elif grounding_metadata := server_content.grounding_metadata:
          await self._publish_event(
              event_bus.EventType.GROUNDING_METADATA, grounding_metadata
          )
        elif url_context_metadata := server_content.url_context_metadata:
          await self._publish_event(
              event_bus.EventType.URL_CONTEXT_METADATA, url_context_metadata
          )
        # Check for input / output transcripts if configured to be emitted.
        elif input_transcription := server_content.input_transcription:
          await self._publish_event(
              event_bus.EventType.INPUT_TRANSCRIPT, input_transcription
          )
        elif output_transcription := server_content.output_transcription:
          await self._publish_event(
              event_bus.EventType.OUTPUT_TRANSCRIPT, output_transcription
          )
        else:
          logging.warning("Unsupported server content: %s", server_content)
      ##########################################################################
      # LiveServerToolCall
      ##########################################################################
      elif tool_call := response.tool_call:
        await self._publish_event(event_bus.EventType.TOOL_CALL, tool_call)
      ##########################################################################
      # LiveServerToolCancellation
      ##########################################################################
      elif tool_call_cancellation := response.tool_call_cancellation:
        await self._publish_event(
            event_bus.EventType.TOOL_CALL_CANCELLATION, tool_call_cancellation
        )
      ##########################################################################
      # UsageMetadata
      ##########################################################################
      elif usage_metadata := response.usage_metadata:
        await self._publish_event(
            event_bus.EventType.USAGE_METADATA, usage_metadata
        )
      ##########################################################################
      # LiveServerGoAway
      ##########################################################################
      elif go_away := response.go_away:
        await self._publish_event(event_bus.EventType.GO_AWAY, go_away)
        if self._go_away_task is None or self._go_away_task.done():
          self._go_away_task = asyncio.create_task(
              self._handle_go_away_and_reconnect(go_away)
          )
      ##########################################################################
      # LiveServerSessionResumptionUpdate
      ##########################################################################
      elif session_resumption_update := response.session_resumption_update:
        if (
            session_resumption_update.resumable
            and session_resumption_update.new_handle
        ):
          self._session_resumption_handle = session_resumption_update.new_handle
          logging.info("Stored session resumption handle.")
        await self._publish_event(
            event_bus.EventType.SESSION_RESUMPTION_UPDATE,
            session_resumption_update,
        )
      else:
        logging.warning("Unsupported response: %s", response)

  async def _send_text(self, text: str) -> None:
    """Sends text to the session."""
    if not self._session:
      return
    if (
        self._config.update_vision_after_fr
        and self._turn_coverage
        == types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY
    ):
      # Send text first, then vision update
      await self._session.send_client_content(
          turns={"role": "user", "parts": [{"text": text}]},
          turn_complete=False,
      )
      parts = _prepare_image_parts(
          self._last_image_received,
          self._config.enable_image_stitching,
          self._config.show_camera_name_in_stitched_image,
      )
      await self._session.send_client_content(
          turns={"role": "user", "parts": parts},
          turn_complete=True,
      )
    else:
      # Simple case: just send the text
      await self._session.send_client_content(
          turns={"role": "user", "parts": [{"text": text}]},
          turn_complete=True,
      )

  async def _send_tool_response(
      self, tool_response: types.LiveClientToolResponse
  ) -> None:
    """Sends tool response to the session."""
    if not self._session:
      return
    if (
        self._config.update_vision_after_fr
        and self._turn_coverage
        == types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY
    ):
      # Send tool response with SILENT scheduling, then vision update
      assert len(tool_response.function_responses) == 1
      response = tool_response.function_responses[0]
      response.scheduling = types.FunctionResponseScheduling.SILENT
      await self._session.send_tool_response(function_responses=response)
      parts = _prepare_image_parts(
          self._last_image_received,
          self._config.enable_image_stitching,
          self._config.show_camera_name_in_stitched_image,
      )
      await self._session.send_client_content(
          turns={"role": "user", "parts": parts},
          turn_complete=True,
      )
    else:
      # Simple case: just send the tool response
      await self._session.send_tool_response(
          function_responses=tool_response.function_responses,
      )

  async def _handle_go_away_and_reconnect(
      self, go_away: types.LiveServerGoAway
  ) -> None:
    """Handles GO_AWAY by counting down and then reconnecting.

    Args:
      go_away: The GO_AWAY message from the server.
    """
    if not self._config.enable_automatic_session_resumption:
      logging.info(
          "GO_AWAY received but automatic session resumption is disabled."
          " Time left from server: %s. Not reconnecting.",
          go_away.time_left,
      )
      await self._publish_event(
          event_bus.EventType.SYSTEM_MESSAGE,
          "GO_AWAY received. Automatic session resumption is disabled.",
      )
      return

    grace_period = 20.0
    logging.info(
        "GO_AWAY received. Time left from server: %s. Starting %ss countdown.",
        go_away.time_left,
        grace_period,
    )

    remaining = grace_period
    while remaining > 0:
      await self._publish_event(
          event_bus.EventType.SYSTEM_MESSAGE,
          f"GO_AWAY received. Reconnecting in {int(remaining)}s...",
      )
      await asyncio.sleep(1.0)
      remaining -= 1.0

    await self._reconnect_session()

  async def _reconnect_session(self) -> None:
    """Reconnects the Live API session using resumption handle if available."""
    handle = self._session_resumption_handle

    if handle:
      logging.info("Attempting to reconnect with session resumption handle.")
    else:
      logging.info("No resumption handle available. Starting fresh session.")

    new_config = self._create_config_with_resumption(handle)

    self._is_active = False
    if self._non_streaming_inputs_queue:
      await self._non_streaming_inputs_queue.put(None)
    if self._streaming_input_queue:
      await self._streaming_input_queue.put(None)

    if self._gemini_live_api_task and not self._gemini_live_api_task.done():
      self._gemini_live_api_task.cancel()
      try:
        await self._gemini_live_api_task
      except asyncio.CancelledError:
        pass

    logging.info("Session disconnected for reconnection.")
    await self._publish_event(
        event_bus.EventType.FRAMEWORK_STATUS,
        "NOT_READY",
    )
    self._session = None
    self._clear_queue(self._non_streaming_inputs_queue)
    self._clear_queue(self._streaming_input_queue)

    self._live_config = new_config
    self._gemini_live_api_task = asyncio.create_task(
        self._create_and_manage_session()
    )

    timeout = 5.0
    start = time.monotonic()
    while time.monotonic() - start < timeout:
      if self._session and self._is_active:
        if handle:
          logging.info("Session resumed successfully with context preserved.")
          await self._publish_event(
              event_bus.EventType.SYSTEM_MESSAGE,
              "Session resumed successfully with context preserved.",
          )
        else:
          logging.info("New session started (no previous context).")
          await self._publish_event(
              event_bus.EventType.SYSTEM_MESSAGE,
              "New session started (no previous context).",
          )
        await self._publish_event(
            event_bus.EventType.FRAMEWORK_STATUS,
            "READY",
        )
        return
      await asyncio.sleep(0.1)

    if handle:
      logging.warning(
          "Session resumption failed. Will retry with fresh session."
      )
      await self._handle_resumption_failure()

  async def _handle_resumption_failure(self) -> None:
    """Handles failure to resume session by warning and starting fresh."""
    grace_period = 20.0
    remaining = grace_period

    while remaining > 0:
      await self._publish_event(
          event_bus.EventType.SYSTEM_MESSAGE,
          "Session resumption failed. Starting fresh session in"
          f" {int(remaining)}s...",
      )
      await asyncio.sleep(2.0)
      remaining -= 2.0

    self._session_resumption_handle = None
    await self._reconnect_session()

  def _create_config_with_resumption(
      self, handle: str | None
  ) -> types.LiveConnectConfig | types.LiveConnectConfigDict:
    """Creates a new live config with session resumption configured.

    Args:
      handle: The session resumption handle to use, or None for fresh session.

    Returns:
      A new LiveConnectConfig with session_resumption set.
    """
    if isinstance(self._live_config, dict):
      new_config = dict(self._live_config)
      new_config["session_resumption"] = {"handle": handle}
      return new_config
    else:
      config_dict = {}
      if self._live_config.generation_config:
        config_dict["generation_config"] = self._live_config.generation_config
      if self._live_config.response_modalities:
        config_dict["response_modalities"] = (
            self._live_config.response_modalities
        )
      if self._live_config.speech_config:
        config_dict["speech_config"] = self._live_config.speech_config
      if self._live_config.system_instruction:
        config_dict["system_instruction"] = self._live_config.system_instruction
      if self._live_config.tools:
        config_dict["tools"] = self._live_config.tools
      if self._live_config.realtime_input_config:
        config_dict["realtime_input_config"] = (
            self._live_config.realtime_input_config
        )
      if self._live_config.context_window_compression:
        config_dict["context_window_compression"] = (
            self._live_config.context_window_compression
        )
      if self._live_config.output_audio_transcription:
        config_dict["output_audio_transcription"] = (
            self._live_config.output_audio_transcription
        )
      if self._live_config.input_audio_transcription:
        config_dict["input_audio_transcription"] = (
            self._live_config.input_audio_transcription
        )
      config_dict["session_resumption"] = types.SessionResumptionConfig(
          handle=handle
      )
      return config_dict
