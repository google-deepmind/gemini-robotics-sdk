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

from absl import logging
from google import genai
from google.genai import types

from safari_sdk.agent.framework import config as framework_config
from safari_sdk.agent.framework import constants
from safari_sdk.agent.framework.event_bus import event_bus


_DEFAULT_MODEL_NAME = "models/gemini-2.0-flash-live-001"


@enum.unique
class LiveAPIHealth(enum.Enum):
  """Class for tracking the type of the external controller endpoint."""

  NORMAL = "NORMAL"
  # live API returned 1011 quota exceeded error.
  QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
  INTERNAL_ERROR_1011 = "INTERNAL_ERROR_1011"
  INTERNAL_ERROR_1007 = "INTERNAL_ERROR_1007"
  # live API returned other error.
  OTHER_ERROR = "OTHER_ERROR"


def _prepare_image_parts(img_dict: dict[str, types.Blob]) -> list[types.Part]:
  """Prepares a list of image parts for sending to the Agent."""
  parts = []
  for blob in img_dict.values():
    if blob.display_name:
      parts.append(
          types.Part(
              text=blob.display_name,
          )
      )
    parts.append(types.Part.from_bytes(data=blob.data, mime_type="image/jpeg"))
  return parts


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
      initial_camera_names: dict[str, str] | None = None,
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
      camera_names: The names of the camera streams available to the model.
        Currently, only the first camera is used.
      initial_camera_names: The names of the cameras to use initially alongside
        how they will be captioned for the model. If not provided, the first
        camera in the list of camera names will be used.
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
    self._live_api_health = LiveAPIHealth.NORMAL
    self._gemini_live_api_task: asyncio.Task | None = None
    self._receive_responses_task: asyncio.Task | None = None
    self._non_streaming_inputs_queue = asyncio.Queue[
        str | types.LiveClientToolResponse | None
    ]()
    self._send_non_streaming_inputs_task: asyncio.Task | None = None
    self._streaming_input_queue = asyncio.Queue[types.Blob | None]()
    self._send_streaming_inputs_task: asyncio.Task | None = None
    self._last_image_input_time = {}
    self._last_image_received = {}
    self._ignore_image_inputs = ignore_image_inputs
    self._camera_names = camera_names
    initial_camera_names = initial_camera_names or {camera_names[0]: ""}
    invalid_cameras = set(initial_camera_names.keys()) - set(camera_names)
    if invalid_cameras:
      raise ValueError(
          f"Initial camera names {invalid_cameras} not found in available"
          f" cameras {camera_names}."
      )
    self._current_camera_names = initial_camera_names
    self._turn_coverage = self._get_turn_coverage()

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
    if stream_name not in self._current_camera_names.keys():
      logging.info(
          "image event stream name %s not in current camera names", stream_name
      )
      return

    try:
      if self._turn_coverage == types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY:
        self._last_image_received[stream_name] = types.Blob(
            display_name=self._current_camera_names[stream_name],
            data=event.data,
            mime_type="image/jpeg",
        )
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
      await self._streaming_input_queue.put(realtime_input)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception("Error queuing audio: %s", e)

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
      self, status: LiveAPIHealth, exception: Exception | None
  ) -> None:
    if self._live_api_health != status:
      self._live_api_health = status
      await self._publish_event(
          event_bus.EventType.LIVE_API_HEALTH,
          {
              "health_status": status.value,
              "exception_message": str(exception) if exception else None,
          },
      )

  def _clear_queue(self, queue: asyncio.Queue) -> None:
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
    if e is None:
      await self._publish_health_status(LiveAPIHealth.NORMAL, None)
    elif "quota".lower() in str(e):
      await self._publish_health_status(LiveAPIHealth.QUOTA_EXCEEDED, e)
    elif "1011" in str(e):
      await self._publish_health_status(LiveAPIHealth.INTERNAL_ERROR_1011, e)
    elif "1007" in str(e):
      await self._publish_health_status(LiveAPIHealth.INTERNAL_ERROR_1007, e)
    else:
      await self._publish_health_status(LiveAPIHealth.OTHER_ERROR, e)

  async def _publish_live_config_to_event_bus(self):
    """Publishes the live config to the event bus."""
    logging.info("Publishing live config to event bus.")
    await self._bus.publish(
        event=event_bus.Event(
            type=event_bus.EventType.LOG_SESSION_METADATA,
            source=event_bus.EventSource.MAIN_AGENT,
            data=f"Live API Config (also in metadata): {self._live_config}",
            metadata=self._live_config,
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
      ##########################################################################
      # LiveServerSessionResumptionUpdate
      ##########################################################################
      elif session_resumption_update := response.session_resumption_update:
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
      parts = _prepare_image_parts(self._last_image_received)
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
      parts = _prepare_image_parts(self._last_image_received)
      await self._session.send_client_content(
          turns={"role": "user", "parts": parts},
          turn_complete=True,
      )
    else:
      # Simple case: just send the tool response
      await self._session.send_tool_response(
          function_responses=tool_response.function_responses,
      )
