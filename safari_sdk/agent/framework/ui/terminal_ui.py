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

"""Terminal UI shell for hosting UI handlers and managing text input."""

import asyncio
from collections.abc import Callable
from typing import Any

from absl import logging

from safari_sdk.agent.framework import config as framework_config
from safari_sdk.agent.framework import types
from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.ui import default_ui
from safari_sdk.agent.framework.ui import operator_text_ui


UIHandler = Callable[[event_bus.Event], None]


def _noop_handler(event: event_bus.Event) -> None:
  del event


EXTERNAL_UI_HANDLERS: dict[types.ExternalUIType, UIHandler] = {
    types.ExternalUIType.NONE: _noop_handler,
    types.ExternalUIType.OPERATOR_DATA_COLLECT: operator_text_ui.handle_event,
}


def parse_user_input_to_event(
    message: str,
    source: event_bus.EventSource,
    default_metadata: dict[str, Any] | None = None,
) -> event_bus.Event:
  """Parses user input message and creates the appropriate event.

  Handles special prefixes:
    - "@d": Creates a DEBUG event
    - "@s": Creates a SUCCESS_SIGNAL event with data=True
    - "@f": Creates a SUCCESS_SIGNAL event with data=False
    - Otherwise: Creates a MODEL_TEXT_INPUT event

  Args:
    message: The user input message.
    source: The source of the event.
    default_metadata: Optional metadata to include for MODEL_TEXT_INPUT events.

  Returns:
    The appropriate Event based on the message content.
  """
  event_type = event_bus.EventType.MODEL_TEXT_INPUT
  data = message
  metadata = default_metadata or {}

  if message.strip() == "/context":
    event_type = event_bus.EventType.DEBUG
    metadata = {"command": "context_dump"}
  elif "@d" in message:
    event_type = event_bus.EventType.DEBUG
    debug_msg = message.replace("@d ", "").replace("@d", "")
    metadata = {"debug_message": debug_msg}
    if "/context" in debug_msg:
      metadata["command"] = "context_dump"
  elif "@s" in message:
    event_type = event_bus.EventType.SUCCESS_SIGNAL
    data = True
  elif "@f" in message:
    event_type = event_bus.EventType.SUCCESS_SIGNAL
    data = False

  return event_bus.Event(
      type=event_type,
      source=source,
      data=data,
      metadata=metadata,
  )


class TerminalUI:
  """Shell for hosting UI handlers and managing text input from the terminal."""

  def __init__(
      self,
      bus: event_bus.EventBus,
      config: framework_config.AgentFrameworkConfig,
  ):
    self._config = config
    self._text_input_listener_task = None
    self._send_reminder_text_input_tasks = []
    self._bus = bus
    self._external_ui_handler = EXTERNAL_UI_HANDLERS.get(
        config.external_ui_type, _noop_handler
    )
    all_events_to_handle = [
        event_bus.EventType.MODEL_TURN,
        event_bus.EventType.MODEL_THOUGHT,
        event_bus.EventType.MODEL_TURN_COMPLETE,
        event_bus.EventType.MODEL_TURN_INTERRUPTED,
        event_bus.EventType.GENERATION_COMPLETE,
        event_bus.EventType.TOOL_CALL,
        event_bus.EventType.TOOL_CALL_CANCELLATION,
        event_bus.EventType.TOOL_RESULT,
        event_bus.EventType.GO_AWAY,
        event_bus.EventType.DEBUG,
    ]
    if self._config.enable_audio_transcription:
      all_events_to_handle.append(event_bus.EventType.OUTPUT_TRANSCRIPT)
    self._bus.subscribe(
        event_types=all_events_to_handle,
        handler=self._handle_event,
    )

  def _handle_event(self, event: event_bus.Event) -> None:
    """Dispatches event to internal UI (always) and external UI (if configured)."""
    default_ui.handle_event(event)
    self._external_ui_handler(event)

  async def connect(self):
    """Connects the event bus to the terminal UI."""
    self._text_input_listener_task = asyncio.create_task(
        self.text_input_loop(self._bus)
    )
    if self._config.reminder_time_in_seconds is not None:
      for i, reminder_time in enumerate(self._config.reminder_time_in_seconds):
        self._send_reminder_text_input_tasks.append(
            asyncio.create_task(
                self._send_reminder_text_input(
                    reminder_time,
                    self._config.reminder_text_list[i],
                )
            )
        )
    print("Type to send a message to the model. Press CTRL+C to exit.")

  async def disconnect(self):
    """Disconnects the event bus from the terminal UI."""
    if (
        self._text_input_listener_task
        and not self._text_input_listener_task.done()
    ):
      self._text_input_listener_task.cancel()
      try:
        await self._text_input_listener_task
      except asyncio.CancelledError:
        logging.debug("Text input loop cancelled.")

    for reminder_task in self._send_reminder_text_input_tasks:
      if self._send_reminder_text_input_tasks and not reminder_task.done():
        reminder_task.cancel()
        try:
          await reminder_task
        except asyncio.CancelledError:
          logging.debug("Send reminder text input task cancelled.")

  async def _send_reminder_text_input(
      self, reminder_time: float, reminder_text: str
  ):
    """Sends the ending user text input to the agent."""
    await asyncio.sleep(reminder_time)
    event = event_bus.Event(
        type=event_bus.EventType.MODEL_TEXT_INPUT,
        source=event_bus.EventSource.USER,
        data=reminder_text,
    )
    await self._bus.publish(event)

  async def text_input_loop(self, bus: event_bus.EventBus):
    """Handles text input events and publishes them to the event bus."""
    while True:
      try:
        message = await asyncio.to_thread(input, "\n[USER]: ")
        if message and message.strip():
          event = parse_user_input_to_event(
              message=message, source=event_bus.EventSource.USER
          )
          await bus.publish(event)
      except (RuntimeError, KeyboardInterrupt):
        logging.info("Text input loop shutting down.")
        return
      except Exception as e:  # pylint: disable=broad-except
        logging.exception("Error in text input loop: %s", e)
        return
