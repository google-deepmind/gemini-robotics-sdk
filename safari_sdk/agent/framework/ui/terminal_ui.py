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

"""Provides a terminal-based user interface for interacting with the event bus."""

import asyncio

from absl import logging

from safari_sdk.agent.framework import config as framework_config
from safari_sdk.agent.framework.event_bus import event_bus


TERMINAL_COLOR_GREEN = "\033[92m"
TERMINAL_COLOR_YELLOW = "\033[93m"
TERMINAL_COLOR_RED = "\033[91m"
TERMINAL_COLOR_BLUE = "\033[94m"
TERMINAL_COLOR_ORANGE = "\033[38;5;208m"
TERMINAL_COLOR_RESET = "\033[0m"


COLOR_MAP = {
    event_bus.EventType.TOOL_CALL: TERMINAL_COLOR_GREEN,
    event_bus.EventType.TOOL_RESULT: TERMINAL_COLOR_ORANGE,
    event_bus.EventType.TOOL_CALL_CANCELLATION: TERMINAL_COLOR_RED,
    event_bus.EventType.MODEL_TURN: TERMINAL_COLOR_BLUE,
}


class TerminalUI:
  """Handles user input from the terminal and displays relevant events from the event bus."""

  def __init__(
      self,
      bus: event_bus.EventBus,
      config: framework_config.AgentFrameworkConfig,
  ):
    self._config = config
    self._text_input_listener_task = None
    self._send_reminder_text_input_tasks = []
    self._bus = bus
    if self._config.use_operator_friendly_terminal_ui:
      print_text_output_events_handler = (
          self._print_text_output_events_operator_friendly
      )
      print_model_turn_event_handler = (
          self._print_model_turn_event_operator_friendly
      )
      print_model_turn_completion_handler = (
          self._print_model_turn_completion_operator_friendly
      )
      text_output_events_to_handle = [
          event_bus.EventType.TOOL_CALL,
          event_bus.EventType.TOOL_CALL_CANCELLATION,
          event_bus.EventType.TOOL_RESULT,
      ]
    else:
      print_text_output_events_handler = self._print_text_output_events
      print_model_turn_event_handler = self._print_model_turn_event
      print_model_turn_completion_handler = (
          self._print_model_turn_completion
      )
      text_output_events_to_handle = [
          event_bus.EventType.TOOL_CALL,
          event_bus.EventType.TOOL_CALL_CANCELLATION,
          event_bus.EventType.TOOL_RESULT,
          event_bus.EventType.GO_AWAY,
          event_bus.EventType.DEBUG,
          event_bus.EventType.OUTPUT_TRANSCRIPT,
      ]
    self._bus.subscribe(
        event_types=[event_bus.EventType.MODEL_TURN],
        handler=print_model_turn_event_handler,
    )
    self._bus.subscribe(
        event_types=[
            event_bus.EventType.MODEL_TURN_COMPLETE,
            event_bus.EventType.MODEL_TURN_INTERRUPTED,
            event_bus.EventType.GENERATION_COMPLETE,
        ],
        handler=print_model_turn_completion_handler,
    )
    self._bus.subscribe(
        event_types=text_output_events_to_handle,
        handler=print_text_output_events_handler,
    )

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

  def _print_text_output_events(self, event: event_bus.Event):
    """Handles text output events and prints them to the terminal."""
    color = COLOR_MAP.get(event.type, "")
    reset_color = TERMINAL_COLOR_RESET if color else ""
    print(
        f"{color}[{event.source.value}, {event.type.value}]:"
        f" {event.data}{reset_color}"
    )

  def _print_text_output_events_operator_friendly(self, event: event_bus.Event):
    """Handles text output events and prints them to the terminal."""
    color = COLOR_MAP.get(event.type, "")
    reset_color = TERMINAL_COLOR_RESET if color else ""
    # Only print TOOL_CALL, TOOL_CALL_CANCELLATION, and TOOL_RESULT.
    match event.type:
      case event_bus.EventType.TOOL_CALL:
        print(
            "\n"
            f"{color}[{event.type.value}]:"
            f"{event.data.function_calls[0].args['instruction']}{reset_color}"
        )
      case event_bus.EventType.TOOL_RESULT:
        print(
            "\n"
            f"{color}[{event.type.value}]:"
            f"{event.data.function_responses[0].response}{reset_color}"
        )
      case event_bus.EventType.TOOL_CALL_CANCELLATION:
        print(
            "\n"
            f"{color}[{event.type.value}]:{reset_color}"
        )
      case _:
        # Only print TOOL_CALL, TOOL_CALL_CANCELLATION and TOOL_RESULT.
        pass

  def _print_model_turn_event(self, event: event_bus.Event):
    """Handles model turn events and prints the parts to the terminal."""
    color = COLOR_MAP.get(event.type, "")
    reset_color = TERMINAL_COLOR_RESET if color else ""
    for part in event.data.parts:
      if part.text:
        print(
            f"{color}[{event.source.value}, {event.type.value} - text]: "
            f"{part.text}{reset_color}"
        )
      if part.code_execution_result:
        print(
            f"{color}[{event.source.value}, {event.type.value} - code execution"
            f" result]: {part.code_execution_result}{reset_color}"
        )
      elif part.executable_code:
        print(
            f"{color}"
            f"[{event.source.value}, {event.type.value} - executable code]:"
            f" {part.executable_code}{reset_color}"
        )

  def _print_model_turn_event_operator_friendly(self, event: event_bus.Event):
    """Handles model turn events and prints the parts to the terminal."""
    color = COLOR_MAP.get(event.type, "")
    reset_color = TERMINAL_COLOR_RESET if color else ""
    for part in event.data.parts:
      # Only print text parts.
      if part.text:
        # No newline to append adjacent text events.
        print(f"{color}{part.text}{reset_color}", end="")

  def _print_model_turn_completion(self, event: event_bus.Event):
    """Handles model turn completion events and prints them to the terminal."""
    print(f"[{event.source.value}, {event.type.value}]")

  def _print_model_turn_completion_operator_friendly(
      self, event: event_bus.Event
  ):
    """Handles model turn completion events and prints them to the terminal."""
    # Only print MODEL_TURN_COMPLETE.
    if event.type == event_bus.EventType.MODEL_TURN_COMPLETE:
      print(f"\n[{event.type.value}]")

  async def text_input_loop(self, bus: event_bus.EventBus):
    """Handles text input events and publishes them to the event bus."""
    while True:
      try:
        # Use asyncio.to_thread to run the blocking input() call
        # in a separate thread without blocking the main event loop.
        message = await asyncio.to_thread(input, "\n[USER]: ")
        if message and message.strip():  # Don't send empty messages
          # Debug message event (starting w/ "@debug"), send as a debug event.
          if "@d" in message:
            debug_message = message.replace("@d ", "").replace("@d", "")
            event = event_bus.Event(
                type=event_bus.EventType.DEBUG,
                source=event_bus.EventSource.USER,
                data=message,
                metadata={"debug_message": debug_message},
            )
          # Send as a success signal event with success=True.
          elif "@s" in message:
            event = event_bus.Event(
                type=event_bus.EventType.SUCCESS_SIGNAL,
                source=event_bus.EventSource.USER,
                data=True,
            )
          # Send as a success signal event with success=False.
          elif "@f" in message:
            event = event_bus.Event(
                type=event_bus.EventType.SUCCESS_SIGNAL,
                source=event_bus.EventSource.USER,
                data=False,
            )
          else:  # Normal event, sent model text input.
            event = event_bus.Event(
                type=event_bus.EventType.MODEL_TEXT_INPUT,
                source=event_bus.EventSource.USER,
                data=message,
            )
          await bus.publish(event)
      except (RuntimeError, KeyboardInterrupt):
        logging.info("Text input loop shutting down.")
        return
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("Error in text input loop: %s", e)
        return

