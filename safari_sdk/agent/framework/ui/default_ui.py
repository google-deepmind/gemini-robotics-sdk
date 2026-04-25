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

"""Default terminal UI handler for the agent framework."""

import threading
from typing import Callable

from safari_sdk.agent.framework.event_bus import event_bus


TERMINAL_COLOR_GREEN = "\033[92m"
TERMINAL_COLOR_YELLOW = "\033[93m"
TERMINAL_COLOR_RED = "\033[91m"
TERMINAL_COLOR_BLUE = "\033[94m"
TERMINAL_COLOR_ORANGE = "\033[38;5;208m"
TERMINAL_COLOR_CYAN = "\033[96m"
TERMINAL_COLOR_RESET = "\033[0m"


COLOR_MAP = {
    event_bus.EventType.TOOL_CALL: TERMINAL_COLOR_GREEN,
    event_bus.EventType.TOOL_RESULT: TERMINAL_COLOR_ORANGE,
    event_bus.EventType.TOOL_CALL_CANCELLATION: TERMINAL_COLOR_RED,
    event_bus.EventType.MODEL_TURN: TERMINAL_COLOR_BLUE,
    event_bus.EventType.MODEL_THOUGHT: TERMINAL_COLOR_YELLOW,
    event_bus.EventType.OUTPUT_TRANSCRIPT: TERMINAL_COLOR_CYAN,
}


class TranscriptBuffer:
  """Buffers transcript fragments and flushes them as stitched text."""

  def __init__(
      self,
      flush_callback: Callable[[str], None],
      flush_timeout_seconds: float = 0.5,
  ):
    self._buffer: list[str] = []
    self._flush_callback = flush_callback
    self._flush_timeout_seconds = flush_timeout_seconds
    self._timer: threading.Timer | None = None
    self._lock = threading.Lock()

  def append(self, text: str, finished: bool = False) -> None:
    """Appends text to the buffer and flushes if finished or after timeout."""

    with self._lock:
      if self._timer:
        self._timer.cancel()
        self._timer = None
      if text:
        self._buffer.append(text)
      if finished:
        self._flush_locked()
      else:
        self._timer = threading.Timer(
            self._flush_timeout_seconds, self._flush_on_timeout
        )
        self._timer.daemon = True
        self._timer.start()

  def _flush_on_timeout(self) -> None:
    with self._lock:
      self._flush_locked()

  def _flush_locked(self) -> None:
    if self._timer:
      self._timer.cancel()
      self._timer = None
    if self._buffer:
      stitched = "".join(self._buffer)
      self._buffer.clear()
      self._flush_callback(stitched)

  def flush(self) -> None:
    with self._lock:
      self._flush_locked()

  def clear(self) -> None:
    """Clears the buffer without flushing."""
    with self._lock:
      if self._timer:
        self._timer.cancel()
        self._timer = None
      self._buffer.clear()


def _print_stitched_transcript(text: str) -> None:
  print(
      f"{TERMINAL_COLOR_CYAN}[STITCHED_AUDIO_TRANSCRIPT]:"
      f" {text}{TERMINAL_COLOR_RESET}"
  )


_output_transcript_buffer = TranscriptBuffer(
    flush_callback=_print_stitched_transcript,
    flush_timeout_seconds=0.5,
)


def _print_to_terminal(message: str, event: event_bus.Event, color: str = ""):
  reset_color = TERMINAL_COLOR_RESET if color else ""
  print(
      f"{color}[{event.source.value}, {event.type.value} - text]:"
      f" {message}{reset_color}"
  )


def handle_event(event: event_bus.Event) -> None:
  """Handles events and prints them to the terminal with default formatting.

  Args:
    event: The event to handle.
  """
  color = COLOR_MAP.get(event.type, "")

  match event.type:
    case event_bus.EventType.MODEL_TURN:
      for part in event.data.parts:
        if part.text:
          _print_to_terminal(f"{part.text}", event=event, color=color)
        if part.code_execution_result:
          _print_to_terminal(
              f"{part.code_execution_result}", event=event, color=color
          )
        elif part.executable_code:
          _print_to_terminal(
              f"{part.executable_code}", event=event, color=color
          )

    case (
        event_bus.EventType.MODEL_TURN_COMPLETE
        | event_bus.EventType.MODEL_TURN_INTERRUPTED
        | event_bus.EventType.GENERATION_COMPLETE
    ):
      _output_transcript_buffer.flush()
      _print_to_terminal("", event=event)

    case event_bus.EventType.OUTPUT_TRANSCRIPT:
      text = getattr(event.data, "text", "") or ""
      finished = getattr(event.data, "finished", False) or False
      _output_transcript_buffer.append(text, finished)

    case (
        event_bus.EventType.TOOL_CALL
        | event_bus.EventType.TOOL_CALL_CANCELLATION
        | event_bus.EventType.TOOL_RESULT
        | event_bus.EventType.GO_AWAY
        | event_bus.EventType.DEBUG
        | event_bus.EventType.MODEL_THOUGHT
    ):
      _print_to_terminal(f"{event.data}", event=event, color=color)

    case _:
      pass
