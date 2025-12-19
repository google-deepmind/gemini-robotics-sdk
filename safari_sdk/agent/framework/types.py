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

"""Type definitions shared across the framework."""

from collections.abc import AsyncIterator
import dataclasses
import datetime
import enum
from typing import Any, Protocol, TypeVar

from google.genai import types


@enum.unique
class EventSource(enum.Enum):
  """Event sources."""

  USER = "USER"
  MAIN_AGENT = "LIVE_AGENT"
  ROBOT = "ROBOT"
  AGENTIC_TOOL = "AGENTIC_TOOL"
  EXTERNAL_CONTROLLER = "EXTERNAL_CONTROLLER"


@enum.unique
class EventType(enum.Enum):
  """Event types."""

  ##############################################################################
  # Events derived from Gemini Live API. Reference:
  # https://github.com/googleapis/python-genai/blob/main/google/genai/types.py.
  ##############################################################################
  SETUP_COMPLETE = "SETUP_COMPLETE"
  TOOL_CALL = "TOOL_CALL"
  TOOL_CALL_CANCELLATION = "TOOL_CALL_CANCELLATION"
  GO_AWAY = "GO_AWAY"
  USAGE_METADATA = "USAGE_METADATA"
  SESSION_RESUMPTION_UPDATE = "SESSION_RESUMPTION_UPDATE"
  MODEL_TURN = "MODEL_TURN"
  MODEL_TURN_COMPLETE = "MODEL_TURN_COMPLETE"
  MODEL_TURN_INTERRUPTED = "MODEL_TURN_INTERRUPTED"
  GROUNDING_METADATA = "GROUNDING_METADATA"
  GENERATION_COMPLETE = "GENERATION_COMPLETE"
  INPUT_TRANSCRIPT = "INPUT_TRANSCRIPT"
  OUTPUT_TRANSCRIPT = "OUTPUT_TRANSCRIPT"
  URL_CONTEXT_METADATA = "URL_CONTEXT_METADATA"
  # Tool result event in response to a tool call that can be emitted by one of
  # the event handlers in the system - this would typically be published by an
  # event handler that is subscribed to the TOOL_CALL event and would be
  # subscribed to by the Live Agent event handler to process.
  TOOL_RESULT = "TOOL_RESULT"
  #############################################################################
  # Other events.
  ##############################################################################
  MODEL_AUDIO_INPUT = "MODEL_AUDIO_INPUT"
  MODEL_IMAGE_INPUT = "MODEL_IMAGE_INPUT"
  MODEL_TEXT_INPUT = "MODEL_TEXT_INPUT"
  # This is a debug event that can be emitted by any entity.
  DEBUG = "DEBUG"
  # This is an event that can be emitted by any entity to indicate a success
  # signal for example a manual success/failure signal via a user interface.
  SUCCESS_SIGNAL = "SUCCESS_SIGNAL"
  # The external controller server subscribes to these event types in order to
  # stream the status of the framework and its main components to anywhere
  # outside of the EAR framework. Any entity can emit this event to indicate
  # the status of the framework. This is typically used to indicate that the
  # agent has finished a overall task.
  FRAMEWORK_STATUS = "FRAMEWORK_STATUS"
  LIVE_API_HEALTH = "LIVE_API_HEALTH"
  GEMINI_CLIENT_HEALTH = "GEMINI_CLIENT_HEALTH"
  # This is an event that can be emitted by any entity to indicate a reset of
  # the framework. This can be used to reset the entire framework.
  RESET = "RESET"
  # This is an event that can be emitted by any entity to indicate a logging
  # session metadata. This can be used to log any metadata that is relevant to
  # the agent framework.
  LOG_SESSION_METADATA = "LOG_SESSION_METADATA"
  # This event indicates a real-time image has been sent to the model.
  # This is used to track the exact images sent to Live API.
  REAL_TIME_IMAGE_SENT = "REAL_TIME_IMAGE_SENT"
  # This event indicates a real-time audio has been sent to the model.
  # This is used to track the exact audio sent to Live API.
  REAL_TIME_AUDIO_SENT = "REAL_TIME_AUDIO_SENT"


@dataclasses.dataclass(frozen=True)
class Event:
  """Event data."""

  # The type of the event.
  type: EventType
  # The source of the event.
  source: EventSource
  # The timestamp of the event.
  timestamp: datetime.datetime = dataclasses.field(
      default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
  )
  # The data associated with the event.
  data: Any = None
  # Additional metadata associated with the event that might be useful for
  # downstream event handlers. This is useful for passing along additional
  # information that is relevant to the event but is not the data itself.
  metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


class EventBusHandlerSignature(Protocol):
  """Protocol for modules that can be registered as event bus handlers."""

  def __call__(self, event: Event, *args, **kwargs) -> Any:
    """Handler for an event."""


class AsyncFunction(Protocol):
  """Protocol for modules that can be registered to execute function calls.

  For either long-running or generator functions, both of which will return
  multiple responses. Long-running functions will first generate an "ack" and
  later a response. Generator functions will generate multiple responses over
  time.
  """

  async def __call__(
      self, call_id: str, *args, **kwargs
  ) -> AsyncIterator[types.FunctionResponse]:
    ...


# Stream of events that can be added to a bus by specialized handlers.
_EventT = TypeVar("_EventT", bound=Event)
EventStream = AsyncIterator[_EventT]


@enum.unique
class ControlMode(enum.Enum):
  """Enum for specifying the IO mode of the framework."""

  TERMINAL_ONLY = "TERMINAL_ONLY"
