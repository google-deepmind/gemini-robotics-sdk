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

"""Handler that executes tool calls from the Live API.

Translates TOOL_CALL events into function executions and publishes the results
as TOOL_RESULT events back to the event bus. Each function call within a tool
call event is processed sequentially and emitted as a separate tool result.
"""

import asyncio
from collections.abc import Mapping
import inspect
from typing import Any, AsyncIterator, Coroutine, cast

from absl import logging
from google.genai import types as genai_types

from safari_sdk.agent.framework import types as framework_types
from safari_sdk.agent.framework.event_bus import event_bus


_AnyFunction = framework_types.AsyncFunction


class ToolCallEventHandler:
  """Handles tool call events by executing registered functions.

  Subscribes to TOOL_CALL events, executes the corresponding async functions
  from the tool dictionary, and publishes TOOL_RESULT events with the function
  responses. Supports async functions and async generators.
  """

  def __init__(
      self,
      bus: event_bus.EventBus,
      tool_dict: Mapping[str, _AnyFunction],
  ):
    self._bus = bus
    self._function_handlers: dict[str, _AnyFunction] = {**tool_dict}
    bus.subscribe(
        event_types=[event_bus.EventType.TOOL_CALL],
        handler=self._handle_tool_call_event,
    )

  async def _handle_tool_call_event(
      self,
      event: event_bus.Event,
  ):
    """Performs all of the function calls contained in a tool call event."""
    for fc in event.data.function_calls:
      logging.debug(
          "Handling function call: %s with args: %s, id: %s",
          fc.name,
          fc.args,
          fc.id,
      )
      # Actually perform the function call.
      async for function_response in self._run_function(
          fc.name,
          fc.args,
          fc.id,
      ):
        # Compile the function call response into a tool result event and send
        # it to the event bus. Functions already return a GenAI response, but we
        # enforce that the ID and name match the original function call in case
        # they don't set it.
        function_response.id = fc.id
        function_response.name = fc.name
        if function_response.scheduling is None:
          function_response.scheduling = (
              genai_types.FunctionResponseScheduling.INTERRUPT
          )
        await self._bus.publish(
            event=event_bus.Event(
                type=event_bus.EventType.TOOL_RESULT,
                source=event_bus.EventSource.ROBOT,
                data=genai_types.LiveClientToolResponse(
                    # We emit a tool result event for each function call
                    # response, instead of aggregating all responses into a
                    # single event.
                    function_responses=[function_response]
                ),
            )
        )

  async def _run_function(
      self, name: str, kwargs: Mapping[str, Any], call_id: str
  ) -> AsyncIterator[genai_types.FunctionResponse]:
    """Runs a function with the given name and arguments."""
    if name not in self._function_handlers:
      raise ValueError(f"Unknown function: {name}")
    fn = self._function_handlers[name]
    result = fn(**kwargs, call_id=call_id)

    # The function may be an async generator or async function.
    if inspect.isasyncgen(result):
      result = cast(AsyncIterator[genai_types.FunctionResponse], result)
      async for fc_result in result:
        yield fc_result
    elif asyncio.iscoroutine(result):
      result = cast(Coroutine[Any, Any, genai_types.FunctionResponse], result)
      yield await result
    else:
      result = cast(genai_types.FunctionResponse, result)
      yield result
