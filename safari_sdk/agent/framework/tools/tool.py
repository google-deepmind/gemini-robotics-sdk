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

"""Defines a base class for functions used by tool calls."""

import asyncio
import inspect
from typing import Any, AsyncIterator, Coroutine, cast

from google.genai import types as genai_types

from safari_sdk.agent.framework import types as framework_types
from safari_sdk.agent.framework.event_bus import event_bus


class Tool:
  """A class that exposes a stateless function to the main agent.

  This class is meant to wrap functions and expose them as stateless
  tools to the main agent. If the function is stateful, it also manages the
  state of the tool by subscribing to events from the event bus.

  Tools will normally be subscribed to tool call events by the
  `ToolCallEventHandler`, so they should not do this themselves. If the tool
  requires additional events for its functioning, it must subscribe to them
  itself.

  All tools are automatically subscribed to tool call cancellation events, and
  should override the `_handle_tool_call_cancellation_events` method . If your
  tool itself cannot be cancelled because it blocks for other wrapped tools to
  complete, make sure that said tools are listening to the same `call_id` so
  accept the same cancellation event.

  Attributes:
    fn: The python function that underlies the tool. The function signature
      must be the same as the declaration.
    declaration: The declaration of the tool that is exposed to the agent's
      prompt. The main agent uses this to decide which tools to call and with
      what parameters. You can think of this as the function signature of the
      tool. It should be as specific as possible so that the main agent can call
      the tool correctly.
  """

  def __init__(
      self,
      bus: event_bus.EventBus,
      # We support only AsyncFunctions.
      fn: framework_types.AsyncFunction,
      declaration: genai_types.FunctionDeclaration,
  ):
    self.fn = fn
    self.declaration = declaration
    self._bus = bus
    self._bus.subscribe(
        event_types=[event_bus.EventType.TOOL_CALL_CANCELLATION],
        handler=self._handle_tool_call_cancellation_events,
    )

  def _handle_tool_call_cancellation_events(self, event: event_bus.Event):
    """Handle a tool call cancellation event.

    Implementations subclasses should override this method to handle tool call
    cancellation events. They need to ensure to filter by the `call_id` of the
    cancellation event and only act on it if it is relevant.

    Args:
      event: The tool call cancellation event. Its data field can be expected to
        have a field `id`, which contains the `call_id` of all (potentially
        multiple) tool calls that are to be cancelled.
    """
    pass

  async def __call__(
      self, *args, **kwargs
  ) -> AsyncIterator[genai_types.FunctionResponse]:
    result = self.fn(*args, **kwargs)
    # The function may be an async generator or awaitable async function
    if inspect.isasyncgen(result):
      result = cast(AsyncIterator[genai_types.FunctionResponse], result)
      async for fc_result in result:
        yield fc_result
    elif asyncio.iscoroutine(result):
      result = cast(Coroutine[Any, Any, genai_types.FunctionResponse], result)
      yield await result
