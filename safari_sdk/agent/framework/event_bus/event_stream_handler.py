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

"""Takes generic event generators and publishes them to the bus."""

import asyncio
from collections.abc import Mapping
from typing import Generic, TypeVar

from absl import logging

from safari_sdk.agent.framework import constants
from safari_sdk.agent.framework import types
from safari_sdk.agent.framework.event_bus import event_bus


_EventT = TypeVar("_EventT", bound=types.Event)


class EventStreamHandler(Generic[_EventT]):
  """Takes event generators and publishes them to the bus."""

  def __init__(
      self,
      bus: event_bus.EventBus,
      streams: Mapping[str, types.EventStream[_EventT]],
  ):
    self._bus = bus
    self._streams = {**streams}
    self._stream_tasks: dict[str, asyncio.Task] = {}

  async def _start_event_stream(
      self,
      stream_name: str,
      bus: event_bus.EventBus,
      event_generator: types.EventStream[_EventT],
  ):
    """Start streaming events to the bus."""
    try:
      async for event in event_generator:
        event.metadata[constants.STREAM_NAME_METADATA_KEY] = stream_name
        await bus.publish(event=event)
    except asyncio.CancelledError:
      logging.info("Event stream %s cancelled.", stream_name)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.info(
          "An unexpected error occurred during event stream %s: %s",
          stream_name,
          e,
      )

  async def connect(
      self,
  ):
    """Connects to the handler and starts streaming events to the event bus."""

    for stream_name, event_generator in self._streams.items():
      if stream_name in self._stream_tasks:
        logging.warning("Stream task already running for %s.", stream_name)
        continue
      self._stream_tasks[stream_name] = asyncio.create_task(
          self._start_event_stream(
              stream_name=stream_name,
              bus=self._bus,
              event_generator=event_generator,
          )
      )
      logging.info("Stream subscription task created: %s.", stream_name)

  async def disconnect(self):
    """Disconnects to the handler and stops streaming events to the bus."""
    logging.info("Disconnecting...")
    if not self._stream_tasks:
      logging.info("No active stream tasks to cancel.")
      return
    for stream_name, stream_task in self._stream_tasks.items():
      if not stream_task.done():
        logging.info("Cancelling stream task %s...", stream_name)
        stream_task.cancel()
        try:
          await stream_task
          logging.info("Stream task cancelled and awaited.")
        except asyncio.CancelledError:
          logging.info("Stream task was successfully cancelled.")
        except Exception as e:  # pylint: disable=broad-exception-caught
          logging.info("Error during stream task cancellation: %s", e)
        finally:
          self._stream_task = None

    self._stream_tasks = {}
    logging.info("Disconnect complete.")
