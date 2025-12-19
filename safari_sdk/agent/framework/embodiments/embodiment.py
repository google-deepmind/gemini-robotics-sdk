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

"""Embodiment base class.

Embodiments are convenience "containers" that bundle the functionality of a
single robot together. They are not required for building an Agent class, but
provide a reusable package for agents using the same robot hardware.
"""

import abc
import dataclasses
from typing import Sequence

from safari_sdk.agent.framework import types
from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.event_bus import event_stream_handler
from safari_sdk.agent.framework.tools import tool


@dataclasses.dataclass(frozen=True)
class EventStreamDefinition:
  """Definition of an event stream."""

  name: str
  stream: types.EventStream[event_bus.Event]
  is_image_stream: bool


class Embodiment(metaclass=abc.ABCMeta):
  """Base class for robot embodiments.

  An embodiment bundles robot-specific tools (e.g., motion control) and event
  streams (e.g., cameras, sensors) into a single reusable component. It manages
  the lifecycle of these streams and provides a unified interface for agents to
  interact with the physical or simulated hardware.
  """

  def __init__(
      self,
      bus: event_bus.EventBus,
  ):
    self._event_streams = self._create_event_streams()
    self._tools = self._create_tools(bus)
    self._stream_handler = event_stream_handler.EventStreamHandler(
        bus=bus,
        streams={
            event_stream_info.name: event_stream_info.stream
            for event_stream_info in self._event_streams
        },
    )

  @property
  def camera_stream_names(self) -> Sequence[str]:
    """Returns the names of the camera streams."""
    return [
        event_stream_info.name
        for event_stream_info in self._event_streams
        if event_stream_info.is_image_stream
    ]

  @abc.abstractmethod
  def _create_event_streams(self) -> Sequence[EventStreamDefinition]:
    """Create the event streams."""

  @property
  def event_streams(self) -> Sequence[EventStreamDefinition]:
    """Returns the event streams."""
    return self._event_streams

  @abc.abstractmethod
  def _create_tools(self, bus: event_bus.EventBus) -> Sequence[tool.Tool]:
    """Create the tools."""

  @property
  def tools(self) -> Sequence[tool.Tool]:
    """Returns the tools."""
    return self._tools

  async def connect(self):
    """Connects to the robot and starts streaming events to the event bus."""
    await self._stream_handler.connect()

  async def disconnect(self):
    """Disconnects to the robot and stops streaming events to the event bus."""
    await self._stream_handler.disconnect()
