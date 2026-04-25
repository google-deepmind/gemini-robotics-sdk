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

"""Launch and run the agent framework."""

import asyncio
from collections.abc import Sequence
import dataclasses
from typing import Protocol

from absl import logging

from safari_sdk.agent.framework import config as framework_config
from safari_sdk.agent.framework import types
from safari_sdk.agent.framework.agents import agent as agent_lib
from safari_sdk.agent.framework.comms import external_controller
from safari_sdk.agent.framework.comms import log_handler
from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.ui import terminal_ui


@dataclasses.dataclass(frozen=True)
class EventSubscriberConfig:
  """Configuration for an event subscriber that can be added to a bus."""

  event_types: list[event_bus.EventType]
  event_handler: event_bus.EventBusHandlerSignature


class FrameworkComponent(Protocol):
  """A component that can be added to the agent framework."""

  async def connect(self) -> None:
    """Connects to the component."""

  async def disconnect(self) -> None:
    """Disconnects from the component."""


class AgentFramework:
  """Launch and run the agent framework."""

  def __init__(
      self,
      bus: event_bus.EventBus,
      config: framework_config.AgentFrameworkConfig,
      agent_instance: agent_lib.Agent,
      additional_components: Sequence[FrameworkComponent] | None = None,
  ):
    """Initializes the agent framework.

    Args:
      bus: The event bus to use for the framework. If not provided, a new event
        bus will be created.
      config: The agent framework configuration object.
      agent_instance: The agent instance to use for the framework.
      additional_components: Additional components to add to the framework. Must
        have a connect() and disconnect() method.
    """

    self._bus = bus
    self._config = config
    self._components = list(additional_components or [])
    self._agent = agent_instance
    self._has_been_shutdown = False

    match config.control_mode:
      case types.ControlMode.TERMINAL_ONLY:
        self._components.append(
            terminal_ui.TerminalUI(bus=self._bus, config=self._config)
        )
      case types.ControlMode.LAUNCH_SERVER:
        self._components.append(
            external_controller.ExternalControllerFastAPIServer(
                bus=self._bus,
                host=config.external_controller_host,
                port=config.external_controller_port,
            )
        )

      case types.ControlMode.SERVER_AND_TERMINAL:
        self._components.append(
            terminal_ui.TerminalUI(bus=self._bus, config=self._config)
        )
        self._components.append(
            external_controller.ExternalControllerFastAPIServer(
                bus=self._bus,
                host=config.external_controller_host,
                port=config.external_controller_port,
            )
        )
      case _:
        raise ValueError(f"Unsupported control mode: {config.control_mode}")

    self._bus.subscribe(
        event_types=[event_bus.EventType.RESET],
        handler=self._handle_reset_events,
    )

  async def run(self):
    """Runs the framework."""
    if self._config.show_system_prompt_then_exit:
      banner_len = 80
      print("\n" + "=" * banner_len)
      print(" " * 33 + "SYSTEM PROMPT")
      print("=" * banner_len)
      print(self._agent.system_prompt)
      print("\n" + "=" * banner_len)
      print(" " * 33 + "EXPOSED TOOLS")
      print("=" * banner_len)
      for tool in self._agent.exposed_tools:
        header = f"[Tool: {tool.declaration.name}]"
        print(f"\n{header}")
        print("-" * len(header))
        print(tool.declaration)
      print("\n" + "=" * banner_len)
      return

    await self._start_framework()
    try:
      while True:
        await asyncio.sleep(1)
    except asyncio.CancelledError:
      logging.info("Run loop cancelled.")
    finally:
      await self._shutdown(shutdown_event_bus=True)

  async def _start_framework(self):
    """Starts the framework."""
    logging.info("Starting framework...")
    if self._has_been_shutdown:
      logging.warning(
          "Framework has already been shutdown, restarting may result in"
          " undefined behavior."
      )

    try:
      if not self._bus.is_running:
        self._bus.start()
      logging.info("Event bus started.")

      if self._config.publish_logging_events:
        event_log_handler = log_handler.EventBusLogHandler(
            self._bus, min_level=logging.INFO
        )
        event_log_handler.set_loop(asyncio.get_running_loop())
        logging.get_absl_logger().addHandler(event_log_handler)

      logging.info("Connecting agent...")
      await self._agent.connect()
      await asyncio.sleep(2.0)
      logging.info("Agent connected.")

      # Publish all active framework configuration flags as a single event.
      # Placed after agent connection to ensure all subscribers are ready.
      await self._bus.publish(
          event=event_bus.Event(
              type=event_bus.EventType.LOG_SESSION_METADATA,
              source=event_bus.EventSource.MAIN_AGENT,
              data="Global active configuration flags.",
              metadata={"ear_flags": self._config.to_dict()},
          )
      )

      for component in self._components:
        logging.info("Connecting component %s.", component)
        await component.connect()
        logging.info("Component %s connected.", component)
    except asyncio.CancelledError:
      logging.info("Main task cancelled, shutting down.")
    except ConnectionError as e:
      logging.exception("Failed to start framework. Error: %s", e)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.error(
          "An unexpected error occurred in main: %s", e, exc_info=True
      )

  async def _shutdown(self, shutdown_event_bus: bool = False):
    """Shuts down the framework and any running components."""
    logging.info("Shutting down framework...")

    # Disconnect agent and components concurrently.
    tasks = [self._agent.disconnect()]
    for component in self._components:
      tasks.append(component.disconnect())
    await asyncio.gather(*tasks, return_exceptions=True)
    if shutdown_event_bus:
      await self._bus.shutdown()
    self._has_been_shutdown = True
    logging.info("Framework shutdown complete.")

  async def reset_framework(self):
    logging.info("Resetting framework...")
    await self._shutdown(shutdown_event_bus=False)
    await asyncio.sleep(1.0)
    await self._start_framework()
    logging.info("Framework reset complete.")

  def register_event_subscribers(
      self, event_subscriber_configs: list[EventSubscriberConfig]
  ):
    """Registers additional event subscribers to the event bus."""
    for event_subscriber_config in event_subscriber_configs:
      self._bus.subscribe(
          event_types=event_subscriber_config.event_types,
          handler=event_subscriber_config.event_handler,
      )

  def get_bus(self) -> event_bus.EventBus:
    """Returns the current event bus."""
    return self._bus

  async def _handle_reset_events(self, event: event_bus.Event) -> None:
    """Callback for reset events."""
    logging.info("Reset event received.")
    del event  # Unused.
    await self.reset_framework()
