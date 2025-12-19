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

"""Run Aloha agent.

Entry point for running various Aloha agent implementations. Select an agent
by passing the --agent_name flag.

Available agents:
  - simple_agent: Basic Aloha agent

Example:
  python run.py --agent_name=simple_agent
"""

import asyncio
import signal

from absl import app
from absl import flags
from absl import logging

import simple_agent
from safari_sdk.agent.framework import agent_framework
from safari_sdk.agent.framework import config as framework_config
from safari_sdk.agent.framework.event_bus import event_bus


def main(_) -> None:
  config = framework_config.AgentFrameworkConfig.create()
  bus = event_bus.EventBus(config=config)
  match config.agent_name:
    case "simple_agent":
      agent_cls = simple_agent.SimpleAlohaAgent
    case _:
      raise ValueError("Unsupported agent name: %s" % config.agent_name)
  agent_instance = agent_cls(
      bus=bus,
      config=config,
  )

  framework = agent_framework.AgentFramework(
      bus=bus,
      config=config,
      agent_instance=agent_instance,
  )

  async def run_main():
    loop = asyncio.get_running_loop()
    main_task = loop.create_task(framework.run())

    def _keyboard_interrupt_handler():
      logging.info("Keyboard interrupt received, cancelling main task.")
      main_task.cancel()
      loop.remove_signal_handler(signal.SIGINT)

    loop.add_signal_handler(signal.SIGINT, _keyboard_interrupt_handler)

    try:
      await main_task
    except asyncio.CancelledError:
      logging.info("Main task cancelled.")

  asyncio.run(run_main())


if __name__ == "__main__":
  flags.FLAGS.mark_as_parsed()
  app.run(main)
