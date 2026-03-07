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
  - run_for_duration_agent: Unified agent (use --use_streaming=False for
  non-streaming)
  - simple_agent: Basic Aloha agent

Example:
  # Streaming mode (default)
  python run.py --agent_name=run_for_duration_agent

  # Non-streaming mode
  python run.py --agent_name=run_for_duration_agent --use_streaming=False
"""

import asyncio
import signal

from absl import app
from absl import flags
from absl import logging

import run_for_duration_agent
import simple_agent
from safari_sdk.agent.framework import agent_framework
from safari_sdk.agent.framework import config as framework_config
from safari_sdk.agent.framework.event_bus import event_bus


def main(_) -> None:
  config = framework_config.AgentFrameworkConfig.create()
  logging.info(
      "Config created: agent_name=%s, api_key=%s, model_name=%s",
      config.agent_name,
      "SET" if config.api_key else "NOT SET",
      config.agent_model_name,
  )
  bus = event_bus.EventBus(config=config)

  agent_instance = None
  match config.agent_name:
    case "aloha_agent" | "run_for_duration_agent":
      is_streaming = config.orchestrator_handler_type.name == "STREAMING"
      agent_instance = run_for_duration_agent.AlohaAgent(
          bus=bus,
          config=config,
      )
      logging.info(
          "Creating AlohaAgent in %s mode",
          "streaming" if is_streaming else "non-streaming",
      )
    case "simple_agent":
      agent_instance = simple_agent.SimpleAlohaAgent(
          bus=bus,
          config=config,
      )
    case _:
      raise ValueError("Unsupported agent name: %s" % config.agent_name)

  logging.info("Creating agent: %s", type(agent_instance).__name__)

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
