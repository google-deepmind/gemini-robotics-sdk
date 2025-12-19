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

"""Run instruction until done tool."""

import asyncio
import time

from absl import logging
from google.genai import types

from safari_sdk.agent.framework import config as framework_config
from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.tools import success_detection
from safari_sdk.agent.framework.tools import tool


_DEFAULT_FUNCTION_DECLARATION = types.FunctionDeclaration(
    name="run_instruction_until_done",
    description="""
  Sends a natural language instruction to the robot. The robot may perform needed actions to follow the instruciton.
  This function will return when the instruction completed or a time limit is reached.
  The language instruction needs to be specific and atomic, e.g.:
    * "bring the cup to the table"
    * "put the book on the table"
    * "put the red dice in the green tray"
    * "push the bowl to the left"
    * "open the middle drawer"
    """,
    behavior=types.Behavior.NON_BLOCKING,
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "instruction": types.Schema(
                type=types.Type.STRING,
                description=(
                    "The natural language instruction sent to the robot which"
                    " should be specific and atomic."
                ),
            ),
        },
        required=["instruction"],
    ),
)


class RunInstructionUntilDoneTool(tool.Tool):
  """A tool that runs an instruction until a success condition is met or a timeout occurs.

  This tool wraps a `run_instruction_tool` and uses a `success_detector_tool`
  to determine when the instruction has been successfully completed. It can
  also stop the robot via a `stop_tool` upon success.
  """

  def __init__(
      self,
      bus: event_bus.EventBus,
      config: framework_config.AgentFrameworkConfig,
      success_detector_tool: success_detection.AbstractSuccessDetectionTool,
      run_instruction_tool: tool.Tool,
      stop_tool: tool.Tool | None = None,
      minimum_run_duration: float = 0.0,
      maximum_run_duration: float | None = None,
      function_declaration: types.FunctionDeclaration = (
          _DEFAULT_FUNCTION_DECLARATION
      ),
      make_success_known_to_agent: bool = True,
  ):
    declaration = function_declaration
    super().__init__(
        fn=self.run_instruction_until_done,
        declaration=declaration,
        bus=bus,
    )
    self._config = config
    self._call_id = None
    self._success_detector_tool = success_detector_tool
    self._run_instruction_tool = run_instruction_tool
    self._stop_tool = stop_tool
    if maximum_run_duration is not None:
      self._success_detector_tool.set_timeout_seconds(maximum_run_duration)
    self._minimum_run_duration = minimum_run_duration
    self._make_success_known_to_agent = make_success_known_to_agent

  def _handle_tool_call_cancellation_events(self, event: event_bus.Event):
    """Handle the subscribed events."""
    # Automatically subscribed to via superclass.
    if self._call_id in event.data.ids:
      self._tool_call_cancelled = True
      logging.info(
          "Tool call cancelled for run_instruction_until_done. id: %s.",
          self._call_id,
      )

  async def run_instruction_until_done(
      self, instruction: str, call_id: str
  ) -> types.FunctionResponse:
    """Runs the robot with an instruction, optional stops the robot when done."""
    self._call_id = call_id
    start_time = time.time()
    # Sends instruction to the robot.
    await self._run_instruction_tool.fn(instruction, call_id)
    # Check if instruction was successful.
    sd_response: types.FunctionResponse = await self._success_detector_tool.fn(
        instruction, call_id
    )
    # Wait until minimum run duration is reached.
    if self._minimum_run_duration is not None:
      await asyncio.sleep(
          self._minimum_run_duration - (time.time() - start_time)
      )
    # Stop the robot if needed.
    if (
        self._stop_tool
        and sd_response.response["task_success"]
        and self._config is not None
        and self._config.stop_on_success
    ):
      await self._stop_tool.fn(call_id)
      stopped = True
    else:
      stopped = False
    # Return response to the agent.
    response = {
        "subtask": instruction,
        "is_robot_stopped": stopped,
    }
    if sd_response.response["timeout"]:
      response["subtask_status"] = "time limit reached"
    elif self._make_success_known_to_agent:
      response["subtask_success"] = sd_response.response["task_success"]
    return types.FunctionResponse(
        response=response,
        will_continue=False,
    )
