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

"""Aloha embodiment."""

from typing import Sequence

from google.genai import types

from safari_sdk.agent.framework.embodiments import aloha_fast_api_endpoints
from safari_sdk.agent.framework.embodiments import embodiment
from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.tools import fast_api_tools
from safari_sdk.agent.framework.tools import tool


_Endpoints = aloha_fast_api_endpoints.FastApiEndpoints

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8888


# List of endpoints provided by this embodiment.
OVERHEAD_ENDPOINT = _Endpoints.OVERHEAD_CAMERA_STREAM.value.path
LEFT_WRIST_ENDPOINT = _Endpoints.LEFT_WRIST_CAMERA_STREAM.value.path
RIGHT_WRIST_ENDPOINT = _Endpoints.RIGHT_WRIST_CAMERA_STREAM.value.path
WORMS_EYE_ENDPOINT = _Endpoints.WORMS_EYE_CAMERA_STREAM.value.path

# List of tools provided by this embodiment.
RUN_INSTRUCTION_TOOL_NAME = "robot_run_instruction"
RESET_TOOL_NAME = "robot_reset"
STOP_TOOL_NAME = "robot_stop"
OPEN_GRIPPER_TOOL_NAME = "robot_open_gripper"
SLEEP_TOOL_NAME = "robot_sleep"


class AlohaEmbodiment(embodiment.Embodiment):
  """Robot handler connecting FastAPI robots to the event bus."""

  def __init__(
      self,
      bus: event_bus.EventBus,
      server: str = DEFAULT_HOST,
      port: int = DEFAULT_PORT,
  ):
    self._robot_server = f"http://{server}:{port}"
    super().__init__(bus=bus)

  def _create_event_streams(self) -> Sequence[embodiment.EventStreamDefinition]:
    return [
        embodiment.EventStreamDefinition(
            name=endpoint,
            stream=fast_api_tools.FastApiVideoStream(
                server=self._robot_server,
                endpoint=endpoint,
                stream_name=endpoint,
            ).stream(),
            is_image_stream=True,
        )
        for endpoint in (
            OVERHEAD_ENDPOINT,
            LEFT_WRIST_ENDPOINT,
            RIGHT_WRIST_ENDPOINT,
            WORMS_EYE_ENDPOINT,
        )
    ]

  def _create_tools(self, bus: event_bus.EventBus) -> Sequence[tool.Tool]:
    """Instantiates a list of tools specific to the Aloha robot."""
    return [
        tool.Tool(
            fn=fast_api_tools.FastApiGet(
                self._robot_server,
                _Endpoints.RUN_INSTRUCTION.value,
                ["instruction"],
            ),
            declaration=types.FunctionDeclaration(
                name=RUN_INSTRUCTION_TOOL_NAME,
                description="""
                    Run a natural language instruction on the robot. This will
                    cause the agent on the robot to execute low-level actions
                    (in a run high frequency run loop) needed to complete the
                    language instruction.
                    This function returns right away (before the language
                    instruction is completed) and the agent on the robot will
                    not stop the robot even if the language instruction is
                    completed.
                    The language instruction needs to be specific, unambiguous,
                    atomic (involving single action with one object)
                    instructions, e.g.:
                        * "bring the cup to the table"
                        * "put the book on the table"
                        * "put the red dice in the green tray"
                        * "push the bowl to the left"
                """,
                behavior=types.Behavior.NON_BLOCKING,
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "instruction": types.Schema(
                            type=types.Type.STRING,
                            description=(
                                "A short, clear instruction for the robot (e.g."
                                " 'pick up cup', 'place on table', 'move left')"
                            ),
                        ),
                    },
                    required=["instruction"],
                ),
            ),
            bus=bus,
        ),
        tool.Tool(
            fn=fast_api_tools.FastApiGet(
                self._robot_server, _Endpoints.RESET.value, []
            ),
            declaration=types.FunctionDeclaration(
                name=RESET_TOOL_NAME,
                description="Reset the robot to its default starting position.",
                behavior=types.Behavior.NON_BLOCKING,
            ),
            bus=bus,
        ),
        tool.Tool(
            fn=fast_api_tools.FastApiGet(
                self._robot_server, _Endpoints.STOP_INSTRUCTION.value, []
            ),
            declaration=types.FunctionDeclaration(
                name=STOP_TOOL_NAME,
                description="Stop the robot in its current position.",
                behavior=types.Behavior.NON_BLOCKING,
            ),
            bus=bus,
        ),
        tool.Tool(
            fn=fast_api_tools.FastApiGet(
                self._robot_server, _Endpoints.OPEN_GRIPPERS.value, []
            ),
            declaration=types.FunctionDeclaration(
                name=OPEN_GRIPPER_TOOL_NAME,
                description="Open the grippers of the robot.",
                behavior=types.Behavior.NON_BLOCKING,
            ),
            bus=bus,
        ),
        tool.Tool(
            fn=fast_api_tools.FastApiGet(
                self._robot_server, _Endpoints.SLEEP.value, []
            ),
            declaration=types.FunctionDeclaration(
                name=SLEEP_TOOL_NAME,
                description=(
                    "Put the robot in a safe resting position for shutdown."
                ),
                behavior=types.Behavior.NON_BLOCKING,
            ),
            bus=bus,
        ),
    ]
