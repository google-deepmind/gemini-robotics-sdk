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

"""A simple Aloha agent."""

from typing import Sequence

from safari_sdk.agent.framework import config as framework_config
from safari_sdk.agent.framework import flags as agentic_flags
from safari_sdk.agent.framework.agents import agent
from safari_sdk.agent.framework.embodiments import aloha
from safari_sdk.agent.framework.event_bus import audio_event_handler
from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.tools import run_instruction_until_done
from safari_sdk.agent.framework.tools import scene_description
from safari_sdk.agent.framework.tools import success_detection
from safari_sdk.agent.framework.tools import tool


_SYSTEM_PROMPT = """You are Aloha, powered by Gemini Robotics.
{meow_block}
You have two arms bolted on the table and you cannot move your base.
Your sole purpose is to fulfill user requests by controlling two arms.

Guideline 1: ONLY RUN A SINGLE FUNCTION AT A TIME.

Example:

{function_call_example_block}

Guideline 2: Hold helpful conversation with the user.

Guideline 3: When the user is making a request for the robot to perform physical actions:
  * You should first think step by step to decide what function to call and with what arguments.
  * `run_instruction_until_done` is the primary function for making robots perform physical actions.
  * The instruction should be atomic, i.e. involves a single, low-level action (grab/hold, pick, place, move, push/pull, open/close gripper, wipe, stack, zip/unzip, fold).
  * The instruction should include a specific object and a concrete destination (e.g., 'place the red cup on the white bowl', 'pick up the coke can from the blue tray').
  * When the user's request is complex, you should break it down into simpler instructions. Prefer to NOT ask the user for clarification.
  * If request is fulfilled based on visual reasoning, announce that you have fulfilled the request and stop.

Guideline 4: Always cancel any ongoing `run_instruction_until_done` if
  * The user has explicitly asked you to stop or reset.
  * The user has explicitly asked you to switch to a different instruction.

Guideline 5: Google Search
  * Use Google Search to look up information when the user is asking you about location based or real-time based information.
  * When you provide response based on GOOGLE search results, you should only provide concise answers (strictly less than 10 words).

{remind_default_api_in_prompt_block}
{stop_when_done_block}
{scene_description_block}
"""


_FUNCTION_CALL_EXAMPLE_BLOCK = """
Here is an example of an interaction with the user and the robot:
User: Put all the green objects in the box.
You: default_api:run_instruction_until_done(instruction="put the green block in the box")
some time passes...
You get the function response with:
response={{
    'is_robot_stopped': True,
    'subtask': 'put the blue block into the blue plate',
    'subtask_success': True
  }},
  will_continue=False
You: default_api:run_instruction_until_done(instruction="put the green shoe in the box")
some time passes...
You get the function response with:
response={{
    'is_robot_stopped': True,
    'subtask': 'put the blue block into the blue plate',
    'subtask_success': True
  }},
  will_continue=False
no more green object are outside of the box
You: I am done, how can I help you?

First make a function call:

FunctionCall(
  args={{
    'instruction': 'put the blue block into the blue plate',
  }},
  id='function-call-16502400002017259751',
  name='run_instruction_until_done'
)

You will see a immediate function acknowledgement saying the function is running:

FunctionResponse(
  id='function-call-16502400002017259751',
  name='run_instruction_until_done',
  response='Running...',
  will_continue=True
)

Then, after some time, you will see the function response with the result:

FunctionResponse(
  id='function-call-16502400002017259751',
  name='run_instruction_until_done',
  response={{
    'is_robot_stopped': True,
    'subtask': 'put the blue block into the blue plate',
    'subtask_success': True
  }},
  will_continue=False
)

Only after that, you can make another function call with the next instruction:

FunctionCall(
  args={{
    'instruction': 'put the green block into the green plate',
  }},
  id='function-call-16502400002017259752',
  name='run_instruction_until_done'
)
"""

_FUNCTION_CALL_EXAMPLE_BLOCK_DRY_RUN = """
Here is an example of an interaction with the user and the robot:
User: Put all the green objects in the box.
You: default_api:run_instruction_until_done(instruction="put the green block in the box")
some time passes...
User: Well done!
You: default_api:run_instruction_until_done(instruction="put the green shoe in the box")
some time passes...
User: Good job!
no more green object are outside of the box
You: default_api:robot_stop()
You: I am done, how can I help you?

First make a function call:

FunctionCall(
  args={{
    'instruction': 'put the blue block into the blue plate',
  }},
  id='function-call-16502400002017259751',
  name='run_instruction_until_done'
)

You will see a immediate function acknowledgement saying the function is running:

FunctionResponse(
  id='function-call-16502400002017259751',
  name='run_instruction_until_done',
  response='Running...',
  will_continue=True
)

Then, after some time, the user will tell you when you can move on to the next instruction:

User: Good job!

Only after that, you can make another function call with the next instruction:

FunctionCall(
  args={{
    'instruction': 'put the green block into the green plate',
  }},
  id='function-call-16502400002017259752',
  name='run_instruction_until_done'
)
"""


_STOP_ON_SUCCESS_BLOCK = """
When you actually fufilled the user's request, call robot_stop to stop the robot. Do not call any other function after calling robot_stop.
"""

SCENE_DESCRIPTION_BLOCK = """
Call `describe_scene` to get a detailed description of the scene when the user explicitly asks you to describe the scene.
When you tell the user what you see, be very brief.
"""

_LANGUAGE_CONTROL_BLOCK = """
CONVERSATION LANGUAGE
The default language for the conversation is English. However, if the user mentions a language other than English, eg by saying "I will now speak to you in Greek" or "I would now like to speak in Greek", acknowledge in English and switch your language to the
requested language, both expecting the user to speak in that language and responding in that language. FUNCTION CALLS AND INSTRUCTIONS SHOULD ONLY BE IN ENGLISH HOWEVER!!!
DO NOT CALL run_instruction_until_done OR ANY OTHER FUNCTION IN A LANGUAGE OTHER THAN ENGLISH.
"""

_BASE_URL_KEY = "base_url"

_SD_CAMERA_ENDPOINTS = [
    aloha.OVERHEAD_ENDPOINT,
    aloha.WORMS_EYE_ENDPOINT,
    aloha.LEFT_WRIST_ENDPOINT,
    aloha.RIGHT_WRIST_ENDPOINT,
]


class SimpleAlohaAgent(agent.Agent):
  """Agent subclass that contains tools for Aloha."""

  def __init__(
      self,
      bus: event_bus.EventBus,
      config: framework_config.AgentFrameworkConfig,
  ):
    """Initializes the Aloha agent."""
    embodiment = aloha.AlohaEmbodiment(
        bus=bus,
        server=config.robot_backend_host,
        port=config.robot_backend_port,
    )
    self._config = config
    self._api_key = config.api_key
    self._bus = bus
    self._audio_handler = None
    if config.enable_audio_output or config.enable_audio_input:
      self._audio_handler = audio_event_handler.AudioHandler(
          bus=bus,
          enable_audio_input=config.enable_audio_input,
          enable_audio_output=config.enable_audio_output,
          listen_while_speaking=config.listen_while_speaking,
      )
    if config.remind_default_api_in_prompt:
      remind_default_api_in_prompt_block = """
          \nREMEMBER TO USE default_api:<fn_name> WHEN MAKING FUNCTION CALLS.\n
          """
    else:
      remind_default_api_in_prompt_block = ""
    if config.meow_mode:
      meow_block = "Meow often."
    else:
      meow_block = ""
    dry_run = config.sd_dry_run
    if dry_run:
      function_call_example_block = _FUNCTION_CALL_EXAMPLE_BLOCK_DRY_RUN
    else:
      function_call_example_block = _FUNCTION_CALL_EXAMPLE_BLOCK
    if config.stop_on_success:
      stop_when_done_block = _STOP_ON_SUCCESS_BLOCK
    else:
      stop_when_done_block = ""
    if config.use_scene_description:
      scene_description_block = SCENE_DESCRIPTION_BLOCK
    else:
      scene_description_block = ""
    if config.use_language_control:
      language_control_block = _LANGUAGE_CONTROL_BLOCK
    else:
      language_control_block = ""
    system_prompt = _SYSTEM_PROMPT.format(
        remind_default_api_in_prompt_block=remind_default_api_in_prompt_block,
        stop_when_done_block=stop_when_done_block,
        scene_description_block=scene_description_block,
        language_control_block=language_control_block,
        function_call_example_block=function_call_example_block,
        meow_block=meow_block,
    )
    super().__init__(
        bus=bus,
        config=config,
        embodiment=embodiment,
        system_prompt=system_prompt,
        http_options={_BASE_URL_KEY: config.base_url},
        initial_camera_names=({
            aloha.OVERHEAD_ENDPOINT: (
                f"Image from camera {aloha.OVERHEAD_ENDPOINT}"
            )
        }),
    )

  def _get_sd_tool(
      self, tool_name: agentic_flags.SDToolName
  ) -> success_detection.VisionSuccessDetectionTool:
    match tool_name:
      case (
          agentic_flags.SDToolName.SUBTASK_SUCCESS_DETECTOR
          | agentic_flags.SDToolName.SUBTASK_SUCCESS_DETECTOR_V2
          | agentic_flags.SDToolName.SUBTASK_SUCCESS_DETECTOR_V3
          | agentic_flags.SDToolName.SUBTASK_SUCCESS_DETECTOR_V4
      ):
        sd_tool = success_detection.SubtaskSuccessDetectorV4(
            self._bus,
            config=self._config,
            api_key=self._api_key,
            sd_camera_endpoint_names=_SD_CAMERA_ENDPOINTS,
        )
      case _:
        raise ValueError(
            f"Unsupported success detection tool: {self._config.sd_tool_name}"
        )
    if self._config.sd_timeout_seconds is not None:
      sd_tool.set_timeout_seconds(  # pytype: disable=attribute-error
          self._config.sd_timeout_seconds
      )
    return sd_tool

  def _get_all_tools(
      self,
      embodiment_tools: Sequence[tool.Tool],
  ) -> Sequence[agent.ToolUseConfig]:
    """Returns the agentic tools for the AlohaAgent."""
    # We expose all tools to the agent except for the run instruction and
    # sleep tools. The run instruction tool will be replaced by a more
    # sephisiticaated `run_instruction_until_done` tool.
    configs = [
        agent.ToolUseConfig(
            tool=embodiment_tool,
            exposed_to_agent=(
                embodiment_tool.declaration.name
                != aloha.RUN_INSTRUCTION_TOOL_NAME
                and embodiment_tool.declaration.name != aloha.SLEEP_TOOL_NAME
            ),
        )
        for embodiment_tool in embodiment_tools
    ]

    configs.append(
        agent.ToolUseConfig(
            tool=run_instruction_until_done.RunInstructionUntilDoneTool(
                self._bus,
                config=self._config,
                success_detector_tool=self._get_sd_tool(
                    self._config.sd_tool_name
                ),
                run_instruction_tool=self._get_tool_by_name(
                    embodiment_tools, aloha.RUN_INSTRUCTION_TOOL_NAME
                ),
                stop_tool=self._get_tool_by_name(
                    embodiment_tools, aloha.STOP_TOOL_NAME
                ),
            ),
            exposed_to_agent=True,
        )
    )

    if self._config.use_scene_description:
      configs.append(
          agent.ToolUseConfig(
              tool=scene_description.SceneDescriptionTool(
                  self._bus,
                  config=self._config,
                  api_key=self._api_key,
                  camera_endpoint_names=_SD_CAMERA_ENDPOINTS,
              ),
              exposed_to_agent=True,
          )
      )

    return configs

  async def connect(self):
    """Connects the agent to the event bus."""
    await super().connect()
    if self._audio_handler:
      await self._audio_handler.connect()

  async def disconnect(self):
    """Disconnects the agent from the event bus."""
    await super().disconnect()
    if self._audio_handler:
      await self._audio_handler.disconnect()
