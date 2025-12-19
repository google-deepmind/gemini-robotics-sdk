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

"""Live API based robotics agent."""

import abc
from collections.abc import Mapping
import dataclasses
from typing import Sequence

from google.genai import types

from safari_sdk.agent.framework import config as framework_config
from safari_sdk.agent.framework.embodiments import embodiment as embodiment_lib
from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.event_bus import tool_call_event_handler
from safari_sdk.agent.framework.live_api import live_handler
from safari_sdk.agent.framework.tools import tool as tool_lib


@dataclasses.dataclass(frozen=True)
class ToolUseConfig:
  """Configuration for how tools are used by the Agent."""

  # The tool to register with the handlers.
  tool: tool_lib.Tool

  # Whether or not the tool is exposed to the Agent. Some tools may be called
  # only by other tools or by human or UI "agents". In the latter case, they may
  # still need to be registered with the event bus, but should not be exposed
  # to the orchestration model.
  exposed_to_agent: bool


class Agent(metaclass=abc.ABCMeta):
  """Live API based robotics agent."""

  def __init__(
      self,
      bus: event_bus.EventBus,
      config: framework_config.AgentFrameworkConfig,
      embodiment: embodiment_lib.Embodiment,
      system_prompt: str,
      http_options: dict[str, str] | None = None,
      initial_camera_names: Mapping[str, str] | None = None,
      ignore_vision_inputs: bool = False,
  ):
    """Initializes the agent.

    Args:
      bus: The event bus to use for communication.
      config: The agent framework configuration object.
      embodiment: The embodiment to use for the agent.
      system_prompt: The system prompt to use for the agent.
      http_options: The HTTP options to use for the Gemini API VLM.
      initial_camera_names: The names of the cameras to use initially alongside
        how they will be captioned for the model. If not provided, the first
        camera in the list of camera names will be used.
      ignore_vision_inputs: Whether to ignore vision inputs. In this mode, the
        handler will not send any images to the model.
    """
    self._config = config
    self._embodiment = embodiment
    all_tools_with_use = self._get_all_tools(
        embodiment_tools=embodiment.tools,
    )
    all_tools = [tool_use.tool for tool_use in all_tools_with_use]
    exposed_tools = [
        tool_use.tool
        for tool_use in all_tools_with_use
        if tool_use.exposed_to_agent
    ]

    # Subscribe all tools to TOOL_CALL events and make them publish TOOL_RESULT
    # events when the tool is called.
    tool_call_event_handler.ToolCallEventHandler(
        bus=bus,
        tool_dict={tool.declaration.name: tool for tool in all_tools},
    )

    live_api_config = self._create_live_api_config(
        config,
        exposed_tools,
        system_prompt=system_prompt,
    )
    self._live_handler = live_handler.GeminiLiveAPIHandler(
        bus=bus,
        config=config,
        live_config=live_api_config,
        camera_names=embodiment.camera_stream_names,
        initial_camera_names=initial_camera_names,
        http_options=http_options,
        ignore_image_inputs=ignore_vision_inputs,
    )
    self._live_handler.register_event_subscribers()

  @abc.abstractmethod
  def _get_all_tools(
      self,
      embodiment_tools: Sequence[tool_lib.Tool],
  ) -> Sequence[ToolUseConfig]:
    """Returns the tools and mark which are exposed to the Agent.

    Note that if the `embodiment_tools` are to be exposed to the Agent, they
    should be included in the returned sequence of tools.

    Args:
      embodiment_tools: The tools from the embodiment.

    Returns:
      The tools to be used by the Agent.
    """
    pass

  def get_camera_stream_names(self) -> Sequence[str]:
    """Returns the camera stream names for the Agent."""
    return self._embodiment.camera_stream_names

  def _get_audio_transcription_configs(
      self,
      enable_audio_input: bool,
      enable_audio_output: bool,
      enable_audio_transcription: bool,
  ) -> tuple[
      types.AudioTranscriptionConfig | None,
      types.AudioTranscriptionConfig | None,
  ]:
    """Returns audio transcription configs for input and output.

    Args:
      enable_audio_input: Whether audio input is enabled.
      enable_audio_output: Whether audio output is enabled.
      enable_audio_transcription: Whether audio transcription is enabled.

    Returns:
      A tuple of (input_audio_transcription, output_audio_transcription).
    """
    input_audio_transcription = None
    output_audio_transcription = None
    if enable_audio_transcription:
      if enable_audio_input:
        input_audio_transcription = types.AudioTranscriptionConfig()
      if enable_audio_output:
        output_audio_transcription = types.AudioTranscriptionConfig()
    return input_audio_transcription, output_audio_transcription

  def _get_speech_config(
      self, output_audio_voice_name: str | None
  ) -> types.SpeechConfig | None:
    if output_audio_voice_name:
      return types.SpeechConfig(
          voice_config=types.VoiceConfig(
              prebuilt_voice_config=types.PrebuiltVoiceConfig(
                  voice_name=output_audio_voice_name
              ),
          ),
      )
    return None

  def _get_turn_coverage(
      self, only_activity_coverage: bool
  ) -> types.TurnCoverage:
    if only_activity_coverage:
      return types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY
    return types.TurnCoverage.TURN_INCLUDES_ALL_INPUT

  def _get_context_window_compression(
      self,
  ) -> types.ContextWindowCompressionConfig | None:
    if self._config.enable_context_window_compression:
      return types.ContextWindowCompressionConfig(
          trigger_tokens=self._config.context_compression_trigger_tokens,
          sliding_window=types.SlidingWindow(
              target_tokens=self._config.context_compression_sliding_window_target,
          ),
      )
    return None

  def _create_live_api_config(
      self,
      config: framework_config.AgentFrameworkConfig,
      toolset: Sequence[tool_lib.Tool],
      system_prompt: str,
  ) -> types.LiveConnectConfigDict:
    """Creates the Live API config."""
    function_declarations = [tool.declaration for tool in toolset]
    response_modality = (
        types.Modality.AUDIO
        if config.enable_audio_output
        else types.Modality.TEXT
    )
    input_audio_transcription, output_audio_transcription = (
        self._get_audio_transcription_configs(
            config.enable_audio_input,
            config.enable_audio_output,
            config.enable_audio_transcription,
        )
    )
    speech_config = self._get_speech_config(config.output_audio_voice_name)
    turn_coverage = self._get_turn_coverage(config.only_activity_coverage)
    context_window_compression = self._get_context_window_compression()

    return types.LiveConnectConfigDict(
        system_instruction=system_prompt,
        tools=[
            types.Tool(function_declarations=tuple(function_declarations)),
            types.Tool(google_search=types.GoogleSearch()),
        ],
        response_modalities=[response_modality],
        realtime_input_config=types.RealtimeInputConfig(
            turn_coverage=turn_coverage,
        ),
        input_audio_transcription=input_audio_transcription,
        output_audio_transcription=output_audio_transcription,
        speech_config=speech_config,
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
        context_window_compression=context_window_compression,
    )

  def _get_tool_by_name(
      self,
      tools: Sequence[tool_lib.Tool],
      name: str,
  ) -> tool_lib.Tool:
    """Returns the tool with the given name from a list of tools."""
    for t in tools:
      if t.declaration.name == name:
        return t
    raise ValueError(f"Tool {name} not found in the provided tools.")

  async def connect(self):
    await self._live_handler.connect()
    await self._embodiment.connect()

  async def disconnect(self):
    await self._embodiment.disconnect()
    await self._live_handler.disconnect()
