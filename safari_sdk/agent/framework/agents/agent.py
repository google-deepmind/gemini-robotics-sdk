# Copyright 2025 Google LLC
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

"""Agent with shared functionality for all orchestrator handler modes.

This module provides the Agent abstract class that contains all the shared
logic for streaming, non-streaming GenAI, and Evergreen handler modes.

Subclasses only need to implement _get_all_tools() and can switch between
modes using config.orchestrator_handler_type.
"""

import abc
from collections.abc import Mapping
import dataclasses
from typing import Sequence

from absl import logging
from google.genai import types as genai_types

from safari_sdk.agent.framework import config as framework_config
from safari_sdk.agent.framework import types as framework_types
from safari_sdk.agent.framework.agents import handler as handler_lib
from safari_sdk.agent.framework.embodiments import embodiment as embodiment_lib
from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.event_bus import tool_call_event_handler
from safari_sdk.agent.framework.inference_handlers import live_handler
from safari_sdk.agent.framework.inference_handlers import unary_genai_handler
from safari_sdk.agent.framework.tools import tool as tool_lib
from safari_sdk.agent.framework.utils import http_options as http_options_util


# Keywords that indicate a Live API (streaming-only) model.
_LIVE_MODEL_KEYWORDS = ("live",)


def is_live_model(model_name: str | None) -> bool:
  """Returns True if the model name indicates a Live API model."""
  if not model_name:
    return False
  model_lower = model_name.lower()
  return any(keyword in model_lower for keyword in _LIVE_MODEL_KEYWORDS)


def validate_model_mode_compatibility(
    model_name: str | None,
    handler_type: framework_types.OrchestratorHandlerType,
) -> None:
  """Validates that the model is compatible with the selected mode.

  Args:
    model_name: The model name from config.
    handler_type: The orchestrator handler type from config.

  Raises:
    ValueError: If there's a clear incompatibility that will cause failure.
  """
  if not model_name:
    return

  is_live = is_live_model(model_name)
  use_streaming = (
      handler_type == framework_types.OrchestratorHandlerType.STREAMING
  )

  if is_live and not use_streaming:
    raise ValueError(
        f"Model '{model_name}' is a Live API model and requires streaming mode."
        " Either:\n  1. Use --agent.orchestrator_handler_type=STREAMING\n"
        "  2. Use a non-streaming model like 'gemini-2.0-flash'"
        " or 'gemini-3-flash-preview'"
    )

  if not is_live and use_streaming:
    logging.warning(
        "Model '%s' may not be optimized for Live API streaming. "
        "If you experience issues, try:\n"
        "  1. Using a Live model like "
        "'gemini-2.5-flash-preview-native-audio-dialog'\n"
        "  2. Using --agent.orchestrator_handler_type=NONSTREAMING_GENAI",
        model_name,
    )


@dataclasses.dataclass(frozen=True)
class ToolUseConfig:
  """Configuration for how tools are used by the Agent.

  This dataclass specifies whether a tool should be exposed to the AI agent
  or only registered internally for use by other components.
  """

  # The tool to register with the handlers.
  tool: tool_lib.Tool

  # Whether or not the tool is exposed to the Agent. Some tools may be called
  # only by other tools or by human or UI "agents". In the latter case, they may
  # still need to be registered with the event bus, but should not be exposed
  # to the orchestration model.
  exposed_to_agent: bool


class Agent(metaclass=abc.ABCMeta):
  """Abstract base class for robotics agents.

  This class provides the shared functionality for agents that interact with
  embodiments via an API handler. It supports streaming (Live API),
  non-streaming (generate_content), and Evergreen modes via the
  config.orchestrator_handler_type setting.

  Subclasses must implement _get_all_tools() to specify which tools are
  available and exposed to the agent.

  Example usage:
    # Streaming mode (default) - set via config
    config = AgentFrameworkConfig.create(
        orchestrator_handler_type=OrchestratorHandlerType.STREAMING)
    agent = MyAgent(bus, config, embodiment, system_prompt)
  """

  def __init__(
      self,
      bus: event_bus.EventBus,
      config: framework_config.AgentFrameworkConfig,
      embodiment: embodiment_lib.Embodiment,
      system_prompt: str,
      stream_name_to_camera_name: Mapping[str, str] | None = None,
      ignore_vision_inputs: bool = False,
  ):
    """Initializes the base agent.

    Args:
      bus: The event bus to use for communication.
      config: The agent framework configuration object.
      embodiment: The embodiment to use for the agent.
      system_prompt: The system prompt to use for the agent.
      stream_name_to_camera_name: Mapping from image stream (endpoint) names to
        camera names. It specifies which camera streams are sent to the
        orchestrator model as well as the names with which to prepend the
        images. If None, the first camera is used and an empty string will be
        prepended. Note that prepending of the camera name is only supported
        under the following conditions for streaming mode:
        `update_vision_after_fr=True` AND
        `turn_coverage=TURN_INCLUDES_ONLY_ACTIVITY`
      ignore_vision_inputs: Whether to ignore vision inputs. In this mode, the
        handler will not send any images to the model.

    Raises:
      ValueError: If the model is incompatible with the selected mode.
    """
    self._config = config
    self._bus = bus
    self._embodiment = embodiment
    self._system_prompt = system_prompt
    self._stream_name_to_camera_name = stream_name_to_camera_name
    self._ignore_vision_inputs = ignore_vision_inputs

    validate_model_mode_compatibility(
        config.agent_model_name,
        config.orchestrator_handler_type,
    )

    # Get the HTTP options.
    self._http_options = http_options_util.get_http_options(config)

    # Get all tools and separate exposed vs internal tools.
    all_tools_with_use = self._get_all_tools(
        embodiment_tools=embodiment.tools,
    )
    self._all_tools = [tool_use.tool for tool_use in all_tools_with_use]
    self._exposed_tools = [
        tool_use.tool
        for tool_use in all_tools_with_use
        if tool_use.exposed_to_agent
    ]

    # Subscribe all tools to TOOL_CALL events and make them publish TOOL_RESULT
    # events when the tool is called.
    tool_call_event_handler.ToolCallEventHandler(
        bus=bus,
        tool_dict={tool.declaration.name: tool for tool in self._all_tools},
    )

    # Create the handler based on streaming mode.
    self._handler = self._create_handler()
    self._handler.register_event_subscribers()

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

  def _create_handler(self) -> handler_lib.Handler:
    """Creates the API handler based on orchestrator_handler_type config.

    Returns:
      The handler instance (Streaming, NonStreaming GenAI, or Evergreen).
    """
    handler_type = self._config.orchestrator_handler_type

    if handler_type == framework_types.OrchestratorHandlerType.STREAMING:
      return self._create_streaming_handler()
    elif (
        handler_type
        == framework_types.OrchestratorHandlerType.NONSTREAMING_GENAI
    ):
      return self._create_unary_genai_handler()
    else:
      raise ValueError(f"Unknown orchestrator handler type: {handler_type}")

  # ---------------------------------------------------------------------------
  # Streaming handler creation (Live API)
  # ---------------------------------------------------------------------------

  def _create_streaming_handler(self) -> handler_lib.Handler:
    """Creates the GeminiLiveAPIHandler for streaming interactions."""
    return live_handler.GeminiLiveAPIHandler(
        bus=self._bus,
        config=self._config,
        live_config=self._create_live_api_config(),
        camera_names=self._embodiment.camera_stream_names,
        stream_name_to_camera_name=self._stream_name_to_camera_name,
        http_options=self._http_options,
        ignore_image_inputs=self._ignore_vision_inputs,
    )

  def _get_audio_transcription_configs(
      self,
      enable_audio_input: bool,
      enable_audio_output: bool,
  ) -> tuple[
      genai_types.AudioTranscriptionConfig | None,
      genai_types.AudioTranscriptionConfig | None,
  ]:
    """Returns audio transcription configs for input and output.

    Audio transcription is always enabled when audio input/output is enabled.
    The `enable_audio_transcription` flag in the config controls only whether
    to display the transcript in the terminal UI, not whether to enable it
    in the Live API.

    Args:
      enable_audio_input: Whether audio input is enabled.
      enable_audio_output: Whether audio output is enabled.

    Returns:
      A tuple of (input_audio_transcription, output_audio_transcription).
    """
    input_audio_transcription = None
    output_audio_transcription = None
    if enable_audio_input:
      input_audio_transcription = genai_types.AudioTranscriptionConfig()
    if enable_audio_output:
      output_audio_transcription = genai_types.AudioTranscriptionConfig()
    return input_audio_transcription, output_audio_transcription

  def _get_speech_config(
      self, output_audio_voice_name: str | None
  ) -> genai_types.SpeechConfig | None:
    if output_audio_voice_name:
      return genai_types.SpeechConfig(
          voice_config=genai_types.VoiceConfig(
              prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                  voice_name=output_audio_voice_name
              ),
          ),
      )
    return None

  def _get_turn_coverage(
      self, only_activity_coverage: bool
  ) -> genai_types.TurnCoverage:
    if only_activity_coverage:
      return genai_types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY
    return genai_types.TurnCoverage.TURN_INCLUDES_ALL_INPUT

  def _get_context_window_compression(
      self,
  ) -> genai_types.ContextWindowCompressionConfig | None:
    if self._config.enable_context_window_compression:
      return genai_types.ContextWindowCompressionConfig(
          trigger_tokens=self._config.context_compression_trigger_tokens,
          sliding_window=genai_types.SlidingWindow(
              target_tokens=self._config.context_compression_sliding_window_target,
          ),
      )
    return None

  def _create_live_api_config(self) -> genai_types.LiveConnectConfigDict:
    """Creates the Live API config."""
    function_declarations = [tool.declaration for tool in self._exposed_tools]
    response_modality = (
        genai_types.Modality.AUDIO
        if self._config.enable_audio_output
        else genai_types.Modality.TEXT
    )
    input_audio_transcription, output_audio_transcription = (
        self._get_audio_transcription_configs(
            self._config.enable_audio_input,
            self._config.enable_audio_output,
        )
    )
    speech_config = self._get_speech_config(
        self._config.output_audio_voice_name
    )
    turn_coverage = self._get_turn_coverage(self._config.only_activity_coverage)
    context_window_compression = self._get_context_window_compression()

    return genai_types.LiveConnectConfigDict(
        system_instruction=self._system_prompt,
        tools=[
            genai_types.Tool(
                function_declarations=tuple(function_declarations)
            ),
            genai_types.Tool(google_search=genai_types.GoogleSearch()),
        ],
        response_modalities=[response_modality],
        realtime_input_config=genai_types.RealtimeInputConfig(
            turn_coverage=turn_coverage,
        ),
        input_audio_transcription=input_audio_transcription,
        output_audio_transcription=output_audio_transcription,
        speech_config=speech_config,
        media_resolution=genai_types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
        context_window_compression=context_window_compression,
    )

  # ---------------------------------------------------------------------------
  # Non-streaming handler creation (generate_content API)
  # ---------------------------------------------------------------------------

  def _create_unary_genai_handler(self) -> handler_lib.Handler:
    """Creates the UnaryGenAIHandler for request-response interactions."""
    tools = self._create_nonstreaming_tools_config()
    tool_config = self._create_nonstreaming_tool_config()
    return unary_genai_handler.UnaryGenAIHandler(
        bus=self._bus,
        config=self._config,
        system_instruction=self._system_prompt,
        tools=tools,
        tool_config=tool_config,
        camera_names=self._embodiment.camera_stream_names,
        stream_name_to_camera_name=self._stream_name_to_camera_name,
        http_options=self._http_options,
        ignore_image_inputs=self._ignore_vision_inputs,
        temperature=self._config.agent_temperature,
        max_output_tokens=self._config.agent_max_output_tokens,
        thinking_budget=self._config.agent_thinking_budget,
        media_resolution=self._config.agent_media_resolution,
    )

  def _create_nonstreaming_tools_config(self) -> list[genai_types.Tool]:
    """Creates the tools configuration for generate_content.

    Note: We strip the 'behavior' field from function declarations because
    it's a Live API specific feature that may not be supported by
    generate_content.

    Returns:
      List of Tool objects with cleaned function declarations.
    """
    # Create clean function declarations without the 'behavior' field
    clean_declarations = []
    for tool in self._exposed_tools:
      decl = tool.declaration
      # Create a new FunctionDeclaration without the behavior field
      clean_decl = genai_types.FunctionDeclaration(
          name=decl.name,
          description=decl.description,
          parameters=decl.parameters,
      )
      clean_declarations.append(clean_decl)

    return [
        genai_types.Tool(function_declarations=tuple(clean_declarations)),
    ]

  def _create_nonstreaming_tool_config(self) -> genai_types.ToolConfig | None:
    """Creates the tool configuration. Override to customize.

    Returns:
      Tool configuration for the generate_content call, or None for default.
    """
    return None

  # ---------------------------------------------------------------------------
  # Common properties and methods
  # ---------------------------------------------------------------------------

  @property
  def use_streaming(self) -> bool:
    """Returns whether the agent is using streaming mode."""
    return (
        self._config.orchestrator_handler_type
        == genai_types.OrchestratorHandlerType.STREAMING
    )

  def get_camera_stream_names(self) -> Sequence[str]:
    """Returns the camera stream names for the Agent."""
    return self._embodiment.camera_stream_names

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

  @property
  def handler(self) -> handler_lib.Handler:
    """Returns the API handler."""
    return self._handler

  async def connect(self):
    """Connects the handler and embodiment."""
    await self._handler.connect()
    await self._embodiment.connect()

  async def disconnect(self):
    """Disconnects the embodiment and handler."""
    await self._embodiment.disconnect()
    await self._handler.disconnect()

  # ---------------------------------------------------------------------------
  # Non-streaming specific methods (no-op for streaming)
  # ---------------------------------------------------------------------------

  @property
  def conversation_history(self):
    """Returns the conversation history."""
    if isinstance(self._handler, unary_genai_handler.UnaryGenAIHandler):
      return list(self._handler.conversation_history)
    return []

  def clear_history(self) -> None:
    """Clears the conversation history."""
    if isinstance(self._handler, unary_genai_handler.UnaryGenAIHandler):
      self._handler.clear_history()
