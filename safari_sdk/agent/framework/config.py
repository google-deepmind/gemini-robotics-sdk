# Copyright 2026 Google LLC
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

"""Configuration data class for the Safari agent framework."""

# pytype: skip-file

import dataclasses

from absl import logging
from google.genai import types as genai_types

from safari_sdk.agent.framework import flags as agentic_flags
from safari_sdk.agent.framework import types


@dataclasses.dataclass
class AgentFrameworkConfig:
  """Configuration for the Safari agent framework.

  This dataclass stores all configuration values that were previously accessed
  directly from flags. Consumers should accept this config object and read
  values from it instead of accessing flags directly.

  Use the `create()` class method to generate a config from flags with optional
  user overrides.
  """

  # General configuration
  # API key for the Gemini Live and Gemini API.
  api_key: str | None = None
  # Base URL of the Gemini Live and Gemini API. For example:
  # prod:
  # https://generativelanguage.googleapis.com,
  base_url: str = 'https://generativelanguage.googleapis.com'
  # The logging level to use.
  log_level: str = 'INFO'

  # Framework configuration
  # The control mode for the framework.
  control_mode: types.ControlMode = types.ControlMode.TERMINAL_ONLY
  # The host to use for the external controller server. Has no effect if
  # control_mode is not set to LAUNCH_SERVER. (Omit the http:// prefix.)
  external_controller_host: str = '127.0.0.1'
  # The port to use for the external controller server. Has no effect if
  # control_mode is not set to LAUNCH_SERVER.
  external_controller_port: int = 8887
  # Whether to publish Python logging events to the event bus.
  publish_logging_events: bool = False
  # The external UI type to use for event printing. This allows customizing
  # how events are printed to external UIs (e.g., robotics operator UI) without
  # modifying the core terminal UI logic. OPERATOR_DATA_COLLECT mode is designed
  # for operator-led data collection or evaluation with simplified output.
  external_ui_type: types.ExternalUIType = types.ExternalUIType.NONE

  # Agent configuration
  # The name of the agent to use.
  agent_name: str = 'simple_agent'
  tool_run_for_duration_second: float = 8.0
  run_until_done_time_limit: float = 60.0
  # Whether to meow.
  meow_mode: bool = False
  # The name of the model to use for the Gemini Live agent.
  agent_model_name: str = 'gemini-2.5-flash-native-audio-preview-12-2025'
  # Whether to enable audio input.
  enable_audio_input: bool = False
  # Whether to enable audio output.
  enable_audio_output: bool = False
  # Whether to handle audio on the robot.
  handle_audio_on_robot: bool = False
  # Whether to handle audio on the GUI.
  handle_audio_on_gui: bool = True
  # Whether to enable listening while speaking.
  listen_while_speaking: bool = False
  # Whether to display audio transcription in the terminal UI. Audio
  # transcription is always enabled in the Live API when audio input/output is
  # enabled; this flag only controls whether the transcription is displayed.
  enable_audio_transcription: bool = False
  # The name of the voice to use for the Gemini Live agent. Has no effect if
  # agent.enable_audio_output is False.
  output_audio_voice_name: str | None = None
  # Whether to only use activity coverage for the Gemini Live agent. In
  # additional to toggling the LiveAPI setting, images are inserted before each
  # test input and function response.
  only_activity_coverage: bool = True
  # Whether to update vision after a function response.
  update_vision_after_fr: bool = True
  # Whether to enable context window compression. Without compression,
  # audio-only sessions are limited to 15 minutes, and audio-video sessions are
  # limited to 2 minutes. Exceeding these limits will terminate the session
  # (and therefore, the connection), but you can use context window compression
  # to extend sessions to an unlimited amount of time. See details here:
  # https://ai.google.dev/api/live#contextwindowcompressionconfig.
  enable_context_window_compression: bool = False
  # The interval in seconds for streaming images to the Gemini Live model.
  gemini_live_image_streaming_interval_seconds: float = 1.0
  # Whether to remind the agent to use default_api.<fn_name> when making
  # function calls.
  remind_default_api_in_prompt: bool = False
  # Whether to use no chat mode.
  no_chat_mode: bool = False
  # The text will be sent to the Gemini Live API when the reminder is
  # triggered. The reminder can be triggered automatically via
  # agent.reminder_time_in_seconds or manually via '@eN' in the terminal UI. (N
  # is an integer from 0) In general, this feature allows users to send any
  # text to gemini live after x seconds. The text can trigger gemini to say
  # something or do other actions such as making function call...
  reminder_text_list: list[str] = dataclasses.field(default_factory=list)
  # The number of seconds to delay before automatically sending a reminder text
  # to the Gemini Live API. If the framework connects after this delay, the
  # reminder will be sent. Set to None to disable this feature.
  reminder_time_in_seconds: list[float] = dataclasses.field(
      default_factory=list
  )
  # Whether to use language control in the prompt.
  use_language_control: bool = False
  # Whether to instruct the agent to focus on fulfilling the given task and not
  # engage in conversation (useful for autonomous or sim eval).
  use_quiet_autonomy_mode: bool = False
  # The number of tokens to trigger context window compression.
  context_compression_trigger_tokens: int = 110000
  # The target number of tokens for the sliding window.
  context_compression_sliding_window_target: int = 60000
  # Whether to log the Gemini query for all tools.
  log_gemini_query: bool = False
  # Whether to stitch multiple camera images into a single grid image before
  # sending to the Gemini Live model. When disabled, images are sent
  # individually.
  enable_image_stitching: bool = False
  # Whether to show camera name labels on stitched images. Only has effect
  # when enable_image_stitching is True.
  show_camera_name_in_stitched_image: bool = True
  # Whether to enable automatic session resumption when receiving GO_AWAY from
  # the Live API. When enabled, the framework will automatically reconnect
  # after a 20 second grace period.
  enable_automatic_session_resumption: bool = True
  # Maximum number of images to keep in conversation history for non-streaming
  # non_streaming_image_pruning_target_amount.
  non_streaming_image_pruning_trigger_amount: int = 60
  # When the number of images exceeds
  # non_streaming_image_pruning_trigger_amount, prune down to this number.
  # This batch pruning preserves prefill cache.
  non_streaming_image_pruning_target_amount: int = 15
  # The interval in seconds for buffering images in the non-streaming handler.
  non_streaming_image_buffering_interval_seconds: float = 1.0
  # Whether to discard buffered images after each turn. When True (default),
  # images are cleared after being used. When False, images accumulate.
  non_streaming_discard_images_after_turn: bool = True
  non_streaming_fr_latest_image_only: bool = True
  non_streaming_user_turn_latest_image_only: bool = False
  non_streaming_include_stream_names: bool = True
  non_streaming_thinking_level: str | None = None
  non_streaming_tool_result_timeout_seconds: float = 300.0

  # Agent model generation parameters (for non-streaming handler).
  # Temperature for the agent model (0.0-2.0). None uses server default.
  agent_temperature: float | None = None
  # Max output tokens for agent responses. None uses server default.
  agent_max_output_tokens: int | None = None
  # Thinking budget for the agent model. 0=disabled, -1=auto, None=default.
  agent_thinking_budget: int | None = None
  # Media resolution for images. Defaults to HIGH (1120 tokens).
  agent_media_resolution: genai_types.MediaResolution = (
      genai_types.MediaResolution.MEDIA_RESOLUTION_MEDIUM
  )

  orchestrator_handler_type: types.OrchestratorHandlerType = (
      types.OrchestratorHandlerType.STREAMING
  )

  # Success detection configuration
  # Whether to run success detection in dry run mode. If True, decide success
  # or not only based on human signals.
  sd_dry_run: bool = False
  # The name of the success detection tool to use.
  sd_tool_name: agentic_flags.SDToolName = (
      agentic_flags.SDToolName.SUBTASK_SUCCESS_DETECTOR
  )
  # The timeout for the success detection tool.
  sd_timeout_seconds: float = 60.0
  # The name of the model to use for the success detection.
  sd_model_name: str = 'gemini-robotics-er-1.5-preview'
  # The thinking budget for the success detection model. 0 is DISABLED. -1 is
  # AUTOMATIC. The default values and allowed ranges are model dependent.
  # Mutually exclusive with sd_thinking_level.
  sd_thinking_budget: int = -1
  # The thinking level for V4/FierceFalcon success detection model (MINIMAL,
  # LOW, MEDIUM, HIGH). Mutually exclusive with sd_thinking_budget.
  sd_thinking_level: str | None = None
  # Whether to use progress prediction for success detection.
  sd_use_progress_prediction: bool = False
  # The threshold for the progress prediction time signal. The seconds left
  # prediction must be less than this threshold to trigger success. Has no
  # effect when use_progress_prediction is False.
  sd_pp_time_threshold: float = 0.6
  # The threshold for the progress prediction percentage signal. The percentage
  # prediction must be larger than this threshold to trigger success. Has no
  # effect when use_progress_prediction is False.
  sd_pp_percent_threshold: float = 90
  # The number of history frames to use for SD.
  sd_num_history_frames: int = 0
  # The interval between history frames to use for SD.
  sd_history_interval_s: float = 1.0
  # Whether to print the final prompt for SD.
  sd_print_final_prompt: bool = False
  # Whether to use start images for SD.
  sd_use_start_images: bool = True
  # Whether to use explicit thinking for SD.
  sd_use_explicit_thinking: bool = True
  # The word limit for guided thinking for SD.
  sd_guided_thinking_word_limit: int = 50
  # Whether to print the raw SD response.
  sd_print_raw_sd_response: bool = True
  # The interval in seconds between async SD runs.
  sd_async_sd_interval_s: float = 0.2
  # The thinking budget for the overall task success detector. Set to 0 to
  # disable, -1 for automatic.
  overall_task_success_detector_thinking_budget: int = -1
  # Whether the run_instruction_until_done tool should stop the robot when the
  # success detector returns True.
  stop_on_success: bool = True
  # The model temperature to use for SD. Recommend to use higher temperature
  # for ensemble SD.
  sd_temperature: float = 0.0
  # The number of parallel SD runs to use.
  sd_ensemble_size: int = 1
  # The threshold for the ensemble size.
  sd_ensemble_threshold: int = 1
  # The sleep interval in seconds for the ensemble SD model.
  sd_sleep_interval_s: float = 0.2
  # Max output tokens for SD responses. None uses server default.
  sd_max_output_tokens: int | None = None
  # Media resolution for SD images (e.g. MEDIA_RESOLUTION_HIGH for 1120 tokens).
  sd_media_resolution: genai_types.MediaResolution = (
      genai_types.MediaResolution.MEDIA_RESOLUTION_HIGH
  )
  # Image stitching mode for SD: 'none', 'camera', or 'time'.
  sd_stitch_mode: types.SDStitchMode = types.SDStitchMode.NONE
  # Path to custom SD prompt file. If set, overrides default prompt.
  sd_prompt_file: str | None = None
  # Prompt name from registry. If unset, uses mode-appropriate default.
  sd_prompt_name: str | None = None

  # Scene description configuration
  # Whether to use scene description.
  use_scene_description: bool = False
  # The name of the model to use for scene description.
  scene_description_model_name: str = 'gemini-robotics-er-1.5-preview'
  # The thinking budget for the scene description model. Set to 0 to disable,
  # -1 for automatic.
  scene_description_thinking_budget: int = 100
  # The number of words to output for scene description.
  scene_description_num_output_words: int = 200

  # Robot backend configuration
  # The hostname of the robot backend server. (Omit the http:// prefix.)
  robot_backend_host: str = 'localhost'
  # The port of the robot backend server.
  robot_backend_port: int = 8888

  # Logging configuration
  # Whether to enable logging.
  enable_logging: bool = False
  # The ID of the robot.
  robot_id: str | None = None
  # The output directory for the logs.
  logging_output_directory: str = '/tmp/safari_logs'
  # The key of the session log type. If logging is enabled then the agent logs
  # from event bus will be saved to SSOT. This is the key which describes the
  # session log type column for the SSOT table.
  logging_session_log_type_key: str | None = None
  # The value of the session log type. Default value is 'agent' to separate
  # them from policy logs.
  logging_session_log_type_value: str = 'agent'
  exclude_model_image_input_logging: bool = False
  agent_session_id: str | None = None
  non_streaming_enable_context_snapshot_logging: bool = True

  def __post_init__(self):
    if (
        self.sd_thinking_budget is not None
        and self.sd_thinking_budget != -1
        and self.sd_thinking_level is not None
    ):
      raise ValueError(
          'sd_thinking_budget and sd_thinking_level are mutually exclusive.'
          ' Please set only one of them. sd_thinking_budget is for Gemini 2.5'
          ' family models, while sd_thinking_level is for Gemini 3 family'
          ' models.'
      )

  @classmethod
  def create(
      cls,
      *,
      api_key: str | None = None,
      base_url: str | None = None,
      log_level: str | None = None,
      control_mode: types.ControlMode | None = None,
      external_controller_host: str | None = None,
      external_controller_port: int | None = None,
      publish_logging_events: bool | None = None,
      external_ui_type: types.ExternalUIType | None = None,
      agent_name: str | None = None,
      tool_run_for_duration_second: float | None = None,
      run_until_done_time_limit: float | None = None,
      meow_mode: bool | None = None,
      agent_model_name: str | None = None,
      enable_audio_input: bool | None = None,
      enable_audio_output: bool | None = None,
      handle_audio_on_robot: bool | None = None,
      handle_audio_on_gui: bool | None = None,
      listen_while_speaking: bool | None = None,
      enable_audio_transcription: bool | None = None,
      output_audio_voice_name: str | None = None,
      only_activity_coverage: bool | None = None,
      update_vision_after_fr: bool | None = None,
      enable_context_window_compression: bool | None = None,
      gemini_live_image_streaming_interval_seconds: float | None = None,
      remind_default_api_in_prompt: bool | None = None,
      no_chat_mode: bool | None = None,
      # NOTE: since `None` causes the flag default to be used, and the flag
      # defaults are "Hi" and "Aww out of time", you must set this to an empty
      # list to disable reminders!
      reminder_text_list: list[str] | None = None,
      reminder_time_in_seconds: list[float] | None = None,
      use_language_control: bool | None = None,
      use_quiet_autonomy_mode: bool | None = None,
      context_compression_trigger_tokens: int | None = None,
      context_compression_sliding_window_target: int | None = None,
      log_gemini_query: bool | None = None,
      enable_image_stitching: bool | None = None,
      show_camera_name_in_stitched_image: bool | None = None,
      enable_automatic_session_resumption: bool | None = None,
      non_streaming_image_pruning_trigger_amount: int | None = None,
      non_streaming_image_buffering_interval_seconds: float | None = None,
      non_streaming_image_pruning_target_amount: int | None = None,
      non_streaming_discard_images_after_turn: bool | None = None,
      non_streaming_fr_latest_image_only: bool | None = None,
      non_streaming_user_turn_latest_image_only: bool | None = None,
      non_streaming_include_stream_names: bool | None = None,
      non_streaming_thinking_level: str | None = None,
      non_streaming_tool_result_timeout_seconds: float | None = None,
      agent_temperature: float | None = None,
      agent_max_output_tokens: int | None = None,
      agent_thinking_budget: int | None = None,
      agent_media_resolution: genai_types.MediaResolution | None = None,
      orchestrator_handler_type: types.OrchestratorHandlerType | None = None,
      sd_dry_run: bool | None = None,
      sd_tool_name: agentic_flags.SDToolName | None = None,
      sd_timeout_seconds: float | None = None,
      sd_model_name: str | None = None,
      sd_thinking_budget: int | None = None,
      sd_thinking_level: str | None = None,
      sd_use_progress_prediction: bool | None = None,
      sd_pp_time_threshold: float | None = None,
      sd_pp_percent_threshold: float | None = None,
      sd_num_history_frames: int | None = None,
      sd_history_interval_s: float | None = None,
      sd_print_final_prompt: bool | None = None,
      sd_use_start_images: bool | None = None,
      sd_use_explicit_thinking: bool | None = None,
      sd_guided_thinking_word_limit: int | None = None,
      sd_print_raw_sd_response: bool | None = None,
      sd_async_sd_interval_s: float | None = None,
      overall_task_success_detector_thinking_budget: int | None = None,
      stop_on_success: bool | None = None,
      sd_temperature: float | None = None,
      sd_ensemble_size: int | None = None,
      sd_ensemble_threshold: int | None = None,
      sd_sleep_interval_s: float | None = None,
      sd_max_output_tokens: int | None = None,
      sd_media_resolution: genai_types.MediaResolution | None = None,
      use_scene_description: bool | None = None,
      scene_description_model_name: str | None = None,
      scene_description_thinking_budget: int | None = None,
      scene_description_num_output_words: int | None = None,
      robot_backend_host: str | None = None,
      robot_backend_port: int | None = None,
      enable_logging: bool | None = None,
      non_streaming_enable_context_snapshot_logging: bool | None = None,
      robot_id: str | None = None,
      logging_output_directory: str | None = None,
      logging_session_log_type_key: str | None = None,
      logging_session_log_type_value: str | None = None,
      exclude_model_image_input_logging: bool | None = None,
      agent_session_id: str | None = None,
  ) -> 'AgentFrameworkConfig':
    """Creates an AgentFrameworkConfig from flags with optional overrides."""
    if (control_mode == types.ControlMode.TERMINAL_ONLY) and (
        handle_audio_on_gui
    ):
      logging.warning(
          'handle_audio_on_gui is set to True while control_mode is'
          ' TERMINAL_ONLY. This combination is not supported. Setting'
          ' handle_audio_on_gui to False.'
      )
      handle_audio_on_gui = False
    return cls(
        api_key=api_key or agentic_flags.AGENTIC_API_KEY.value,
        base_url=base_url or agentic_flags.AGENTIC_BASE_URL.value,
        log_level=log_level or agentic_flags.AGENTIC_LOG_LEVEL.value,
        control_mode=control_mode or agentic_flags.AGENTIC_CONTROL_MODE.value,
        external_controller_host=(
            external_controller_host
            or agentic_flags.AGENTIC_EXTERNAL_CONTROLLER_HOST.value
        ),
        external_controller_port=(
            external_controller_port
            or agentic_flags.AGENTIC_EXTERNAL_CONTROLLER_PORT.value
        ),
        publish_logging_events=(
            publish_logging_events
            or agentic_flags.AGENTIC_PUBLISH_LOGGING_EVENTS.value
        ),
        external_ui_type=(
            external_ui_type or agentic_flags.AGENTIC_EXTERNAL_UI_TYPE.value
        ),
        agent_name=agent_name or agentic_flags.AGENTIC_AGENT_NAME.value,
        tool_run_for_duration_second=(
            tool_run_for_duration_second
            if tool_run_for_duration_second is not None
            else agentic_flags.AGENTIC_TOOL_RUN_FOR_DURATION_SECOND.value
        ),
        run_until_done_time_limit=(
            run_until_done_time_limit
            if run_until_done_time_limit is not None
            else agentic_flags.AGENTIC_RUN_UNTIL_DONE_TIME_LIMIT.value
        ),
        meow_mode=meow_mode or agentic_flags.AGENTIC_MEOW_MODE.value,
        agent_model_name=(
            agent_model_name or agentic_flags.AGENTIC_AGENT_MODEL_NAME.value
        ),
        enable_audio_input=(
            enable_audio_input or agentic_flags.AGENTIC_ENABLE_AUDIO_INPUT.value
        ),
        enable_audio_output=(
            enable_audio_output
            or agentic_flags.AGENTIC_ENABLE_AUDIO_OUTPUT.value
        ),
        handle_audio_on_robot=(
            handle_audio_on_robot
            or agentic_flags.AGENTIC_HANDLE_AUDIO_ON_ROBOT.value
        ),
        handle_audio_on_gui=(
            handle_audio_on_gui
            or agentic_flags.AGENTIC_HANDLE_AUDIO_ON_GUI.value
        ),
        listen_while_speaking=(
            listen_while_speaking
            or agentic_flags.AGENTIC_LISTEN_WHILE_SPEAKING.value
        ),
        enable_audio_transcription=(
            enable_audio_transcription
            or agentic_flags.AGENTIC_ENABLE_AUDIO_TRANSCRIPTION.value
        ),
        output_audio_voice_name=(
            output_audio_voice_name
            or agentic_flags.AGENTIC_OUTPUT_AUDIO_VOICE_NAME.value
        ),
        only_activity_coverage=(
            only_activity_coverage
            or agentic_flags.AGENTIC_ONLY_ACTIVITY_COVERAGE.value
        ),
        update_vision_after_fr=(
            update_vision_after_fr
            or agentic_flags.AGENTIC_UPDATE_VISION_AFTER_FR.value
        ),
        enable_context_window_compression=(
            enable_context_window_compression
            or agentic_flags.AGENTIC_ENABLE_CONTEXT_WINDOW_COMPRESSION.value
        ),
        gemini_live_image_streaming_interval_seconds=(
            gemini_live_image_streaming_interval_seconds
            or agentic_flags.AGENTIC_GEMINI_LIVE_IMAGE_STREAMING_INTERVAL_SECONDS.value
        ),
        remind_default_api_in_prompt=(
            remind_default_api_in_prompt
            or agentic_flags.AGENTIC_REMIND_DEFAULT_API_IN_PROMPT.value
        ),
        no_chat_mode=(no_chat_mode or agentic_flags.AGENTIC_NO_CHAT_MODE.value),
        reminder_text_list=(
            reminder_text_list
            if reminder_text_list is not None
            else agentic_flags.AGENTIC_REMINDER_TEXT_LIST.value
        ),
        reminder_time_in_seconds=(
            reminder_time_in_seconds
            if reminder_time_in_seconds is not None
            else agentic_flags.AGENTIC_REMINDER_TIME_IN_SECONDS.value
        ),
        use_language_control=(
            use_language_control
            or agentic_flags.AGENTIC_USE_LANGUAGE_CONTROL.value
        ),
        use_quiet_autonomy_mode=(
            use_quiet_autonomy_mode
            or agentic_flags.AGENTIC_USE_QUIET_AUTONOMY_MODE.value
        ),
        context_compression_trigger_tokens=(
            context_compression_trigger_tokens
            or agentic_flags.AGENTIC_CONTEXT_COMPRESSION_TRIGGER_TOKENS.value
        ),
        context_compression_sliding_window_target=(
            context_compression_sliding_window_target
            or agentic_flags.AGENTIC_CONTEXT_COMPRESSION_SLIDING_WINDOW_TARGET.value
        ),
        log_gemini_query=(
            log_gemini_query or agentic_flags.AGENTIC_LOG_GEMINI_QUERY.value
        ),
        enable_image_stitching=(
            enable_image_stitching
            or agentic_flags.AGENTIC_ENABLE_IMAGE_STITCHING.value
        ),
        show_camera_name_in_stitched_image=(
            show_camera_name_in_stitched_image
            or agentic_flags.AGENTIC_SHOW_CAMERA_NAME_IN_STITCHED_IMAGE.value
        ),
        enable_automatic_session_resumption=(
            enable_automatic_session_resumption
            if enable_automatic_session_resumption is not None
            else agentic_flags.AGENTIC_ENABLE_AUTOMATIC_SESSION_RESUMPTION.value
        ),
        non_streaming_image_pruning_trigger_amount=(
            non_streaming_image_pruning_trigger_amount
            or agentic_flags.AGENTIC_NON_STREAMING_IMAGE_PRUNING_TRIGGER_AMOUNT.value
        ),
        non_streaming_image_pruning_target_amount=(
            non_streaming_image_pruning_target_amount
            if non_streaming_image_pruning_target_amount is not None
            else agentic_flags.AGENTIC_NON_STREAMING_IMAGE_PRUNING_TARGET_AMOUNT.value
        ),
        non_streaming_image_buffering_interval_seconds=(
            non_streaming_image_buffering_interval_seconds
            or agentic_flags.AGENTIC_NON_STREAMING_IMAGE_BUFFERING_INTERVAL_SECONDS.value
        ),
        non_streaming_discard_images_after_turn=(
            non_streaming_discard_images_after_turn
            if non_streaming_discard_images_after_turn is not None
            else agentic_flags.AGENTIC_NON_STREAMING_DISCARD_IMAGES_AFTER_TURN.value
        ),
        non_streaming_fr_latest_image_only=(
            non_streaming_fr_latest_image_only
            if non_streaming_fr_latest_image_only is not None
            else agentic_flags.AGENTIC_NON_STREAMING_FR_LATEST_IMAGE_ONLY.value
        ),
        non_streaming_user_turn_latest_image_only=(
            non_streaming_user_turn_latest_image_only
            if non_streaming_user_turn_latest_image_only is not None
            else agentic_flags.AGENTIC_NON_STREAMING_USER_TURN_LATEST_IMAGE_ONLY.value
        ),
        non_streaming_include_stream_names=(
            non_streaming_include_stream_names
            if non_streaming_include_stream_names is not None
            else agentic_flags.AGENTIC_NON_STREAMING_INCLUDE_STREAM_NAMES.value
        ),
        non_streaming_thinking_level=(
            non_streaming_thinking_level
            or agentic_flags.AGENTIC_NON_STREAMING_THINKING_LEVEL.value
        ),
        non_streaming_tool_result_timeout_seconds=(
            non_streaming_tool_result_timeout_seconds
            if non_streaming_tool_result_timeout_seconds is not None
            else agentic_flags.AGENTIC_NON_STREAMING_TOOL_RESULT_TIMEOUT_SECONDS.value
        ),
        agent_temperature=(
            agent_temperature
            if agent_temperature is not None
            else agentic_flags.AGENTIC_AGENT_TEMPERATURE.value
        ),
        agent_max_output_tokens=(
            agent_max_output_tokens
            if agent_max_output_tokens is not None
            else agentic_flags.AGENTIC_AGENT_MAX_OUTPUT_TOKENS.value
        ),
        agent_thinking_budget=(
            agent_thinking_budget
            if agent_thinking_budget is not None
            else agentic_flags.AGENTIC_AGENT_THINKING_BUDGET.value
        ),
        agent_media_resolution=(
            agent_media_resolution
            or agentic_flags.AGENTIC_AGENT_MEDIA_RESOLUTION.value
        ),
        orchestrator_handler_type=(
            orchestrator_handler_type
            or agentic_flags.AGENTIC_ORCHESTRATOR_HANDLER_TYPE.value
        ),
        sd_dry_run=sd_dry_run or agentic_flags.AGENTIC_SD_DRY_RUN.value,
        sd_tool_name=sd_tool_name or agentic_flags.AGENTIC_SD_TOOL_NAME.value,
        sd_timeout_seconds=(
            sd_timeout_seconds or agentic_flags.AGENTIC_SD_TIMEOUT_SECONDS.value
        ),
        sd_model_name=(
            sd_model_name or agentic_flags.AGENTIC_SD_MODEL_NAME.value
        ),
        sd_thinking_budget=(
            sd_thinking_budget or agentic_flags.AGENTIC_SD_THINKING_BUDGET.value
        ),
        sd_thinking_level=(
            sd_thinking_level or agentic_flags.AGENTIC_SD_THINKING_LEVEL.value
        ),
        sd_use_progress_prediction=(
            sd_use_progress_prediction
            or agentic_flags.AGENTIC_SD_USE_PROGRESS_PREDICTION.value
        ),
        sd_pp_time_threshold=(
            sd_pp_time_threshold
            or agentic_flags.AGENTIC_SD_PP_TIME_THRESHOLD.value
        ),
        sd_pp_percent_threshold=(
            sd_pp_percent_threshold
            or agentic_flags.AGENTIC_SD_PP_PERCENT_THRESHOLD.value
        ),
        sd_num_history_frames=(
            sd_num_history_frames
            or agentic_flags.AGENTIC_SD_NUM_HISTORY_FRAMES.value
        ),
        sd_history_interval_s=(
            sd_history_interval_s
            or agentic_flags.AGENTIC_SD_HISTORY_INTERVAL_S.value
        ),
        sd_print_final_prompt=(
            sd_print_final_prompt
            or agentic_flags.AGENTIC_SD_PRINT_FINAL_PROMPT.value
        ),
        sd_use_start_images=(
            sd_use_start_images
            or agentic_flags.AGENTIC_SD_USE_START_IMAGES.value
        ),
        sd_use_explicit_thinking=(
            sd_use_explicit_thinking
            or agentic_flags.AGENTIC_SD_USE_EXPLICIT_THINKING.value
        ),
        sd_guided_thinking_word_limit=(
            sd_guided_thinking_word_limit
            or agentic_flags.AGENTIC_SD_GUIDED_THINKING_WORD_LIMIT.value
        ),
        sd_print_raw_sd_response=(
            sd_print_raw_sd_response
            or agentic_flags.AGENTIC_SD_PRINT_RAW_SD_RESPONSE.value
        ),
        sd_async_sd_interval_s=(
            sd_async_sd_interval_s
            or agentic_flags.AGENTIC_SD_ASYNC_SD_INTERVAL_S.value
        ),
        overall_task_success_detector_thinking_budget=(
            overall_task_success_detector_thinking_budget
            or agentic_flags.AGENTIC_OVERALL_TASK_SUCCESS_DETECTOR_THINKING_BUDGET.value
        ),
        stop_on_success=(
            stop_on_success or agentic_flags.AGENTIC_STOP_ON_SUCCESS.value
        ),
        sd_temperature=(
            sd_temperature or agentic_flags.AGENTIC_SD_TEMPERATURE.value
        ),
        sd_ensemble_size=(
            sd_ensemble_size or agentic_flags.AGENTIC_SD_ENSEMBLE_SIZE.value
        ),
        sd_ensemble_threshold=(
            sd_ensemble_threshold
            or agentic_flags.AGENTIC_SD_ENSEMBLE_THRESHOLD.value
        ),
        sd_sleep_interval_s=(
            sd_sleep_interval_s
            or agentic_flags.AGENTIC_SD_SLEEP_INTERVAL_S.value
        ),
        sd_max_output_tokens=(
            sd_max_output_tokens
            if sd_max_output_tokens is not None
            else agentic_flags.AGENTIC_SD_MAX_OUTPUT_TOKENS.value
        ),
        sd_media_resolution=(
            sd_media_resolution
            or agentic_flags.AGENTIC_SD_MEDIA_RESOLUTION.value
        ),
        sd_stitch_mode=agentic_flags.AGENTIC_SD_STITCH_MODE.value,
        sd_prompt_file=agentic_flags.AGENTIC_SD_PROMPT_FILE.value,
        sd_prompt_name=agentic_flags.AGENTIC_SD_PROMPT_NAME.value,
        use_scene_description=(
            use_scene_description
            or agentic_flags.AGENTIC_USE_SCENE_DESCRIPTION.value
        ),
        scene_description_model_name=(
            scene_description_model_name
            or agentic_flags.AGENTIC_SCENE_DESCRIPTION_MODEL_NAME.value
        ),
        scene_description_thinking_budget=(
            scene_description_thinking_budget
            or agentic_flags.AGENTIC_SCENE_DESCRIPTION_THINKING_BUDGET.value
        ),
        scene_description_num_output_words=(
            scene_description_num_output_words
            or agentic_flags.AGENTIC_SCENE_DESCRIPTION_NUM_OUTPUT_WORDS.value
        ),
        robot_backend_host=(
            robot_backend_host or agentic_flags.AGENTIC_ROBOT_BACKEND_HOST.value
        ),
        robot_backend_port=(
            robot_backend_port or agentic_flags.AGENTIC_ROBOT_BACKEND_PORT.value
        ),
        enable_logging=(
            enable_logging or agentic_flags.AGENTIC_ENABLE_LOGGING.value
        ),
        robot_id=robot_id or agentic_flags.AGENTIC_ROBOT_ID.value,
        logging_output_directory=(
            logging_output_directory
            or agentic_flags.AGENTIC_LOGGING_OUTPUT_DIRECTORY.value
        ),
        logging_session_log_type_key=(
            logging_session_log_type_key
            or agentic_flags.AGENTIC_LOGGING_SESSION_LOG_TYPE_KEY.value
        ),
        logging_session_log_type_value=(
            logging_session_log_type_value
            or agentic_flags.AGENTIC_LOGGING_SESSION_LOG_TYPE_VALUE.value
        ),
        exclude_model_image_input_logging=(
            exclude_model_image_input_logging
            or agentic_flags.AGENTIC_EXCLUDE_MODEL_IMAGE_INPUT_LOGGING.value
        ),
        agent_session_id=(
            agent_session_id
            or agentic_flags.AGENTIC_LOGGING_AGENT_SESSION_ID.value
        ),
        non_streaming_enable_context_snapshot_logging=(
            non_streaming_enable_context_snapshot_logging
            if non_streaming_enable_context_snapshot_logging is not None
            else agentic_flags.AGENTIC_NON_STREAMING_ENABLE_CONTEXT_SNAPSHOT_LOGGING.value
        ),
    )
