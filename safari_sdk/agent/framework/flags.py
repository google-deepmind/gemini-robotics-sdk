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

"""Flags for the agentic framework.

All agentic framework and user code flags should be defined here. To avoid flag
name collisions, please prefix all flag variables with "AGENTIC_" and all actual
flag value with "agent.".
"""

import enum
from absl import flags
from safari_sdk.agent.framework import types


class SDToolName(enum.Enum):
  # Set `SUBTASK_SUCCESS_DETECTOR` will use `SubtaskSuccessDetectorV4`.
  SUBTASK_SUCCESS_DETECTOR = "SubtaskSuccessDetector"
  # Set `SUBTASK_SUCCESS_DETECTOR_V2` will use `SubtaskSuccessDetectorV4`.
  SUBTASK_SUCCESS_DETECTOR_V2 = "SubtaskSuccessDetectorV2"
  # Set `SUBTASK_SUCCESS_DETECTOR_V3` will use `SubtaskSuccessDetectorV4`.
  SUBTASK_SUCCESS_DETECTOR_V3 = "SubtaskSuccessDetectorV3"
  SUBTASK_SUCCESS_DETECTOR_V4 = "SubtaskSuccessDetectorV4"
  ENSEMBLE_SUBTASK_SUCCESS_DETECTOR_V2 = "EnsembleSubtaskSuccessDetectorV2"

AGENTIC_API_KEY = flags.DEFINE_string(
    "general.api_key",
    None,
    "API key for the Gemini Live and Gemini API.",
)

AGENTIC_BASE_URL = flags.DEFINE_string(
    "general.base_url",
    "https://generativelanguage.googleapis.com",
    "Base URL of the Gemini Live and Gemini API. For example:"
    " - prod: https://generativelanguage.googleapis.com"
    " - autopush: https://autopush-generativelanguage.sandbox.googleapis.com"
    " - staging: https://staging-generativelanguage.sandbox.googleapis.com"
    " - preprod: https://preprod-generativelanguage.googleapis.com",
)

AGENTIC_LOG_LEVEL = flags.DEFINE_enum(
    "general.log_level",
    "INFO",
    ["DEBUG", "INFO", "WARNING", "ERROR", "FATAL"],
    "The logging level to use.",
)


# ------------------------
# Agentic framework flags.
# ------------------------
AGENTIC_CONTROL_MODE = flags.DEFINE_enum_class(
    "framework.control_mode",
    default=types.ControlMode.TERMINAL_ONLY,
    enum_class=types.ControlMode,
    help="The control mode for the framework.",
)

AGENTIC_EXTERNAL_CONTROLLER_HOST = flags.DEFINE_string(
    "framework.external_controller_host",
    "127.0.0.1",
    "The host to use for the external controller server. Has no effect if"
    " control_mode is not set to LAUNCH_SERVER. (Omit the http:// prefix.)",
)

AGENTIC_EXTERNAL_CONTROLLER_PORT = flags.DEFINE_integer(
    "framework.external_controller_port",
    8887,
    "The port to use for the external controller server. Has no effect if"
    " control_mode is not set to LAUNCH_SERVER.",
)

AGENTIC_USE_OPERATOR_FRIENDLY_TERMINAL_UI = flags.DEFINE_bool(
    "framework.use_operator_friendly_terminal_ui",
    default=False,
    help="Whether to use the operator friendly terminal UI.",
)

# ------------------------
# Agent flags.
# ------------------------
AGENTIC_AGENT_NAME = flags.DEFINE_string(
    "agent.name",
    "simple_agent",
    "The name of the agent to use.",
)

AGENTIC_MEOW_MODE = flags.DEFINE_bool(
    "agent.meow",
    False,
    "Whether to meow.",
)


AGENTIC_AGENT_MODEL_NAME = flags.DEFINE_string(
    "agent.model_name",
    "gemini-live-2.5-flash-preview",
    "The name of the model to use for the Gemini Live agent.",
)

AGENTIC_ENABLE_AUDIO_INPUT = flags.DEFINE_bool(
    "agent.enable_audio_input",
    False,
    "Whether to enable audio input.",
)

AGENTIC_ENABLE_AUDIO_OUTPUT = flags.DEFINE_bool(
    "agent.enable_audio_output",
    False,
    "Whether to enable audio output.",
)

AGENTIC_LISTEN_WHILE_SPEAKING = flags.DEFINE_bool(
    "agent.listen_while_speaking",
    False,
    "Whether to enable listening while speaking.",
)

AGENTIC_ENABLE_AUDIO_TRANSCRIPTION = flags.DEFINE_bool(
    "agent.enable_audio_transcription",
    False,
    "Whether to enable audio transcription.",
)

AGENTIC_OUTPUT_AUDIO_VOICE_NAME = flags.DEFINE_string(
    "agent.voice_name",
    None,
    "The name of the voice to use for the Gemini Live agent. Has no effect if"
    " agent.enable_audio_output is False.",
)

AGENTIC_ONLY_ACTIVITY_COVERAGE = flags.DEFINE_bool(
    "agent.only_activity_coverage",
    True,
    "Whether to only use activity coverage for the Gemini Live agent."
    "In additional to toggling the LiveAPI setting, images are inserted before"
    "each test input and function response.",
)

AGENTIC_UPDATE_VISION_AFTER_FR = flags.DEFINE_bool(
    "agent.update_vision_after_fr",
    True,
    "Whether to update vision after a function response.",
)

AGENTIC_ENABLE_CONTEXT_WINDOW_COMPRESSION = flags.DEFINE_bool(
    "agent.enable_context_window_compression",
    False,
    "Whether to enable context window compression. Without compression,"
    " audio-only sessions are limited to 15 minutes, and audio-video sessions"
    " are limited to 2 minutes. Exceeding these limits will terminate the"
    " session (and therefore, the connection), but you can use context window"
    " compression to extend sessions to an unlimited amount of time. See"
    " details here:"
    " https://ai.google.dev/api/live#contextwindowcompressionconfig.",
)

AGENTIC_GEMINI_LIVE_IMAGE_STREAMING_INTERVAL_SECONDS = flags.DEFINE_float(
    "agent.gemini_live_image_streaming_interval_seconds",
    1.0,
    "The interval in seconds for streaming images to the Gemini Live model.",
)

AGENTIC_REMIND_DEFAULT_API_IN_PROMPT = flags.DEFINE_bool(
    "agent.remind_default_api_in_prompt",
    False,
    "Whether to remind the agent to use default_api.<fn_name> when making"
    " function calls.",
)

AGENTIC_NO_CHAT_MODE = flags.DEFINE_bool(
    "agent.no_chat_mode",
    False,
    "Whether to use no chat mode.",
)

AGENTIC_REMINDER_TEXT_LIST = flags.DEFINE_multi_string(
    "agent.reminder_text",
    [
        "Repeat what I said exactly: 'Hi!'",
        (
            "Repeat what I said exactly: 'Aw, look at that, it is time for the"
            " next participant. Thanks for checking out Gemini Robotics!'"
        ),
    ],
    "The text will be sent to the Gemini Live API when the reminder is"
    " triggered. The reminder can be triggered by automatically via"
    " `agent.reminder_time_in_seconds` or manually via '@eN' in the terminal"
    " UI. (N is an integer from 0) In general, this feature allows users to"
    " send any text to gemini live after x seconds. the text can trigger gemini"
    " to say something or do other actions such as making function call...",
)

AGENTIC_REMINDER_TIME_IN_SECONDS = flags.DEFINE_multi_float(
    "agent.reminder_time_in_seconds",
    [0.5, 360],
    "The number of seconds to delay before automatically sending a reminder"
    " text to the Gemini Live API. If the framework connects after this delay,"
    " the reminder will be sent. Set to None to disable this feature.",
    lower_bound=0,
)

AGENTIC_USE_LANGUAGE_CONTROL = flags.DEFINE_bool(
    "agent.use_language_control",
    False,
    "Whether to use language control in the prompt.",
)

AGENTIC_CONTEXT_COMPRESSION_TRIGGER_TOKENS = flags.DEFINE_integer(
    "agent.context_compression_trigger_tokens",
    110000,
    "The number of tokens to trigger context window compression.",
)

AGENTIC_CONTEXT_COMPRESSION_SLIDING_WINDOW_TARGET = flags.DEFINE_integer(
    "agent.context_compression_sliding_window_target",
    60000,
    "The target number of tokens for the sliding window.",
)

AGENTIC_LOG_GEMINI_QUERY = flags.DEFINE_bool(
    "agent.log_gemini_query",
    False,
    "Whether to log the Gemini query for all tools.",
)

# ------------------------
# Success detection flags.
# ------------------------
AGENTIC_SD_DRY_RUN = flags.DEFINE_bool(
    "sd.dry_run",
    False,
    "Whether to run success detection in dry run mode. If True, decide success"
    " or not only based on human signals.",
)


AGENTIC_SD_TOOL_NAME = flags.DEFINE_enum_class(
    "sd.tool_name",
    SDToolName.SUBTASK_SUCCESS_DETECTOR,
    SDToolName,
    "The name of the success detection tool to use.",
)

AGENTIC_SD_TIMEOUT_SECONDS = flags.DEFINE_float(
    "sd.timeout_seconds",
    60.0,
    "The timeout for the success detection tool.",
)

AGENTIC_SD_MODEL_NAME = flags.DEFINE_string(
    "sd.model_name",
    "gemini-robotics-er-1.5-preview",
    "The name of the model to use for the success detection.",
)

AGENTIC_SD_THINKING_BUDGET = flags.DEFINE_integer(
    "sd.thinking_budget",
    -1,
    "The thinking budget for the success detection model. 0 is DISABLED. -1 is"
    " AUTOMATIC. The default values and allowed ranges are model dependent.",
)

AGENTIC_SD_USE_PROGRESS_PREDICTION = flags.DEFINE_bool(
    "sd.use_progress_prediction",
    False,
    "Whether to use progress prediction for success detection.",
)

AGENTIC_SD_PP_TIME_THRESHOLD = flags.DEFINE_float(
    "sd.pp_time_threshold",
    0.6,
    "The threshold for the progress prediction time signal. The seconds left"
    " prediction must be less than this threshold to trigger success. Has no"
    " effect when use_progress_prediction is False.",
)

AGENTIC_SD_PP_PERCENT_THRESHOLD = flags.DEFINE_float(
    "sd.pp_percent_threshold",
    90,
    "The threshold for the progress prediction percentage signal. The"
    " percentage prediction must be larger than this threshold to trigger"
    " success. Has no effect when use_progress_prediction is False.",
)

AGENTIC_SD_NUM_HISTORY_FRAMES = flags.DEFINE_integer(
    "sd.num_history_frames",
    0,
    "The number of history frames to use for SD.",
)

AGENTIC_SD_HISTORY_INTERVAL_S = flags.DEFINE_float(
    "sd.history_interval_s",
    1.0,
    "The interval between history frames to use for SD.",
)

AGENTIC_SD_PRINT_FINAL_PROMPT = flags.DEFINE_bool(
    "sd.print_final_prompt",
    False,
    "Whether to print the final prompt for SD.",
)

AGENTIC_SD_USE_START_IMAGES = flags.DEFINE_bool(
    "sd.use_start_images",
    True,
    "Whether to use start images for SD.",
)

AGENTIC_SD_USE_EXPLICIT_THINKING = flags.DEFINE_bool(
    "sd.use_explicit_thinking",
    True,
    "Whether to use explicit thinking for SD.",
)

AGENTIC_SD_GUIDED_THINKING_WORD_LIMIT = flags.DEFINE_integer(
    "sd.guided_thinking_word_limit",
    50,
    "The word limit for guided thinking for SD.",
)

AGENTIC_SD_PRINT_RAW_SD_RESPONSE = flags.DEFINE_bool(
    "sd.print_raw_sd_response",
    True,
    "Whether to print the raw SD response.",
)

AGENTIC_SD_ASYNC_SD_INTERVAL_S = flags.DEFINE_float(
    "sd.async_sd_interval_s",
    0.2,
    "The interval in seconds between async SD runs.",
)

AGENTIC_OVERALL_TASK_SUCCESS_DETECTOR_THINKING_BUDGET = flags.DEFINE_integer(
    "sd.overall_task_success_detector_thinking_budget",
    -1,
    "The thinking budget for the overall task success detector. Set to 0 to"
    " disable, -1 for automatic.",
)

AGENTIC_STOP_ON_SUCCESS = flags.DEFINE_bool(
    "sd.stop_on_success",
    True,
    "Whether the run_instruction_until_done tool should stop the robot when the"
    " success detector returns True.",
)

AGENTIC_SD_TEMPERATURE = flags.DEFINE_float(
    "sd.temperature",
    0.0,
    "The model temperature to use for SD. Recommend to use higher temperature"
    " for ensamble SD.",
)

AGENTIC_SD_ENSEMBLE_SIZE = flags.DEFINE_integer(
    "sd.ensemble_size",
    1,
    "The number of parallel SD runs to use.",
)

AGENTIC_SD_ENSEMBLE_THRESHOLD = flags.DEFINE_integer(
    "sd.ensemble_threshold",
    1,
    "The threshold for the ensemble size.",
)

AGENTIC_SD_SLEEP_INTERVAL_S = flags.DEFINE_float(
    "sd.sleep_interval_s",
    0.2,
    "The sleep interval in seconds for the ensemble SD model.",
)

# ------------------------
# Scene description flags.
# ------------------------

AGENTIC_USE_SCENE_DESCRIPTION = flags.DEFINE_bool(
    "agent.use_scene_description",
    False,
    "Whether to use scene description.",
)

AGENTIC_SCENE_DESCRIPTION_MODEL_NAME = flags.DEFINE_string(
    "scene_description.model_name",
    "gemini-robotics-er-1.5-preview",
    "The name of the model to use for scene description.",
)

AGENTIC_SCENE_DESCRIPTION_THINKING_BUDGET = flags.DEFINE_integer(
    "scene_description.thinking_budget",
    100,
    "The thinking budget for the scene description model. Set to 0 to disable,"
    " -1 for automatic.",
)

AGENTIC_SCENE_DESCRIPTION_NUM_OUTPUT_WORDS = flags.DEFINE_integer(
    "scene_description.num_output_words",
    200,
    "The number of words to output for scene description.",
)

# ------------------------
# Robot backend flags.
# ------------------------
AGENTIC_ROBOT_BACKEND_HOST = flags.DEFINE_string(
    "backend.robot_backend_host",
    "localhost",
    "The hostname of the robot backend server. (Omit the http:// prefix.)",
)

AGENTIC_ROBOT_BACKEND_PORT = flags.DEFINE_integer(
    "backend.robot_backend_port",
    8888,
    "The port of the robot backend server.",
)


# ------------------------
# Logging flags.
# ------------------------
AGENTIC_ENABLE_LOGGING = flags.DEFINE_bool(
    "logging.enable_logging",
    False,
    "Whether to enable logging.",
)

AGENTIC_ROBOT_ID = flags.DEFINE_string(
    "logging.robot_id",
    None,
    "The ID of the robot.",
)

AGENTIC_LOGGING_OUTPUT_DIRECTORY = flags.DEFINE_string(
    "logging.output_directory",
    "/tmp/safari_logs",
    "The output directory for the logs.",
)
