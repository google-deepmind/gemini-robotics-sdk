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

"""Flags for the agentic framework.

All agentic framework and user code flags should be defined here. To avoid flag
name collisions, please prefix all flag variables with "AGENTIC_" and all actual
flag value with "agent.".
"""

import enum
from absl import flags
from google.genai import types as genai_types
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
  # This success detector queries the robot backend to determine if the success
  # pedal (i.e. by human operator) has been triggered.
  PEDAL_TRIGGERED_SUCCESS_DETECTOR = "PedalTriggeredSuccessDetector"


SdStitchMode = types.SDStitchMode

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
    ,
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
    default=types.ControlMode.SERVER_AND_TERMINAL,
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

AGENTIC_KILL_PORT_PROCESS = flags.DEFINE_bool(
    "framework.kill_port_process",
    False,
    "If True, kill any process using the external controller port before"
    " starting the server. First attempts graceful SIGTERM, then SIGKILL"
    " after a timeout. Helps when restarting the framework and a previous"
    " instance is still holding the port.",
)

AGENTIC_EXTERNAL_UI_TYPE = flags.DEFINE_enum_class(
    "framework.external_ui_type",
    default=types.ExternalUIType.NONE,
    enum_class=types.ExternalUIType,
    help=(
        "The external UI type to use for event printing. For example,"
        " OPERATOR_DATA_COLLECT prints to robotics UI."
    ),
)

AGENTIC_PUBLISH_LOGGING_EVENTS = flags.DEFINE_bool(
    "framework.publish_logging_events",
    default=False,
    help="Whether to publish Python logging events to the event bus.",
)

AGENTIC_HANDLE_AUDIO_ON_ROBOT = flags.DEFINE_boolean(
    "framework.handle_audio_on_robot",
    False,
    "Whether to stream audio recording from and playback to the robot server.",
)

AGENTIC_HANDLE_AUDIO_ON_GUI = flags.DEFINE_boolean(
    "framework.handle_audio_on_gui",
    True,
    "Whether to handle audio on the GUI. This is only applicable when running"
    "the agent as external server and interacting through the web GUI.",
)
AGENTIC_LISTEN_WHILE_SPEAKING = flags.DEFINE_boolean(
    "framework.listen_while_speaking",
    False,
    "Whether to listen while speaking.",
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
    "Whether to display audio transcription in the terminal UI. Note: Audio"
    " transcription is always enabled in the Live API when audio input/output"
    " is enabled; this flag only controls whether the transcription is"
    " displayed in the terminal.",
)

AGENTIC_OUTPUT_AUDIO_VOICE_NAME = flags.DEFINE_string(
    "agent.voice_name",
    None,
    "The name of the voice to use for the Gemini Live agent. Has no effect if"
    " agent.enable_audio_output is False.",
)

AGENTIC_ONLY_ACTIVITY_COVERAGE = flags.DEFINE_bool(
    "agent.only_activity_coverage",
    False,
    "Whether to only use activity coverage for the Gemini Live agent."
    "In additional to toggling the LiveAPI setting, images are inserted before"
    "each test input and function response.",
)

AGENTIC_UPDATE_VISION_AFTER_FR = flags.DEFINE_bool(
    "agent.update_vision_after_fr",
    False,
    "Whether to update vision after a function response.",
)

AGENTIC_ENABLE_CONTEXT_WINDOW_COMPRESSION = flags.DEFINE_bool(
    "agent.enable_context_window_compression",
    True,
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

AGENTIC_USE_QUIET_AUTONOMY_MODE = flags.DEFINE_bool(
    "agent.use_quiet_autonomy_mode",
    False,
    "Whether to use quiet autonomy mode in the prompt.",
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

AGENTIC_ENABLE_IMAGE_STITCHING = flags.DEFINE_bool(
    "agent.enable_image_stitching",
    False,
    "Whether to stitch multiple camera images into a single grid image before"
    " sending to the Gemini Live model. When disabled, images are sent"
    " individually.",
)

AGENTIC_SHOW_CAMERA_NAME_IN_STITCHED_IMAGE = flags.DEFINE_bool(
    "agent.show_camera_name_in_stitched_image",
    True,
    "Whether to show camera name labels on stitched images. Only has effect"
    " when agent.enable_image_stitching is True.",
)

AGENTIC_ENABLE_AUTOMATIC_SESSION_RESUMPTION = flags.DEFINE_bool(
    "agent.enable_automatic_session_resumption",
    True,
    "Whether to enable automatic session resumption when receiving GO_AWAY"
    " from the Live API. When enabled, the framework will automatically"
    " reconnect after a 20 second grace period.",
)

AGENTIC_NON_STREAMING_IMAGE_PRUNING_TRIGGER_AMOUNT = flags.DEFINE_integer(
    "agent.non_streaming_image_pruning_trigger_amount",
    60,
    "Maximum number of images to keep in conversation history for non-streaming"
    " API. When this limit is exceeded, images are pruned down to"
    " agent.non_streaming_image_pruning_target_amount.",
)

AGENTIC_NON_STREAMING_IMAGE_BUFFERING_INTERVAL_SECONDS = flags.DEFINE_float(
    "agent.non_streaming_image_buffering_interval_seconds",
    1.0,
    "The interval in seconds for buffering images in the non-streaming handler."
    " Images received faster than this interval are dropped.",
)

AGENTIC_NON_STREAMING_IMAGE_PRUNING_TARGET_AMOUNT = flags.DEFINE_integer(
    "agent.non_streaming_image_pruning_target_amount",
    15,
    "When the number of images in context exceeds"
    " agent.non_streaming_image_pruning_trigger_amount, prune down to this"
    " number. This batch pruning strategy preserves the prefill cache by"
    " keeping the conversation prefix stable between pruning events.",
)

AGENTIC_NON_STREAMING_DISCARD_IMAGES_AFTER_TURN = flags.DEFINE_bool(
    "agent.non_streaming_discard_images_after_turn",
    True,
    "Whether to discard buffered images after each turn (user message or"
    " function response). When True (default), images are cleared after being"
    " used. When False, images accumulate and are sent with subsequent turns.",
)

AGENTIC_NON_STREAMING_FR_LATEST_IMAGE_ONLY = flags.DEFINE_bool(
    "agent.non_streaming_fr_latest_image_only",
    True,
    "When enabled, only the latest image per stream (or the latest stitched"
    " frame) captured during tool execution is attached to the function"
    " response. When disabled, all images from the FC-FR window are attached.",
)

AGENTIC_NON_STREAMING_USER_TURN_LATEST_IMAGE_ONLY = flags.DEFINE_bool(
    "agent.non_streaming_user_turn_latest_image_only",
    False,
    "When enabled, only the latest image per stream (or the latest stitched"
    " frame) buffered before the user turn is attached to the user turn."
    " When disabled, all buffered images are attached.",
)

AGENTIC_NON_STREAMING_INCLUDE_STREAM_NAMES = flags.DEFINE_bool(
    "agent.non_streaming_include_stream_names",
    True,
    "Whether to prepend stream/camera name text labels before each image in"
    " non-stitching mode. When True (default), a text part with the camera"
    " name is added before each image. When False, only the raw image bytes"
    " are included, saving tokens.",
)

AGENTIC_NON_STREAMING_THINKING_LEVEL = flags.DEFINE_string(
    "agent.non_streaming_thinking_level",
    None,
    "The thinking level for the non-streaming handler"
    " (MINIMAL, LOW, MEDIUM, HIGH).",
)

AGENTIC_NON_STREAMING_TOOL_RESULT_TIMEOUT_SECONDS = flags.DEFINE_float(
    "agent.non_streaming_tool_result_timeout_seconds",
    300.0,
    "The maximum amount of time in seconds to wait for tool results from the"
    " event bus after the model emits a function call before producing an error"
    " response.",
)

# Agent model generation parameters (for non-streaming handler).
AGENTIC_AGENT_TEMPERATURE = flags.DEFINE_float(
    "agent.temperature",
    None,
    "Temperature for the agent model (0.0-2.0). None uses server default.",
)

AGENTIC_AGENT_MAX_OUTPUT_TOKENS = flags.DEFINE_integer(
    "agent.max_output_tokens",
    None,
    "Max output tokens for agent responses. None uses server default.",
)

AGENTIC_AGENT_THINKING_BUDGET = flags.DEFINE_integer(
    "agent.thinking_budget",
    None,
    "Thinking budget for the agent model. 0=disabled, -1=auto, None=default.",
)

AGENTIC_AGENT_MEDIA_RESOLUTION = flags.DEFINE_enum_class(
    "agent.media_resolution",
    default=genai_types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
    enum_class=genai_types.MediaResolution,
    help=(
        "Media resolution for images. LOW=280, MEDIUM=560, HIGH=1120,"
        " ULTRA_HIGH=2240 tokens."
    ),
)

AGENTIC_ORCHESTRATOR_HANDLER_TYPE = flags.DEFINE_enum_class(
    "agent.orchestrator_handler_type",
    default=types.OrchestratorHandlerType.NONSTREAMING_GENAI,
    enum_class=types.OrchestratorHandlerType,
    help=(
        "The type of orchestrator handler to use. STREAMING uses the Live API,"
        " NONSTREAMING_GENAI uses the GenAI generate_content API,"
        " NONSTREAMING_EVERGREEN uses the internal Evergreen model_client API."
    ),
)


AGENTIC_NON_STREAMING_ENABLE_CONTEXT_SNAPSHOT_LOGGING = flags.DEFINE_bool(
    "agent.enable_non_streaming_context_snapshot_logging",
    True,
    "Whether to log full conversation context snapshots to the event bus on"
    " every model call. When enabled, the entire conversation history"
    " (including images, function calls, and thought signatures) is logged"
    " as a CONTEXT_SNAPSHOT event.",
)

# ------------------------
# Tool flags.
# ------------------------
AGENTIC_TOOL_RUN_FOR_DURATION_SECOND = flags.DEFINE_float(
    "run_for_duration.duration",
    8.0,
    "The default duration in seconds for the run_instruction_for_duration"
    " tool.",
)

AGENTIC_RUN_UNTIL_DONE_TIME_LIMIT = flags.DEFINE_float(
    "run_until_done.time_limit",
    60.0,
    "The time limit in seconds for the run_instruction_until_done tool.",
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
    "The thinking budget for Gemini 2.5 family success detection model. 0 is"
    " DISABLED. -1 is AUTOMATIC. The default values and allowed ranges are"
    " model dependent. Mutually exclusive with sd.thinking_level.",
)

AGENTIC_SD_THINKING_LEVEL = flags.DEFINE_string(
    "sd.thinking_level",
    None,
    "The thinking level for Gemini 3 family success detection model "
    "(MINIMAL, LOW, MEDIUM, HIGH). Mutually exclusive with sd.thinking_budget.",
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
    " for ensemble SD.",
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

AGENTIC_SD_MAX_OUTPUT_TOKENS = flags.DEFINE_integer(
    "sd.max_output_tokens",
    None,
    "Max output tokens for SD responses. None uses server default.",
)

AGENTIC_SD_MEDIA_RESOLUTION = flags.DEFINE_enum_class(
    "sd.media_resolution",
    default=genai_types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
    enum_class=genai_types.MediaResolution,
    help=(
        "Media resolution for SD images. LOW=280, MEDIUM=560, HIGH=1120,"
        " ULTRA_HIGH=2240 tokens."
    ),
)

AGENTIC_SD_STITCH_MODE = flags.DEFINE_enum_class(
    "sd.stitch_mode",
    default=SdStitchMode.NONE,
    enum_class=SdStitchMode,
    help="Image stitching mode for success detection. Options:"
    " 'none' (send individual images),"
    " 'camera' (stitch all cameras into one image per timepoint),"
    " 'time' (stitch start+current per camera).",
)

AGENTIC_SD_PROMPT_FILE = flags.DEFINE_string(
    "sd.prompt_file",
    None,
    "Path to a file containing the SD question prompt. If set, overrides the"
    " default prompt. Use {subtask} placeholder for the task name.",
)

AGENTIC_SD_PROMPT_NAME = flags.DEFINE_string(
    "sd.prompt_name",
    None,
    "Prompt name from sd_prompts.py registry. If unset, uses mode-appropriate"
    " default (new_no_stitch, new_camera, new_time).",
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

AGENTIC_LOGGING_SESSION_LOG_TYPE_KEY = flags.DEFINE_string(
    "logging.session_log_type_key",
    None,
    "The key of the session log type.",
)

AGENTIC_LOGGING_SESSION_LOG_TYPE_VALUE = flags.DEFINE_string(
    "logging.session_log_type_value",
    "agent",
    "The value of the session log type.",
)

AGENTIC_EXCLUDE_MODEL_IMAGE_INPUT_LOGGING = flags.DEFINE_bool(
    "logging.exclude_model_image_input_logging",
    False,
    "Whether to exclude MODEL_IMAGE_INPUT events from being logged to the"
    " event stream. These are the most frequent event type and are currently"
    " unused by downstream consumers. Enabling this reduces log size and"
    " improves logging performance.",
)

AGENTIC_LOGGING_AGENT_SESSION_ID = flags.DEFINE_string(
    "logging.agent_session_id",
    None,
    "The agent session ID to use when starting the event bus. If not set, a"
    " random UUID will be generated.",
)
