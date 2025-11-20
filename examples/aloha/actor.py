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

"""Actor class that owns policy, env, and runloop."""

from collections.abc import Sequence
import logging
import threading

from absl import flags
import env
from gdm_robotics.adapters import gymnasium_env_to_gdmr_env_wrapper as gym_wrapper
from gdm_robotics.runtime import runloop as runloop_lib
import rclpy

from safari_sdk import utils
from safari_sdk.logging.python import episodic_logger as episodic_logger_lib
from safari_sdk.model import constants
from safari_sdk.model import gemini_robotics_policy


# Flags
_ENABLE_LOGGING = flags.DEFINE_bool(
    "logging.enable_logging",
    False,
    "Whether to enable logging.",
)
_ROBOT_ID = flags.DEFINE_string(
    "logging.robot_id",
    "unused_robot_id",
    "The ID of the robot.",
)

_LOGGING_OUTPUT_DIRECTORY = flags.DEFINE_string(
    "logging.output_directory",
    "/persistent/robot_backend/episodic_logs",
    "The output directory for the logs.",
)
_IMAGE_COMPRESSION_JPEG_QUALITY = flags.DEFINE_integer(
    "image_compression_jpeg_quality",
    95,
    "JPEG quality to use for images sent to the model.",
)

# Robot constants
ROBOT_CONFIG_NAME = "aloha_stationary"
CONFIG_BASE_PATH = "/home/juggler/interbotix_ws/src/aloha/config/"
MAX_NUM_STEPS = 5000
IMAGE_OBSERVATION_KEYS = (
    "overhead_cam",
    "worms_eye_cam",
    "wrist_cam_left",
    "wrist_cam_right",
)
PROPRIOCEPTIVE_OBSERVATION_KEYS = ("joints_pos",)
DEFAULT_INSTRUCTION = "do nothing"
UNUSED_TASK_ID = "unused_task_id"


class Actor:
  """Manages the robot environment, policy, and execution runloop."""

  def __init__(
      self,
      serve_id: str,
      inference_mode: constants.InferenceMode,
      robotics_api_connection: constants.RoboticsApiConnectionType,
  ):
    """Initializes the Actor, creating the environment and policy."""
    # Uninstalls ros signal handlers (signal.SIGINT, signal.SIGTERM) to avoid
    # automatic ROS shutdown during keyboard interrupt.
    rclpy.signals.uninstall_signal_handlers()
    # Create the Aloha environment.
    self.environment = env.create_aloha_environment(
        robot_config_name=ROBOT_CONFIG_NAME,
        config_base_path=CONFIG_BASE_PATH,
        max_num_steps=MAX_NUM_STEPS,
        default_instruction=DEFAULT_INSTRUCTION,
    )
    # Create the Gemini Robotics Policy.
    self.policy = self.create_policy(
        serve_id, inference_mode, robotics_api_connection
    )
    # Create the episodic logger.
    self.episodic_logger = None
    if _ENABLE_LOGGING.value:
      robot_id = _ROBOT_ID.value
      if not robot_id:
        robot_id = utils.get_robot_id_from_system_env()
        if not robot_id:
          raise ValueError(
              "Robot ID is not set by flag or system environment variable."
              "Please set '--logging.robot_id' flag or 'ROBOT_ID' environment"
              " variable."
          )
      print(f"Using robot id: {robot_id}")
      self.episodic_logger = episodic_logger_lib.EpisodicLogger.create(
          agent_id=_ROBOT_ID.value,
          task_id=UNUSED_TASK_ID,
          proprioceptive_observation_keys=PROPRIOCEPTIVE_OBSERVATION_KEYS,
          output_directory=_LOGGING_OUTPUT_DIRECTORY.value,
          action_spec=self.environment.action_spec(),
          timestep_spec=self.environment.timestep_spec(),
          image_observation_keys=IMAGE_OBSERVATION_KEYS,
          policy_extra_spec={},
          dynamic_metadata_provider=self.get_dynamic_metadata,
      )
    # Create the runloop.
    self.runloop = self.create_runloop(
        self.environment, self.policy, self.get_loggers()
    )
    # Threading primitives for controlling the runloop and instruction.
    self._runloop_thread = None
    self._instruction_lock = threading.Lock()  # Protects access to instruction.
    self._instruction = DEFAULT_INSTRUCTION
    # Dynamic metadata
    self._agent_session_id = ""

  def _runloop(self):
    """The main loop where the policy interacts with the environment.

    This method is run in a separate thread. It resets the environment and
    policy state, then continuously steps the policy and environment until
    the stop event is set.
    """
    logging.info("Runloop started.")
    self.runloop.reset()
    self.runloop.run_single_episode()
    logging.info("Runloop stopped.")

  def start_runloop(self):
    """Starts the actor's runloop in a new thread.

    If the runloop thread is already running, this method does nothing.
    """
    if self._runloop_thread is None or not self._runloop_thread.is_alive():
      self._runloop_thread = threading.Thread(target=self._runloop)
      self._runloop_thread.start()
      logging.info("Runloop thread started.")
    else:
      logging.info("Runloop thread already running.")

  def stop_runloop(self):
    """Signals the runloop to stop and waits for the thread to join."""
    logging.info("Stopping runloop...")
    if self._runloop_thread and self._runloop_thread.is_alive():
      self.runloop.stop()
      self._runloop_thread.join()
    self._runloop_thread = None
    logging.info("Runloop stopped.")

  def get_instruction(self) -> str:
    """Returns the current task instruction."""
    with self._instruction_lock:
      return self._instruction

  def set_instruction(self, instruction: str):
    """Sets the task instruction for the policy.

    This method is thread-safe.

    Args:
      instruction: The new instruction string.
    """
    with self._instruction_lock:
      self._instruction = instruction
    logging.info("Instruction set to: %s", instruction)

  def get_agent_session_id(self) -> str:
    """Returns the agent session ID."""
    return self._agent_session_id

  def set_agent_session_id(self, agent_session_id: str):
    """Sets the agent session ID."""
    self._agent_session_id = agent_session_id

  def get_dynamic_metadata(self) -> dict[str, str]:
    """Returns the dynamic metadata for the runloop."""
    dynamic_metadata = {}
    if self._agent_session_id:
      dynamic_metadata["agent_session_id"] = self._agent_session_id
    return dynamic_metadata

  def reset(self):
    """Resets the actor's runloop and environment."""
    self.stop_runloop()
    logging.info("Resetting environment...")
    self.environment.wrapped_env.reset(
        options={
            env.SHOULD_MOVE_TO_HOME_RESET_OPTION_KEY: True,
            env.INSTRUCTION_RESET_OPTION_KEY: DEFAULT_INSTRUCTION,
        }
    )
    logging.info("Environment reset.")

  def reset_from(
      self,
      serve_id: str,
      inference_mode: constants.InferenceMode,
      robotics_api_connection: constants.RoboticsApiConnectionType,
  ):
    """Resets the actor with a new policy and runloop configuration.

    This method stops the current runloop, resets the environment, and then
    re-initializes the policy and runloop with the provided connection details.

    Args:
      serve_id: The ID of the model serving the policy.
      inference_mode: The inference mode for the Gemini Robotics Policy.
      robotics_api_connection: The connection type for the robotics API.
    """
    self.stop_runloop()
    logging.info("Resetting environment...")
    self.environment.wrapped_env.reset(
        options={
            env.SHOULD_MOVE_TO_HOME_RESET_OPTION_KEY: True,
            env.INSTRUCTION_RESET_OPTION_KEY: DEFAULT_INSTRUCTION,
        }
    )
    logging.info("Environment reset.")
    logging.info("Resetting policy...")
    self.policy = self.create_policy(
        serve_id, inference_mode, robotics_api_connection
    )
    self.policy.step_spec(self.environment.timestep_spec())
    logging.info("Policy reset.")
    logging.info("Resetting runloop...")
    self.runloop = self.create_runloop(
        self.environment, self.policy, self.get_loggers()
    )
    logging.info("Runloop reset.")

  def shutdown(self):
    """Stops the runloop and closes the environment."""
    self.stop_runloop()
    logging.info("Shutting down environment...")
    self.environment.close()
    logging.info("Actor shutdown complete.")

  def create_policy(
      self,
      serve_id: str,
      inference_mode: constants.InferenceMode,
      robotics_api_connection: constants.RoboticsApiConnectionType,
  ):
    """Creates a new policy with the provided connection details."""
    policy = gemini_robotics_policy.GeminiRoboticsPolicy(
        serve_id=serve_id,
        task_instruction_key=env.INSTRUCTION_RESET_OPTION_KEY,
        image_observation_keys=IMAGE_OBSERVATION_KEYS,
        proprioceptive_observation_keys=PROPRIOCEPTIVE_OBSERVATION_KEYS,
        inference_mode=inference_mode,
        robotics_api_connection=robotics_api_connection,
        min_replan_interval=15,
        image_compression_jpeg_quality=_IMAGE_COMPRESSION_JPEG_QUALITY.value,
    )
    policy.step_spec(self.environment.timestep_spec())
    return policy

  def create_runloop(
      self,
      environment: gym_wrapper.GymnasiumEnvToGdmrEnvWrapper,
      policy: gemini_robotics_policy.GeminiRoboticsPolicy,
      loggers: Sequence[episodic_logger_lib.EpisodicLogger],
  ):
    """Creates a new runloop."""
    return runloop_lib.Runloop(
        environment=environment,
        policy=policy,
        loggers=loggers,
        reset_options_provider=lambda: gym_wrapper.GymnasiumEnvResetOptions(
            options={
                env.INSTRUCTION_RESET_OPTION_KEY: self.get_instruction(),
            }
        ),
    )

  def get_loggers(self) -> list[episodic_logger_lib.EpisodicLogger]:
    """Returns the loggers for the runloop."""
    if self.episodic_logger is None:
      return []
    return [self.episodic_logger]
