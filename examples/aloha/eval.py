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

"""Run Robotics Policy Eval on Aloha Robot."""

import socket
from absl import app
from absl import flags
import env
from gdm_robotics.runtime import runloop as runloop_lib
import rclpy
from safari_sdk.logging.python import episodic_logger
from safari_sdk.model import constants
from safari_sdk.model import gemini_robotics_policy

# Robot constants
ROBOT_CONFIG_NAME = 'aloha_stationary'
CONFIG_BASE_PATH = '/home/juggler/interbotix_ws/src/aloha/config/'

SERVE_ID = flags.DEFINE_string(
    'serve_id',
    'gemini_robotics_on_device',
    'The serve id to use for the Gemini Robotics Policy.',
)
INFERENCE_MODE = flags.DEFINE_enum_class(
    'inference_mode',
    constants.InferenceMode.SYNCHRONOUS,
    constants.InferenceMode,
    'The inference mode to use for the Gemini Robotics Policy.',
)
ROBOTS_API_CONNECTION = flags.DEFINE_enum_class(
    'robotics_api_connection',
    constants.RoboticsApiConnectionType.LOCAL,
    constants.RoboticsApiConnectionType,
    'The robotics API connection type to use.',
)
INSTRUCTION = flags.DEFINE_string(
    'instruction', None, 'Specify the instruction to give to the robot.'
)
MAX_NUM_STEPS = flags.DEFINE_integer(
    'steps', 5000, 'Number of steps to run the episode for.'
)
AGENT_ID = flags.DEFINE_string(
    'agent_id',
    socket.gethostname(),
    'The agent id to use for the episodic logger.',
)


class UserInputRunloopOperations(runloop_lib.RunloopRuntimeOperations):
  """Runloop runtime operations that handle user input."""

  def __init__(self, default_instruction: str):
    self._instruction = default_instruction
    self._has_quit = False

  @property
  def instruction(self) -> str:
    return self._instruction

  @property
  def has_quit(self) -> bool:
    return self._has_quit

  def before_episode_reset(self) -> bool:
    # Reset the quit flag.
    self._has_quit = False

    new_input = instruction = input(
        "\nEnter a new instruction or 'quit' to cleanly exit: "
    ).lower()

    if new_input == 'quit':
      self._has_quit = True
      return False

    # It is an instruction. Save it.
    self._instruction = instruction
    return True


def main(argv):
  del argv  # Unused.
  if SERVE_ID.value is None:
    raise ValueError('serve_id must be specified.')

  if INSTRUCTION.value is None:
    print('Script started. Enter an instruction to begin.')
  else:
    print(f'Script started with instruction: {INSTRUCTION.value}')

  # Create environment and policy.
  environment = env.create_aloha_environment(
      robot_config_name=ROBOT_CONFIG_NAME,
      config_base_path=CONFIG_BASE_PATH,
      max_num_steps=MAX_NUM_STEPS.value,
  )
  # Uninstalls ros signal handlers (signal.SIGINT, signal.SIGTERM) to avoid
  # automatic ROS shutdown during keyboard interrupt.
  rclpy.signals.uninstall_signal_handlers()
  policy = gemini_robotics_policy.GeminiRoboticsPolicy(
      serve_id=SERVE_ID.value,
      task_instruction_key=env.INSTRUCTION_RESET_OPTION_KEY,
      image_observation_keys=(
          'overhead_cam',
          'worms_eye_cam',
          'wrist_cam_left',
          'wrist_cam_right',
      ),
      proprioceptive_observation_keys=('joints_pos',),
      inference_mode=INFERENCE_MODE.value,
      robotics_api_connection=ROBOTS_API_CONNECTION.value,
  )
  policy.step_spec(environment.timestep_spec())

  user_input_ops = UserInputRunloopOperations(INSTRUCTION.value)

  def _update_instruction_on_reset():
    return env.ResetOptions(
        options={env.INSTRUCTION_RESET_OPTION_KEY: user_input_ops.instruction}
    )

  logger = episodic_logger.EpisodicLogger.create(
      agent_id=AGENT_ID.value,
      task_id=user_input_ops.instruction,
      proprioceptive_observation_keys=['joints_pos'],
      output_directory='/tmp/eval_logs',
      action_spec=environment.action_spec(),
      timestep_spec=environment.timestep_spec(),
      image_observation_keys=[
          'overhead_cam',
          'worms_eye_cam',
          'wrist_cam_left',
          'wrist_cam_right',
      ],
      policy_extra_spec={},
  )

  runloop = runloop_lib.Runloop(
      environment=environment,
      policy=policy,
      loggers=[logger],
      runloop_runtime_operations=(user_input_ops,),
      reset_options_provider=_update_instruction_on_reset,
  )

  print('Script started. Enter an instruction to begin.')

  while True:
    try:
      runloop.reset()
      runloop.run_single_episode()
      if user_input_ops.has_quit:
        break
    except KeyboardInterrupt:
      runloop.stop()

  environment.close()


if __name__ == '__main__':
  app.run(main)
