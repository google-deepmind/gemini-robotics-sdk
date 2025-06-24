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

import argparse

import env
import rclpy

from safari_sdk.model import gemini_robotics_policy
from safari_sdk.model import genai_robotics

AlohaEnv = env.AlohaEnv

_SERVE_ID = 'gemini_robotics_on_device'

_IMAGE_SIZE = (480, 848)
_ALOHA_CAMERAS = {
    'overhead_cam': _IMAGE_SIZE,
    'worms_eye_cam': _IMAGE_SIZE,
    'wrist_cam_left': _IMAGE_SIZE,
    'wrist_cam_right': _IMAGE_SIZE,
}
_ALOHA_JOINTS = {'joints_pos': 14}

# Robot constants
ROBOT_CONFIG_NAME = 'aloha_stationary'
CONFIG_BASE_PATH = '/home/juggler/interbotix_ws/src/aloha/config/'


def run_single_episode(aloha_env, instruction, steps=5000):
  """Run a single episode of AlohaEnv."""
  # 2. Reset the environment with a task instruction
  obs, info = aloha_env.reset(options={'instruction': instruction})
  print(f"Starting episode with instruction: {info['instruction']}")
  print(
      'Initial observation received. Joint positions:'
      f" {len(obs['joints_pos'])}"
  )
  # 3. Initialize the policy using GeminiRoboticsPolicy
  try:
    print('Creating policy...')
    policy = gemini_robotics_policy.GeminiRoboticsPolicy(
        serve_id=_SERVE_ID,
        task_instruction=instruction,
        inference_mode=gemini_robotics_policy.InferenceMode.SYNCHRONOUS,
        cameras=_ALOHA_CAMERAS,
        joints=_ALOHA_JOINTS,
        robotics_api_connection=genai_robotics.RoboticsApiConnectionType.LOCAL,
    )
    policy.setup()  # Initialize the policy
    print('GeminiRoboticsPolicy initialized successfully.')
  except ValueError as e:
    print(f'Error initializing policy: {e}')
    raise
  except Exception as e:  # pylint: disable=broad-except
    print(f'An unexpected error occurred during initialization: {e}')
    raise

  # 4. Run a loop (e.g., for one episode)
  for _ in range(steps):  # Default: Run for 5000 steps (100 seconds at 50Hz)
    obs['task_instruction'] = info['instruction']
    action = policy.step(obs)
    # 5. Step the environment with the chosen action
    obs, _, _, _, _ = aloha_env.step(action)


def main():
  """Main function.

  Usage: python3 eval.py [--instruction <instruction>] [--steps <steps>]

    If --instruction is specified, all episodes will use the same instruction.
    Otherwise, specify task instruction when prompted.

    You may stop the task using ctrl-c or by providing a KeyboardInterrupt.
    Exit the program by typing 'quit' or ctrl-c between episodes.
  """
  parser = argparse.ArgumentParser(
      description='Run Robotics Policy Eval on Aloha Robot.'
  )
  parser.add_argument(
      '--instruction',
      type=str,
      help='Specify the instruction to give to the robot.',
  )
  parser.add_argument(
      '--steps',
      type=int,
      default=5000,
      help='Number of steps to run the episode for.',
  )
  args = parser.parse_args()

  if args.instruction is None:
    print('Script started. Enter an instruction to begin.')
  else:
    print(f'Script started with instruction: {args.instruction}')

  # 1. Initialize the environment
  aloha_env = AlohaEnv(
      robot_config_name=ROBOT_CONFIG_NAME, config_base_path=CONFIG_BASE_PATH
  )

  # uninstalls ros signal handlers (signal.SIGINT, signal.SIGTERM) to avoid
  # automatic ROS shutdown during keyboard interrupt.
  rclpy.signals.uninstall_signal_handlers()

  # Uses reset to bring the robot to a known home position before starting the
  # eval loop.
  aloha_env.reset(options={'instruction': ''})

  print('Script started. Enter an instruction to begin.')
  # The main application loop. It continues until the user decides to quit.
  episode_started = False
  while True:
    try:
      episode_started = False
      if args.instruction:
        instruction = args.instruction.lower()
        if (
            input(
                "\nEnter to start an episode, 'quit' to cleanly exit."
            ).lower()
            == 'quit'
        ):
          raise KeyboardInterrupt
      else:
        instruction = input(
            "\nEnter a new instruction or 'quit' to cleanly exit: "
        ).lower()
        if instruction == 'quit':
          raise KeyboardInterrupt

      episode_started = True
      print('Episode started, Ctrl-C to stop')
      run_single_episode(aloha_env, instruction, steps=args.steps)
    except KeyboardInterrupt:
      if not episode_started:
        print('\nExiting program.')
        aloha_env.close()
        break


if __name__ == '__main__':
  main()
