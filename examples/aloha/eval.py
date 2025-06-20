"""Run Robotics Policy Eval on Aloha Robot."""

import argparse
import json
import time
import env
import grpc
import numpy as np
import rclpy

AlohaEnv = env.AlohaEnv

# GRPC server constants
SERVER_ADDRESS = 'localhost:60061'
SERVICE_NAME = 'gemini_robotics'
METHOD_NAME = 'sample_actions_json_flat'
FULL_METHOD_NAME = f'/{SERVICE_NAME}/{METHOD_NAME}'

# Robot constants
ROBOT_CONFIG_NAME = 'aloha_stationary'
CONFIG_BASE_PATH = '/home/juggler/interbotix_ws/src/aloha/config/'

HOME_POSITION = [[0.0, -0.96, 1.16, 0.0, -0.3, 0.0, 1.0]]


def run_single_episode(aloha_env, instruction, steps=5000):
  """Run a single episode of AlohaEnv."""
  # 2. Reset the environment with a task instruction
  obs, info = aloha_env.reset(options={'instruction': instruction})
  print(f"Starting episode with instruction: {info['instruction']}")
  print(
      'Initial observation received. Joint positions:'
      f" {len(obs['joints_pos'])}"
  )
  # 3. setup the gRPC server
  with grpc.insecure_channel(
      SERVER_ADDRESS,
  ) as channel:
    # Prepare request bytes (client-side serialization)
    query_model = channel.unary_unary(
        FULL_METHOD_NAME,
        request_serializer=lambda v: v,  # Already bytes
        response_deserializer=lambda v: v,  # Expecting bytes back
    )
    future_actions = np.zeros((0, 14), dtype=np.float32).tolist()

    # 4. Run a loop (e.g., for one episode)
    for _ in range(steps):  # Run for 5000 stsps (100 seconds at 50Hz)
      start_time = time.time()
      obs['conditioning'] = future_actions
      obs['task_instruction'] = info['instruction']
      obs_json = json.dumps(obs).encode('utf-8')
      request_bytes = obs_json

      response_bytes = query_model(request_bytes, timeout=5)
      response = json.loads(response_bytes.decode('utf-8'))
      future_actions = response['conditioning']
      action_chunk = np.asarray(response['action_chunk'])
      # 5. Step the environment with the chosen action
      obs, _, _, _, _ = aloha_env.step(action_chunk)

      time_elapsed = time.time() - start_time
      if time_elapsed < aloha_env.dt:
        time.sleep(aloha_env.dt - time_elapsed)


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
