# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Interactive Aloha Simulation eval of Gemini Robotics On device model.

Instructions:

- I = enter new instruction.
- space bar = pause/restart.
- backspace = reset environment.
- mouse right moves the camera
- mouse left rotates the camera
- double click to select an object

When the environment is not running:
- ctrl + mouse left rotates a selected object
- ctrl + mouse right moves a selected object

When the environment is running:
- ctrl + mouse left applies torque to an object
- ctrl + mouse right applies force to an object
"""

import time
from typing import Sequence, TypeAlias

from absl import app
from absl import flags
from absl import logging
from aloha_sim import task_suite
from dm_control import composer
import dm_env
from dm_env import specs
import mujoco
import mujoco.viewer
import numpy as np
from rich import prompt
import tree

from safari_sdk.model import gemini_robotics_policy
from safari_sdk.model import genai_robotics

ActionSpec: TypeAlias = specs.Array
ExtraOutputSpec: TypeAlias = tree.Structure[specs.Array]
StateSpec: TypeAlias = tree.Structure[specs.Array]
State: TypeAlias = tree.Structure[np.typing.ArrayLike]
Action: TypeAlias = np.typing.ArrayLike
ExtraOutput: TypeAlias = tree.Structure[np.typing.ArrayLike]

_TASK_NAME = flags.DEFINE_enum(
    'task_name',
    'HandOverBanana',
    task_suite.TASK_FACTORIES.keys(),
    'Task name.',
)
_INSTRUCTION = flags.DEFINE_string(
    'instruction',
    None,
    'Override the default instruction.',
)

# --- Global State for Viewer Interaction ---
_GLOBAL_STATE = {
    '_IS_RUNNING': True,
    '_SHOULD_RESET': False,
    '_SINGLE_STEP': False,
    '_ASKING_INSTRUCTION': False,
}
_LOG_STEPS = 100
_DT = 0.02
_IMAGE_SIZE = (480, 848)
_ALOHA_CAMERAS = {
    'overhead_cam': _IMAGE_SIZE,
    'worms_eye_cam': _IMAGE_SIZE,
    'wrist_cam_left': _IMAGE_SIZE,
    'wrist_cam_right': _IMAGE_SIZE,
}
_ALOHA_JOINTS = {'joints_pos': 14}
_SERVE_ID = 'gemini_robotics_on_device'


def _key_callback(key: int) -> None:
  """Viewer callbacks for key-presses."""
  if key == 32:  # Space bar
    _GLOBAL_STATE['_IS_RUNNING'] = not _GLOBAL_STATE['_IS_RUNNING']
    logging.info('RUNNING = %s', _GLOBAL_STATE['_IS_RUNNING'])
  elif key == 259:  # Backspace
    _GLOBAL_STATE['_SHOULD_RESET'] = True
    logging.info('RESET = %s', _GLOBAL_STATE['_SHOULD_RESET'])
  elif key == 262:  # Right arrow
    _GLOBAL_STATE['_SINGLE_STEP'] = True
    _GLOBAL_STATE['_IS_RUNNING'] = True  # Allow one step to proceed
    logging.info('_SINGLE_STEP = %s', _GLOBAL_STATE['_SINGLE_STEP'])
  elif key == 73:  # I key
    _GLOBAL_STATE['_IS_RUNNING'] = False
    _GLOBAL_STATE['_ASKING_INSTRUCTION'] = True
  else:
    logging.info('UNKNOWN KEY PRESS = %s', key)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.info('Initializing %s environment...', _TASK_NAME.value)
  if _TASK_NAME.value not in task_suite.TASK_FACTORIES.keys():
    raise ValueError(
        f'Unknown task_name: {_TASK_NAME.value}. Available tasks:'
        f' {list(task_suite.TASK_FACTORIES.keys())}'
    )
  task_class, kwargs = task_suite.TASK_FACTORIES[_TASK_NAME.value]
  task = task_class(
      cameras=_ALOHA_CAMERAS, control_timestep=_DT, update_interval=25, **kwargs
  )
  # Override the instruction and reward function of tasks.
  if _INSTRUCTION.value:
    print(
        f'Overriding instruction to "{_INSTRUCTION.value}", this also disables'
        ' reward calculation and success detection.'
    )
    task.get_instruction = lambda: _INSTRUCTION.value
    task.get_reward = lambda _: 0.0
  env = composer.Environment(
      task=task,
      time_limit=float('inf'),  # No explicit time limit from the environment
      random_state=np.random.RandomState(0),  # For reproducibility
      recompile_mjcf_every_episode=False,
      strip_singleton_obs_buffer_dim=True,
      delayed_observation_padding=composer.ObservationPadding.INITIAL_VALUE,
  )
  env.reset()

  # Instantiate the policy.
  try:
    print('Creating policy...')
    policy = gemini_robotics_policy.GeminiRoboticsPolicy(
        serve_id=_SERVE_ID,
        task_instruction=env.task.get_instruction(),
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

  logging.info('Running policy...')

  logging.info('Launching viewer...')
  viewer_model = env.physics.model.ptr
  viewer_data = env.physics.data.ptr
  with mujoco.viewer.launch_passive(
      viewer_model, viewer_data, key_callback=_key_callback
  ) as viewer_handle:
    viewer_handle.sync()
    logging.info(
        'Viewer started. Press Space to play/pause, Backspace to reset.'
    )
    while viewer_handle.is_running():
      timestep = env.reset()
      instruction = task.get_instruction()
      viewer_handle.sync()

      steps = 0
      time_inference = 0
      time_stepping = 0
      sync_time = 0

      while not timestep.last():
        steps += 1
        if _GLOBAL_STATE['_ASKING_INSTRUCTION']:
          instruction = prompt.Prompt.ask(
              'Enter new instruction. Press enter to use current instruction',
              default=instruction,
          )
          logging.info('Using instruction: %s', instruction)
          _GLOBAL_STATE['_ASKING_INSTRUCTION'] = False
          _GLOBAL_STATE['_IS_RUNNING'] = True
        if _GLOBAL_STATE['_IS_RUNNING'] or _GLOBAL_STATE['_SINGLE_STEP']:
          frame_start_time = time.time()
          action = policy.step(timestep.observation)
          query_end_time = time.time()
          time_inference += query_end_time - frame_start_time

          current_timestep = env.step(action)
          step_end_time = time.time()
          time_stepping += step_end_time - query_end_time

          viewer_handle.sync()

          sync_time += time.time() - step_end_time

          if steps % _LOG_STEPS == 0:
            logging.info('Step: %s', steps)
            logging.info(
                'Inference time per step:\t%ss, total:\t%ss',
                time_inference / _LOG_STEPS,
                time_inference,
            )
            logging.info(
                'Stepping time per step:\t%ss, total:\t%ss',
                time_stepping / _LOG_STEPS,
                time_stepping,
            )
            logging.info(
                'Sync time per step:\t%ss, total:\t%ss',
                sync_time / _LOG_STEPS,
                sync_time,
            )
            time_inference = 0
            time_stepping = 0
            sync_time = 0

          if _GLOBAL_STATE['_SHOULD_RESET']:
            # Reset was pressed mid-episode
            _GLOBAL_STATE['_SHOULD_RESET'] = False
            current_timestep = current_timestep._replace(
                step_type=dm_env.StepType.LAST
            )

          assert (
              not current_timestep.first()
          ), 'Environment auto-reseted mid-episode unexpectedly.'
          timestep = current_timestep

          if _GLOBAL_STATE['_SINGLE_STEP']:
            _GLOBAL_STATE['_SINGLE_STEP'] = False
            _GLOBAL_STATE['_IS_RUNNING'] = False  # Pause after single step

        with viewer_handle.lock():
          # Apply perturbations if active (e.g. mouse drag)
          if viewer_handle.perturb.active:
            if _GLOBAL_STATE['_IS_RUNNING']:
              mujoco.mjv_applyPerturbForce(
                  env.physics.model.ptr,
                  env.physics.data.ptr,
                  viewer_handle.perturb,
              )
            else:
              mujoco.mjv_applyPerturbPose(
                  env.physics.model.ptr,
                  env.physics.data.ptr,
                  viewer_handle.perturb,
                  flg_paused=1,
              )
              mujoco.mj_kinematics(env.physics.model.ptr, env.physics.data.ptr)
          viewer_handle.sync()

        if not _GLOBAL_STATE['_IS_RUNNING']:
          time.sleep(0.01)  # Yield to other threads if paused

      if _GLOBAL_STATE[
          '_SHOULD_RESET'
      ]:  # Reset pressed at the very end of an episode
        _GLOBAL_STATE['_SHOULD_RESET'] = False
  logging.info('Viewer exited.')
  env.close()


if __name__ == '__main__':
  app.run(main)
