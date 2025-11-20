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

r"""Robotics Policy Eval on Aloha Robot in simulation.

This will launch a Mujoco viewer that allows the user to interact with the
policy evaluation.

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

from collections.abc import Sequence
import copy
import time

from absl import app
from absl import flags
from absl import logging
from aloha_sim import task_suite
from dm_control import composer
import dm_env
from dm_env import specs
from gdm_robotics.adapters import dm_env_to_gdmr_env_wrapper
from gdm_robotics.interfaces import policy as gdmr_policy
from gdm_robotics.interfaces import types as gdmr_types
import mujoco
import mujoco.viewer
import numpy as np
from rich import prompt
from typing_extensions import override

from safari_sdk.model import constants as gemini_robotics_constants
from safari_sdk.model import gemini_robotics_policy


_TASK_NAME = flags.DEFINE_enum(
    'task_name',
    'HandOverBanana',
    task_suite.TASK_FACTORIES.keys(),
    'Task name.',
)
_POLICY = flags.DEFINE_enum(
    'policy',
    'gemini_robotics_on_device',
    ['gemini_robotics_on_device', 'no_policy'],
    'Policy to use.',
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
_INIT_ACTION = np.asarray([
    0.0,
    -0.96,
    1.16,
    0.0,
    -0.3,
    0.0,
    1.5,
    0.0,
    -0.96,
    1.16,
    0.0,
    -0.3,
    0.0,
    1.5,
])
_SERVE_ID = 'gemini_robotics_on_device'


class NoPolicy(gdmr_policy.Policy[np.ndarray]):
  """A no-op policy that always returns the initial action."""

  def __init__(self):
    self._dummy_state = np.zeros(())

  @override
  def step(
      self,
      timestep: dm_env.TimeStep,
      prev_state: gdmr_types.StateStructure[np.ndarray],
  ) -> tuple[
      tuple[
          gdmr_types.ActionType,
          gdmr_types.ExtraOutputStructure[np.ndarray],
      ],
      gdmr_types.StateStructure[np.ndarray],
  ]:
    return (_INIT_ACTION, {}), self._dummy_state

  @override
  def initial_state(
      self,
  ) -> gdmr_types.StateStructure[np.ndarray]:
    """Returns the policy initial state."""
    return self._dummy_state

  @override
  def step_spec(self, timestep_spec: gdmr_types.TimeStepSpec) -> tuple[
      tuple[gdmr_types.ActionSpec, gdmr_types.ExtraOutputSpec],
      gdmr_types.StateSpec,
  ]:
    """Returns the spec of the ((action, extra), state) from `step` method."""
    return (
        gdmr_types.UnboundedArraySpec(shape=(14,), dtype=np.float32),
        {},
    ), specs.Array(shape=(), dtype=np.float32)


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


def _append_task_instruction(
    timestep: dm_env.TimeStep, instruction: str
) -> dm_env.TimeStep:
  """Appends the task instruction to timestep observation."""
  new_observations = timestep.observation
  new_observations.update({'instruction': np.array(instruction)})
  return timestep._replace(observation=new_observations)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 2:
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
  env = composer.Environment(
      task=task,
      time_limit=float('inf'),  # No explicit time limit from the environment
      random_state=np.random.RandomState(0),  # For reproducibility
      recompile_mjcf_every_episode=False,
      strip_singleton_obs_buffer_dim=True,
      delayed_observation_padding=composer.ObservationPadding.INITIAL_VALUE,
  )
  env.reset()
  viewer_model = env.physics.model.ptr
  viewer_data = env.physics.data.ptr
  env = dm_env_to_gdmr_env_wrapper.DmEnvToGdmrEnvWrapper(env)
  # Update the spec to include the instruction as we add it manually in our
  # runloop.
  timestep_spec = copy.deepcopy(env.timestep_spec())
  assert isinstance(timestep_spec.observation, dict)
  timestep_spec.observation.update({'instruction': specs.StringArray(shape=())})

  # Instantiate the policy.
  if _POLICY.value == 'no_policy':
    policy = NoPolicy()
  else:
    try:
      print('Creating policy...')
      policy = gemini_robotics_policy.GeminiRoboticsPolicy(
          serve_id=_SERVE_ID,
          task_instruction_key='instruction',
          image_observation_keys=_ALOHA_CAMERAS.keys(),
          proprioceptive_observation_keys=_ALOHA_JOINTS.keys(),
          min_replan_interval=25,
          inference_mode=gemini_robotics_constants.InferenceMode.SYNCHRONOUS,
          robotics_api_connection=gemini_robotics_constants.RoboticsApiConnectionType.LOCAL,
      )
      policy.step_spec(timestep_spec)  # Initialize the policy
      print('GeminiRoboticsPolicy initialized successfully.')
    except ValueError as e:
      print(f'Error initializing policy: {e}')
      raise
    except Exception as e:  # pylint: disable=broad-except
      print(f'An unexpected error occurred during initialization: {e}')
      raise

  logging.info('Running policy...')

  logging.info('Launching viewer...')

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
      policy_state = policy.initial_state()
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
          timestep = _append_task_instruction(timestep, instruction)
          (action, _), policy_state = policy.step(timestep, policy_state)
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
                  viewer_model,
                  viewer_data,
                  viewer_handle.perturb,
              )
            else:
              mujoco.mjv_applyPerturbPose(
                  viewer_model,
                  viewer_data,
                  viewer_handle.perturb,
                  flg_paused=1,
              )
              mujoco.mj_kinematics(viewer_model, viewer_data)
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
