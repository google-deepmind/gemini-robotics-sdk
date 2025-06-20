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

"""Logger for LeRobot data."""

import concurrent.futures

from absl import logging
import dm_env
from dm_env import specs
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np

from safari_sdk.logging.python import mcap_episodic_logger

# LeRobot step keys.
_ACTION_KEY = "action"
_FRAME_INDEX_KEY = "frame_index"
_NEXT_DONE_KEY = "next.done"
_OBSERVATION_KEY_PREFIX = "observation."
# _TASK_KEY maps to _INSTRUCTION_KEY and eventually will be used to populate the
# instruction label in the SSOT Session.
_TASK_KEY = "task"

# LeRobot feature keys.
_DTYPE_VIDEO = "video"
_DTYPE_FLOAT32 = "float32"
_DTYPE_INT64 = "int64"
_DTYPE_BOOL = "bool"
_DTYPE_KEY = "dtype"

# MCAP Logger spec keys.
_REWARD_KEY = "reward"
_DISCOUNT_KEY = "discount"
_STEP_TYPE_KEY = "step_type"
_SHAPE_KEY = "shape"
_INSTRUCTION_KEY = "instruction"

# Others.
_AGENT_ID_PREFIX = "robot_episode_"


class LeRobotEpisodicLogger:
  """An episodic logger that writes LeRobot episodes to MCAP files."""

  def __init__(
      self,
      task_id: str,
      output_directory: str,
      camera_names: list[str] | None = None,
      proprio_key: str | None = None,
      features: dict | None = None,
  ):
    """Initializes the logger.

    Args:
      task_id: The task ID.
      output_directory: The output directory for MCAP files.
      camera_names: A list of camera keys for image encoding.
      proprio_key: The key for the proprioceptive data.
      features: A dictionary of dataset features, used to generate specs for
        validation.
    """
    self._task_id = task_id
    self._output_directory = output_directory
    self._camera_names = camera_names or []
    self._proprio_key = proprio_key
    self._timestep_spec = None
    self._action_spec = None

    if features:
      self._timestep_spec, self._action_spec = self._parse_features_to_specs(
          features
      )

    self._mcap_episodic_logger: (
        mcap_episodic_logger.McapEpisodicLogger | None
    ) = None
    self._current_episode_id: int = -1
    self._previous_action: np.ndarray | None = None

  def _parse_features_to_specs(
      self,
      features: dict,
  ) -> tuple[mcap_episodic_logger.TimeStepSpec, specs.BoundedArray]:
    """Converts dataset features to dm_env specs."""
    action_spec = None
    observation_spec = {}

    # Mapping from LeRobot dtype to numpy dtype.
    def _dtype_map(dtype: str) -> str:
      if dtype == _DTYPE_VIDEO:
        return "uint8"
      else:
        return dtype

    for key, feature_info in features.items():
      dtype = _dtype_map(feature_info[_DTYPE_KEY])

      if key == _ACTION_KEY:
        shape = tuple(feature_info[_SHAPE_KEY])
        action_spec = specs.BoundedArray(
            shape=shape,
            dtype=dtype,
            minimum=-2.0,
            maximum=3.0,
            name=key,
        )
      elif key.startswith(_OBSERVATION_KEY_PREFIX):
        obs_key = key.replace(_OBSERVATION_KEY_PREFIX, "", 1)
        observation_spec[obs_key] = specs.Array(
            shape=tuple(feature_info[_SHAPE_KEY]),
            dtype=dtype,
            name=obs_key,
        )

    if action_spec is None:
      raise ValueError("Action spec not found in features.")
    if not observation_spec:
      raise ValueError("Observation spec not found in features.")
    observation_spec[_INSTRUCTION_KEY] = specs.Array(
        shape=(), dtype=object, name=_INSTRUCTION_KEY
    )
    # Create timestep spec.
    timestep_spec = mcap_episodic_logger.TimeStepSpec(
        observation=observation_spec,
        reward=specs.Array(shape=(), dtype=np.float32, name=_REWARD_KEY),
        discount=specs.Array(shape=(), dtype=np.float32, name=_DISCOUNT_KEY),
        step_type=specs.BoundedArray(
            shape=(),
            dtype=int,
            minimum=min(dm_env.StepType),
            maximum=max(dm_env.StepType),
            name=_STEP_TYPE_KEY,
        ),
    )

    return timestep_spec, action_spec

  def start_episode(self, episode_id: int) -> None:
    """Starts a new episode session."""
    if self._mcap_episodic_logger is not None:
      raise ValueError(
          "Cannot start a new episode, the previous one has not been finished."
      )

    self._current_episode_id = episode_id
    agent_id = f"{_AGENT_ID_PREFIX}{episode_id}"
    logging.info("Starting episode %s with agent id %s", episode_id, agent_id)
    self._mcap_episodic_logger = mcap_episodic_logger.McapEpisodicLogger(
        agent_id=agent_id,
        task_id=self._task_id,
        output_directory=self._output_directory,
        proprio_key=self._proprio_key,
        camera_names=self._camera_names,
        timestep_spec=self._timestep_spec,
        action_spec=self._action_spec,
        policy_extra_spec={},
        validate_data_with_spec=True,
    )
    self._previous_action = None

  def finish_episode(self) -> None:
    """Finishes the current episode and writes the data to a file."""
    if self._mcap_episodic_logger is None:
      return

    self._mcap_episodic_logger.write()
    self._mcap_episodic_logger = None

  def record_step(self, step_data: dict[str, np.ndarray]) -> None:
    """Records a single step."""
    assert (
        self._mcap_episodic_logger is not None
    ), "Cannot record step, episode not started. Call start_episode() first."

    observation = {}
    for k, v in step_data.items():
      if k.startswith(_OBSERVATION_KEY_PREFIX):
        obs_key = k.replace(_OBSERVATION_KEY_PREFIX, "", 1)
        # Transpose image data from (C, H, W) (PyTorch) to (H, W, C).
        if obs_key in self._camera_names and v.ndim == 3:
          v = np.transpose(v, (1, 2, 0))
          v = (v * 255).astype(np.uint8)
        observation[obs_key] = v
    observation[_INSTRUCTION_KEY] = np.array(step_data[_TASK_KEY], dtype=object)

    action = step_data[_ACTION_KEY]
    frame_index = int(step_data[_FRAME_INDEX_KEY])

    if frame_index == 0:
      step_type = dm_env.StepType.FIRST
    elif bool(step_data[_NEXT_DONE_KEY]):
      step_type = dm_env.StepType.LAST
    else:
      step_type = dm_env.StepType.MID

    timestep = dm_env.TimeStep(
        step_type=step_type,
        reward=np.float32(0.0),
        discount=np.float32(1.0),
        observation=observation,
    )
    if step_type == dm_env.StepType.FIRST:
      self._mcap_episodic_logger.reset(
          timestep, episode_uuid=str(self._current_episode_id)
      )
    else:
      self._mcap_episodic_logger.record_action_and_next_timestep(
          action=self._previous_action,
          next_timestep=timestep,
          policy_extra={},
      )
    self._previous_action = action


def convert_lerobot_data_to_mcap(
    *,
    dataset: LeRobotDataset,
    task_id: str,
    output_directory: str,
    proprio_key: str,
    episodes_limit: int,
    max_workers: int,
) -> None:
  """Converts LeRobot data to MCAP files, processing episodes in parallel."""
  episode_indices = dataset.episode_data_index
  num_episodes = len(episode_indices["from"])
  if episodes_limit <= 0:
    num_episodes_to_process = num_episodes
  else:
    num_episodes_to_process = min(episodes_limit, num_episodes)

  if max_workers <= 0:
    raise ValueError("max_workers must be greater than 0.")

  max_workers = min(max_workers, num_episodes_to_process)
  logging.info(
      "Will process the first %d episodes with %d workers.",
      num_episodes_to_process,
      max_workers,
  )

  camera_names = [
      key.replace(f"{_OBSERVATION_KEY_PREFIX}", "", 1)
      for key in dataset.meta.camera_keys
  ]

  def _process_episode(episode_id: int):
    """Processes a single episode."""
    thread_logger = LeRobotEpisodicLogger(
        task_id=task_id,
        output_directory=output_directory,
        camera_names=camera_names,
        proprio_key=proprio_key,
        features=dataset.features,
    )
    start_index = episode_indices["from"][episode_id]
    end_index = episode_indices["to"][episode_id]
    logging.info(
        "Processing episode %d from index %d to %d",
        episode_id,
        start_index,
        end_index,
    )

    thread_logger.start_episode(episode_id=episode_id)

    for step_index in range(start_index, end_index):
      step = dataset[step_index]
      step_np = {k: np.array(v) for k, v in step.items()}
      thread_logger.record_step(step_np)

    thread_logger.finish_episode()

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=max_workers
  ) as executor:
    futures = [
        executor.submit(_process_episode, i)
        for i in range(num_episodes_to_process)
    ]
    for future in concurrent.futures.as_completed(futures):
      try:
        future.result()
      except Exception as e:
        logging.exception("Error processing episode: %s", e)
