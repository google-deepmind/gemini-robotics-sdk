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

"""MCAP episode loader with auto-detection of task metadata.

Loads episodes from MCAP files produced by EpisodicLogger and auto-detects
task_instruction, image_keys, proprio_keys, action_dim from the data.
"""

from collections.abc import Sequence
import dataclasses
import glob
import os
import re

from absl import logging
import numpy as np

from safari_sdk.logging.python import constants as log_constants
from safari_sdk.logging.python import mcap_parser_utils
from safari_sdk.protos.logging import metadata_pb2
from tensorflow.core.example import example_pb2


@dataclasses.dataclass
class DetectedMetadata:
  """Auto-detected metadata from MCAP session and episode data."""

  task_id: str | None = None
  task_instruction: str | None = None
  image_keys: list[str] | None = None
  proprio_keys: list[str] | None = None
  action_dim: int | None = None


@dataclasses.dataclass
class Episode:
  """A single evaluation episode loaded from MCAP data."""

  observations: list[dict[str, np.ndarray]]
  actions: np.ndarray  # (num_steps, action_dim)
  task_instruction: str
  num_steps: int


def _extract_label_list(
    session: metadata_pb2.Session, key: str
) -> list[str] | None:
  """Extract a list-valued label from Session proto labels."""
  for label in session.labels:
    if label.key == key and label.HasField('label_value'):
      lv = label.label_value
      if lv.HasField('list_value'):
        return [v.string_value for v in lv.list_value.values]
  return None


def _group_mcap_by_episode(mcap_paths: list[str]) -> dict[str, list[str]]:
  """Group MCAP file paths by episode UUID and sort shards.

  MCAP files are named like: episode-<uuid>-shard<N>.mcap

  Args:
    mcap_paths: List of MCAP file paths.

  Returns:
    Dict mapping episode UUID to list of shard file paths.
  """
  groups: dict[str, list[tuple[str, int]]] = {}
  for path in mcap_paths:
    basename = os.path.basename(path).split('.')[0]
    # Split on 'shard' to get the episode UUID prefix
    match = re.fullmatch(r'(.+?)shard(\d+)', basename)
    if match:
      episode_uuid = match.group(1)
      shard_num = int(match.group(2))
    else:
      episode_uuid = basename
      shard_num = 0
    groups.setdefault(episode_uuid, []).append((path, shard_num))

  # Sort by shard number and extract paths
  sorted_groups: dict[str, list[str]] = {}
  for uuid, paths_with_shards in groups.items():
    paths_with_shards.sort(key=lambda x: x[1])
    sorted_groups[uuid] = [p[0] for p in paths_with_shards]
  return sorted_groups


def _parse_observation_from_example(
    example: example_pb2.Example,
    image_keys: list[str],
    proprio_keys: list[str],
) -> tuple[dict[str, np.ndarray], str | None]:
  """Parse observation dict and instruction from a tf.train.Example."""
  features = example.features.feature
  obs = {}
  instruction = None

  # Images
  for cam_key in image_keys:
    feat_key = f'observation/{cam_key}'
    if feat_key in features and features[feat_key].bytes_list.value:
      decoded = mcap_parser_utils._maybe_decode_image(  # pylint: disable=protected-access
          list(features[feat_key].bytes_list.value), key=feat_key
      )
      if decoded is not None:
        obs[cam_key] = decoded

  # Proprioception
  for proprio_key in proprio_keys:
    feat_key = f'observation/{proprio_key}'
    if feat_key in features:
      obs[proprio_key] = np.array(
          features[feat_key].float_list.value, dtype=np.float32
      )

  # Instruction
  inst_key = 'observation/instruction'
  if inst_key in features and features[inst_key].bytes_list.value:
    instruction = features[inst_key].bytes_list.value[0].decode('utf-8')

  return obs, instruction


def _parse_action_from_example(example: example_pb2.Example) -> np.ndarray:
  """Parse flat action vector from a tf.train.Example."""
  features = example.features.feature
  if 'action' in features:
    return np.array(features['action'].float_list.value, dtype=np.float32)
  return np.array([], dtype=np.float32)


def detect_metadata_from_session(
    mcap_paths: list[str],
) -> DetectedMetadata:
  """Read /session topic from MCAP files to auto-detect metadata."""
  sessions = mcap_parser_utils.read_and_parse_mcap_messages(
      mcap_paths, log_constants.SESSION_TOPIC_NAME, metadata_pb2.Session
  )
  if not sessions:
    logging.warning('No /session messages found in MCAP files.')
    return DetectedMetadata()

  session = sessions[0]
  metadata = DetectedMetadata(
      task_id=session.task_id if session.HasField('task_id') else None,
      image_keys=_extract_label_list(session, 'image_observation_keys'),
      proprio_keys=_extract_label_list(
          session, 'proprioceptive_observation_keys'
      ),
  )
  return metadata


def load_mcap_episodes(
    dataset_path: str,
    max_episodes: int | None = None,
    image_keys: Sequence[str] | None = None,
    proprio_keys: Sequence[str] | None = None,
    task_instruction_override: str | None = None,
) -> tuple[list[Episode], DetectedMetadata]:
  """Load episodes from MCAP files and auto-detect metadata.

  Args:
    dataset_path: Path to directory of MCAP files or a single .mcap file.
    max_episodes: Maximum number of episodes to load (None = all).
    image_keys: Override for image observation keys.
    proprio_keys: Override for proprioceptive observation keys.
    task_instruction_override: Override for task instruction string.

  Returns:
    Tuple of (list of Episodes, DetectedMetadata).
  """
  if os.path.isdir(dataset_path):
    mcap_paths = glob.glob(
        os.path.join(dataset_path, '**', '*.mcap'), recursive=True
    )
    if not mcap_paths:
      raise ValueError(f'No mcap files found in directory {dataset_path}')
  elif dataset_path.endswith('.mcap'):
    mcap_paths = [dataset_path]
  else:
    raise ValueError(f'Invalid dataset_path: {dataset_path}')

  logging.info('Found %d MCAP files.', len(mcap_paths))

  # Group by episode
  episode_groups = _group_mcap_by_episode(mcap_paths)
  logging.info('Found %d episodes.', len(episode_groups))

  # Auto-detect metadata from first episode's /session
  first_group = list(episode_groups.values())[0]
  metadata = detect_metadata_from_session(first_group)

  # Apply overrides
  if image_keys is not None:
    metadata.image_keys = list(image_keys)
  if proprio_keys is not None:
    metadata.proprio_keys = list(proprio_keys)
  if task_instruction_override:
    metadata.task_instruction = task_instruction_override

  # Validate we have keys
  if not metadata.image_keys:
    raise ValueError(
        'Could not auto-detect image_observation_keys from MCAP session. '
        'Please provide --image_keys.'
    )
  if not metadata.proprio_keys:
    raise ValueError(
        'Could not auto-detect proprioceptive_observation_keys from MCAP '
        'session. Please provide --proprioception_keys.'
    )

  logging.info('Image keys: %s', metadata.image_keys)
  logging.info('Proprio keys: %s', metadata.proprio_keys)
  logging.info('Task ID: %s', metadata.task_id)

  # Load episodes
  episodes = []
  episode_uuids = list(episode_groups.keys())
  if max_episodes is not None:
    episode_uuids = episode_uuids[:max_episodes]

  for ep_idx, uuid in enumerate(episode_uuids):
    shard_paths = episode_groups[uuid]
    logging.info(
        'Loading episode %d/%d (UUID: %s, %d shards)...',
        ep_idx + 1,
        len(episode_uuids),
        uuid[:12],
        len(shard_paths),
    )

    # Read timesteps and actions
    timestep_protos = mcap_parser_utils.read_and_parse_mcap_messages(
        shard_paths, log_constants.TIMESTEP_TOPIC_NAME, example_pb2.Example
    )
    action_protos = mcap_parser_utils.read_and_parse_mcap_messages(
        shard_paths, log_constants.ACTION_TOPIC_NAME, example_pb2.Example
    )

    if not timestep_protos:
      logging.warning('Episode %d has no timesteps, skipping.', ep_idx)
      continue

    observations = []
    ep_instruction = None
    for ts_proto in timestep_protos:
      obs, inst = _parse_observation_from_example(
          ts_proto, metadata.image_keys, metadata.proprio_keys
      )
      observations.append(obs)
      if ep_instruction is None and inst:
        ep_instruction = inst

    # Auto-detect instruction from first episode
    if metadata.task_instruction is None and ep_instruction:
      metadata.task_instruction = ep_instruction
      logging.info('Auto-detected task instruction: %s', ep_instruction)

    actions_list = [_parse_action_from_example(a) for a in action_protos]
    if actions_list:
      actions = np.stack(actions_list)
      # Auto-detect action dim
      if metadata.action_dim is None:
        metadata.action_dim = actions.shape[1]
        logging.info('Auto-detected action_dim: %d', metadata.action_dim)
    else:
      actions = np.zeros((len(observations), 0), dtype=np.float32)

    # Align lengths (actions may be 1 shorter than observations)
    min_len = min(len(observations), len(actions))
    observations = observations[:min_len]
    actions = actions[:min_len]

    instruction = ep_instruction or metadata.task_instruction or ''
    episodes.append(
        Episode(
            observations=observations,
            actions=actions,
            task_instruction=instruction,
            num_steps=min_len,
        )
    )
    logging.info(
        '  Episode %d: %d steps, instruction: "%s"',
        ep_idx,
        min_len,
        instruction[:60],
    )

  if not episodes:
    raise ValueError('No valid episodes loaded from MCAP data.')

  logging.info(
      'Loaded %d episodes. Metadata: action_dim=%s, task="%s"',
      len(episodes),
      metadata.action_dim,
      (metadata.task_instruction or '')[:60],
  )
  return episodes, metadata
