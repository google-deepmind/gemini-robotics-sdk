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

"""Utility functions for parsing mcap files containing episodic data."""

from collections.abc import Iterable, Mapping
import dataclasses
import glob
import os
from typing import Any, Type, TypeVar, cast

from absl import logging
import dm_env
from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
from mcap import reader as mcap_reader
from mcap import records
from mcap_protobuf import decoder as mcap_decoder
import numpy as np

from safari_sdk.protos.logging import metadata_pb2
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2

# The type of the proto message that can be logged in the mcap file.
_LogProtoType = TypeVar(
    "_LogProtoType", example_pb2.Example, metadata_pb2.Session
)


def get_mcap_file_paths(root_path: str) -> list[str]:
  """Returns the mcap file paths in the root path or a single mcap file."""
  if root_path.endswith(".mcap"):  # compatible with single mcap file
    logging.info("the root path is a single mcap file")
    episode_paths = [root_path]
  else:
    logging.info("the root path is a directory")
    episode_paths = glob.glob(
        os.path.join(root_path, "*", "*", "*", "*.mcap")
    )  # Adjusted path glob for mcap
  if not episode_paths:
    raise ValueError("No mcap files found in episode path")

  # Sort the file paths by episode UUID and shard number.
  def _get_episode_uuid_and_shard_number(file_path):
    file_name_without_extension = os.path.basename(file_path).split(".")[0]
    episode_uuid, shard_number = file_name_without_extension.split("shard")
    shard_number = int(shard_number)
    return episode_uuid, shard_number

  return sorted(episode_paths, key=_get_episode_uuid_and_shard_number)


def _iter_mcap_records(
    mcap_file_paths: Iterable[str], topic: str
) -> Iterable[records.Message]:
  """Yields mcap message records from files for a given topic."""
  for mcap_file_path in mcap_file_paths:
    with open(mcap_file_path, "rb") as mcap_file:
      reader = mcap_reader.make_reader(
          mcap_file, decoder_factories=[mcap_decoder.DecoderFactory()]
      )
      for _, _, message in reader.iter_messages(topics=[topic]):
        yield message


def read_raw_mcap_messages(
    mcap_root_path: str, topic: str
) -> list[records.Message]:
  """Reads mcap messages from the given root path for the given topic."""
  mcap_file_paths = get_mcap_file_paths(mcap_root_path)
  return list(_iter_mcap_records(mcap_file_paths, topic))


def read_and_parse_mcap_messages(
    mcap_file_paths: Iterable[str], topic: str, proto_type: Type[_LogProtoType]
) -> list[_LogProtoType]:
  """Reads mcap messages and parses them into the given proto type."""
  messages = []
  for message in _iter_mcap_records(mcap_file_paths, topic):
    log_proto = proto_type()
    log_proto.ParseFromString(message.data)
    messages.append(log_proto)
  return messages


@dataclasses.dataclass(frozen=True)
class McapEpisodicProtoData:
  """The result of reading data from mcap files using `read_proto_data`.

  Attributes:
    timesteps: The timesteps of the episode.
    actions: The actions of the episode.
    policy_extra: The policy extra of the episode.
  """

  timesteps: list[example_pb2.Example]
  actions: list[example_pb2.Example]
  policy_extra: list[example_pb2.Example]


def read_proto_data(
    mcap_root_path: str,
    timestep_topic_name: str,
    action_topic_name: str,
    policy_extra_topic_name: str,
) -> McapEpisodicProtoData:
  """Reads proto data from the given mcap root path."""
  mcap_files = get_mcap_file_paths(mcap_root_path)
  timesteps = read_and_parse_mcap_messages(
      mcap_files, timestep_topic_name, example_pb2.Example
  )
  actions = read_and_parse_mcap_messages(
      mcap_files, action_topic_name, example_pb2.Example
  )
  policy_extra = read_and_parse_mcap_messages(
      mcap_files, policy_extra_topic_name, example_pb2.Example
  )
  return McapEpisodicProtoData(
      timesteps=timesteps, actions=actions, policy_extra=policy_extra
  )


def read_session_proto_data(
    mcap_root_path: str, session_topic_name: str
) -> list[metadata_pb2.Session]:
  """Reads session proto data from the given mcap root path."""
  mcap_files = get_mcap_file_paths(mcap_root_path)
  session_messages = read_and_parse_mcap_messages(
      mcap_files, session_topic_name, metadata_pb2.Session
  )

  if not session_messages:
    raise ValueError("No session messages found in mcap files.")

  return session_messages


def parse_examples_to_dm_env_types(
    timestep_spec: gdmr_types.TimeStepSpec,
    action_spec: gdmr_types.ActionSpec,
    policy_extra_spec: gdmr_types.ExtraOutputSpec,
    timesteps_example: list[example_pb2.Example],
    actions_example: list[example_pb2.Example],
    policy_extra_example: list[example_pb2.Example],
    step_type_key: str,
    observation_key_prefix: str,
    reward_key: str,
    discount_key: str,
    action_key_prefix: str,
    policy_extra_key_prefix: str,
) -> tuple[
    list[dm_env.TimeStep],
    list[gdmr_types.ActionType],
    list[Mapping[str, Any]],
]:
  """Parses examples to dm env types."""
  timesteps = [
      _parse_timestep_from_example(
          timestep_example,
          timestep_spec,
          step_type_key,
          observation_key_prefix,
          reward_key,
          discount_key,
      )
      for timestep_example in timesteps_example
  ]
  actions = [
      _parse_action_from_example(action_example, action_spec, action_key_prefix)
      for action_example in actions_example
  ]
  policy_extra = [
      _parse_policy_extra_from_example(
          policy_extra_example, policy_extra_spec, policy_extra_key_prefix
      )
      for policy_extra_example in policy_extra_example
  ]
  return timesteps, actions, policy_extra


def _parse_timestep_from_example(
    timestep_example: example_pb2.Example,
    timestep_spec: gdmr_types.TimeStepSpec,
    step_type_key: str,
    observation_key_prefix: str,
    reward_key: str,
    discount_key: str,
) -> dm_env.TimeStep:
  """Parses a timestep from an example."""

  # Step type is a scalar. Squeeze its singleton dimension.
  step_type = np.asarray(
      timestep_example.features.feature[step_type_key].int64_list.value,
      dtype=np.uint8,
  ).squeeze()

  candidate_rewards = {}
  candidate_discounts = {}
  candidate_observations = {}

  # Split the features into the observation, reward, and discount keys.
  for key, value in timestep_example.features.feature.items():
    python_value = _python_value_from_example_feature(value)

    if key.startswith(observation_key_prefix):
      candidate_observations[key] = python_value
    elif key.startswith(reward_key):
      candidate_rewards[key] = python_value
    elif key.startswith(discount_key):
      candidate_discounts[key] = python_value

  observations = _parse_and_match_spec(
      timestep_spec.observation,
      observation_key_prefix,
      candidate_observations,
  )
  reward = _parse_and_match_spec(
      timestep_spec.reward, reward_key, candidate_rewards
  )
  discount = _parse_and_match_spec(
      timestep_spec.discount,
      discount_key,
      candidate_discounts,
  )

  return dm_env.TimeStep(
      step_type=step_type,
      reward=reward,
      discount=discount,
      observation=observations,
  )


def _parse_action_from_example(
    action_example: example_pb2.Example,
    action_spec: gdmr_types.ActionSpec,
    action_key_prefix: str,
) -> gdmr_types.ActionType:
  """Parses an action from an example."""
  actions = {}
  # Filter for only the action keys. We expect the example to contain only
  # action keys but we check anyway.
  for key, value in action_example.features.feature.items():
    python_value = _python_value_from_example_feature(value)

    if key.startswith(action_key_prefix):
      actions[key] = python_value

  if len(actions) != len(action_example.features.feature):
    raise ValueError(
        "Action example has more features than expected. "
        f"Expected {len(action_example.features.feature)} but got"
        f" {len(actions)}."
    )

  return _parse_and_match_spec(action_spec, action_key_prefix, actions)


def _parse_policy_extra_from_example(
    policy_extra_example: example_pb2.Example,
    policy_extra_spec: gdmr_types.ExtraOutputSpec,
    policy_extra_key_prefix: str,
) -> Mapping[str, Any]:
  """Parses policy extra from an example."""
  policy_extra = {}
  # Filter for only the policy extra keys. We expect the example to contain
  # only policy extra keys but we check anyway.
  for key, value in policy_extra_example.features.feature.items():
    if key.startswith(policy_extra_key_prefix):
      policy_extra[key] = _python_value_from_example_feature(value)

  if len(policy_extra) != len(policy_extra_example.features.feature):
    raise ValueError(
        "Policy extra example has more features than expected. "
        f"Expected {len(policy_extra_example.features.feature)} but got"
        f" {len(policy_extra)}."
    )

  return _parse_and_match_spec(
      policy_extra_spec, policy_extra_key_prefix, policy_extra
  )


def _parse_and_match_spec(
    values_spec: specs.Array | Mapping[str, specs.Array],
    prefix: str,
    values: Mapping[str, Any],
) -> ...:
  """Parses and matches a value to a spec."""
  if len(values) == 1 and list(values.keys())[0] == prefix:
    spec = cast(specs.Array, values_spec)
    # Handle strings.
    value = values[prefix]
    if isinstance(spec, specs.StringArray):
      # We expect a single string inside the value but the parser returns a
      # list.
      value = value[0]
      if not isinstance(value, bytes):
        raise ValueError(f"Expected bytes but got {type(value)}")
      value = cast(bytes, value).decode("utf-8")

    parsed_value = np.asarray(value).astype(spec.dtype)
    if not spec.shape:
      parsed_value = np.squeeze(parsed_value)
    return parsed_value

  # Dictionary of values.
  parsed_values = {}
  for key, value in values.items():
    stripped_key = key.removeprefix(f"{prefix}/")
    spec = cast(specs.Array, values_spec[stripped_key])

    if isinstance(spec, specs.StringArray):
      # We expect a single string inside the value but the parser returns a
      # list.
      value = value[0]
      if not isinstance(value, bytes):
        raise ValueError(f"Expected bytes but got {type(value)}")
      value = cast(bytes, value).decode("utf-8")

    parsed_values[stripped_key] = np.asarray(value).astype(spec.dtype)
    if not spec.shape:
      parsed_values[stripped_key] = np.squeeze(parsed_values[stripped_key])
  return parsed_values


def _python_value_from_example_feature(
    feature: feature_pb2.Feature,
) -> Any:
  """Parses a python value from an example feature."""
  if feature.WhichOneof("kind") == "float_list":
    return feature.float_list.value
  elif feature.WhichOneof("kind") == "int64_list":
    return feature.int64_list.value
  elif feature.WhichOneof("kind") == "bytes_list":
    return feature.bytes_list.value
  else:
    raise ValueError(f"Unsupported feature type: {feature.WhichOneof('kind')}")
