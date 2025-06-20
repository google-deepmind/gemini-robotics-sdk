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

"""Logger for Episodic data."""

from collections.abc import Mapping
import copy
import io
import time
from typing import Any
from typing import NamedTuple
import uuid
import warnings

from absl import logging
import dm_env
from dm_env import specs
from dm_env import specs
import numpy as np
from PIL import Image
import tree

from google.protobuf import struct_pb2
from safari_sdk.logging.python import base_logger
from safari_sdk.logging.python import constants
from safari_sdk.protos import label_pb2
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2

STEP_TYPE_SPEC = specs.BoundedArray(
    shape=(),
    dtype=np.uint8,
    minimum=min(dm_env.StepType),
    maximum=max(dm_env.StepType),
    name="step_type",
)


class TimeStepSpec(NamedTuple):
  step_type: specs.BoundedArray
  reward: tree.Structure[specs.Array]
  discount: tree.Structure[specs.Array]
  observation: tree.Structure[specs.Array]


ActionSpec = tree.Structure[specs.Array]
ExtraOutputSpec = tree.Structure[specs.Array]
ActionType = tree.Structure[np.ndarray]

_OBSERVATION_KEY_PREFIX = "observation"
_ACTION_KEY_PREFIX = "action"
_STEP_TYPE_KEY = "step_type"
_REWARD_KEY = "reward"
_DISCOUNT_KEY = "discount"
_POLICY_EXTRA_PREFIX = "extra/policy_extra"


class McapEpisodicLogger:
  """Episodic Logger implementation, accumulating data in memory."""

  def __init__(
      self,
      agent_id: str,
      task_id: str,
      output_directory: str,
      camera_names: list[str],
      proprio_key: str,
      timestep_spec: TimeStepSpec,
      action_spec: ActionSpec,
      policy_extra_spec: ExtraOutputSpec,
      file_shard_size_limits_bytes: int = 2 * 1024 * 1024 * 1024,
      validate_data_with_spec: bool = True,
  ):
    """Initializes the episodic logger.

    Args:
      agent_id: The agent ID.
      task_id: The task ID.
      output_directory: The output directory. Note that episodes will be written
        to a subdirectory of this directory. The directory structure will be
        YYYY/MM/DD.
      camera_names: A list of camera names in observation, used to encode images
        as jpeg.
      proprio_key: The key of the proprio data in the observation.
      timestep_spec: TimeStep spec, used for validation and metadata logging.
      action_spec: Action spec, used for validation and metadata logging.
      policy_extra_spec: Policy extra spec, used for validation and metadata
        logging.
      file_shard_size_limits_bytes: The file shard size limits in bytes. Default
        is 2GB.
      validate_data_with_spec: Whether to validate all the logged data with the
        provided spec.
    """
    _validate_metadata(camera_names, proprio_key, timestep_spec, action_spec)

    self._base_logger = base_logger.BaseLogger(
        agent_id=agent_id,
        output_directory=output_directory,
        required_topics=[],
        internal_topics=[
            constants.ACTION_TOPIC_NAME,
            constants.TIMESTEP_TOPIC_NAME,
            constants.POLICY_EXTRA_TOPIC_NAME,
        ],
        file_shard_size_limit_bytes=file_shard_size_limits_bytes,
    )

    self._episode_raw_timesteps: list[dm_env.TimeStep] = []
    self._episode_raw_actions: list[tree.Structure[np.ndarray]] = []
    self._episode_raw_policy_extra: list[Mapping[str, Any]] = []

    self._timestep_publish_time_ns: list[int] = []

    self._episode_start_time_ns = 0
    self._current_episode_step = 0
    self._task_id = task_id
    self._camera_names = camera_names
    self._proprio_key = proprio_key

    self._timestep_spec = timestep_spec
    self._action_spec = action_spec
    self._policy_extra_spec = policy_extra_spec
    self._validate_data_with_spec = validate_data_with_spec

  def reset(
      self, timestep: dm_env.TimeStep, episode_uuid: str | None = None
  ) -> None:
    """Resets the logger with a starting TimeStep.

    All existing data will be flushed to the current episode.

    Args:
      timestep: The starting timestep of the episode.
      episode_uuid: The uuid of the episode. If None, a new uuid will be
        generated.
    """

    # Call write to flush previous episode.
    if self._current_episode_step > 0:
      self.write()

    self._clear_saved_data()

    self._episode_raw_timesteps.append(timestep)
    self._timestep_publish_time_ns.append(time.time_ns())
    self._current_episode_step += 1

    episode_uuid = uuid.uuid4() if episode_uuid is None else episode_uuid
    if not self._base_logger.start_session(
        start_nsec=time.time_ns(),
        task_id=self._task_id,
        output_file_prefix=f"episode_log_{episode_uuid}",
    ):
      raise ValueError(
          "Unknown error while starting session for logging episode."
      )

  def record_action_and_next_timestep(
      self,
      action: ActionType,
      next_timestep: dm_env.TimeStep,
      policy_extra: Mapping[str, Any],
  ) -> None:
    """Logs an action and the resulting timestep.

    Note that this method assumes recorded actions and timesteps have the same
    length. Please don't use it together with the reset method.

    Args:
      action: The action taken in the current step.
      next_timestep: The resulting timestep of the action.
      policy_extra: The extra output from the policy.
    """
    self._episode_raw_actions.append(action)
    self._episode_raw_timesteps.append(next_timestep)
    self._episode_raw_policy_extra.append(policy_extra)
    self._current_episode_step += 1
    self._timestep_publish_time_ns.append(time.time_ns())

  def write(self) -> None:
    """Writes the current episode logged data by converting accumulated data to protos."""
    if self._current_episode_step <= 1:
      logging.info("No episode data to write.")
      return

    logging.info("Writing episode with %d steps.", self._current_episode_step)

    # Pad the last action and policy extra with the last corresponding values so
    # as to have the same length for all repeated fields. This is because the
    # last environment transition does not have an associated action and policy
    # extra.
    padded_action = copy.deepcopy(self._episode_raw_actions[-1])
    padded_policy_extra = copy.deepcopy(self._episode_raw_policy_extra[-1])

    self._episode_raw_actions.append(padded_action)
    self._episode_raw_policy_extra.append(padded_policy_extra)

    if self._validate_data_with_spec:
      for raw_action, raw_policy_extra, raw_timestep in zip(
          self._episode_raw_actions,
          self._episode_raw_policy_extra,
          self._episode_raw_timesteps,
      ):
        self._validate_timestep(raw_timestep)
        self._validate_action(raw_action)
        self._validate_policy_extra(raw_policy_extra)

    self._base_logger.add_session_label(
        label_pb2.LabelMessage(
            key="camera_names",
            label_value=struct_pb2.Value(
                list_value=struct_pb2.ListValue(
                    values=[
                        struct_pb2.Value(string_value=camera_name)
                        for camera_name in self._camera_names
                    ]
                )
            ),
        )
    )
    self._base_logger.add_session_label(
        label_pb2.LabelMessage(
            key="proprio_key",
            label_value=struct_pb2.Value(string_value=self._proprio_key),
        )
    )
    for raw_action, policy_extra, raw_timestep, publish_time_ns in zip(
        self._episode_raw_actions,
        self._episode_raw_policy_extra,
        self._episode_raw_timesteps,
        self._timestep_publish_time_ns,
    ):
      action_example = _dict_to_example(
          self._action_to_dict(raw_action)
      )  # Convert action to proto
      timestep_example = _dict_to_example(
          self._timestep_to_dict(raw_timestep),
          camera_names=self._camera_names,
      )  # Convert timestep to proto
      policy_extra_example = _dict_to_example(
          self._policy_extra_to_dict(policy_extra)
      )  # Convert policy extra to proto

      self._base_logger.write_proto_message(
          topic=constants.ACTION_TOPIC_NAME,
          message=action_example,
          publish_time_nsec=publish_time_ns,
      )
      self._base_logger.write_proto_message(
          topic=constants.TIMESTEP_TOPIC_NAME,
          message=timestep_example,
          publish_time_nsec=publish_time_ns,
      )
      self._base_logger.write_proto_message(
          topic=constants.POLICY_EXTRA_TOPIC_NAME,
          message=policy_extra_example,
          publish_time_nsec=publish_time_ns,
      )

    episode_end_time_ns = time.time_ns()
    self._base_logger.stop_session(stop_nsec=episode_end_time_ns)

    logging.info(
        "Episode written to mcap. Episode steps: %d", self._current_episode_step
    )
    self._clear_saved_data()

  def _clear_saved_data(self) -> None:
    self._episode_raw_timesteps.clear()
    self._episode_raw_actions.clear()
    self._episode_raw_policy_extra.clear()
    self._timestep_publish_time_ns.clear()
    self._current_episode_step = 0

  def _validate_timestep(self, timestep: dm_env.TimeStep) -> None:
    """Validates a timestep."""
    observation_spec = self._timestep_spec.observation
    reward_spec = self._timestep_spec.reward
    discount_spec = self._timestep_spec.discount
    try:
      tree.map_structure(
          lambda obs, spec: spec.validate(obs),
          timestep.observation,
          observation_spec,
      )
    except ValueError as validation_error:
      logging.exception("Observation validation failed for timestep.")
      validation_error.add_note("Observation validation failed for timestep.")
      raise

    try:
      tree.map_structure(
          lambda reward, spec: spec.validate(reward),
          timestep.reward,
          reward_spec,
      )
    except ValueError as validation_error:
      logging.exception("Reward validation failed for timestep.")
      validation_error.add_note("Reward validation failed for timestep.")
      raise

    try:
      tree.map_structure(
          lambda discount, spec: spec.validate(discount),
          timestep.discount,
          discount_spec,
      )
    except ValueError as validation_error:
      logging.exception("Discount validation failed for timestep.")
      validation_error.add_note("Discount validation failed for timestep.")
      raise

  def _validate_action(self, raw_action: ActionType) -> None:
    try:
      tree.map_structure(
          lambda action, spec: spec.validate(action),
          raw_action,
          self._action_spec,
      )
    except ValueError as validation_error:
      logging.exception("Action validation failed for action.")
      validation_error.add_note("Action validation failed for timestep.")
      raise

  def _validate_policy_extra(self, raw_policy_extra: Mapping[str, Any]) -> None:
    try:
      tree.map_structure(
          lambda policy_extra, spec: spec.validate(policy_extra),
          raw_policy_extra,
          self._policy_extra_spec,
      )
    except ValueError as validation_error:
      logging.exception("Policy extra validation failed for policy extra.")
      validation_error.add_note("Policy extra validation failed for timestep.")
      raise

  def _timestep_to_dict(
      self, timestep: dm_env.TimeStep
  ) -> Mapping[str, np.ndarray]:
    """Converts a dm_env.TimeStep to a dictionary of numpy arrays."""
    obs_dict = {}
    # Prefix the observation keys with a common prefix.
    if isinstance(timestep.observation, Mapping):
      for key, value in timestep.observation.items():
        obs_dict[f"{_OBSERVATION_KEY_PREFIX}/{key}"] = value
    else:
      raise TypeError(
          f"Unsupported observation type: {type(timestep.observation)}"
      )

    timestep_dict = {
        _STEP_TYPE_KEY: np.asarray(
            # Step_type is a uint8 but we upscale to avoid being treated as
            # bytes later when converting to tf.Example.
            timestep.step_type,
            dtype=np.int32,
        ),  # Ensure scalar is an array
        **obs_dict,  # Add observations
    }

    if isinstance(timestep.reward, Mapping):
      for key, value in timestep.reward.items():
        timestep_dict[f"{_REWARD_KEY}/{key}"] = value
    else:
      reward = np.asarray(timestep.reward)
      # Reward is a float. If the `asarray` converted it to something different,
      # cast it back to a float.
      if reward.dtype != np.float64 or reward.dtype != np.float32:
        reward = reward.astype(np.float32, copy=False)
      timestep_dict[_REWARD_KEY] = reward

    if isinstance(timestep.discount, Mapping):
      for key, value in timestep.discount.items():
        timestep_dict[f"{_DISCOUNT_KEY}/{key}"] = value
    else:
      discount = np.asarray(timestep.discount)
      # Discount is a float. If the `asarray` converted it to something
      # different, cast it back to a float. Casting should be always safe.
      if discount.dtype != np.float64 or discount.dtype != np.float32:
        discount = discount.astype(np.float32, copy=False)
      timestep_dict[_DISCOUNT_KEY] = discount

    return timestep_dict

  def _action_to_dict(
      self, action: Mapping[str, np.ndarray] | np.ndarray
  ) -> Mapping[str, np.ndarray]:
    """Converts an ActionType to a dictionary of numpy arrays."""
    action_dict = {}
    if isinstance(action, Mapping):
      # If action is already a dict, prefix it.
      action_dict = {
          f"{_ACTION_KEY_PREFIX}/{key}": value for key, value in action.items()
      }
    elif isinstance(action, np.ndarray):
      action_dict["action"] = (
          action  # If it's a numpy array, put it under "action" key
      )
    else:
      raise TypeError(f"Unsupported action type: {type(action)}")
    return action_dict

  def _policy_extra_to_dict(
      self, policy_extra: Mapping[str, Any]
  ) -> Mapping[str, np.ndarray]:
    """Prefix the keys of the policy extra with a common prefix."""
    policy_extra_dict = {}
    for key, value in policy_extra.items():
      policy_extra_dict[f"{_POLICY_EXTRA_PREFIX}/{key}"] = value
    return policy_extra_dict


def _np_array_to_feature(array: np.ndarray) -> feature_pb2.Feature:
  """Converts a numpy array to a TensorFlow Feature."""
  if array.dtype == np.float32 or array.dtype == np.float64:
    return feature_pb2.Feature(
        float_list=feature_pb2.FloatList(value=array.flatten())
    )
  elif array.dtype == np.int32 or array.dtype == np.int64:
    return feature_pb2.Feature(
        int64_list=feature_pb2.Int64List(value=array.flatten())
    )
  elif array.dtype == np.uint8 or array.dtype == np.uint16:
    return feature_pb2.Feature(
        bytes_list=feature_pb2.BytesList(value=[array.tobytes()])
    )
  elif array.dtype == np.bool_:
    return feature_pb2.Feature(
        int64_list=feature_pb2.Int64List(value=array.astype(np.int64).flatten())
    )
  elif np.issubdtype(array.dtype, np.str_) or array.dtype == np.object_:
    # Handle string types. Environment spec defines string array as
    # dtype=object.
    byte_arrays = [s.encode("utf-8") for s in array.flatten()]
    return feature_pb2.Feature(
        bytes_list=feature_pb2.BytesList(value=byte_arrays)
    )
  else:
    raise ValueError(f"Unsupported numpy dtype: {array.dtype}")


def _encode_image_to_jpeg(array: np.ndarray) -> bytes:
  """Encodes a numpy array to a JPEG image."""
  img = Image.fromarray(array)
  with io.BytesIO() as output_stream:
    img.save(output_stream, format="JPEG")
    jpeg_bytes = output_stream.getvalue()
  return jpeg_bytes


def _dict_to_example(
    data_dict: Mapping[str, np.ndarray],
    camera_names: list[str] | None = None,
) -> example_pb2.Example:
  """Converts a dictionary of numpy arrays to a TensorFlow Example."""
  features = {}
  for key, value in data_dict.items():
    if (
        camera_names
        and key.startswith(_OBSERVATION_KEY_PREFIX)
        and key.split("/")[-1] in camera_names
    ):
      features[key] = feature_pb2.Feature(
          bytes_list=feature_pb2.BytesList(value=[_encode_image_to_jpeg(value)])
      )
    else:
      features[key] = _np_array_to_feature(value)
  return example_pb2.Example(features=feature_pb2.Features(feature=features))


def _validate_metadata(
    camera_names: list[str],
    proprio_key: str,
    timestep_spec: TimeStepSpec,
    action_spec: ActionSpec,
) -> None:
  """Validates that the metadata to comply with the specs we currently support."""
  _validate_observation_is_mapping(timestep_spec)
  _validate_instruction_in_timestep(timestep_spec)
  _validate_camera_names(timestep_spec, camera_names)
  _validate_proprio_key(timestep_spec, proprio_key)
  _validate_action(action_spec)


def _validate_observation_is_mapping(
    timestep_spec: TimeStepSpec,
) -> None:
  if not isinstance(timestep_spec.observation, Mapping):
    raise TypeError("Observation in timestep_spec must be a Mapping.")


def _validate_instruction_in_timestep(
    timestep_spec: TimeStepSpec,
) -> None:
  if "instruction" not in timestep_spec.observation:
    warnings.warn(
        "'instruction' is not found in timestep_spec.observation. Dummy"
        " instruction will be used."
    )


def _validate_action(action_spec: ActionSpec):
  if not isinstance(action_spec, specs.BoundedArray):
    raise TypeError("action_spec must be a BoundedArray.")


def _validate_camera_names(
    timestep_spec: TimeStepSpec, camera_names: list[str]
) -> None:
  """Validates that the camera names are listed in the observation spec."""
  if not camera_names:
    return
  for camera_name in camera_names:
    if camera_name not in timestep_spec.observation:
      raise KeyError(
          f"Camera name {camera_name} not found in observation spec."
      )


def _validate_proprio_key(
    timestep_spec: TimeStepSpec, proprio_key: str
) -> None:
  """Validates that the proprio key is listed in the observation spec."""
  if proprio_key not in timestep_spec.observation:
    raise KeyError(f"Proprio key {proprio_key} not found in observation spec.")

  if not isinstance(timestep_spec.observation[proprio_key], specs.Array):
    raise TypeError(
        f"Proprio data {proprio_key} must be a specs.Array in observation spec."
    )
