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

from collections.abc import Callable, Mapping
import copy
import io
import time
from typing import Any
import uuid

from absl import logging
import dm_env
from dm_env import specs
from gdm_robotics.interfaces import episodic_logger
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np
from PIL import Image
import tree

from google.protobuf import struct_pb2
from safari_sdk.logging.python import constants
from safari_sdk.logging.python import file_handler as file_handler_lib
from safari_sdk.logging.python import mcap_message_writer as mcap_message_writer_lib
from safari_sdk.logging.python import metadata_utils
from safari_sdk.logging.python import session_manager as session_manager_lib
from safari_sdk.protos import label_pb2
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2


class EpisodicLogger(episodic_logger.EpisodicLogger):
  """Episodic Logger implementation, accumulating data in memory."""

  @classmethod
  def create(
      cls,
      agent_id: str,
      task_id: str,
      output_directory: str,
      image_observation_keys: list[str],
      proprioceptive_observation_keys: list[str],
      timestep_spec: gdmr_types.TimeStepSpec,
      action_spec: gdmr_types.ActionSpec,
      policy_extra_spec: gdmr_types.ExtraOutputSpec,
      file_shard_size_limit_bytes: int = 2 * 1024 * 1024 * 1024,
      validate_data_with_spec: bool = True,
      timestamp_key: str | None = None,
      dynamic_metadata_provider: Callable[[], Mapping[str, str]] | None = None,
  ) -> "EpisodicLogger":
    """Creates a EpisodicLogger with its dependencies.

    Args:
      agent_id: The ID of the agent.
      task_id: The ID of the task.
      output_directory: The output directory. Note that episodes will be written
        to a subdirectory of this directory. The directory structure will be
        YYYY/MM/DD.
      image_observation_keys: A container of camera names in observation, used
        to encode images as jpeg (without th the /observation prefix).
      proprioceptive_observation_keys: A container of keys for the
        proprioceptive data in the observations.
      timestep_spec: The timestep spec.
      action_spec: The action spec.
      policy_extra_spec: The policy extra spec.
      file_shard_size_limit_bytes: The file shard size limits in bytes. Default
        is 2GB.
      validate_data_with_spec: Whether to validate the data with the spec.
      timestamp_key: The observation key that maps to the the timestamps of each
        step in an episode. This is used to set the publish time of each MCAP
        message. This is optional, if not provided, the logger will generate
        timestamps using time.time_ns().
      dynamic_metadata_provider: A function that provides dynamic metadata to be
        logged as session labels. The function is called when `write` is called.

    Returns:
        A EpisodicLogger instance.
    """
    if not agent_id:
      raise ValueError("agent_id must be provided as a non-empty string.")

    topics = {
        constants.ACTION_TOPIC_NAME,
        constants.TIMESTEP_TOPIC_NAME,
        constants.POLICY_EXTRA_TOPIC_NAME,
    }
    required_topics = set()

    file_handler = file_handler_lib.FileHandler(
        agent_id=agent_id,
        topics=topics,
        output_directory=output_directory,
        file_shard_size_limit_bytes=file_shard_size_limit_bytes,
    )
    mcap_message_writer = mcap_message_writer_lib.McapMessageWriter(
        file_handler=file_handler,
    )

    # Validate the specs are of the correct type before we pass to the session
    # manager and the logger.
    _validate_metadata(
        image_observation_keys,
        proprioceptive_observation_keys,
        timestep_spec,
        action_spec,
    )

    session_manager = session_manager_lib.SessionManager(
        topics=topics,
        required_topics=required_topics,
        policy_environment_metadata_params=metadata_utils.PolicyEnvironmentMetadataParams(
            jpeg_compression_keys=image_observation_keys,
            observation_spec=timestep_spec.observation,
            reward_spec=timestep_spec.reward,
            discount_spec=timestep_spec.discount,
            action_spec=action_spec,
            policy_extra_spec=policy_extra_spec,
        ),
    )
    return cls(
        task_id=task_id,
        image_observation_keys=image_observation_keys,
        proprioceptive_observation_keys=proprioceptive_observation_keys,
        timestep_spec=timestep_spec,
        action_spec=action_spec,
        policy_extra_spec=policy_extra_spec,
        mcap_message_writer=mcap_message_writer,
        session_manager=session_manager,
        validate_data_with_spec=validate_data_with_spec,
        timestamp_key=timestamp_key,
        dynamic_metadata_provider=dynamic_metadata_provider,
    )

  def __init__(
      self,
      task_id: str,
      image_observation_keys: list[str],
      proprioceptive_observation_keys: list[str],
      timestep_spec: gdmr_types.TimeStepSpec,
      action_spec: gdmr_types.ActionSpec,
      policy_extra_spec: gdmr_types.ExtraOutputSpec,
      mcap_message_writer: mcap_message_writer_lib.McapMessageWriter,
      session_manager: session_manager_lib.SessionManager,
      validate_data_with_spec: bool = True,
      timestamp_key: str | None = None,
      dynamic_metadata_provider: Callable[[], Mapping[str, str]] | None = None,
  ):
    """Initializes the episodic logger.

    Args:
      task_id: The task ID.
      image_observation_keys: A container of camera names in observation, used
        to encode images as jpeg.
      proprioceptive_observation_keys: A container of keys of the proprio data
        in the observations.
      timestep_spec: TimeStep spec, used for validation and metadata logging.
      action_spec: Action spec, used for validation and metadata logging.
      policy_extra_spec: Policy extra spec, used for validation and metadata
        logging.
      mcap_message_writer: The message writer for writing logs to mcap files.
      session_manager: The session manager for managing session metadata.
      validate_data_with_spec: Whether to validate all the logged data with the
        provided spec.
      timestamp_key: The observation key that maps to the the timestamps of each
        step in an episode. This is used to set the publish time of each MCAP
        message. This is optional, if not provided, the logger will generate
        timestamps using time.time_ns().
      dynamic_metadata_provider: A function that provides dynamic metadata to be
        logged as session labels. The function is called when `write` is called.
    """
    if not task_id:
      raise ValueError("task_id must be provided as a non-empty string.")

    self._mcap_message_writer = mcap_message_writer
    self._session_manager = session_manager

    self._episode_raw_timesteps: list[dm_env.TimeStep] = []
    self._episode_raw_actions: list[tree.Structure[np.ndarray]] = []
    self._episode_raw_policy_extra: list[Mapping[str, Any]] = []

    self._timestep_publish_time_ns: list[int] = []

    self._episode_start_time_ns = 0
    self._current_episode_step = 0
    self._task_id = task_id
    self._image_observation_keys = image_observation_keys
    self._proprioceptive_observation_keys = proprioceptive_observation_keys

    self._timestep_spec = timestep_spec
    self._action_spec = action_spec
    self._policy_extra_spec = policy_extra_spec
    self._validate_data_with_spec = validate_data_with_spec
    self._timestamp_key = timestamp_key
    self._dynamic_metadata_provider = dynamic_metadata_provider

    # Whether the logger is currently recording data.
    # We mark as True when reset is called and False after writing an episode
    # has been completed.
    self._is_recording = False

  def reset(
      self, timestep: dm_env.TimeStep, episode_uuid: str | None = None
  ) -> None:
    """Resets the logger with a starting TimeStep.

    All existing data will be flushed to the current episode.

    In this method, we mark the logger as recording (i.e. set _is_recording to
    True).

    Args:
      timestep: The starting timestep of the episode.
      episode_uuid: The uuid of the episode. If None, a new uuid will be
        generated.
    """

    # Call write to flush previous episode.
    if self._current_episode_step > 0:
      self.write()

    self._clear_saved_data()

    if self._timestamp_key:
      timestamp_ns = timestep.observation[self._timestamp_key]
    else:
      timestamp_ns = time.time_ns()

    self._episode_raw_timesteps.append(timestep)
    self._timestep_publish_time_ns.append(timestamp_ns)
    self._current_episode_step += 1

    episode_uuid = uuid.uuid4() if episode_uuid is None else episode_uuid

    output_file_prefix = f"episode_log_{episode_uuid}"

    # Try to start a new session.
    try:
      self._session_manager.start_session(
          start_timestamp_nsec=timestamp_ns, task_id=self._task_id
      )
    except ValueError:
      logging.exception("Failed to start session for logging episode.")
      raise

    # Reset the file handler and start the worker thread for writing logs.
    self._mcap_message_writer.reset_file_handler(
        output_file_prefix=output_file_prefix,
        start_timestamp_nsec=timestamp_ns,
    )
    self._mcap_message_writer.start()

    # Mark the logger as recording.
    self._is_recording = True

  def record_action_and_next_timestep(
      self,
      action: gdmr_types.ActionType,
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
    if self._timestamp_key:
      timestamp_ns = next_timestep.observation[self._timestamp_key]
    else:
      timestamp_ns = time.time_ns()

    self._episode_raw_actions.append(action)
    self._episode_raw_timesteps.append(next_timestep)
    self._episode_raw_policy_extra.append(policy_extra)
    self._current_episode_step += 1
    self._timestep_publish_time_ns.append(timestamp_ns)

  def write(self) -> None:
    """Writes the current episode logged data by converting accumulated data to protos."""
    # If the logger is not recording data, we should not write.
    # This protects against cases whwere we try to call write() without calling
    # reset() first.
    if not self._is_recording:
      logging.info("Logger is not recording data. Skipping write.")
      return

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
          image_observation_keys=self._image_observation_keys,
      )  # Convert timestep to proto
      policy_extra_example = _dict_to_example(
          self._policy_extra_to_dict(policy_extra)
      )  # Convert policy extra to proto

      self._mcap_message_writer.write_proto_message(
          topic=constants.ACTION_TOPIC_NAME,
          message=action_example,
          publish_time_nsec=publish_time_ns,
          log_time_nsec=publish_time_ns,
      )
      self._mcap_message_writer.write_proto_message(
          topic=constants.TIMESTEP_TOPIC_NAME,
          message=timestep_example,
          publish_time_nsec=publish_time_ns,
          log_time_nsec=publish_time_ns,
      )
      self._mcap_message_writer.write_proto_message(
          topic=constants.POLICY_EXTRA_TOPIC_NAME,
          message=policy_extra_example,
          publish_time_nsec=publish_time_ns,
          log_time_nsec=publish_time_ns,
      )

    self._write_session()

    # Mark the logger as not recording once the episode has been written.
    self._is_recording = False

    logging.info(
        "Episode written to mcap. Episode steps: %d", self._current_episode_step
    )
    self._clear_saved_data()

  def _write_session(self) -> None:
    """Writes the Session message to an mcap file.

    Also logs the camera names and proprio key as session labels.
    """
    self._session_manager.add_session_label(
        label_pb2.LabelMessage(
            key="image_observation_keys",
            label_value=struct_pb2.Value(
                list_value=struct_pb2.ListValue(
                    values=[
                        struct_pb2.Value(string_value=camera_name)
                        for camera_name in self._image_observation_keys
                    ]
                )
            ),
        )
    )
    self._session_manager.add_session_label(
        label_pb2.LabelMessage(
            key="proprioceptive_observation_keys",
            label_value=struct_pb2.Value(
                list_value=struct_pb2.ListValue(
                    values=[
                        struct_pb2.Value(string_value=proprio_key)
                        for proprio_key in self._proprioceptive_observation_keys
                    ]
                )
            ),
        )
    )

    # TODO: Use standard metadata logging method.
    if self._dynamic_metadata_provider is not None:
      additional_metadata = self._dynamic_metadata_provider()
      for key, value in additional_metadata.items():
        self._session_manager.add_session_label(
            label_pb2.LabelMessage(
                key=key,
                label_value=struct_pb2.Value(string_value=value),
            )
        )

    episode_end_time_ns = self._timestep_publish_time_ns[-1]
    session = self._session_manager.stop_session(
        stop_timestamp_nsec=episode_end_time_ns
    )
    # Write the Session proto to the log file.
    self._mcap_message_writer.write_proto_message(
        topic=constants.SESSION_TOPIC_NAME,
        message=session,
        publish_time_nsec=episode_end_time_ns,
        log_time_nsec=episode_end_time_ns,
    )
    # Flush the queue and stop the worker thread.
    self._mcap_message_writer.stop(stop_timestamp_nsec=episode_end_time_ns)

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
    except ValueError:
      logging.exception("Observation validation failed for timestep.")
      raise

    try:
      tree.map_structure(
          lambda reward, spec: spec.validate(reward),
          timestep.reward,
          reward_spec,
      )
    except ValueError:
      logging.exception("Reward validation failed for timestep.")
      raise

    try:
      tree.map_structure(
          lambda discount, spec: spec.validate(discount),
          timestep.discount,
          discount_spec,
      )
    except ValueError:
      logging.exception("Discount validation failed for timestep.")
      raise

  def _validate_action(self, raw_action: gdmr_types.ActionType) -> None:
    try:
      tree.map_structure(
          lambda action, spec: spec.validate(action),
          raw_action,
          self._action_spec,
      )
    except ValueError:
      logging.exception("Action validation failed for action.")
      raise

  def _validate_policy_extra(self, raw_policy_extra: Mapping[str, Any]) -> None:
    try:
      tree.map_structure(
          lambda policy_extra, spec: spec.validate(policy_extra),
          raw_policy_extra,
          self._policy_extra_spec,
      )
    except ValueError:
      logging.exception("Policy extra validation failed for policy extra.")
      raise

  def _timestep_to_dict(
      self, timestep: dm_env.TimeStep
  ) -> Mapping[str, np.ndarray]:
    """Converts a dm_env.TimeStep to a dictionary of numpy arrays."""
    obs_dict = {}
    # Prefix the observation keys with a common prefix.
    if isinstance(timestep.observation, Mapping):
      for key, value in timestep.observation.items():
        obs_dict[constants.OBSERVATION_KEY_TEMPLATE.format(key)] = value
    else:
      raise TypeError(
          f"Unsupported observation type: {type(timestep.observation)}"
      )

    timestep_dict = {
        constants.STEP_TYPE_KEY: np.asarray(
            # Step_type is a uint8 but we upscale to avoid being treated as
            # bytes later when converting to tf.Example.
            timestep.step_type,
            dtype=np.int32,
        ),  # Ensure scalar is an array
        **obs_dict,  # Add observations
    }

    if isinstance(timestep.reward, Mapping):
      for key, value in timestep.reward.items():
        timestep_dict[constants.REWARD_KEY_TEMPLATE.format(key)] = value
    else:
      reward = np.asarray(timestep.reward)
      # Reward is a float. If the `asarray` converted it to something different,
      # cast it back to a float.
      if reward.dtype != np.float64 or reward.dtype != np.float32:
        reward = reward.astype(np.float32, copy=False)
      timestep_dict[constants.REWARD_KEY] = reward

    if isinstance(timestep.discount, Mapping):
      for key, value in timestep.discount.items():
        timestep_dict[constants.DISCOUNT_KEY_TEMPLATE.format(key)] = value
    else:
      discount = np.asarray(timestep.discount)
      # Discount is a float. If the `asarray` converted it to something
      # different, cast it back to a float. Casting should be always safe.
      if discount.dtype != np.float64 or discount.dtype != np.float32:
        discount = discount.astype(np.float32, copy=False)
      timestep_dict[constants.DISCOUNT_KEY] = discount

    return timestep_dict

  def _action_to_dict(
      self, action: Mapping[str, np.ndarray] | np.ndarray
  ) -> Mapping[str, np.ndarray]:
    """Converts an ActionType to a dictionary of numpy arrays."""

    if not isinstance(action, np.ndarray):
      raise TypeError(
          f"Unsupported action type: {type(action)}. Only np.ndarray is"
          " supported."
      )

    return {constants.ACTION_KEY_PREFIX: action}

  def _policy_extra_to_dict(
      self, policy_extra: Mapping[str, Any]
  ) -> Mapping[str, np.ndarray]:
    """Prefix the keys of the policy extra with a common prefix."""
    policy_extra_dict = {}
    for key, value in policy_extra.items():
      policy_extra_dict[constants.POLICY_EXTRA_KEY_TEMPLATE.format(key)] = value
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
    image_observation_keys: list[str] | None = None,
) -> example_pb2.Example:
  """Converts a dictionary of numpy arrays to a TensorFlow Example."""
  features = {}
  for key, value in data_dict.items():
    if (
        image_observation_keys
        and key.startswith(constants.OBSERVATION_KEY_PREFIX)
        and key.split("/")[-1] in image_observation_keys
    ):
      features[key] = feature_pb2.Feature(
          bytes_list=feature_pb2.BytesList(value=[_encode_image_to_jpeg(value)])
      )
    else:
      features[key] = _np_array_to_feature(value)
  return example_pb2.Example(features=feature_pb2.Features(feature=features))


def _validate_metadata(
    image_observation_keys: list[str],
    proprioceptive_observation_keys: list[str],
    timestep_spec: gdmr_types.TimeStepSpec,
    action_spec: gdmr_types.ActionSpec,
) -> None:
  """Validates that the metadata to comply with the specs we currently support."""
  _validate_observation_is_mapping(timestep_spec)
  _validate_instruction_in_timestep(timestep_spec)
  _validate_image_observation_keys(timestep_spec, image_observation_keys)
  _validate_proprioceptive_observation_keys(
      timestep_spec, proprioceptive_observation_keys
  )
  _validate_action(action_spec)


def _validate_observation_is_mapping(
    timestep_spec: gdmr_types.TimeStepSpec,
) -> None:
  if not isinstance(timestep_spec.observation, Mapping):
    raise TypeError("Observation in timestep_spec must be a Mapping.")


def _validate_instruction_in_timestep(
    timestep_spec: gdmr_types.TimeStepSpec,
) -> None:
  if "instruction" not in timestep_spec.observation:
    raise KeyError(
        "'instruction' is required in timestep_spec.observation."
    )


def _validate_action(action_spec: gdmr_types.ActionSpec):
  if not isinstance(action_spec, specs.BoundedArray):
    raise TypeError("action_spec must be a BoundedArray.")


def _validate_image_observation_keys(
    timestep_spec: gdmr_types.TimeStepSpec, image_observation_keys: list[str]
) -> None:
  """Validates that the camera names are listed in the observation spec."""
  if not image_observation_keys:
    return
  for image_key in image_observation_keys:
    if image_key not in timestep_spec.observation:
      raise KeyError(
          f"Image observation key {image_key} not found in observation spec."
      )


def _validate_proprioceptive_observation_keys(
    timestep_spec: gdmr_types.TimeStepSpec,
    proprioceptive_observation_keys: list[str],
) -> None:
  """Validates that the proprio key is listed in the observation spec."""
  if not proprioceptive_observation_keys:
    return

  for proprio_key in proprioceptive_observation_keys:
    if proprio_key not in timestep_spec.observation:
      raise KeyError(
          f"Proprio key {proprio_key} not found in observation spec."
      )

    if not isinstance(timestep_spec.observation[proprio_key], specs.Array):
      raise TypeError(
          f"Proprio data {proprio_key} must be a specs.Array in observation"
          " spec."
      )
