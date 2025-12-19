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

import collections
from collections.abc import Callable, Mapping
import copy
import time
from typing import Any
import uuid

from absl import logging
import dm_env
from dm_env import specs
from gdm_robotics.interfaces import episodic_logger
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np
import tree

from google.protobuf import struct_pb2
from safari_sdk.logging.cc.python import log_writer
from safari_sdk.logging.python import constants
from safari_sdk.logging.python import metadata_utils
from safari_sdk.logging.python import session_manager as session_manager_lib
from safari_sdk.protos import label_pb2


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
      batch_size: int | None = None,
      num_workers: int | None = None,
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
      batch_size: The number of steps to batch before writing to disk. If None,
        no batching will be done and the entire episode will stored in memory
        and written to disk in a single batch.
      num_workers: The number of workers to use for writing to disk. If None,
        the default number of workers will be used.

    Returns:
        A EpisodicLogger instance.
    """
    if not agent_id:
      raise ValueError("agent_id must be provided as a non-empty string.")

    if batch_size is not None and batch_size < constants.MIN_BATCH_SIZE:
      raise ValueError(
          f"Batch size must be at least {constants.MIN_BATCH_SIZE}. If you want"
          " to disable batching, please set batch_size to None."
      )
    topics = {
        constants.ACTION_TOPIC_NAME,
        constants.TIMESTEP_TOPIC_NAME,
        constants.POLICY_EXTRA_TOPIC_NAME,
    }
    required_topics = set()

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

    max_num_workers = (
        num_workers if num_workers else constants.DEFAULT_NUM_WORKERS
    )
    thread_pool_log_writer_config = log_writer.ThreadPoolLogWriterConfig(
        max_num_workers=max_num_workers,
        image_observation_keys=[
            constants.OBSERVATION_KEY_TEMPLATE.format(key)
            for key in image_observation_keys
        ],
        mcap_file_config=log_writer.McapFileConfig(
            output_dir=output_directory,
            file_metadata_topic=constants.FILE_METADATA_TOPIC_NAME,
            agent_id=agent_id,
            file_shard_size_limit_bytes=file_shard_size_limit_bytes,
        ),
    )

    writer = log_writer.create_log_writer(config=thread_pool_log_writer_config)
    if writer is None:
      raise ValueError("Failed to create log writer.")

    return cls(
        task_id=task_id,
        image_observation_keys=image_observation_keys,
        proprioceptive_observation_keys=proprioceptive_observation_keys,
        timestep_spec=timestep_spec,
        action_spec=action_spec,
        policy_extra_spec=policy_extra_spec,
        session_manager=session_manager,
        writer=writer,
        validate_data_with_spec=validate_data_with_spec,
        timestamp_key=timestamp_key,
        dynamic_metadata_provider=dynamic_metadata_provider,
        batch_size=batch_size,
        output_directory=output_directory,
    )

  def __init__(
      self,
      task_id: str,
      image_observation_keys: list[str],
      proprioceptive_observation_keys: list[str],
      timestep_spec: gdmr_types.TimeStepSpec,
      action_spec: gdmr_types.ActionSpec,
      policy_extra_spec: gdmr_types.ExtraOutputSpec,
      session_manager: session_manager_lib.SessionManager,
      writer: log_writer.LogWriter,
      output_directory: str,
      validate_data_with_spec: bool = True,
      timestamp_key: str | None = None,
      dynamic_metadata_provider: Callable[[], Mapping[str, str]] | None = None,
      batch_size: int | None = None,
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
      session_manager: The session manager for managing session metadata.
      writer: The log writer.
        output_directory: The output directory.
      validate_data_with_spec: Whether to validate all the logged data with the
        provided spec.
      timestamp_key: The observation key that maps to the the timestamps of each
        step in an episode. This is used to set the publish time of each MCAP
        message. This is optional, if not provided, the logger will generate
        timestamps using time.time_ns().
      dynamic_metadata_provider: A function that provides dynamic metadata to be
        logged as session labels. The function is called when `write` is called.
      batch_size: The number of steps to batch before writing to disk. If None,
        no batching will be done and the entire episode will stored in memory
        and written to disk in a single batch.
    """
    if not task_id:
      raise ValueError("task_id must be provided as a non-empty string.")

    self._writer = writer
    self._batch_size = batch_size

    if self._batch_size is not None:
      logging.info(
          "Batching logging is enabled with batch size: %s steps.",
          self._batch_size,
      )
    else:
      logging.info("Batching logging is disabled.")

    self._current_batch_size = 0
    self._batch_number = 0

    self._timestep_batch = collections.defaultdict(list)
    self._action_batch = collections.defaultdict(list)
    self._policy_extra_batch = collections.defaultdict(list)

    self._session_manager = session_manager

    self._output_directory = output_directory

    self._timestep_publish_time_ns: list[int] = []
    self._action_publish_time_ns: list[int] = []
    self._last_timestep_publish_time_ns = 0

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

    self._episode_uuid: str = ""

    # Whether the logger is currently recording data.
    # We mark as True when reset is called and False after writing an episode
    # has been completed.
    self._is_recording = False
    # Whether the logger has been stopped. This is set to True when stop is
    # called.
    self._stopped = False

  def __del__(self):
    """Stops the logger and writes any remaining data."""
    self.stop()

  def stop(self) -> None:
    """Stops the logger and writes any remaining data."""
    self._writer.stop()
    self._is_recording = False
    self._stopped = True

  def flush(self) -> None:
    """Flushes the logger and writes any remaining data."""
    self._writer.stop()
    self._writer.start()

  def reset(self, timestep: dm_env.TimeStep) -> None:
    """Resets the logger with a starting TimeStep.

    All existing data will be flushed to the current episode.

    In this method, we mark the logger as recording (i.e. set _is_recording to
    True).

    Args:
      timestep: The starting timestep of the episode.
    """

    # Call write to flush previous episode.
    if self._current_episode_step > 0:
      self.write()

    self._reset_saved_data()

    if self._timestamp_key:
      timestamp_ns = timestep.observation[self._timestamp_key]
    else:
      timestamp_ns = time.time_ns()

    # Try to start a new session.
    try:
      self._session_manager.start_session(
          start_timestamp_nsec=timestamp_ns, task_id=self._task_id
      )
    except ValueError:
      logging.exception("Failed to start session for logging episode.")
      raise

    self._add_timestep_to_batch(timestep)
    self._timestep_publish_time_ns.append(timestamp_ns)

    self._current_episode_step += 1
    self._current_batch_size += 1

    self._episode_uuid = str(uuid.uuid4())

    logging.info("Resetting logger for Episode uuid: %s", self._episode_uuid)

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
    if self._stopped:
      logging.warning("Logger is stopped. Ignoring the current step.")
      return

    if self._timestamp_key:
      timestamp_ns = next_timestep.observation[self._timestamp_key]
    else:
      timestamp_ns = time.time_ns()

    # The next action push time should be equal to the previous timestep
    # publish time.
    if self._timestep_publish_time_ns:
      self._action_publish_time_ns.append(self._timestep_publish_time_ns[-1])

    self._timestep_publish_time_ns.append(timestamp_ns)

    # Validate the data before adding it to the batch.
    if self._validate_data_with_spec:
      self._validate_timestep(next_timestep)
      self._validate_action(action)
      self._validate_policy_extra(policy_extra)

    self._add_timestep_to_batch(next_timestep)
    self._add_action_to_batch(action)
    self._add_policy_extra_to_batch(policy_extra)

    self._current_batch_size += 1
    self._current_episode_step += 1

    last_batch = next_timestep.step_type == dm_env.StepType.LAST
    # Write the batch if the batch size is reached.
    # Or if the next timestep is the last timestep of the episode.
    if self._batch_size and (
        self._current_batch_size == self._batch_size or last_batch
    ):
      self._write_batch(last_batch)

  def _add_timestep_to_batch(self, timestep: dm_env.TimeStep) -> None:
    # Adds the timestep to the current batch.

    if not isinstance(timestep.observation, Mapping):
      raise TypeError(
          f"Unsupported observation type: {type(timestep.observation)}"
      )
    # Append the step type to the batch.
    self._timestep_batch[constants.STEP_TYPE_KEY].append(
        np.asarray(timestep.step_type, dtype=np.int32)
    )

    # Append the observations to the batch.
    for key, value in timestep.observation.items():
      self._timestep_batch[
          constants.OBSERVATION_KEY_TEMPLATE.format(key)
      ].append(value)

    # Append the rewards to the batch.
    if isinstance(timestep.reward, Mapping):
      for key, value in timestep.reward.items():
        self._timestep_batch[constants.REWARD_KEY_TEMPLATE.format(key)].append(
            value
        )
    else:
      reward = np.asarray(timestep.reward)
      # Reward is a float. If the `asarray` converted it to something different,
      # cast it back to a float.
      if reward.dtype != np.float64 or reward.dtype != np.float32:
        reward = reward.astype(np.float32, copy=False)
      self._timestep_batch[constants.REWARD_KEY].append(reward)

    # Append the discounts to the batch.
    if isinstance(timestep.discount, Mapping):
      for key, value in timestep.discount.items():
        self._timestep_batch[
            constants.DISCOUNT_KEY_TEMPLATE.format(key)
        ].append(value)
    else:
      discount = np.asarray(timestep.discount)
      # Discount is a float. If the `asarray` converted it to something
      # different, cast it back to a float. Casting should be always safe.
      if discount.dtype != np.float64 or discount.dtype != np.float32:
        discount = discount.astype(np.float32, copy=False)
      self._timestep_batch[constants.DISCOUNT_KEY].append(discount)

  def _add_action_to_batch(self, action: gdmr_types.ActionType) -> None:
    # Adds the action to the current batch.
    if not isinstance(action, np.ndarray):
      raise TypeError(
          f"Unsupported action type: {type(action)}. Only np.ndarray is"
          " supported."
      )

    self._action_batch[constants.ACTION_KEY_PREFIX].append(action)

  def _add_policy_extra_to_batch(self, policy_extra: Mapping[str, Any]) -> None:
    # Adds the policy extra to the current batch.
    for key, value in policy_extra.items():
      self._policy_extra_batch[
          constants.POLICY_EXTRA_KEY_TEMPLATE.format(key)
      ].append(value)

  def _pad_action_and_policy_extra_batch(self) -> None:
    # Pad the last action and policy extra with the last corresponding values so
    # as to have the same length for all repeated fields. This is because the
    # last environment transition does not have an associated action and policy
    # extra.
    padded_action = copy.deepcopy(
        self._action_batch[constants.ACTION_KEY_PREFIX][-1]
    )
    self._action_batch[constants.ACTION_KEY_PREFIX].append(padded_action)

    for key in self._policy_extra_batch.keys():
      self._policy_extra_batch[key].append(self._policy_extra_batch[key][-1])

    self._action_publish_time_ns.append(self._timestep_publish_time_ns[-1])

  def _write_batch(self, last_batch: bool = False) -> None:
    # Writes the current batch of data to disk.
    # If the last batch is true, we will pad the action and policy extra batch.
    if self._current_batch_size == 0:
      logging.info("No data to write. Batch size is 0.")
      return

    timestep_options = log_writer.EnqueueMcapFileOptions(
        episode_uuid=self._episode_uuid,
        topic=constants.TIMESTEP_TOPIC_NAME,
        timestamp_ns=self._timestep_publish_time_ns[-1],
    )

    for key, value in self._timestep_batch.items():
      self._timestep_batch[key] = np.asarray(value)

    self._writer.enqueue_episode_data(
        self._timestep_batch, self._timestep_publish_time_ns, timestep_options
    )

    # Pad the last action and policy extra with the last corresponding values
    # so as to have the same length for all repeated fields. This is because the
    # last environment transition does not have an associated action and policy
    # extra.
    if last_batch:
      self._pad_action_and_policy_extra_batch()
      self._last_timestep_publish_time_ns = self._timestep_publish_time_ns[-1]

    action_options = log_writer.EnqueueMcapFileOptions(
        episode_uuid=self._episode_uuid,
        topic=constants.ACTION_TOPIC_NAME,
        timestamp_ns=self._action_publish_time_ns[-1],
    )
    self._action_batch[constants.ACTION_KEY_PREFIX] = np.asarray(
        self._action_batch[constants.ACTION_KEY_PREFIX]
    )

    self._writer.enqueue_episode_data(
        self._action_batch, self._action_publish_time_ns, action_options
    )

    policy_extra_options = log_writer.EnqueueMcapFileOptions(
        episode_uuid=self._episode_uuid,
        topic=constants.POLICY_EXTRA_TOPIC_NAME,
        timestamp_ns=self._action_publish_time_ns[-1],
    )
    for key, value in self._policy_extra_batch.items():
      self._policy_extra_batch[key] = np.asarray(value)

    self._writer.enqueue_episode_data(
        self._policy_extra_batch,
        self._action_publish_time_ns,
        policy_extra_options,
    )

    # Reset the batch variables. Increase the batch number for the same episode.
    self._current_batch_size = 0
    self._batch_number += 1

    self._reset_batch_data()

  def write(self) -> None:
    """Writes the current episode logged data by converting accumulated data to protos."""
    # If the logger is not recording data, we should not write.
    # This protects against cases where we try to call write() without calling
    # reset() first.
    if not self._is_recording:
      logging.info("Logger is not recording data. Skipping write.")
      return

    if self._current_episode_step <= 1:
      logging.info("No episode data to write.")
      return

    logging.info("Writing episode with %d steps.", self._current_episode_step)

    self._write_batch(last_batch=True)

    self._write_session()

    # Mark the logger as not recording once the episode has been written.
    self._is_recording = False

    logging.info(
        "Episode written to mcap. Episode steps: %d", self._current_episode_step
    )
    self._reset_saved_data()

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

    episode_end_time_ns = self._last_timestep_publish_time_ns
    logging.info("Episode end time ns: %d", episode_end_time_ns)
    session = self._session_manager.stop_session(
        stop_timestamp_nsec=episode_end_time_ns
    )

    self._writer.enqueue_session_data(
        session,
        log_writer.EnqueueMcapFileOptions(
            episode_uuid=self._episode_uuid,
            topic=constants.SESSION_TOPIC_NAME,
            timestamp_ns=episode_end_time_ns,
        ),
    )

  def _reset_batch_data(self, reset_episode: bool = False) -> None:
    # If the episode is not being reset, we can reuse the previous timestep
    # publish time as the action publish time for the current step.
    if not reset_episode and self._timestep_publish_time_ns:
      self._action_publish_time_ns = [self._timestep_publish_time_ns[-1]]
    else:
      self._action_publish_time_ns = []
    self._timestep_publish_time_ns = []
    self._timestep_batch = collections.defaultdict(list)
    self._action_batch = collections.defaultdict(list)
    self._policy_extra_batch = collections.defaultdict(list)

  def _reset_saved_data(self) -> None:
    self._reset_batch_data(reset_episode=True)
    self._current_episode_step = 0
    self._current_batch_size = 0
    self._batch_number = 0

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
