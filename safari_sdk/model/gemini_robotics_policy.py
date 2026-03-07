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

"""Gemini Robotics Policy."""

from collections.abc import Iterable, Sequence
from concurrent import futures
import copy
import dataclasses
import logging
import threading

import dm_env
from dm_env import specs
from gdm_robotics.interfaces import policy as gdmr_policy
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np
import tree
from typing_extensions import override

from safari_sdk.model import additional_observations_provider
from safari_sdk.model import constants
from safari_sdk.model import remote_model_interface


@dataclasses.dataclass(kw_only=True)
class EpisodeStatistics:
  """Statistics for an episode.

  Attributes:
    action_stall_count: The number of times during an episode the policy had to
      wait for a new action chunk from the model because it had run out of
      buffered actions. This metric is used to measure the performance of the
      asynchronous inference, as ideally the policy should never have to wait
      for the model. It is only incremented in ASYNCHRONOUS mode when
      the policy needs to block execution to wait for the next action from
      the model.
  """
  action_stall_count: int


class GeminiRoboticsPolicy(gdmr_policy.Policy[np.ndarray]):
  """Policy which uses the Gemini Robotics API."""

  def __init__(
      self,
      serve_id: str,
      task_instruction_key: str,
      image_observation_keys: Iterable[str],
      proprioceptive_observation_keys: Iterable[str],
      min_replan_interval: int = 15,
      inference_mode: constants.InferenceMode = constants.InferenceMode.ASYNCHRONOUS,
      additional_observations_providers: Sequence[
          additional_observations_provider.AdditionalObservationsProvider
      ] = (),
      robotics_api_connection: constants.RoboticsApiConnectionType = constants.RoboticsApiConnectionType.CLOUD,
      image_compression_jpeg_quality: int = 95,
      num_retries: int = 1,
      action_conditioning_chunk_length: int | None = None,
  ):
    """Initializes the evaluation policy.

    Note: this is policy has an implicit state which is not returned by the
    functions.

    Important: Before using the policy (i.e. calling `initial_state` and `step`)
    you must initialize it by providing the timestep spec by calling
    `step_spec`.

    Args:
      serve_id: The serve ID to use for the policy.
      task_instruction_key: The key of the task instruction in the observation.
      image_observation_keys: A list of observation keys that are related to
        images.
      proprioceptive_observation_keys: The list of observation keys that are
        related to proprioceptive sensors (e.g. joints).
      min_replan_interval: The minimum number of steps to wait before replanning
        the task instruction.
      inference_mode: Whether to use an async or sync implementation of the
        policy.
      additional_observations_providers: A sequence of providers for additional
        observations.
      robotics_api_connection: Connection type for the Robotics API.
      image_compression_jpeg_quality: The JPEG quality for encoding images.
      num_retries: The number of retries for inference calls to the server when
        the connection is CLOUD.
      action_conditioning_chunk_length: Controls the length of the action chunk
        that is used to condition the model from the available action chunk when
        the inference mode is ASYNCHRONOUS. If None, the model will receive all
        remaining actions not consumed so far from the available action chunk.
    """

    self._string_observations_keys = [task_instruction_key]
    self._task_instruction_key = task_instruction_key
    self._image_observation_keys = list(image_observation_keys)
    self._proprioceptive_observation_keys = list(
        proprioceptive_observation_keys
    )
    self._min_replan_interval = min_replan_interval
    self._additional_observations_providers = list(
        additional_observations_providers
    )

    # Go through the additional observation observations spec and
    # augment the image, string and proprioceptive keys.
    for provider in self._additional_observations_providers:
      additional_specs = provider.get_additional_observations_spec()
      for key, spec in additional_specs.items():
        if isinstance(spec, specs.StringArray):
          self._string_observations_keys.append(key)
        elif isinstance(spec, specs.Array):
          if len(spec.shape) == 3:
            self._image_observation_keys.append(key)
          elif len(spec.shape) == 1 or len(spec.shape) == 2:
            self._proprioceptive_observation_keys.append(key)

    self._dummy_state = np.zeros(())

    self._model = remote_model_interface.RemoteModelInterface(
        serve_id=serve_id,
        robotics_api_connection=robotics_api_connection,
        string_observations_keys=self._string_observations_keys,
        task_instruction_key=self._task_instruction_key,
        proprioceptive_observation_keys=self._proprioceptive_observation_keys,
        image_observation_keys=self._image_observation_keys,
        image_compression_jpeg_quality=image_compression_jpeg_quality,
        num_of_retries=num_retries,
    )

    self._model_output = np.array([])
    self._action_spec: gdmr_types.UnboundedArraySpec | None = None
    self._timestep_spec: gdmr_types.TimeStepSpec | None = None
    self._num_of_actions_per_request = 0

    # Threading setup
    self._inference_mode = inference_mode
    if inference_mode == constants.InferenceMode.ASYNCHRONOUS:
      self._executor = futures.ThreadPoolExecutor(max_workers=1)
      self._future: futures.Future[np.ndarray] | None = None
      self._model_output_lock = threading.Lock()
      self._actions_executed_during_inference = 0

    self._action_conditioning_chunk_length = action_conditioning_chunk_length

    self._initialize_episode_statistics()

  def _initialize_episode_statistics(self):
    """Initializes the episode statistics."""
    self._episode_statistics = EpisodeStatistics(action_stall_count=0)

  @property
  def additional_observations_providers(
      self,
  ) -> Sequence[
      additional_observations_provider.AdditionalObservationsProvider
  ]:
    """Returns the additional observations providers."""
    return self._additional_observations_providers

  @override
  def initial_state(
      self,
  ) -> gdmr_types.StateStructure[np.ndarray]:
    """Resets the policy and returns the policy initial state."""
    if self._action_spec is None:
      raise ValueError('Cannot call initial_state before calling step_spec.')

    if self._inference_mode == constants.InferenceMode.ASYNCHRONOUS:
      # Cancel any pending futures on reset
      if self._future and self._future.running():
        self._future.cancel()
      self._future = None

    self._model_output = np.array([])
    for provider in self._additional_observations_providers:
      provider.reset()

    self._initialize_episode_statistics()

    return self._dummy_state

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
    """Takes a step with the policy given an environment timestep.

    Args:
      timestep: An instance of environment `TimeStep`.
      prev_state: This is ignored.

    Returns:
      A tuple of ((action, extra), state) with `action` indicating the action to
      be executed, extra an empty dict and state the dummy policy state.
    """
    del prev_state  # Unused.

    # Add additional observations.
    should_replan = self._should_replan()
    for provider in self._additional_observations_providers:
      additional_obs = provider.get_additional_observations(
          timestep, should_replan
      )
      timestep.observation.update(additional_obs)

    if 'thinking' in timestep.observation:
      logging.info('thinking: %s', str(timestep.observation['thinking']))
    if self._inference_mode == constants.InferenceMode.ASYNCHRONOUS:
      action = self._step_async(timestep)
    else:
      action = self._step_sync(timestep)

    return (action, {}), self._dummy_state

  @override
  def step_spec(self, timestep_spec: gdmr_types.TimeStepSpec) -> tuple[
      tuple[gdmr_types.ActionSpec, gdmr_types.ExtraOutputSpec],
      gdmr_types.StateSpec,
  ]:
    """Returns the spec of the ((action, extra), state) from `step` method."""

    observation_spec = dict(timestep_spec.observation)

    # Add additional observations specs provided by the users.
    extra_specs = {}
    for provider in self._additional_observations_providers:
      extra_specs.update(provider.get_additional_observations_spec())

    if extra_specs:
      observation_spec.update(extra_specs)

      self._timestep_spec = gdmr_types.TimeStepSpec(
          observation=observation_spec,
          step_type=timestep_spec.step_type,
          reward=timestep_spec.reward,
          discount=timestep_spec.discount,
      )
    else:
      self._timestep_spec = timestep_spec

    assert self._timestep_spec is not None
    observation_spec = self._timestep_spec.observation

    # Validate that the timestep_spec contains the required keys.
    if self._string_observations_keys and not all(
        string_obs_key in observation_spec
        for string_obs_key in self._string_observations_keys
    ):
      raise ValueError(
          'timestep_spec does not contain all string observation keys.'
          f' Expected: {self._string_observations_keys}, actual:'
          f' {observation_spec}'
      )

    if self._image_observation_keys and not all(
        image_obs_key in observation_spec
        for image_obs_key in self._image_observation_keys
    ):
      raise ValueError(
          'timestep_spec does not contain all image observation keys.'
          f' Expected: {self._image_observation_keys}, actual:'
          f' {observation_spec}'
      )
    if self._proprioceptive_observation_keys and not all(
        proprio_obs_key in observation_spec
        for proprio_obs_key in self._proprioceptive_observation_keys
    ):
      raise ValueError(
          'timestep_spec does not contain all proprioceptive observation keys.'
          f' Expected: {self._proprioceptive_observation_keys}, actual:'
          f' {observation_spec}'
      )

    if self._action_spec is None:
      logging.warning('action_spec is None, initializing policy.')
      self._setup()
    if self._action_spec is None:
      raise ValueError('action_spec is None, setup failed')

    return (self._action_spec, {}), specs.Array(shape=(), dtype=np.float32)

  def _setup(self):
    """Initializes the policy."""
    if self._timestep_spec is None:
      raise ValueError('timestep_spec is None. Call step_spec first.')

    empty_observation = tree.map_structure(
        lambda s: s.generate_value(), self._timestep_spec.observation
    )

    # Some models require a task instruction to be present
    for string_obs_key in self._string_observations_keys:
      empty_observation[string_obs_key] = np.array('non empty string')

    self._actions_buffer = self._query_model(empty_observation, np.array([]))

    # We support only sequence of actions, that is a 2D array with (num_actions,
    # action_dim) shape.
    if self._actions_buffer.ndim != 2:
      raise ValueError(
          'Action returned by the model must be a 2D array, got'
          f' {self._actions_buffer.shape}.'
      )
    # First axis is the number of actions.
    self._num_of_actions_per_request = self._actions_buffer.shape[0]
    # Assert that the num_of_actions_per_request would be greater than the min
    # replan interval + action_conditioning_chunk_length. If not, the length of
    # the conditioning used to query the model in ASYNCHRONOUS mode would end up
    # being less than desired action_conditioning_chunk_length.
    if (
        self._inference_mode == constants.InferenceMode.ASYNCHRONOUS
        and self._action_conditioning_chunk_length is not None
    ):
      required_buffer_size = (
          self._action_conditioning_chunk_length + self._min_replan_interval
      )
      if self._num_of_actions_per_request < required_buffer_size:
        raise ValueError(
            'Number of actions per request must be greater than the sum of the'
            ' action conditioning chunk length and the minimum replan'
            f' interval. Got {self._num_of_actions_per_request} actions per'
            f' request, but require more than {required_buffer_size} actions.'
            ' (Action conditioning chunk length:'
            f' {self._action_conditioning_chunk_length}, min replan interval:'
            f' {self._min_replan_interval})'
        )

    self._action_spec = gdmr_types.UnboundedArraySpec(
        shape=self._actions_buffer.shape[1:],
        dtype=np.float32,
    )

  def _should_replan(self) -> bool:
    """Returns whether the policy should replan."""
    assert self._action_spec is not None
    actions_left = self._model_output.shape[0]
    if (
        self._num_of_actions_per_request - actions_left
    ) >= self._min_replan_interval:
      return True
    if actions_left == 0:
      return True
    return False

  def _step_sync(self, timestep: dm_env.TimeStep) -> np.ndarray:
    """Computes an action from observations."""
    observation = timestep.observation
    if self._should_replan():
      self._model_output = self._query_model(observation, self._model_output)
      assert self._model_output.shape[0] > 0

    action = self._model_output[0]
    self._model_output = self._model_output[1:]
    return action

  def _step_async(self, timestep: dm_env.TimeStep) -> np.ndarray:
    """Computes an action from the given observation.

    Method:
    1. If Gemini Returned an action chunk, update the action buffer.
    2. If no Gemini query is pending and the action buffer is less than the
       minimum replan interval, trigger a new query.
    3. If the action buffer is empty (first query) trigger a new query.
    4. If there is more than one action in the buffer, consume the first
       action and remove it from the buffer.
    5. If only one action is in the buffer, consume it without removing it (we
    will keep outputting this action until a new action is generated, this is an
    edge case that should not happen in practice). This results in a quasi-async
    implementation.

    Args:
      timestep: An instance of environment `TimeStep`.

    Returns:
        The next action to take.

    Raises:
        ValueError: If no actions are available and no future to generate them
        is present.
    """
    observation = timestep.observation
    is_initial_step = timestep.step_type == dm_env.StepType.FIRST
    with self._model_output_lock:
      # If new model output is available, update the buffer.
      if self._future and self._future.done():
        new_model_output = self._future.result()
        # Remove the actions that were executed while the future was running.
        self._model_output = new_model_output[
            self._actions_executed_during_inference :
        ]
        self._future = None
      actions_left = self._model_output.shape[0]

      # If not enough actions left and not generating, trigger a replan.
      if self._should_replan() and self._future is None:
        self._future = self._executor.submit(
            self._query_model,
            copy.deepcopy(observation),
            copy.deepcopy(self._model_output),
        )
        self._actions_executed_during_inference = 0

    # If no actions left (first query), block until the future is done.
    if actions_left == 0:
      if not self._future:
        raise ValueError('No actions left and no future to generate them.')
      result_from_blocking_wait = self._future.result()
      with self._model_output_lock:
        self._model_output = result_from_blocking_wait[
            self._actions_executed_during_inference :
        ]
        self._future = None
      if not is_initial_step:
        self._episode_statistics.action_stall_count += 1

    # Consume the action.
    with self._model_output_lock:
      action = self._model_output[0]
      self._model_output = self._model_output[1:]
      self._actions_executed_during_inference += 1
    return action

  def _query_model(
      self,
      observation: dict[str, np.ndarray],
      model_output: np.ndarray,
  ) -> np.ndarray:
    """Queries the model with the given observation and task instruction."""
    # Conditioning on what the model has left to output in ASYNCHRONOUS mode
    # and accounting for the action_conditioning_chunk_length if set.
    if (
        self._inference_mode == constants.InferenceMode.ASYNCHRONOUS
        and model_output.size > 0
    ):
      # Trim the model output to the action conditioning chunk length if set.
      # If None, the model will still receive all remaining actions not consumed
      # so far from the available action chunk.
      # NOTE: It is safe to use and reassign the model_output variable here
      # because its value is a deep copy of the original dictionary which
      # means that we don't need the lock and does not affect the current
      # model_output variable used in the step method.
      model_output = model_output[
          : self._action_conditioning_chunk_length
      ]
      observation[constants.CONDITIONING_ENCODED_OBS_KEY] = (
          model_output.tolist()
      )
    return self._model.query_model(observation)

  @property
  def episode_statistics(self) -> EpisodeStatistics:
    return copy.deepcopy(self._episode_statistics)
