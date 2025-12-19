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
import json
import logging
import threading
from typing import Any

from absl import flags
import dm_env
from dm_env import specs
from gdm_robotics.interfaces import policy as gdmr_policy
from gdm_robotics.interfaces import types as gdmr_types
import grpc
import numpy as np
import tree
from typing_extensions import override

from safari_sdk.model import additional_observations_provider
from safari_sdk.model import constants
from safari_sdk.model import genai_robotics
from safari_sdk.model import observation_to_model_query_contents


_ENABLE_SERVER_INIT = flags.DEFINE_bool(
    'safari_enable_server_init',
    False,
    'Whether to enable server init.',
)


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
    """
    self._serve_id = serve_id
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
    self._robotics_api_connection = robotics_api_connection
    self._image_compression_jpeg_quality = image_compression_jpeg_quality

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

    self._model_output = np.array([])
    self._action_spec: gdmr_types.UnboundedArraySpec | None = None
    self._timestep_spec: gdmr_types.TimeStepSpec | None = None
    self._num_of_actions_per_request = 0

    # Initialize the genai_robotics client
    self._client = genai_robotics.Client(
        robotics_api_connection=robotics_api_connection,
    )

    # TODO: Remove when this is fixed.
    if (
        _ENABLE_SERVER_INIT.value
        and self._robotics_api_connection
        == constants.RoboticsApiConnectionType.LOCAL
    ):
      # Create an additional client connection for the reset method.
      local_credentials = grpc.local_channel_credentials()
      self._grpc_channel = grpc.secure_channel(
          genai_robotics._LOCAL_GRPC_URL.removeprefix('grpc://'),
          local_credentials,
      )
      self._initial_state_stub = self._grpc_channel.unary_unary(
          method='/gemini_robotics/initial_state',
          request_serializer=lambda v: v,
          response_deserializer=lambda v: v,
      )

      def _query_encoder(query: dict[str, Any]) -> str:
        encoded_query = json.dumps(query).encode('utf-8')
        return self._initial_state_stub(encoded_query).decode('utf-8')

      self._initial_state_method = _query_encoder

    # Threading setup
    self._inference_mode = inference_mode
    if inference_mode == constants.InferenceMode.ASYNCHRONOUS:
      self._executor = futures.ThreadPoolExecutor(max_workers=1)
      self._future: futures.Future[np.ndarray] | None = None
      self._model_output_lock = threading.Lock()
      self._actions_executed_during_inference = 0

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

    # TODO: Remove when this is fixed.
    if (
        _ENABLE_SERVER_INIT.value
        and self._robotics_api_connection
        == constants.RoboticsApiConnectionType.LOCAL
    ):
      self._initial_state_method({})

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
      action = self._step_async(timestep.observation)
    else:
      action = self._step_sync(timestep.observation)

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

    # Validate that the timestep_spec contains the required keys.
    if self._string_observations_keys and not all(
        string_obs_key in self._timestep_spec.observation
        for string_obs_key in self._string_observations_keys
    ):
      raise ValueError(
          'timestep_spec does not contain all string observation keys.'
          f' Expected: {self._string_observations_keys}, actual:'
          f' {self._timestep_spec.observation}'
      )

    if self._image_observation_keys and not all(
        image_obs_key in self._timestep_spec.observation
        for image_obs_key in self._image_observation_keys
    ):
      raise ValueError(
          'timestep_spec does not contain all image observation keys.'
          f' Expected: {self._image_observation_keys}, actual:'
          f' {self._timestep_spec.observation}'
      )
    if self._proprioceptive_observation_keys and not all(
        proprio_obs_key in self._timestep_spec.observation
        for proprio_obs_key in self._proprioceptive_observation_keys
    ):
      raise ValueError(
          'timestep_spec does not contain all proprioceptive observation keys.'
          f' Expected: {self._proprioceptive_observation_keys}, actual:'
          f' {self._timestep_spec.observation}'
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
    # First axis is the number of actions.
    self._num_of_actions_per_request = self._actions_buffer.shape[0]

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

  def _step_sync(self, observation: dict[str, np.ndarray]) -> np.ndarray:
    """Computes an action from observations."""
    if self._should_replan():
      self._model_output = self._query_model(observation, self._model_output)
      assert self._model_output.shape[0] > 0

    action = self._model_output[0]
    self._model_output = self._model_output[1:]
    return action

  def _step_async(self, observation: dict[str, np.ndarray]) -> np.ndarray:
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
        observation: A dictionary of observations from the environment.

    Returns:
        The next action to take.

    Raises:
        ValueError: If no actions are available and no future to generate them
        is present.
    """
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
    contents = observation_to_model_query_contents.observation_to_model_query_contents(
        observation=observation,
        model_output=model_output,
        string_observations_keys=self._string_observations_keys,
        task_instruction_key=self._task_instruction_key,
        proprioceptive_observation_keys=self._proprioceptive_observation_keys,
        image_observation_keys=self._image_observation_keys,
        inference_mode=self._inference_mode,
    )
    if (
        self._robotics_api_connection
        == constants.RoboticsApiConnectionType.CLOUD_GENAI
    ):
      contents = genai_robotics.update_robotics_content_to_genai_format(
          contents,
          image_compression_jpeg_quality=self._image_compression_jpeg_quality,
      )

    response = self._client.models.generate_content(
        model=self._serve_id,
        contents=contents,
    )

    # Parse the response text (assuming its JSON containing the action)
    if response.text:
      response_data = json.loads(response.text)
    elif response.candidates:
      response_data = json.loads(
          response.candidates[0].content.parts[0].inline_data.data
      )
    else:
      raise ValueError('Response does not contain text or candidates.')

    # Assuming the structure is {'action_chunk': [...]}
    action_chunk = response_data.get('action_chunk')
    if action_chunk is None:
      raise ValueError("Response JSON does not contain 'action_chunk'")
    action_chunk = np.array(action_chunk)
    if action_chunk.ndim == 3:
      action_chunk = action_chunk[0]

    return action_chunk

  @property
  def policy_type(self) -> str:
    if self._inference_mode == constants.InferenceMode.ASYNCHRONOUS:
      return f'gemini_robotics_async[{self._serve_id}]'
    return f'gemini_robotics[{self._serve_id}]'
