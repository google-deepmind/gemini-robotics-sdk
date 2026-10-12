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

"""Gemini Robotics Policy using bidirectional streaming."""

from collections.abc import Sequence
import copy
import dataclasses
import logging
import time

import dm_env
from dm_env import specs
from gdm_robotics.interfaces import policy as gdmr_policy
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np
import tree
from typing_extensions import override

from safari_sdk.model import action_chunk_resampling
from safari_sdk.model import additional_observations_provider
from safari_sdk.model import bidi_remote_model_interface
from safari_sdk.model import constants


# Latency metrics reported through the `extra` output. Declared once so that
# the spec, the populated dict, and the sentinel dict cannot drift apart.
_LATENCY_METRIC_KEYS = (
    # Total time from query start to query return on the robot client.
    'inference_total_ms',
    # Remote model execution duration inside the cloud backend.
    'remote_inference_ms',
    # Pure physical wire transit RTT excluding server compute.
    'client_wire_transit_ms',
    # Client workstation CPU processing time (image encode, prep, parsing).
    'client_processing_ms',
    # Client CPU time spent JPEG compressing camera observation images.
    'client_image_encode_ms',
    # Legacy aggregate non-server overhead (total - remote).
    'network_overhead_ms',
)


@dataclasses.dataclass(kw_only=True)
class EpisodeStatistics:
  """Statistics for an episode.

  Attributes:
    action_stall_count: The number of times during an episode the policy had to
      wait for a new action chunk from the model because it had run out of
      buffered actions. This metric is used to measure the performance of the
      asynchronous inference, as ideally the policy should never have to wait
      for the model. It is only incremented in ASYNCHRONOUS mode when the policy
      needs to block execution to wait for the next action from the model.
  """

  action_stall_count: int


@dataclasses.dataclass
class _InFlightQuery:
  """Bundles an `ActionHorizon` with per-query policy metadata.

  Attributes:
    horizon: The streamed `ActionHorizon` returned by `query_model()`.
    started_at_s: `time.perf_counter()` taken as the query was issued.
    stats_dict: The latency metrics dictionary built when the first chunk lands.
    stats_reported: Whether `stats_dict` has been emitted in `extra` after this
      query became active.
  """

  horizon: bidi_remote_model_interface.ActionHorizon
  started_at_s: float
  stats_dict: dict[str, np.ndarray] | None = None
  stats_reported: bool = False


class BidiGeminiRoboticsPolicy(gdmr_policy.Policy[np.ndarray]):
  """Policy which uses the Gemini Robotics bidirectional streaming API."""

  def __init__(
      self,
      serve_id: str,
      task_instruction_key: str,
      image_observation_keys: Sequence[str],
      proprioceptive_observation_keys: Sequence[str],
      min_replan_interval: int = 15,
      inference_mode: constants.InferenceMode = constants.InferenceMode.ASYNCHRONOUS,
      additional_observations_providers: Sequence[
          additional_observations_provider.AdditionalObservationsProvider
      ] = (),
      robotics_api_connection: constants.RoboticsApiConnectionType = constants.RoboticsApiConnectionType.CLOUD,
      image_compression_jpeg_quality: int = 95,
      action_conditioning_chunk_length: int | None = None,
      action_conditioning_resample_factor: float | None = None,
      request_timeout: float = 30.0,
  ):
    """Initializes the evaluation policy.

    Note: this policy has an implicit state which is not returned by the
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
      robotics_api_connection: Connection type for the Robotics API. Must be
        CLOUD.
      image_compression_jpeg_quality: The JPEG quality for encoding images.
      action_conditioning_chunk_length: Controls the length of the action chunk
        that is used to condition the model from the available action chunk when
        the inference mode is ASYNCHRONOUS. If None, the model will receive all
        remaining actions not consumed so far from the available action chunk.
      action_conditioning_resample_factor: A resampling factor applied to the
        action chunk to be used to condition the next model query. When set and
        > 1, performs a linear interpolation between the actions in the
        conditioning chunk to resample them in accordance with the factor
        specified. (for example, a factor of 2 will effectively select every
        other action from the conditioning chunk). If None or 1.0, no resampling
        is performed. Currently only supported in ASYNCHRONOUS mode and must be
        >= 1.
      request_timeout: Seconds to wait for each individual response chunk, not
        for the whole request. The policy's own stall backstop is derived from
        this and is deliberately more permissive, so that a slow chunk is
        reported by the model interface rather than by the policy.

    Raises:
      ValueError: `robotics_api_connection` is not CLOUD, or `request_timeout`
        is not positive.
    """
    if robotics_api_connection != constants.RoboticsApiConnectionType.CLOUD:
      raise ValueError(
          'BidiGeminiRoboticsPolicy only supports robotics_api_connection '
          f'{constants.RoboticsApiConnectionType.CLOUD}, but got '
          f'{robotics_api_connection}.'
      )

    self._model = bidi_remote_model_interface.BidiRemoteModelInterface(
        serve_id=serve_id,
        robotics_api_connection=robotics_api_connection,
        task_instruction_key=task_instruction_key,
        proprioceptive_observation_keys=proprioceptive_observation_keys,
        image_observation_keys=image_observation_keys,
        image_compression_jpeg_quality=image_compression_jpeg_quality,
        request_timeout=request_timeout,
        additional_observations_providers=additional_observations_providers,
    )

    # These are used only to validate the timestep spec in step_spec.
    self._string_observations_keys = [task_instruction_key]
    self._image_observation_keys = list(image_observation_keys)
    self._proprioceptive_observation_keys = list(
        proprioceptive_observation_keys
    )
    self._additional_observations_providers = list(
        additional_observations_providers
    )

    self._min_replan_interval = min_replan_interval
    self._action_conditioning_chunk_length = action_conditioning_chunk_length
    self._action_conditioning_resample_factor = (
        action_conditioning_resample_factor or 1.0
    )
    if self._action_conditioning_resample_factor < 1.0:
      raise ValueError(
          'Only action conditioning resample factor >= 1.0 is supported.'
      )

    self._dummy_state = np.zeros(())

    # Actions of the query issued by `_setup()`, kept only to derive specs.
    self._spec_probe_actions = np.array([])
    self._action_spec: gdmr_types.UnboundedArraySpec | None = None
    self._timestep_spec: gdmr_types.TimeStepSpec | None = None

    self._inference_mode = inference_mode
    self._initial_state_call_count = 0
    # `_active_query` owns the ActionHorizon currently being executed;
    # `_pending_query` holds a replan query in flight until its first actionable
    # chunk arrives and supersedes `_active_query`.
    self._active_query: _InFlightQuery | None = None
    self._pending_query: _InFlightQuery | None = None

    self._last_extra = self._get_empty_extra_dict()
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
  ) -> gdmr_types.StateStructure[np.ndarray]:  # pyrefly: ignore[invalid-type-var]
    """Resets the policy and returns the policy initial state."""
    if self._action_spec is None:
      raise ValueError('Cannot call initial_state before calling step_spec.')

    self._cleanup_previous_state()

    if self._initial_state_call_count >= 1:
      self._model.reset()
    self._initial_state_call_count += 1

    for provider in self._additional_observations_providers:
      provider.reset()

    self._initialize_episode_statistics()

    return self._dummy_state

  @override
  def step(
      self,
      timestep: dm_env.TimeStep,
      prev_state: gdmr_types.StateStructure[np.ndarray],  # pyrefly: ignore[invalid-type-var]
  ) -> tuple[  # pyrefly: ignore[invalid-type-var]
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
      be executed, extra a dictionary containing inference latency metrics, and
      state the dummy policy state.
    """
    del prev_state  # Unused.

    if self._inference_mode == constants.InferenceMode.ASYNCHRONOUS:
      self._promote_pending_query_if_ready()

    should_replan = self._should_replan()
    new_observation = dict(timestep.observation)
    for provider in self._additional_observations_providers:
      additional_obs = provider.get_additional_observations(
          timestep, should_replan
      )
      new_observation.update(additional_obs)
    timestep = timestep._replace(observation=new_observation)

    if 'thinking' in timestep.observation:
      logging.info('thinking: %s', str(timestep.observation['thinking']))
    if self._inference_mode == constants.InferenceMode.ASYNCHRONOUS:
      action = self._step_async(timestep)
    else:
      action = self._step_sync(timestep)

    return (action, self._last_extra), self._dummy_state  # pyrefly: ignore[bad-return]

  @override
  def step_spec(self, timestep_spec: gdmr_types.TimeStepSpec) -> tuple[  # pyrefly: ignore[invalid-type-var]
      tuple[gdmr_types.ActionSpec, gdmr_types.ExtraOutputSpec],
      gdmr_types.StateSpec,
  ]:
    """Returns the spec of the ((action, extra), state) from `step` method."""

    observation_spec = dict(timestep_spec.observation)  # pyrefly: ignore[no-matching-overload]

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
        string_obs_key in observation_spec  # pyrefly: ignore[not-iterable]
        for string_obs_key in self._string_observations_keys
    ):
      raise ValueError(
          'timestep_spec does not contain all string observation keys.'
          f' Expected: {self._string_observations_keys}, actual:'
          f' {observation_spec}'
      )

    if self._image_observation_keys and not all(
        image_obs_key in observation_spec  # pyrefly: ignore[not-iterable]
        for image_obs_key in self._image_observation_keys
    ):
      raise ValueError(
          'timestep_spec does not contain all image observation keys.'
          f' Expected: {self._image_observation_keys}, actual:'
          f' {observation_spec}'
      )
    if self._proprioceptive_observation_keys and not all(
        proprio_obs_key in observation_spec  # pyrefly: ignore[not-iterable]
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

    extra_spec = {
        key: specs.Array(shape=(), dtype=np.float32)
        for key in _LATENCY_METRIC_KEYS
    }
    extra_spec.update({
        'inference_sent': specs.Array(shape=(), dtype=np.uint8),
        'actions_left': specs.Array(shape=(), dtype=np.int32),
        'action_chunk': specs.Array(
            shape=self._spec_probe_actions.shape,
            dtype=self._spec_probe_actions.dtype,
        ),
    })

    return (  # pyrefly: ignore[bad-return]
        self._action_spec,
        extra_spec,
    ), specs.Array(shape=(), dtype=np.float32)

  def _setup(self):
    """Initializes the policy."""
    if self._timestep_spec is None:
      raise ValueError('timestep_spec is None. Call step_spec first.')

    empty_observation = tree.map_structure(
        lambda s: s.generate_value(), self._timestep_spec.observation
    )

    # Some models require a non-empty task instruction to be present
    observation_spec = dict(self._timestep_spec.observation)  # pyrefly: ignore[no-matching-overload]
    non_empty_strings = {
        key: np.array('non empty string', dtype=np.dtypes.StringDType())  # pyrefly: ignore[missing-attribute]
        for key, spec in observation_spec.items()
        if isinstance(spec, specs.StringArray)
    }
    empty_observation.update(non_empty_strings)

    self._model.reset()
    probe_horizon = self._model.query_model(empty_observation)
    self._spec_probe_actions = probe_horizon.snapshot(wait_until_exhausted=True)

    # We support only sequence of actions, that is a 2D array with (num_actions,
    # action_dim) shape.
    if self._spec_probe_actions.ndim != 2:
      raise ValueError(
          'Action returned by the model must be a 2D array, got'
          f' {self._spec_probe_actions.shape}.'
      )
    # First axis is the number of actions.
    num_of_actions_per_request = self._spec_probe_actions.shape[0]
    if (
        self._inference_mode == constants.InferenceMode.ASYNCHRONOUS
        and self._action_conditioning_chunk_length is not None
    ):
      required_buffer_size = (
          action_chunk_resampling.resampled_to_min_source_length(
              resampled_length=self._action_conditioning_chunk_length,
              resample_factor=self._action_conditioning_resample_factor,
          )
          + self._min_replan_interval
      )
      if num_of_actions_per_request < required_buffer_size:
        raise ValueError(
            'Number of actions per request must be greater than the sum of the'
            ' action conditioning chunk length (after resampling if needed) and'
            ' the minimum replan interval. Got'
            f' {num_of_actions_per_request} actions per request, but'
            f' require more than {required_buffer_size} actions.'
            ' (Action conditioning chunk length:'
            f' {self._action_conditioning_chunk_length},'
            ' min replan interval: {self._min_replan_interval})'
        )

    self._action_spec = gdmr_types.UnboundedArraySpec(
        shape=self._spec_probe_actions.shape[1:],
        dtype=self._spec_probe_actions.dtype,
    )

  def _promote_pending_query_if_ready(self) -> None:
    """Promotes `_pending_query` to `_active_query` when actionable or needed."""
    if self._pending_query is None:
      return
    if self._pending_query.horizon.chunks_received > 0 and (
        self._pending_query.stats_dict is None
    ):
      self._record_query_stats(self._pending_query)
    # Promote when needed (cold start or active query ran dry) or actionable
    # (pending query has arrived actions ready to execute past the cursor).
    if (
        self._active_query is None
        or self._active_query.horizon.is_exhausted
        or self._pending_query.horizon.has_arrived_actions()
    ):
      self._active_query = self._pending_query
      self._pending_query = None

  def _should_replan(self) -> bool:
    """Returns whether the policy should replan."""
    assert self._action_spec is not None
    # Has pending query.
    if self._pending_query:
      return False
    # Initial state.
    if self._active_query is None:
      return True
    # Should refresh with actions generated from a newer observation.
    return (
        self._active_query.horizon.is_exhausted
        or self._active_query.horizon.actions_consumed()
        >= self._min_replan_interval
    )

  def _step_sync(self, timestep: dm_env.TimeStep) -> np.ndarray:
    """Computes an action from observations."""
    observation = timestep.observation
    extra = self._get_empty_query_stats_dict()
    inference_sent = False
    if self._should_replan():
      query_start = time.perf_counter()
      horizon = self._model.query_model(observation)
      horizon.snapshot(wait_until_exhausted=True)
      extra = self._build_query_stats(
          (time.perf_counter() - query_start) * 1000.0
      )
      self._log_inference_timing(extra)
      self._active_query = _InFlightQuery(
          horizon=horizon,
          started_at_s=query_start,
          stats_dict=extra,
          stats_reported=True,
      )
      inference_sent = True

    assert self._active_query is not None
    action = self._active_query.horizon.take_next_action()

    extra['inference_sent'] = np.array(inference_sent, dtype=np.uint8)
    extra['actions_left'] = np.array(
        self._active_query.horizon.available_actions_count, dtype=np.int32
    )
    extra['action_chunk'] = self._active_query.horizon.snapshot()

    self._last_extra = extra
    return action

  def _step_async(self, timestep: dm_env.TimeStep) -> np.ndarray:
    """Computes an action from the given timestep in asynchronous mode."""
    observation = timestep.observation
    is_initial_step = timestep.step_type == dm_env.StepType.FIRST

    inference_sent = False
    if self._should_replan():
      self._pending_query = self._issue_query(observation)
      inference_sent = True

    self._promote_pending_query_if_ready()

    if self._active_query is None:
      raise ValueError(
          'No actions left and no query in flight to produce them.'
      )

    if (
        not self._active_query.horizon.has_arrived_actions()
        and not is_initial_step
    ):
      self._episode_statistics.action_stall_count += 1

    action = self._active_query.horizon.take_next_action()
    if self._pending_query is not None:
      self._pending_query.horizon.advance_cursor()

    extra = self._get_empty_query_stats_dict()
    if (
        not self._active_query.stats_reported
        and self._active_query.horizon.chunks_received > 0
    ):
      if self._active_query.stats_dict is None:
        self._record_query_stats(self._active_query)
      if self._active_query.stats_dict is not None:
        extra = dict(self._active_query.stats_dict)
      self._active_query.stats_reported = True

    extra['inference_sent'] = np.array(inference_sent, dtype=np.uint8)
    extra['actions_left'] = np.array(
        self._active_query.horizon.available_actions_count, dtype=np.int32
    )
    extra['action_chunk'] = self._active_query.horizon.snapshot()

    self._last_extra = extra
    return action

  def _issue_query(self, observation: dict[str, np.ndarray]) -> _InFlightQuery:
    """Puts a query on the wire and returns its `_InFlightQuery` bundle."""
    started_at_s = time.perf_counter()
    return _InFlightQuery(
        horizon=self._model.query_model(observation),
        started_at_s=started_at_s,
    )

  def _record_query_stats(self, query: _InFlightQuery) -> None:
    """Records the latency of `query` as measured up to now."""
    stats = self._build_query_stats(
        (time.perf_counter() - query.started_at_s) * 1000.0
    )
    query.stats_dict = stats
    self._log_inference_timing(stats)

  def _cleanup_previous_state(self) -> None:
    """Drops any queries left over from the execution that just ended."""
    self._active_query = None
    self._pending_query = None

  def _build_query_stats(
      self, query_duration_ms: float
  ) -> dict[str, np.ndarray]:
    """Returns the latency metrics for the query that just produced a chunk.

    Pure: performs no logging and mutates no state, so it is safe to call
    repeatedly as chunks stream in.

    Args:
      query_duration_ms: Wall-clock time since the query was issued.

    Returns:
      A dict keyed by `_LATENCY_METRIC_KEYS`. Metrics the backend did not
      report are filled with the -1.0 sentinel.
    """

    def _to_arr(value: float | None) -> np.ndarray:
      return np.array(-1.0 if value is None else value, dtype=np.float32)

    telemetry = self._model.get_telemetry()
    return {
        'inference_total_ms': np.array(query_duration_ms, dtype=np.float32),
        'remote_inference_ms': _to_arr(
            telemetry.remote_inference_ms if telemetry else None
        ),
        'client_wire_transit_ms': _to_arr(
            telemetry.wire_transit_ms if telemetry else None
        ),
        'client_processing_ms': _to_arr(
            telemetry.client_processing_ms if telemetry else None
        ),
        'client_image_encode_ms': _to_arr(
            telemetry.client_image_encode_ms if telemetry else None
        ),
        'network_overhead_ms': _to_arr(
            telemetry.network_overhead_ms if telemetry else None
        ),
    }

  def _log_inference_timing(self, stats: dict[str, np.ndarray]) -> None:
    """Logs one summary line for a completed query.

    Args:
      stats: A dict as returned by `_build_query_stats()`.
    """
    image_encode_ms = float(stats['client_image_encode_ms'])
    logging.info(
        'Inference: total=%.1fms, remote=%.1fms, wire=%.1fms,'
        ' client_proc=%.1fms (img_encode=%s)',
        float(stats['inference_total_ms']),
        float(stats['remote_inference_ms']),
        float(stats['client_wire_transit_ms']),
        float(stats['client_processing_ms']),
        'N/A' if image_encode_ms < 0.0 else f'{image_encode_ms:.1f}ms',
    )

  def _get_empty_query_stats_dict(self) -> dict[str, np.ndarray]:
    """Returns a dictionary with sentinel inference latency metrics."""
    sentinel = np.array(-1.0, dtype=np.float32)
    return {key: sentinel for key in _LATENCY_METRIC_KEYS}

  def _get_empty_extra_dict(self) -> dict[str, np.ndarray]:
    """Returns a dictionary with sentinel extra output metrics."""
    extra = self._get_empty_query_stats_dict()
    extra['inference_sent'] = np.array(0, dtype=np.uint8)
    extra['actions_left'] = np.array(0, dtype=np.int32)
    return extra

  @property
  def episode_statistics(self) -> EpisodeStatistics:
    return copy.deepcopy(self._episode_statistics)

  def close(self) -> None:
    """Closes the policy resources and underlying model interface."""
    self._cleanup_previous_state()
    self._model.close()

  def __del__(self) -> None:
    # Releases only this policy's own resources. Unlike `close()`, it must not
    # close the model, which the caller may still own and use.
    try:
      self._cleanup_previous_state()
    except AttributeError:
      # `__init__` can reject its arguments before assigning any attribute,
      # and the resulting partially constructed object is still finalized.
      pass
