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

"""A policy which is loaded from a TF SavedModel."""

import numpy as np
import tensorflow as tf

from safari_sdk.model import tf_agents_interface as tfa_interface


_MIN_SUPPORTED_API_VERSION = 0
_MAX_SUPPORTED_API_VERSION = 0


class SavedModelPolicy:
  """Policy which is loaded from a TF SavedModel."""

  def __init__(
      self,
      path: str,
      policy_type: str = 'act',
  ):
    """Initializes the evaluation policy.

    Args:
      path: The path to the directory representing the TF SavedModel.
      policy_type: The type of policy to load (currently only 'act').
    """
    self._path = path
    self._policy = None
    self._policy_state = None
    self._observation_spec = None
    self._policy_type = policy_type

  def setup(self):
    """Initializes the policy."""
    self._policy = tf.saved_model.load(self._path)

    metadata = self._policy.get_metadata()
    if 'api_version' not in metadata:
      raise RuntimeError(
          'SavedModel does not have an API version. This model must not be'
          ' compatible with this binary.'
      )
    if (
        metadata['api_version'] < _MIN_SUPPORTED_API_VERSION
        or metadata['api_version'] > _MAX_SUPPORTED_API_VERSION
    ):
      raise RuntimeError(
          f'SavedModel has unsupported API version: {metadata["api_version"]}.'
          ' This binary supports API versions between (inclusive)'
          f' {_MIN_SUPPORTED_API_VERSION} and {_MAX_SUPPORTED_API_VERSION}.'
      )

    self._observation_spec = self._policy.action.input_signature[0].observation

  @property
  def observation_spec(self) -> dict[str, tf.TensorSpec]:
    if self._policy is None:
      raise RuntimeError('Policy not yet initialized. Call setup().')

    return self._observation_spec

  def reset(self) -> dict[str, np.ndarray]:
    """Resets the policy."""
    if self._policy is None:
      raise RuntimeError('Policy not yet initialized. Call setup().')

    observation_0 = tf.nest.map_structure(
        lambda x: tf.zeros(x.shape, x.dtype),
        self.observation_spec,
    )
    self._policy_state = self._policy.get_initial_state()

    return observation_0

  def step(
      self, observation: dict[str, np.ndarray]
  ) -> tfa_interface.ActionType:
    """Computes an action from observations."""
    if self._policy is None:
      raise RuntimeError('Policy not yet initialized. Call setup().')

    tfa_step = tfa_interface.transition(
        observation, reward=np.zeros((), dtype=np.float32)
    )
    policy_step = self._policy.action(tfa_step, self._policy_state)
    self._policy_state = policy_step.state

    return policy_step.action

  @property
  def policy_type(self) -> str:
    return self._policy_type
