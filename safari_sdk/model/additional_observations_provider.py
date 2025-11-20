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

"""Interface for adding additional observations to a timestep."""

import abc

import dm_env
from dm_env import specs
import numpy as np


class AdditionalObservationsProvider(abc.ABC):
  """Abstract class for adding new observations to the existing timestep."""

  @abc.abstractmethod
  def get_additional_observations(
      self, timestep: dm_env.TimeStep, should_replan: bool
  ) -> dict[str, np.ndarray]:
    """Returns a dictionary of additional observations."""

  @abc.abstractmethod
  def get_additional_observations_spec(self) -> dict[str, specs.Array]:
    """Returns the spec for the additional observations."""

  def reset(self) -> None:
    """Resets the internal state of the provider."""
