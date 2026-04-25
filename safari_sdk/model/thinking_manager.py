# Copyright 2026 Google LLC
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

"""Manages the thinking logic for a robotics policy."""

from collections.abc import Sequence
from concurrent import futures
import copy
import enum
import json
import logging
import time

import dm_env
from dm_env import specs
import numpy as np
from typing_extensions import override

from safari_sdk.model import additional_observations_provider
from safari_sdk.model import constants
from safari_sdk.model import genai_robotics
from safari_sdk.model import observation_to_model_query_contents

THINKING_KEY = "thinking"


# Constants for method names.
class MethodNames:
  MOTION_DESCRIPTION = "generate_motion_description"
  NEXT_STEP = "generate_next_step"
  GENERATE = "generate"


class ThinkingStrategy(enum.StrEnum):
  """How to interpret the model output.

  Synchronous means action call waits for thinking call to complete.
  """

  # No thinking
  NONE = "none"
  # Think every chunk.
  THINK_START_OF_CHUNK_SYNCHRONOUS = "think_start_of_chunk_synchronous"
  THINK_START_OF_CHUNK_ASYNCHRONOUS = "think_start_of_chunk_asynchronous"
  # Think every chunk, only for motion description.
  THINK_MOTION_DESCRIPTION_START_OF_CHUNK_SYNCHRONOUS = (
      "think_motion_description_start_of_chunk_synchronous"
  )
  THINK_MOTION_DESCRIPTION_START_OF_CHUNK_ASYNCHRONOUS = (
      "think_motion_description_start_of_chunk_asynchronous"
  )
  # Think every chunk, only for next step.
  THINK_NEXT_STEP_START_OF_CHUNK_SYNCHRONOUS = (
      "think_next_step_start_of_chunk_synchronous"
  )
  THINK_NEXT_STEP_START_OF_CHUNK_ASYNCHRONOUS = (
      "think_next_step_start_of_chunk_asynchronous"
  )

  # Think once per episode.
  THINK_START_OF_EPISODE_SYNCHRONOUS = "think_start_of_episode_synchronous"
  # Think once per episode, only for motion description.
  THINK_MOTION_DESCRIPTION_START_OF_EPISODE_SYNCHRONOUS = (
      "think_motion_description_start_of_episode_synchronous"
  )
  # Think once per episode, only for next step.
  THINK_NEXT_STEP_START_OF_EPISODE_SYNCHRONOUS = (
      "think_next_step_start_of_episode_synchronous"
  )

  # Think every 3 seconds.
  THINK_EVERY_3_SECONDS_SYNCHRONOUS = "think_every_3_seconds_synchronous"
  THINK_EVERY_3_SECONDS_ASYNCHRONOUS = "think_every_3_seconds_asynchronous"
  # Think every 3 seconds, only for motion description.
  THINK_MOTION_DESCRIPTION_EVERY_3_SECONDS_SYNCHRONOUS = (
      "think_motion_description_every_3_seconds_synchronous"
  )
  THINK_MOTION_DESCRIPTION_EVERY_3_SECONDS_ASYNCHRONOUS = (
      "think_motion_description_every_3_seconds_asynchronous"
  )
  # Think every 3 seconds, only for next step.
  THINK_NEXT_STEP_EVERY_3_SECONDS_SYNCHRONOUS = (
      "think_next_step_every_3_seconds_synchronous"
  )
  THINK_NEXT_STEP_EVERY_3_SECONDS_ASYNCHRONOUS = (
      "think_next_step_every_3_seconds_asynchronous"
  )

  # Think every 5 seconds.
  THINK_EVERY_5_SECONDS_SYNCHRONOUS = "think_every_5_seconds_synchronous"
  THINK_EVERY_5_SECONDS_ASYNCHRONOUS = "think_every_5_seconds_asynchronous"
  # Think every 5 seconds, only for motion description.
  THINK_MOTION_DESCRIPTION_EVERY_5_SECONDS_SYNCHRONOUS = (
      "think_motion_description_every_5_seconds_synchronous"
  )
  THINK_MOTION_DESCRIPTION_EVERY_5_SECONDS_ASYNCHRONOUS = (
      "think_motion_description_every_5_seconds_asynchronous"
  )
  # Think every 5 seconds, only for next step.
  THINK_NEXT_STEP_EVERY_5_SECONDS_SYNCHRONOUS = (
      "think_next_step_every_5_seconds_synchronous"
  )
  THINK_NEXT_STEP_EVERY_5_SECONDS_ASYNCHRONOUS = (
      "think_next_step_every_5_seconds_asynchronous"
  )

  # Think every 10 seconds, only for next step.
  THINK_NEXT_STEP_EVERY_10_SECONDS_SYNCHRONOUS = (
      "think_next_step_every_10_seconds_synchronous"
  )
  THINK_NEXT_STEP_EVERY_10_SECONDS_ASYNCHRONOUS = (
      "think_next_step_every_10_seconds_asynchronous"
  )

  @property
  def is_thinking_enabled(self) -> bool:
    return self != self.NONE

  @property
  def inference_mode(self) -> constants.InferenceMode:
    if self.is_asynchronous:
      return constants.InferenceMode.ASYNCHRONOUS
    return constants.InferenceMode.SYNCHRONOUS

  @property
  def is_asynchronous(self) -> bool:
    return self in (
        self.THINK_START_OF_CHUNK_ASYNCHRONOUS,
        self.THINK_EVERY_3_SECONDS_ASYNCHRONOUS,
        self.THINK_EVERY_5_SECONDS_ASYNCHRONOUS,
        self.THINK_MOTION_DESCRIPTION_START_OF_CHUNK_ASYNCHRONOUS,
        self.THINK_MOTION_DESCRIPTION_EVERY_3_SECONDS_ASYNCHRONOUS,
        self.THINK_MOTION_DESCRIPTION_EVERY_5_SECONDS_ASYNCHRONOUS,
        self.THINK_NEXT_STEP_START_OF_CHUNK_ASYNCHRONOUS,
        self.THINK_NEXT_STEP_EVERY_3_SECONDS_ASYNCHRONOUS,
        self.THINK_NEXT_STEP_EVERY_5_SECONDS_ASYNCHRONOUS,
        self.THINK_NEXT_STEP_EVERY_10_SECONDS_ASYNCHRONOUS,
    )

  @property
  def is_synchronous(self) -> bool:
    return not self.is_asynchronous

  @property
  def think_every_chunk(self) -> bool:
    return self in (
        self.THINK_START_OF_CHUNK_SYNCHRONOUS,
        self.THINK_START_OF_CHUNK_ASYNCHRONOUS,
        self.THINK_MOTION_DESCRIPTION_START_OF_CHUNK_SYNCHRONOUS,
        self.THINK_MOTION_DESCRIPTION_START_OF_CHUNK_ASYNCHRONOUS,
        self.THINK_NEXT_STEP_START_OF_CHUNK_SYNCHRONOUS,
        self.THINK_NEXT_STEP_START_OF_CHUNK_ASYNCHRONOUS,
    )

  def should_think(
      self,
      time_step: dm_env.TimeStep,
      inference_step: int,
      last_thinking_time: float,
  ) -> bool:
    """Checks if the model should think now.

    Args:
      time_step: The current time step.
      inference_step: The current inference step.
      last_thinking_time: The last time the model thought.

    Returns:
      True if the model should think, False otherwise.
    """
    if self == self.NONE:
      return False
    # If the first step, always think.
    if time_step.first():
      return True
    # If think_every_chunk, think when inference_step is 0.
    if self.think_every_chunk and inference_step == 0:
      return True
    # If think_every_n_seconds, think when n seconds have passed.
    # Note: Thinking is not triggered at the start of an action chunk.
    if (
        self.think_every_n_seconds
        and time.time() - last_thinking_time > self.think_interval_seconds
    ):
      return True
    return False

  @property
  def think_every_n_seconds(self) -> bool:
    return self in (
        self.THINK_EVERY_3_SECONDS_ASYNCHRONOUS,
        self.THINK_EVERY_3_SECONDS_SYNCHRONOUS,
        self.THINK_EVERY_5_SECONDS_ASYNCHRONOUS,
        self.THINK_EVERY_5_SECONDS_SYNCHRONOUS,
        self.THINK_MOTION_DESCRIPTION_EVERY_3_SECONDS_ASYNCHRONOUS,
        self.THINK_MOTION_DESCRIPTION_EVERY_3_SECONDS_SYNCHRONOUS,
        self.THINK_MOTION_DESCRIPTION_EVERY_5_SECONDS_ASYNCHRONOUS,
        self.THINK_MOTION_DESCRIPTION_EVERY_5_SECONDS_SYNCHRONOUS,
        self.THINK_NEXT_STEP_EVERY_3_SECONDS_ASYNCHRONOUS,
        self.THINK_NEXT_STEP_EVERY_3_SECONDS_SYNCHRONOUS,
        self.THINK_NEXT_STEP_EVERY_5_SECONDS_ASYNCHRONOUS,
        self.THINK_NEXT_STEP_EVERY_5_SECONDS_SYNCHRONOUS,
        self.THINK_NEXT_STEP_EVERY_10_SECONDS_ASYNCHRONOUS,
        self.THINK_NEXT_STEP_EVERY_10_SECONDS_SYNCHRONOUS,
    )

  @property
  def think_interval_seconds(self) -> int:
    """Returns the thinking interval in seconds."""
    if self in (
        self.THINK_EVERY_3_SECONDS_ASYNCHRONOUS,
        self.THINK_EVERY_3_SECONDS_SYNCHRONOUS,
        self.THINK_MOTION_DESCRIPTION_EVERY_3_SECONDS_ASYNCHRONOUS,
        self.THINK_MOTION_DESCRIPTION_EVERY_3_SECONDS_SYNCHRONOUS,
        self.THINK_NEXT_STEP_EVERY_3_SECONDS_ASYNCHRONOUS,
        self.THINK_NEXT_STEP_EVERY_3_SECONDS_SYNCHRONOUS,
    ):
      return 3
    if self in (
        self.THINK_EVERY_5_SECONDS_ASYNCHRONOUS,
        self.THINK_EVERY_5_SECONDS_SYNCHRONOUS,
        self.THINK_MOTION_DESCRIPTION_EVERY_5_SECONDS_ASYNCHRONOUS,
        self.THINK_MOTION_DESCRIPTION_EVERY_5_SECONDS_SYNCHRONOUS,
        self.THINK_NEXT_STEP_EVERY_5_SECONDS_ASYNCHRONOUS,
        self.THINK_NEXT_STEP_EVERY_5_SECONDS_SYNCHRONOUS,
    ):
      return 5
    if self in (
        self.THINK_NEXT_STEP_EVERY_10_SECONDS_ASYNCHRONOUS,
        self.THINK_NEXT_STEP_EVERY_10_SECONDS_SYNCHRONOUS,
    ):
      return 10
    else:
      raise ValueError(f"No known thinking interval for: {self}")

  @property
  def thinking_order(self) -> int:
    """Returns the order of the thinking strategy."""
    match self.method_name:
      case MethodNames.MOTION_DESCRIPTION:
        return 2
      case MethodNames.NEXT_STEP:
        return 1
      case MethodNames.GENERATE:
        return 0
      case _:
        return 0

  @property
  def method_name(self) -> str:
    """Returns the method name for the thinking client."""
    if "motion_description" in self:
      return MethodNames.MOTION_DESCRIPTION
    if "next_step" in self:
      return MethodNames.NEXT_STEP
    return MethodNames.GENERATE


class ThinkingManager(
    additional_observations_provider.AdditionalObservationsProvider
):
  """Manages the thinking logic for a robotics policy."""

  def __init__(
      self,
      thinking_strategy: ThinkingStrategy,
      thinking_serve_id: str | None,
      robotics_api_connection: constants.RoboticsApiConnectionType,
      task_instruction_key: str,
      image_observation_keys: Sequence[str],
      proprioceptive_observation_keys: Sequence[str],
  ):
    """Initializes the ThinkingManager.

    Args:
      thinking_strategy: The thinking strategy to use for the policy.
      thinking_serve_id: The serve ID to use for thinking.
      robotics_api_connection: Connection type for the Robotics API.
      task_instruction_key: The key for the task instruction in the observation.
      image_observation_keys: A list of observation keys that are related to
        images.
      proprioceptive_observation_keys: The list of observation keys that are
        related to proprioceptive sensors (e.g. joints).
    """
    self._thinking_strategy = thinking_strategy
    if self._thinking_strategy.is_thinking_enabled and not thinking_serve_id:
      raise ValueError(
          "thinking_serve_id must be provided if thinking_strategy is not NONE."
      )
    self._thinking_serve_id = thinking_serve_id
    self._task_instruction_key = task_instruction_key
    self._image_observation_keys = image_observation_keys
    self._proprioceptive_observation_keys = proprioceptive_observation_keys

    self._thinking_client = genai_robotics.Client(
        robotics_api_connection=robotics_api_connection,
        method_name=self._thinking_strategy.method_name,
    )
    self._last_thinking_response_value: str | None = None
    self._last_thinking_time: float = 0.0
    self._thinking_future: futures.Future[str] | None = None
    self._thinking_executor = None
    if self._thinking_strategy.is_asynchronous:
      self._thinking_executor = futures.ThreadPoolExecutor(max_workers=1)

  @property
  def last_thinking_response(self) -> str | None:
    if self._last_thinking_response_value is None:
      return None
    return self._last_thinking_response_value.replace("<ctrl95>", "")

  @property
  def last_thinking_time(self) -> float:
    """Returns the timestamp of the last thinking query."""
    return self._last_thinking_time

  @override
  def get_additional_observations(
      self, timestep: dm_env.TimeStep, should_replan: bool
  ) -> dict[str, np.ndarray]:
    """Returns a dictionary of additional observations."""
    if self._thinking_strategy == ThinkingStrategy.NONE:
      return {}
    if self._thinking_strategy.is_thinking_enabled:
      try:
        self._manage_thinking_cycle(timestep.observation, should_replan)
      except Exception as e:  # pylint: disable=broad-except
        logging.exception(
            "Failed to manage thinking cycle: %s, using the last thinking"
            " response.",
            e,
        )
    return {THINKING_KEY: np.array(self.last_thinking_response)}

  @override
  def get_additional_observations_spec(self) -> dict[str, specs.Array]:
    if self._thinking_strategy == ThinkingStrategy.NONE:
      return {}
    return {THINKING_KEY: specs.StringArray(shape=())}

  @override
  def reset(self):
    """Resets the thinking manager."""
    self._last_thinking_response_value = None
    self._last_thinking_time = 0.0
    if self._thinking_future and self._thinking_future.running():
      self._thinking_future.cancel()
    self._thinking_future = None

  def _manage_thinking_cycle(
      self,
      observation: dict[str, np.ndarray],
      is_replan: bool,
  ) -> None:
    """Manages the synchronous or asynchronous thinking logic.

    Args:
      observation: The dictionary of observations.
      is_replan: Whether the policy is replanning.
    """
    # For async strategies, first check if a thinking query returns.
    if self._thinking_strategy.is_asynchronous and self._thinking_future:
      if self._thinking_future.done():
        try:
          self._last_thinking_response_value = self._thinking_future.result()
        except Exception as e:  # pylint: disable=broad-except
          logging.exception("Asynchronous thinking query failed: %s", e)
        finally:
          self._thinking_future = None

    if self._should_think(self._is_first_step, is_replan):
      self._last_thinking_time = time.time()

      # The first call is always synchronous.
      if self._thinking_strategy.is_synchronous or self._is_first_step:
        self._last_thinking_response_value = self._query_thinking(observation)
        return

      if self._thinking_future is None:
        assert self._thinking_executor is not None
        self._thinking_future = self._thinking_executor.submit(
            self._query_thinking, copy.deepcopy(observation)
        )

  @property
  def _is_first_step(self) -> bool:
    return self._last_thinking_response_value is None

  @property
  def thinking_strategy(self) -> ThinkingStrategy:
    return self._thinking_strategy

  def _should_think(self, is_first_step: bool, should_replan: bool) -> bool:
    """Checks if the model should think now."""
    if not self._thinking_strategy.is_thinking_enabled:
      return False
    # If the first step, always think.
    if is_first_step:
      return True
    # If think_every_chunk, think when it's time to replan.
    if self._thinking_strategy.think_every_chunk and should_replan:
      return True
    # If think_every_n_seconds, think when n seconds have passed.
    if (
        self._thinking_strategy.think_every_n_seconds
        and time.time() - self._last_thinking_time
        > self._thinking_strategy.think_interval_seconds
    ):
      return True
    return False

  def _query_thinking(self, observation: dict[str, np.ndarray]) -> str:
    """Queries the model for thinking."""
    contents = observation_to_model_query_contents.observation_to_model_query_contents(
        observation=observation,
        string_observations_keys=[self._task_instruction_key],
        task_instruction_key=self._task_instruction_key,
        proprioceptive_observation_keys=self._proprioceptive_observation_keys,
        image_observation_keys=self._image_observation_keys,
    )
    response = self._thinking_client.models.generate_content(
        model=self._thinking_serve_id,
        contents=contents,
    )
    return json.loads(response.text)["text"]


class MultiThinkingManager(
    additional_observations_provider.AdditionalObservationsProvider
):
  """Manages the thinking logic for a robotics policy."""

  def __init__(
      self,
      thinking_strategies: Sequence[ThinkingStrategy],
      thinking_serve_id: str | None,
      robotics_api_connection: constants.RoboticsApiConnectionType,
      task_instruction_key: str,
      image_observation_keys: Sequence[str],
      proprioceptive_observation_keys: Sequence[str],
  ):
    """Initializes the MultiThinkingManager.

    Note: We use the same `thinking_serve_id` for all thinking strategies for
    now.

    Args:
      thinking_strategies: A sequence of thinking strategies to use.
      thinking_serve_id: The serve ID to use for all thinking strategies.
      robotics_api_connection: Connection type for the Robotics API.
      task_instruction_key: The key for the task instruction in the observation.
      image_observation_keys: A list of observation keys that are related to
        images.
      proprioceptive_observation_keys: The list of observation keys that are
        related to proprioceptive sensors (e.g. joints).
    """
    self._task_instruction_key = task_instruction_key
    self._managers = [
        ThinkingManager(
            thinking_strategy=strategy,
            thinking_serve_id=thinking_serve_id,
            robotics_api_connection=robotics_api_connection,
            task_instruction_key=task_instruction_key,
            image_observation_keys=image_observation_keys,
            proprioceptive_observation_keys=proprioceptive_observation_keys,
        )
        for strategy in thinking_strategies
        if strategy.is_thinking_enabled
    ]
    if not self._managers:
      raise ValueError("At least one thinking strategy must be provided.")
    self._managers.sort(key=lambda m: m.thinking_strategy.thinking_order)
    self._latest_next_step: np.ndarray | None = None
    self._last_thinking_response_value: str | None = None

  def _remove_next_step_prefix(self, next_step: np.ndarray) -> np.ndarray:
    """Formats the next step string to remove the prefix."""
    next_step = next_step.item()
    return np.array(next_step.replace("next step:", "").strip(), dtype=np.dtypes.StringDType())  # pytype: disable=module-attr

  def _get_thinking_observation_for_manager(
      self,
      manager: ThinkingManager,
      timestep: dm_env.TimeStep,
      should_replan: bool,
  ) -> np.ndarray:
    """Gets the thinking observation for a single manager, handling instruction updates.

    Args:
      manager: The ThinkingManager instance.
      timestep: The current dm_env.TimeStep.
      should_replan: Whether the policy is replanning.

    Returns:
      A tuple containing the thinking order and the thinking observation.
    """
    current_timestep = copy.deepcopy(timestep)
    if (
        manager.thinking_strategy.method_name == MethodNames.MOTION_DESCRIPTION
        and self._latest_next_step is not None
    ):
      observation = copy.deepcopy(timestep.observation)
      observation[self._task_instruction_key] = self._latest_next_step
      current_timestep = timestep._replace(observation=observation)

    thinking_obs = manager.get_additional_observations(
        current_timestep, should_replan
    )[THINKING_KEY]

    # If manager is next_step, save its result.
    if manager.thinking_strategy.method_name == MethodNames.NEXT_STEP:
      self._latest_next_step = self._remove_next_step_prefix(
          thinking_obs
      )

    return thinking_obs

  @override
  def get_additional_observations(
      self, timestep: dm_env.TimeStep, should_replan: bool
  ) -> dict[str, np.ndarray]:
    """Returns a dictionary of additional observations from all strategies."""
    responses_with_order = []

    for manager in self._managers:
      thinking_obs = self._get_thinking_observation_for_manager(
          manager, timestep, should_replan
      )
      order = manager.thinking_strategy.thinking_order
      responses_with_order.append((order, thinking_obs))

    sorted_responses = sorted(responses_with_order, key=lambda item: item[0])
    self._last_thinking_response_value = " ".join(
        [response.item() for _, response in sorted_responses]
    )
    return {THINKING_KEY: np.array(self._last_thinking_response_value)}

  @override
  @property
  def last_thinking_response(self) -> str | None:
    """Returns the latest thinking string."""
    return self._last_thinking_response_value

  @property
  def managers(self) -> Sequence[ThinkingManager]:
    """Returns the list of underlying thinking managers."""
    return self._managers

  @property
  def last_thinking_time(self) -> float:
    """Returns the max last_thinking_time across all managers."""
    if not self._managers:
      return 0.0
    return max(m.last_thinking_time for m in self._managers)

  @override
  def get_additional_observations_spec(self) -> dict[str, specs.Array]:
    """Returns the spec for the single aggregated thinking string."""
    return self._managers[0].get_additional_observations_spec()

  @override
  def reset(self):
    """Resets all underlying thinking managers."""
    self._latest_next_step = None
    self._last_thinking_response_value = None
    for manager in self._managers:
      manager.reset()
