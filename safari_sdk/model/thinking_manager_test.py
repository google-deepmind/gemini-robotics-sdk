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

import threading
import time
from unittest import mock

import dm_env
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized
from safari_sdk.model import constants
from safari_sdk.model import genai_robotics
from safari_sdk.model import thinking_manager


class ThinkingManagerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_client = mock.Mock(spec=genai_robotics.Client)
    mock_models = mock.Mock()
    mock_models.generate_content.return_value = mock.Mock(
        text='{"text": "mocked thinking"}'
    )
    self.mock_client.models = mock_models

    self.task_instruction_key = "instruction"
    self.image_keys = ["image"]
    self.proprio_keys = ["proprio"]

    self.observation = {
        self.task_instruction_key: np.array("do the task"),
        self.image_keys[0]: np.zeros((64, 64, 3), dtype=np.uint8),
        self.proprio_keys[0]: np.array([1.0, 2.0]),
    }
    self.timestep = dm_env.TimeStep(
        step_type=dm_env.StepType.FIRST,
        reward=0.0,
        discount=1.0,
        observation=self.observation,
    )

  def _create_manager(
      self,
      thinking_strategy: thinking_manager.ThinkingStrategy,
  ) -> thinking_manager.ThinkingManager:
    with mock.patch.object(genai_robotics, "Client") as mock_client_constructor:
      mock_client_constructor.return_value = self.mock_client
      return thinking_manager.ThinkingManager(
          thinking_strategy=thinking_strategy,
          thinking_serve_id="test_serve_id",
          robotics_api_connection=constants.RoboticsApiConnectionType.CLOUD,
          task_instruction_key=self.task_instruction_key,
          image_observation_keys=self.image_keys,
          proprioceptive_observation_keys=self.proprio_keys,
      )

  def _create_multi_manager(
      self,
      thinking_strategies: list[thinking_manager.ThinkingStrategy],
  ) -> thinking_manager.MultiThinkingManager:
    return thinking_manager.MultiThinkingManager(
        thinking_strategies=thinking_strategies,
        thinking_serve_id="test_serve_id",
        robotics_api_connection=constants.RoboticsApiConnectionType.CLOUD,
        task_instruction_key=self.task_instruction_key,
        image_observation_keys=self.image_keys,
        proprioceptive_observation_keys=self.proprio_keys,
    )

  def test_synchronous_thinking_at_start(self):
    """Verify synchronous thinking happens on the first step."""
    manager = self._create_manager(
        thinking_manager.ThinkingStrategy.THINK_START_OF_CHUNK_SYNCHRONOUS
    )

    with mock.patch.object(manager, "_should_think", return_value=True) as sm:
      obs = manager.get_additional_observations(
          self.timestep, should_replan=False
      )
      sm.assert_called_once_with(True, False)

    self.mock_client.models.generate_content.assert_called_once()
    self.assertEqual(obs[thinking_manager.THINKING_KEY], "mocked thinking")

  def test_asynchronous_thinking_flow(self):
    """Test the asynchronous thinking cycle."""
    manager = self._create_manager(
        thinking_manager.ThinkingStrategy.THINK_START_OF_CHUNK_ASYNCHRONOUS
    )

    # First call should be synchronous because it's the first time we call it.
    obs = manager.get_additional_observations(self.timestep, should_replan=True)
    self.assertEqual(obs[thinking_manager.THINKING_KEY], "mocked thinking")
    self.mock_client.models.generate_content.assert_called_once()

    # Set up for second call.
    event_started = threading.Event()
    event_continue = threading.Event()
    started_event = threading.Event()

    def wait_and_return(*args, **kwargs):
      del args, kwargs  # Unused.
      started_event.set()
      event_started.set()
      event_continue.wait()
      return mock.Mock(text='{"text": "new mocked thinking"}')

    self.mock_client.models.generate_content.side_effect = wait_and_return
    self.timestep = self.timestep._replace(step_type=dm_env.StepType.MID)

    # Second call. Asynchronous.
    with mock.patch.object(manager, "_should_think", return_value=True):
      obs = manager.get_additional_observations(
          self.timestep, should_replan=True
      )
    started_event.wait(timeout=5.0)
    self.assertEqual(
        obs[thinking_manager.THINKING_KEY], "mocked thinking"
    )  # returns stale thinking

    self.assertTrue(event_started.wait(timeout=5.0))
    self.assertEqual(self.mock_client.models.generate_content.call_count, 2)
    future = manager._thinking_future
    self.assertIsNotNone(future)
    self.assertTrue(future.running())

    # Allow thread to complete and wait for it
    event_continue.set()
    future.result()  # this blocks until done.

    # Third call.
    obs = manager.get_additional_observations(
        self.timestep, should_replan=False
    )
    self.assertEqual(obs[thinking_manager.THINKING_KEY], "new mocked thinking")
    self.assertEqual(
        self.mock_client.models.generate_content.call_count, 2
    )  # no new call

  def test_thinking_progress_flow(self):
    """Test the THINKING_PROGRESS thinking cycle."""
    manager = self._create_manager(
        thinking_manager.ThinkingStrategy.THINKING_PROGRESS
    )

    # First call should be synchronous because it's the first time we call it.
    obs = manager.get_additional_observations(
        self.timestep, should_replan=False)
    self.assertEqual(obs[thinking_manager.THINKING_KEY], "mocked thinking")
    self.mock_client.models.generate_content.assert_called_once()

    # Set up for subsequent calls.
    event_2_started = threading.Event()
    event_2_continue = threading.Event()
    event_3_started = threading.Event()

    side_effect_calls = [0]

    def dynamic_side_effect(*args, **kwargs):
      del args, kwargs
      side_effect_calls[0] += 1
      if side_effect_calls[0] == 1:
        event_2_started.set()
        event_2_continue.wait()
        return mock.Mock(text='{"text": "new mocked thinking"}')
      elif side_effect_calls[0] == 2:
        event_3_started.set()
        return mock.Mock(text='{"text": "third mocked thinking"}')
      return mock.Mock(text='{"text": "default"}')

    self.mock_client.models.generate_content.side_effect = dynamic_side_effect
    self.timestep = self.timestep._replace(step_type=dm_env.StepType.MID)

    # Second call. Asynchronous.
    obs = manager.get_additional_observations(
        self.timestep, should_replan=False
    )
    self.assertEqual(
        obs[thinking_manager.THINKING_KEY], "mocked thinking"
    )  # returns stale thinking

    # Wait for the second call to at least start in the background thread
    self.assertTrue(event_2_started.wait(timeout=5.0))
    self.assertEqual(self.mock_client.models.generate_content.call_count, 2)

    future = manager._thinking_future
    self.assertIsNotNone(future)
    self.assertTrue(future.running())

    # Allow 2nd call to complete
    event_2_continue.set()
    future.result()  # wait for it to finish

    # Third call. It should collect the result and trigger a new one.
    obs = manager.get_additional_observations(
        self.timestep, should_replan=False
    )
    self.assertEqual(obs[thinking_manager.THINKING_KEY], "new mocked thinking")

    # Wait for the third call to start in the background thread
    self.assertTrue(event_3_started.wait(timeout=5.0))
    self.assertEqual(
        self.mock_client.models.generate_content.call_count, 3
    )
    self.assertIsNotNone(manager._thinking_future)

  def test_no_thinking_when_strategy_is_none(self):
    """Ensure no thinking occurs when the strategy is NONE."""
    manager = self._create_manager(thinking_manager.ThinkingStrategy.NONE)
    obs = manager.get_additional_observations(self.timestep, should_replan=True)
    self.assertEmpty(obs)
    self.mock_client.models.generate_content.assert_not_called()

  def test_get_additional_observations_spec(self):
    """Tests the spec for additional observations."""
    # With thinking enabled.
    manager = self._create_manager(
        thinking_manager.ThinkingStrategy.THINK_START_OF_CHUNK_SYNCHRONOUS
    )
    spec = manager.get_additional_observations_spec()
    self.assertIn(thinking_manager.THINKING_KEY, spec)
    self.assertEqual(
        spec[thinking_manager.THINKING_KEY], dm_env.specs.StringArray(shape=())
    )

    # With thinking disabled.
    manager = self._create_manager(thinking_manager.ThinkingStrategy.NONE)
    spec = manager.get_additional_observations_spec()
    self.assertEmpty(spec)

  def test_should_think_logic(self):
    """Test the _should_think logic for various scenarios."""
    manager = self._create_manager(
        thinking_manager.ThinkingStrategy.THINK_EVERY_3_SECONDS_SYNCHRONOUS
    )
    manager._last_thinking_time = time.time()
    self.assertFalse(manager._should_think(False, should_replan=False))
    self.assertTrue(manager._should_think(True, should_replan=False))

    manager._last_thinking_time = time.time() - 4
    self.assertTrue(manager._should_think(False, should_replan=False))

  @parameterized.named_parameters(
      dict(
          testcase_name="motion_description",
          strategy=thinking_manager.ThinkingStrategy.THINK_MOTION_DESCRIPTION_START_OF_CHUNK_SYNCHRONOUS,
          expected_method_name="generate_motion_description",
      ),
      dict(
          testcase_name="next_step",
          strategy=thinking_manager.ThinkingStrategy.THINK_NEXT_STEP_START_OF_CHUNK_SYNCHRONOUS,
          expected_method_name="generate_next_step",
      ),
      dict(
          testcase_name="generate",
          strategy=thinking_manager.ThinkingStrategy.THINK_START_OF_CHUNK_SYNCHRONOUS,
          expected_method_name="generate",
      ),
  )
  def test_thinking_manager_client_method_name(
      self, strategy, expected_method_name
  ):
    with mock.patch.object(genai_robotics, "Client") as mock_client_constructor:
      mock_client_constructor.return_value = self.mock_client
      thinking_manager.ThinkingManager(
          thinking_strategy=strategy,
          thinking_serve_id="test_serve_id",
          robotics_api_connection=constants.RoboticsApiConnectionType.CLOUD,
          task_instruction_key=self.task_instruction_key,
          image_observation_keys=self.image_keys,
          proprioceptive_observation_keys=self.proprio_keys,
      )
      mock_client_constructor.assert_called_once_with(
          robotics_api_connection=constants.RoboticsApiConnectionType.CLOUD,
          method_name=expected_method_name,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="motion_description_sync",
          strategy=thinking_manager.ThinkingStrategy.THINK_MOTION_DESCRIPTION_START_OF_CHUNK_SYNCHRONOUS,
          is_async=False,
          think_every_chunk=True,
          think_every_n_seconds=False,
          method_name="generate_motion_description",
      ),
      dict(
          testcase_name="motion_description_async",
          strategy=thinking_manager.ThinkingStrategy.THINK_MOTION_DESCRIPTION_START_OF_CHUNK_ASYNCHRONOUS,
          is_async=True,
          think_every_chunk=True,
          think_every_n_seconds=False,
          method_name="generate_motion_description",
      ),
      dict(
          testcase_name="next_step_sync",
          strategy=thinking_manager.ThinkingStrategy.THINK_NEXT_STEP_START_OF_CHUNK_SYNCHRONOUS,
          is_async=False,
          think_every_chunk=True,
          think_every_n_seconds=False,
          method_name="generate_next_step",
      ),
      dict(
          testcase_name="next_step_async",
          strategy=thinking_manager.ThinkingStrategy.THINK_NEXT_STEP_START_OF_CHUNK_ASYNCHRONOUS,
          is_async=True,
          think_every_chunk=True,
          think_every_n_seconds=False,
          method_name="generate_next_step",
      ),
      dict(
          testcase_name="motion_description_3_sec_sync",
          strategy=thinking_manager.ThinkingStrategy.THINK_MOTION_DESCRIPTION_EVERY_3_SECONDS_SYNCHRONOUS,
          is_async=False,
          think_every_chunk=False,
          think_every_n_seconds=True,
          think_interval_seconds=3,
          method_name="generate_motion_description",
      ),
      dict(
          testcase_name="motion_description_3_sec_async",
          strategy=thinking_manager.ThinkingStrategy.THINK_MOTION_DESCRIPTION_EVERY_3_SECONDS_ASYNCHRONOUS,
          is_async=True,
          think_every_chunk=False,
          think_every_n_seconds=True,
          think_interval_seconds=3,
          method_name="generate_motion_description",
      ),
      dict(
          testcase_name="next_step_5_sec_sync",
          strategy=thinking_manager.ThinkingStrategy.THINK_NEXT_STEP_EVERY_5_SECONDS_SYNCHRONOUS,
          is_async=False,
          think_every_chunk=False,
          think_every_n_seconds=True,
          think_interval_seconds=5,
          method_name="generate_next_step",
      ),
      dict(
          testcase_name="next_step_5_sec_async",
          strategy=thinking_manager.ThinkingStrategy.THINK_NEXT_STEP_EVERY_5_SECONDS_ASYNCHRONOUS,
          is_async=True,
          think_every_chunk=False,
          think_every_n_seconds=True,
          think_interval_seconds=5,
          method_name="generate_next_step",
      ),
      dict(
          testcase_name="thinking_progress",
          strategy=thinking_manager.ThinkingStrategy.THINKING_PROGRESS,
          is_async=True,
          think_every_chunk=False,
          think_every_n_seconds=False,
          think_every_step=True,
          method_name="generate_next_step",
      ),
  )
  def test_thinking_strategy_properties(
      self,
      strategy: thinking_manager.ThinkingStrategy,
      is_async: bool,
      think_every_chunk: bool,
      think_every_n_seconds: bool,
      method_name: str,
      think_interval_seconds: int = -1,
      think_every_step: bool = False,
  ):
    self.assertEqual(strategy.is_asynchronous, is_async)
    self.assertEqual(strategy.think_every_chunk, think_every_chunk)
    self.assertEqual(strategy.think_every_n_seconds, think_every_n_seconds)
    self.assertEqual(strategy.think_every_step, think_every_step)
    self.assertEqual(strategy.method_name, method_name)
    if think_every_n_seconds:
      self.assertEqual(strategy.think_interval_seconds, think_interval_seconds)

  def test_multi_thinking_manager_orders_and_concatenates(self):
    strat1 = (
        thinking_manager.ThinkingStrategy.THINK_MOTION_DESCRIPTION_START_OF_CHUNK_SYNCHRONOUS
    )
    strat2 = (
        thinking_manager.ThinkingStrategy.THINK_NEXT_STEP_START_OF_CHUNK_SYNCHRONOUS
    )

    mock_client_motion = mock.Mock(spec=genai_robotics.Client)
    mock_models_motion = mock.Mock()
    mock_models_motion.generate_content.return_value = mock.Mock(
        text='{"text": "motion_description: motion description"}'
    )
    mock_client_motion.models = mock_models_motion

    mock_client_step = mock.Mock(spec=genai_robotics.Client)
    mock_models_step = mock.Mock()
    mock_models_step.generate_content.return_value = mock.Mock(
        text='{"text": "next_step: next step"}'
    )
    mock_client_step.models = mock_models_step

    def client_side_effect(*args, **kwargs):
      del args
      if kwargs["method_name"] == "generate_motion_description":
        return mock_client_motion
      elif kwargs["method_name"] == "generate_next_step":
        return mock_client_step
      return mock.Mock()

    with mock.patch.object(genai_robotics, "Client") as mock_client_constructor:
      mock_client_constructor.side_effect = client_side_effect
      manager = self._create_multi_manager([strat1, strat2])
      obs = manager.get_additional_observations(self.timestep, False)

    self.assertEqual(
        obs[thinking_manager.THINKING_KEY],
        "next_step: next step motion_description: motion description",
    )
    mock_models_motion.generate_content.assert_called_once()
    mock_models_step.generate_content.assert_called_once()

  def test_multi_thinking_manager_reset(self):
    strat1 = (
        thinking_manager.ThinkingStrategy.THINK_MOTION_DESCRIPTION_START_OF_CHUNK_SYNCHRONOUS
    )
    strat2 = (
        thinking_manager.ThinkingStrategy.THINK_NEXT_STEP_START_OF_CHUNK_SYNCHRONOUS
    )

    with mock.patch.object(genai_robotics, "Client"):
      manager = self._create_multi_manager([strat1, strat2])
      for m in manager._managers:
        m.reset = mock.Mock()
      manager.reset()
      manager._managers[0].reset.assert_called_once()
      manager._managers[1].reset.assert_called_once()

  def test_multi_thinking_manager_spec(self):
    strat1 = (
        thinking_manager.ThinkingStrategy.THINK_MOTION_DESCRIPTION_START_OF_CHUNK_SYNCHRONOUS
    )
    with mock.patch.object(genai_robotics, "Client"):
      manager = self._create_multi_manager([strat1])
      spec = manager.get_additional_observations_spec()
      self.assertIn(thinking_manager.THINKING_KEY, spec)
      self.assertEqual(
          spec[thinking_manager.THINKING_KEY],
          dm_env.specs.StringArray(shape=()),
      )

  def test_multi_thinking_manager_no_strategy(self):
    with self.assertRaises(ValueError):
      self._create_multi_manager([])

    with self.assertRaises(ValueError):
      self._create_multi_manager([thinking_manager.ThinkingStrategy.NONE])

  def test_multi_thinking_manager_next_step_feeds_motion_description(self):
    strat_md = (
        thinking_manager.ThinkingStrategy.THINK_MOTION_DESCRIPTION_START_OF_CHUNK_SYNCHRONOUS
    )
    strat_ns = (
        thinking_manager.ThinkingStrategy.THINK_NEXT_STEP_START_OF_CHUNK_SYNCHRONOUS
    )

    mock_client_md = mock.Mock(spec=genai_robotics.Client)
    mock_models_md = mock.Mock()
    mock_models_md.generate_content.return_value = mock.Mock(
        text='{"text": "motion description: md_output"}'
    )
    mock_client_md.models = mock_models_md

    mock_client_ns = mock.Mock(spec=genai_robotics.Client)
    mock_models_ns = mock.Mock()
    mock_models_ns.generate_content.return_value = mock.Mock(
        text='{"text": "next step: ns_output"}'
    )
    mock_client_ns.models = mock_models_ns

    def client_side_effect(*args, **kwargs):
      del args
      if (
          kwargs["method_name"]
          == thinking_manager.MethodNames.MOTION_DESCRIPTION
      ):
        return mock_client_md
      elif kwargs["method_name"] == thinking_manager.MethodNames.NEXT_STEP:
        return mock_client_ns
      return mock.Mock()

    with mock.patch.object(
        genai_robotics, "Client"
    ) as mock_client_constructor, mock.patch.object(
        thinking_manager.observation_to_model_query_contents,
        "observation_to_model_query_contents",
    ) as mock_obs_to_contents:
      mock_client_constructor.side_effect = client_side_effect
      manager = self._create_multi_manager([strat_md, strat_ns])
      manager.get_additional_observations(self.timestep, False)

      self.assertEqual(
          manager._managers[0].thinking_strategy.method_name,
          thinking_manager.MethodNames.NEXT_STEP,
      )
      self.assertEqual(
          manager._managers[1].thinking_strategy.method_name,
          thinking_manager.MethodNames.MOTION_DESCRIPTION,
      )

      # The first call to observation_to_model_query_contents should be for
      # next_step, with original instruction.
      # The second call should be for motion_description, with instruction
      # from next_step manager.
      self.assertEqual(mock_obs_to_contents.call_count, 2)
      kwargs_ns = mock_obs_to_contents.call_args_list[0][1]
      kwargs_md = mock_obs_to_contents.call_args_list[1][1]

      np.testing.assert_array_equal(
          kwargs_ns["observation"][self.task_instruction_key],
          np.array("do the task"),
      )
      # motion description should be called with "ns_output" as instruction
      np.testing.assert_array_equal(
          kwargs_md["observation"][self.task_instruction_key],
          np.array("ns_output"),
      )


if __name__ == "__main__":
  absltest.main()
