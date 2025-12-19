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

from collections.abc import Mapping
import glob
import io
import os
import shutil
from typing import Any, cast

import dm_env
from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
from gdm_robotics.testing import specs_utils
import numpy as np
from PIL import Image

from absl.testing import absltest
from absl.testing import parameterized
from safari_sdk.logging.python import constants
from safari_sdk.logging.python import episodic_logger
from safari_sdk.logging.python import mcap_parser_utils


_TEST_AGENT_ID = "fake_agent_id_for_test"
_TEST_TASK_ID = "fake_task_id_for_test"
_TEST_PROPRIO_KEY = "joints_pos"
_DEFAULT_NUMBER_STEPS = 3


class EpisodicLoggerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._episode_path = self.create_tempdir()  # Use create_tempdir

  def test_is_recording_is_set_correctly(self):
    timestep_spec = gdmr_types.TimeStepSpec(
        step_type=gdmr_types.STEP_TYPE_SPEC,
        reward=specs.Array(shape=(), dtype=np.float32),
        discount=specs.Array(shape=(), dtype=np.float32),
        observation={
            "instruction": specs.StringArray(shape=(), name="instruction"),
            "feature1": specs.Array(shape=(4,), dtype=np.float32),
            "feature2": specs.Array(shape=(3,), dtype=np.int32),
            "feature3": specs.Array(shape=(), dtype=np.float64),
            _TEST_PROPRIO_KEY: specs.Array(shape=(14,), dtype=np.float64),
        },
    )
    action_spec = specs.BoundedArray(
        shape=(5,),
        dtype=np.float32,
        minimum=np.array([-1.0, -2.0, -1.0, -2.0, 0.0], dtype=np.float32),
        maximum=np.array([1.0, 1.0, 2.0, 3.0, 1.0], dtype=np.float32),
    )

    logger = episodic_logger.EpisodicLogger.create(
        agent_id=_TEST_AGENT_ID,
        task_id=_TEST_TASK_ID,
        proprioceptive_observation_keys=[_TEST_PROPRIO_KEY],
        output_directory=self._episode_path.full_path,
        action_spec=action_spec,
        timestep_spec=timestep_spec,
        image_observation_keys=[],
        policy_extra_spec={},
    )

    initial_timestep = self._generate_timestep(
        timestep_spec, dm_env.StepType.FIRST
    )
    logger.reset(initial_timestep)
    self.assertTrue(logger._is_recording)

    for _ in range(_DEFAULT_NUMBER_STEPS):
      next_timestep = self._generate_timestep(
          timestep_spec, dm_env.StepType.MID
      )
      action = specs_utils.valid_value_for_spec(action_spec)

      policy_extra = {}
      logger.record_action_and_next_timestep(
          action=action, next_timestep=next_timestep, policy_extra=policy_extra
      )

    logger.write()
    self.assertFalse(logger._is_recording)

  def test_data_is_written_correctly(self):
    timestep_spec = gdmr_types.TimeStepSpec(
        step_type=gdmr_types.STEP_TYPE_SPEC,
        reward=specs.Array(shape=(), dtype=np.float32),
        discount=specs.Array(shape=(), dtype=np.float32),
        observation={
            "instruction": specs.StringArray(shape=(), name="instruction"),
            "feature1": specs.Array(shape=(4,), dtype=np.float32),
            "feature2": specs.Array(shape=(3,), dtype=np.int32),
            "feature3": specs.Array(shape=(), dtype=np.float64),
            _TEST_PROPRIO_KEY: specs.Array(shape=(14,), dtype=np.float64),
        },
    )

    action_spec = specs.BoundedArray(
        shape=(5,),
        dtype=np.float32,
        minimum=np.array([-1.0, -2.0, -1.0, -2.0, 0.0], dtype=np.float32),
        maximum=np.array([1.0, 1.0, 2.0, 3.0, 1.0], dtype=np.float32),
    )

    logger = episodic_logger.EpisodicLogger.create(
        agent_id=_TEST_AGENT_ID,
        task_id=_TEST_TASK_ID,
        proprioceptive_observation_keys=[_TEST_PROPRIO_KEY],
        output_directory=self._episode_path.full_path,
        action_spec=action_spec,
        timestep_spec=timestep_spec,
        image_observation_keys=[],
        policy_extra_spec={},
    )

    expected_timesteps = []
    expected_actions = []

    initial_timestep = self._generate_timestep(
        timestep_spec, dm_env.StepType.FIRST
    )
    expected_timesteps.append(initial_timestep)
    logger.reset(initial_timestep)

    for _ in range(_DEFAULT_NUMBER_STEPS):
      next_timestep = self._generate_timestep(
          timestep_spec, dm_env.StepType.MID
      )
      action = specs_utils.valid_value_for_spec(action_spec)

      expected_timesteps.append(next_timestep)
      expected_actions.append(action)

      policy_extra = {}
      logger.record_action_and_next_timestep(
          action=action, next_timestep=next_timestep, policy_extra=policy_extra
      )

    last_timestep = self._generate_timestep(timestep_spec, dm_env.StepType.LAST)
    expected_timesteps.append(last_timestep)
    action = specs_utils.valid_value_for_spec(action_spec)
    expected_actions.append(action)
    logger.record_action_and_next_timestep(
        action=action,
        next_timestep=last_timestep,
        policy_extra={},
    )

    logger.write()

    # Append the last action to the expected actions as the logger will pad the
    # last action with the last corresponding value.
    expected_actions.append(action)

    mcap_proto_data = mcap_parser_utils.read_proto_data(
        self._episode_path.full_path,
        constants.TIMESTEP_TOPIC_NAME,
        constants.ACTION_TOPIC_NAME,
        constants.POLICY_EXTRA_TOPIC_NAME,
    )
    timesteps_examples = mcap_proto_data.timesteps
    actions_examples = mcap_proto_data.actions
    policy_extras_examples = mcap_proto_data.policy_extra

    timesteps, actions, policy_extras = (
        mcap_parser_utils.parse_examples_to_dm_env_types(
            timestep_spec,
            action_spec,
            {},
            timesteps_examples,
            actions_examples,
            policy_extras_examples,
            constants.STEP_TYPE_KEY,
            constants.OBSERVATION_KEY_PREFIX,
            constants.REWARD_KEY,
            constants.DISCOUNT_KEY,
            constants.ACTION_KEY_PREFIX,
            constants.POLICY_EXTRA_PREFIX,
        )
    )

    for idx, _ in enumerate(timesteps):
      self._assert_timestep_is_close(timesteps[idx], expected_timesteps[idx])

    for idx, _ in enumerate(actions):
      if isinstance(actions[idx], Mapping):
        keys = actions[idx].keys()
        expected_actions_keys = cast(
            Mapping[str, np.ndarray], expected_actions[idx]
        ).keys()
        self.assertSameElements(keys, expected_actions_keys)
        for key in keys:
          np.testing.assert_allclose(
              actions[idx][key], expected_actions[idx][key]
          )
      else:
        np.testing.assert_allclose(actions[idx], expected_actions[idx])

    expected_policy_extras = [{}] * (_DEFAULT_NUMBER_STEPS + 2)
    self.assertEqual(policy_extras, expected_policy_extras)

  def test_publish_times_are_within_session_interval(self):
    timestep_spec = gdmr_types.TimeStepSpec(
        step_type=gdmr_types.STEP_TYPE_SPEC,
        reward=specs.Array(shape=(), dtype=np.float32),
        discount=specs.Array(shape=(), dtype=np.float32),
        observation={
            "feature1": specs.Array(shape=(4,), dtype=np.float32),
            "feature2": specs.Array(shape=(3,), dtype=np.int32),
            "feature3": specs.Array(shape=(), dtype=np.float64),
            "instruction": specs.StringArray(shape=(), name="instruction"),
            _TEST_PROPRIO_KEY: specs.Array(shape=(14,), dtype=np.float64),
        },
    )

    action_spec = specs.BoundedArray(
        shape=(5,),
        dtype=np.float32,
        minimum=np.array([-1.0, -2.0, -1.0, -2.0, 0.0], dtype=np.float32),
        maximum=np.array([1.0, 1.0, 2.0, 3.0, 1.0], dtype=np.float32),
    )

    logger = episodic_logger.EpisodicLogger.create(
        agent_id=_TEST_AGENT_ID,
        task_id=_TEST_TASK_ID,
        proprioceptive_observation_keys=[_TEST_PROPRIO_KEY],
        output_directory=self._episode_path.full_path,
        action_spec=action_spec,
        timestep_spec=timestep_spec,
        image_observation_keys=[],
        policy_extra_spec={},
    )

    initial_timestep = self._generate_timestep(
        timestep_spec, dm_env.StepType.FIRST
    )
    logger.reset(initial_timestep)

    for _ in range(_DEFAULT_NUMBER_STEPS):
      next_timestep = self._generate_timestep(
          timestep_spec, dm_env.StepType.MID
      )
      action = specs_utils.valid_value_for_spec(action_spec)

      policy_extra = {}
      logger.record_action_and_next_timestep(
          action=action, next_timestep=next_timestep, policy_extra=policy_extra
      )

    last_timestep = self._generate_timestep(timestep_spec, dm_env.StepType.LAST)
    action = specs_utils.valid_value_for_spec(action_spec)
    logger.record_action_and_next_timestep(
        action=action,
        next_timestep=last_timestep,
        policy_extra={},
    )

    logger.write()
    sessions = mcap_parser_utils.read_session_proto_data(
        self._episode_path.full_path, constants.SESSION_TOPIC_NAME
    )
    # There should only be one session for a single episode.
    self.assertLen(sessions, 1)
    session = sessions[0]

    # Read the raw mcap messages to get the publish times.
    # The publish times should be within the session interval.
    messages = mcap_parser_utils.read_raw_mcap_messages(
        self._episode_path.full_path, constants.TIMESTEP_TOPIC_NAME
    )

    self.assertEqual(session.interval.start_nsec, messages[0].publish_time)
    self.assertGreaterEqual(
        session.interval.stop_nsec, messages[-1].publish_time
    )

  def test_reward_is_scalar(self):
    timestep_spec = gdmr_types.TimeStepSpec(
        step_type=gdmr_types.STEP_TYPE_SPEC,
        reward=specs.Array(shape=(), dtype=np.float64),
        discount=specs.Array(shape=(), dtype=np.float32),
        observation={
            "instruction": specs.StringArray(shape=(), name="instruction"),
            "feature1": specs.Array(shape=(4,), dtype=np.float32),
            _TEST_PROPRIO_KEY: specs.Array(shape=(14,), dtype=np.float64),
        },
    )

    action_spec = specs.BoundedArray(
        shape=(5,),
        dtype=np.float32,
        minimum=np.array([-1.0, -2.0, -1.0, -2.0, 0.0], dtype=np.float32),
        maximum=np.array([1.0, 1.0, 2.0, 3.0, 1.0], dtype=np.float32),
    )

    logger = episodic_logger.EpisodicLogger.create(
        agent_id=_TEST_AGENT_ID,
        task_id=_TEST_TASK_ID,
        output_directory=self._episode_path.full_path,
        action_spec=action_spec,
        timestep_spec=timestep_spec,
        proprioceptive_observation_keys=[_TEST_PROPRIO_KEY],
        image_observation_keys=[],
        policy_extra_spec={},
    )

    expected_timesteps = []

    initial_timestep = self._generate_timestep(
        timestep_spec, dm_env.StepType.FIRST
    )
    # Override the reward to be a scalar.
    initial_timestep = initial_timestep._replace(reward=1.0)
    expected_timesteps.append(initial_timestep)
    logger.reset(initial_timestep)

    for _ in range(_DEFAULT_NUMBER_STEPS):
      next_timestep = self._generate_timestep(
          timestep_spec, dm_env.StepType.MID
      )
      # Override the reward to be a scalar.
      next_timestep = next_timestep._replace(reward=1.0)
      action = specs_utils.valid_value_for_spec(action_spec)

      expected_timesteps.append(next_timestep)

      policy_extra = {}
      logger.record_action_and_next_timestep(
          action=action, next_timestep=next_timestep, policy_extra=policy_extra
      )

    last_timestep = self._generate_timestep(timestep_spec, dm_env.StepType.LAST)
    expected_timesteps.append(last_timestep)
    logger.record_action_and_next_timestep(
        action=specs_utils.valid_value_for_spec(action_spec),
        next_timestep=last_timestep,
        policy_extra={},
    )

    logger.write()

    mcap_proto_data = mcap_parser_utils.read_proto_data(
        self._episode_path.full_path,
        constants.TIMESTEP_TOPIC_NAME,
        constants.ACTION_TOPIC_NAME,
        constants.POLICY_EXTRA_TOPIC_NAME,
    )
    timesteps_examples = mcap_proto_data.timesteps
    actions_examples = mcap_proto_data.actions
    policy_extras_examples = mcap_proto_data.policy_extra

    timesteps, _, _ = mcap_parser_utils.parse_examples_to_dm_env_types(
        timestep_spec,
        action_spec,
        {},
        timesteps_examples,
        actions_examples,
        policy_extras_examples,
        constants.STEP_TYPE_KEY,
        constants.OBSERVATION_KEY_PREFIX,
        constants.REWARD_KEY,
        constants.DISCOUNT_KEY,
        constants.ACTION_KEY_PREFIX,
        constants.POLICY_EXTRA_PREFIX,
    )

    for idx, _ in enumerate(timesteps):
      self._assert_timestep_is_close(timesteps[idx], expected_timesteps[idx])

  def test_reward_is_dict(self):
    timestep_spec = gdmr_types.TimeStepSpec(
        step_type=gdmr_types.STEP_TYPE_SPEC,
        reward={
            "reward1": specs.Array(shape=(), dtype=np.int32),
            "reward2": specs.Array(shape=(3,), dtype=np.float32),
        },
        discount=specs.Array(shape=(), dtype=np.float32),
        observation={
            "instruction": specs.StringArray(shape=(), name="instruction"),
            "feature1": specs.Array(shape=(4,), dtype=np.float32),
            _TEST_PROPRIO_KEY: specs.Array(shape=(14,), dtype=np.float64),
        },
    )

    action_spec = specs.BoundedArray(
        shape=(5,),
        dtype=np.float32,
        minimum=np.array([-1.0, -2.0, -1.0, -2.0, 0.0], dtype=np.float32),
        maximum=np.array([1.0, 1.0, 2.0, 3.0, 1.0], dtype=np.float32),
    )

    logger = episodic_logger.EpisodicLogger.create(
        agent_id=_TEST_AGENT_ID,
        task_id=_TEST_TASK_ID,
        output_directory=self._episode_path.full_path,
        action_spec=action_spec,
        timestep_spec=timestep_spec,
        proprioceptive_observation_keys=[_TEST_PROPRIO_KEY],
        image_observation_keys=[],
        policy_extra_spec={},
    )

    expected_timesteps = []

    initial_timestep = self._generate_timestep(
        timestep_spec, dm_env.StepType.FIRST
    )
    expected_timesteps.append(initial_timestep)
    logger.reset(initial_timestep)

    for _ in range(_DEFAULT_NUMBER_STEPS):
      next_timestep = self._generate_timestep(
          timestep_spec, dm_env.StepType.MID
      )
      action = specs_utils.valid_value_for_spec(action_spec)

      expected_timesteps.append(next_timestep)

      policy_extra = {}
      logger.record_action_and_next_timestep(
          action=action, next_timestep=next_timestep, policy_extra=policy_extra
      )

    last_timestep = self._generate_timestep(timestep_spec, dm_env.StepType.LAST)
    expected_timesteps.append(last_timestep)
    logger.record_action_and_next_timestep(
        action=specs_utils.valid_value_for_spec(action_spec),
        next_timestep=last_timestep,
        policy_extra={},
    )

    logger.write()

    mcap_proto_data = mcap_parser_utils.read_proto_data(
        self._episode_path.full_path,
        constants.TIMESTEP_TOPIC_NAME,
        constants.ACTION_TOPIC_NAME,
        constants.POLICY_EXTRA_TOPIC_NAME,
    )
    timesteps_examples = mcap_proto_data.timesteps
    actions_examples = mcap_proto_data.actions
    policy_extras_examples = mcap_proto_data.policy_extra

    timesteps, _, _ = mcap_parser_utils.parse_examples_to_dm_env_types(
        timestep_spec,
        action_spec,
        {},
        timesteps_examples,
        actions_examples,
        policy_extras_examples,
        constants.STEP_TYPE_KEY,
        constants.OBSERVATION_KEY_PREFIX,
        constants.REWARD_KEY,
        constants.DISCOUNT_KEY,
        constants.ACTION_KEY_PREFIX,
        constants.POLICY_EXTRA_PREFIX,
    )

    for idx, _ in enumerate(timesteps):
      self._assert_timestep_is_close(timesteps[idx], expected_timesteps[idx])

  def test_discount_is_scalar(self):
    timestep_spec = gdmr_types.TimeStepSpec(
        step_type=gdmr_types.STEP_TYPE_SPEC,
        reward=specs.Array(shape=(), dtype=np.float32),
        discount=specs.Array(shape=(), dtype=np.float32),
        observation={
            "instruction": specs.StringArray(shape=(), name="instruction"),
            "feature1": specs.Array(shape=(4,), dtype=np.float32),
            _TEST_PROPRIO_KEY: specs.Array(shape=(14,), dtype=np.float64),
        },
    )

    action_spec = specs.BoundedArray(
        shape=(5,),
        dtype=np.float32,
        minimum=np.array([-1.0, -2.0, -1.0, -2.0, 0.0], dtype=np.float32),
        maximum=np.array([1.0, 1.0, 2.0, 3.0, 1.0], dtype=np.float32),
    )

    logger = episodic_logger.EpisodicLogger.create(
        agent_id=_TEST_AGENT_ID,
        task_id=_TEST_TASK_ID,
        output_directory=self._episode_path.full_path,
        action_spec=action_spec,
        timestep_spec=timestep_spec,
        proprioceptive_observation_keys=[_TEST_PROPRIO_KEY],
        image_observation_keys=[],
        policy_extra_spec={},
    )

    expected_timesteps = []

    initial_timestep = self._generate_timestep(
        timestep_spec, dm_env.StepType.FIRST
    )
    # Override the discount to be a scalar.
    initial_timestep = initial_timestep._replace(discount=np.float32(1.0))
    expected_timesteps.append(initial_timestep)
    logger.reset(initial_timestep)

    for _ in range(_DEFAULT_NUMBER_STEPS):
      next_timestep = self._generate_timestep(
          timestep_spec, dm_env.StepType.MID
      )
      # Override the discount to be a scalar.
      next_timestep = next_timestep._replace(discount=np.float32(1.0))
      action = specs_utils.valid_value_for_spec(action_spec)

      expected_timesteps.append(next_timestep)

      policy_extra = {}
      logger.record_action_and_next_timestep(
          action=action, next_timestep=next_timestep, policy_extra=policy_extra
      )

    last_timestep = self._generate_timestep(timestep_spec, dm_env.StepType.LAST)
    expected_timesteps.append(last_timestep)
    logger.record_action_and_next_timestep(
        action=specs_utils.valid_value_for_spec(action_spec),
        next_timestep=last_timestep,
        policy_extra={},
    )

    logger.write()

    mcap_proto_data = mcap_parser_utils.read_proto_data(
        self._episode_path.full_path,
        constants.TIMESTEP_TOPIC_NAME,
        constants.ACTION_TOPIC_NAME,
        constants.POLICY_EXTRA_TOPIC_NAME,
    )
    timesteps_examples = mcap_proto_data.timesteps
    actions_examples = mcap_proto_data.actions
    policy_extras_examples = mcap_proto_data.policy_extra

    timesteps, _, _ = mcap_parser_utils.parse_examples_to_dm_env_types(
        timestep_spec,
        action_spec,
        {},
        timesteps_examples,
        actions_examples,
        policy_extras_examples,
        constants.STEP_TYPE_KEY,
        constants.OBSERVATION_KEY_PREFIX,
        constants.REWARD_KEY,
        constants.DISCOUNT_KEY,
        constants.ACTION_KEY_PREFIX,
        constants.POLICY_EXTRA_PREFIX,
    )

    for idx, _ in enumerate(timesteps):
      self._assert_timestep_is_close(timesteps[idx], expected_timesteps[idx])

  def test_discount_is_dict(self):
    timestep_spec = gdmr_types.TimeStepSpec(
        step_type=gdmr_types.STEP_TYPE_SPEC,
        reward=specs.Array(shape=(), dtype=np.float32),
        discount={
            "discount1": specs.Array(shape=(), dtype=np.int32),
            "discount2": specs.Array(shape=(3,), dtype=np.float32),
        },
        observation={
            "instruction": specs.StringArray(shape=(), name="instruction"),
            "feature1": specs.Array(shape=(4,), dtype=np.float32),
            _TEST_PROPRIO_KEY: specs.Array(shape=(14,), dtype=np.float64),
        },
    )

    action_spec = specs.BoundedArray(
        shape=(5,),
        dtype=np.float32,
        minimum=np.array([-1.0, -2.0, -1.0, -2.0, 0.0], dtype=np.float32),
        maximum=np.array([1.0, 1.0, 2.0, 3.0, 1.0], dtype=np.float32),
    )

    logger = episodic_logger.EpisodicLogger.create(
        agent_id=_TEST_AGENT_ID,
        task_id=_TEST_TASK_ID,
        output_directory=self._episode_path.full_path,
        action_spec=action_spec,
        timestep_spec=timestep_spec,
        image_observation_keys=[],
        proprioceptive_observation_keys=[_TEST_PROPRIO_KEY],
        policy_extra_spec={},
    )

    expected_timesteps = []

    initial_timestep = self._generate_timestep(
        timestep_spec, dm_env.StepType.FIRST
    )
    expected_timesteps.append(initial_timestep)
    logger.reset(initial_timestep)

    for _ in range(_DEFAULT_NUMBER_STEPS):
      next_timestep = self._generate_timestep(
          timestep_spec, dm_env.StepType.MID
      )
      action = specs_utils.valid_value_for_spec(action_spec)

      expected_timesteps.append(next_timestep)

      policy_extra = {}
      logger.record_action_and_next_timestep(
          action=action, next_timestep=next_timestep, policy_extra=policy_extra
      )

    last_timestep = self._generate_timestep(timestep_spec, dm_env.StepType.LAST)
    expected_timesteps.append(last_timestep)
    logger.record_action_and_next_timestep(
        action=specs_utils.valid_value_for_spec(action_spec),
        next_timestep=last_timestep,
        policy_extra={},
    )

    logger.write()

    mcap_proto_data = mcap_parser_utils.read_proto_data(
        self._episode_path.full_path,
        constants.TIMESTEP_TOPIC_NAME,
        constants.ACTION_TOPIC_NAME,
        constants.POLICY_EXTRA_TOPIC_NAME,
    )
    timesteps_examples = mcap_proto_data.timesteps
    actions_examples = mcap_proto_data.actions
    policy_extras_examples = mcap_proto_data.policy_extra

    timesteps, _, _ = mcap_parser_utils.parse_examples_to_dm_env_types(
        timestep_spec,
        action_spec,
        {},
        timesteps_examples,
        actions_examples,
        policy_extras_examples,
        constants.STEP_TYPE_KEY,
        constants.OBSERVATION_KEY_PREFIX,
        constants.REWARD_KEY,
        constants.DISCOUNT_KEY,
        constants.ACTION_KEY_PREFIX,
        constants.POLICY_EXTRA_PREFIX,
    )

    for idx, _ in enumerate(timesteps):
      self._assert_timestep_is_close(timesteps[idx], expected_timesteps[idx])

  def test_has_policy_extra(self):
    timestep_spec = gdmr_types.TimeStepSpec(
        step_type=gdmr_types.STEP_TYPE_SPEC,
        reward=specs.Array(shape=(), dtype=np.float32),
        discount=specs.Array(shape=(), dtype=np.float32),
        observation={
            "instruction": specs.StringArray(shape=(), name="instruction"),
            "feature1": specs.Array(shape=(4,), dtype=np.float32),
            _TEST_PROPRIO_KEY: specs.Array(shape=(14,), dtype=np.float64),
        },
    )

    action_spec = specs.BoundedArray(
        shape=(5,),
        dtype=np.float32,
        minimum=np.array([-1.0, -2.0, -1.0, -2.0, 0.0], dtype=np.float32),
        maximum=np.array([1.0, 1.0, 2.0, 3.0, 1.0], dtype=np.float32),
    )

    policy_extra_spec = {
        "extra1": specs.Array(shape=(3,), dtype=np.float32),
        "extra2": specs.Array(shape=(4,), dtype=np.float64),
    }

    logger = episodic_logger.EpisodicLogger.create(
        agent_id=_TEST_AGENT_ID,
        task_id=_TEST_TASK_ID,
        output_directory=self._episode_path.full_path,
        action_spec=action_spec,
        timestep_spec=timestep_spec,
        image_observation_keys=[],
        proprioceptive_observation_keys=[_TEST_PROPRIO_KEY],
        policy_extra_spec=policy_extra_spec,
    )

    expected_timesteps = []
    expected_actions = []
    expected_policy_extras = []

    initial_timestep = self._generate_timestep(
        timestep_spec, dm_env.StepType.FIRST
    )
    expected_timesteps.append(initial_timestep)
    logger.reset(initial_timestep)

    for _ in range(_DEFAULT_NUMBER_STEPS):
      next_timestep = self._generate_timestep(
          timestep_spec, dm_env.StepType.MID
      )
      action = specs_utils.valid_value_for_spec(action_spec)
      policy_extra = specs_utils.valid_value_for_spec(policy_extra_spec)

      expected_timesteps.append(next_timestep)
      expected_actions.append(action)
      expected_policy_extras.append(policy_extra)

      logger.record_action_and_next_timestep(
          action=action, next_timestep=next_timestep, policy_extra=policy_extra
      )

    last_timestep = self._generate_timestep(timestep_spec, dm_env.StepType.LAST)
    expected_timesteps.append(last_timestep)
    action = action_spec.generate_value()
    policy_extra = specs_utils.valid_value_for_spec(policy_extra_spec)

    logger.record_action_and_next_timestep(
        action=action,
        next_timestep=last_timestep,
        policy_extra=policy_extra,
    )
    expected_actions.append(action)
    expected_policy_extras.append(policy_extra)
    logger.write()

    # Append the action and last policy extra to the expected policy extras as
    # internally the logger will pad the last values.
    expected_actions.append(action)
    expected_policy_extras.append(policy_extra)

    mcap_proto_data = mcap_parser_utils.read_proto_data(
        self._episode_path.full_path,
        constants.TIMESTEP_TOPIC_NAME,
        constants.ACTION_TOPIC_NAME,
        constants.POLICY_EXTRA_TOPIC_NAME,
    )
    timesteps_examples = mcap_proto_data.timesteps
    actions_examples = mcap_proto_data.actions
    policy_extras_examples = mcap_proto_data.policy_extra

    timesteps, actions, policy_extras = (
        mcap_parser_utils.parse_examples_to_dm_env_types(
            timestep_spec,
            action_spec,
            policy_extra_spec,
            timesteps_examples,
            actions_examples,
            policy_extras_examples,
            constants.STEP_TYPE_KEY,
            constants.OBSERVATION_KEY_PREFIX,
            constants.REWARD_KEY,
            constants.DISCOUNT_KEY,
            constants.ACTION_KEY_PREFIX,
            constants.POLICY_EXTRA_PREFIX,
        )
    )

    for idx, _ in enumerate(timesteps):
      self._assert_timestep_is_close(timesteps[idx], expected_timesteps[idx])

    for idx, _ in enumerate(actions):
      if isinstance(actions[idx], Mapping):
        keys = actions[idx].keys()
        expected_actions_keys = cast(
            Mapping[str, np.ndarray], expected_actions[idx]
        ).keys()
        self.assertSameElements(keys, expected_actions_keys)
        for key in keys:
          np.testing.assert_allclose(
              actions[idx][key], expected_actions[idx][key]
          )
      else:
        np.testing.assert_allclose(actions[idx], expected_actions[idx])

    for idx, _ in enumerate(policy_extras):
      keys = policy_extras[idx].keys()
      expected_policy_extras_keys = cast(
          Mapping[str, Any], expected_policy_extras[idx]
      ).keys()
      self.assertSameElements(keys, expected_policy_extras_keys)
      for key in keys:
        np.testing.assert_allclose(
            policy_extras[idx][key], expected_policy_extras[idx][key]
        )

  def test_string_values_are_written_correctly(self):
    timestep_spec = gdmr_types.TimeStepSpec(
        step_type=gdmr_types.STEP_TYPE_SPEC,
        reward={
            "num_reward": specs.Array(shape=(), dtype=np.float32),
            "string_reward": specs.StringArray(shape=()),
        },
        discount={
            "num_discount": specs.Array(shape=(), dtype=np.float32),
            "string_discount": specs.StringArray(shape=()),
        },
        observation={
            "instruction": specs.StringArray(shape=(), name="instruction"),
            "string_feature": specs.StringArray(shape=()),
            "feature": specs.Array(shape=(3,), dtype=np.int32),
            _TEST_PROPRIO_KEY: specs.Array(shape=(14,), dtype=np.float64),
        },
    )

    action_spec = specs.BoundedArray(
        shape=(5,),
        dtype=np.float32,
        minimum=np.array([-1.0, -2.0, -1.0, -2.0, 0.0], dtype=np.float32),
        maximum=np.array([1.0, 1.0, 2.0, 3.0, 1.0], dtype=np.float32),
    )

    policy_extra_spec = {
        "string_extra": specs.StringArray(shape=()),
        "num_extra": specs.Array(
            shape=(),
            dtype=np.float32,
        ),
    }

    logger = episodic_logger.EpisodicLogger.create(
        agent_id=_TEST_AGENT_ID,
        task_id=_TEST_TASK_ID,
        output_directory=self._episode_path.full_path,
        action_spec=action_spec,
        timestep_spec=timestep_spec,
        image_observation_keys=[],
        proprioceptive_observation_keys=[_TEST_PROPRIO_KEY],
        policy_extra_spec=policy_extra_spec,
    )

    expected_timesteps = []
    expected_actions = []
    expected_policy_extras = []

    initial_timestep = self._generate_timestep(
        timestep_spec, dm_env.StepType.FIRST
    )
    expected_timesteps.append(initial_timestep)
    logger.reset(initial_timestep)

    for _ in range(_DEFAULT_NUMBER_STEPS):
      next_timestep = self._generate_timestep(
          timestep_spec, dm_env.StepType.MID
      )
      action = specs_utils.valid_value_for_spec(action_spec)
      policy_extra = specs_utils.valid_value_for_spec(policy_extra_spec)

      expected_timesteps.append(next_timestep)
      expected_actions.append(action)
      expected_policy_extras.append(policy_extra)

      logger.record_action_and_next_timestep(
          action=action,
          next_timestep=next_timestep,
          policy_extra=policy_extra,
      )

    last_timestep = self._generate_timestep(timestep_spec, dm_env.StepType.LAST)
    action = specs_utils.valid_value_for_spec(action_spec)
    policy_extra = specs_utils.valid_value_for_spec(policy_extra_spec)

    expected_timesteps.append(last_timestep)
    expected_actions.append(action)
    expected_policy_extras.append(policy_extra)

    logger.record_action_and_next_timestep(
        action=action,
        next_timestep=last_timestep,
        policy_extra=policy_extra,
    )

    logger.write()

    # Append the action and last policy extra to the expected policy extras as
    # internally the logger will pad the last values.
    expected_actions.append(action)
    expected_policy_extras.append(policy_extra)

    mcap_proto_data = mcap_parser_utils.read_proto_data(
        self._episode_path.full_path,
        constants.TIMESTEP_TOPIC_NAME,
        constants.ACTION_TOPIC_NAME,
        constants.POLICY_EXTRA_TOPIC_NAME,
    )
    timesteps_examples = mcap_proto_data.timesteps
    actions_examples = mcap_proto_data.actions
    policy_extras_examples = mcap_proto_data.policy_extra

    timesteps, actions, policy_extras = (
        mcap_parser_utils.parse_examples_to_dm_env_types(
            timestep_spec,
            action_spec,
            policy_extra_spec,
            timesteps_examples,
            actions_examples,
            policy_extras_examples,
            constants.STEP_TYPE_KEY,
            constants.OBSERVATION_KEY_PREFIX,
            constants.REWARD_KEY,
            constants.DISCOUNT_KEY,
            constants.ACTION_KEY_PREFIX,
            constants.POLICY_EXTRA_PREFIX,
        )
    )

    # Numpy testing does not work with string arrays. We need to manually check
    # all the features.

    for idx, _ in enumerate(timesteps):
      timestep = timesteps[idx]
      expected_timestep = expected_timesteps[idx]

      np.testing.assert_equal(timestep.step_type, expected_timestep.step_type)

      keys = timestep.reward.keys()
      self.assertSameElements(keys, expected_timestep.reward.keys())
      for key in keys:
        if timestep.reward[key].dtype == np.object_:
          # String.
          self.assertEqual(timestep.reward[key], expected_timestep.reward[key])
        else:
          # Any other type.
          np.testing.assert_allclose(
              timestep.reward[key], expected_timestep.reward[key]
          )

      keys = timestep.discount.keys()
      self.assertSameElements(keys, expected_timestep.discount.keys())
      for key in keys:
        if timestep.discount[key].dtype == np.object_:
          # String.
          self.assertEqual(
              timestep.discount[key], expected_timestep.discount[key]
          )
        else:
          # Any other type.
          np.testing.assert_allclose(
              timestep.discount[key], expected_timestep.discount[key]
          )

      keys = timestep.observation.keys()
      self.assertSameElements(keys, expected_timestep.observation.keys())
      for key in keys:
        if timestep.observation[key].dtype == np.object_:
          # String.
          self.assertEqual(
              timestep.observation[key], expected_timestep.observation[key]
          )
        else:
          # Any other type.
          np.testing.assert_allclose(
              timestep.observation[key], expected_timestep.observation[key]
          )

    for idx, _ in enumerate(actions):
      # action = cast(Mapping[str, np.ndarray], actions[idx])
      np.testing.assert_allclose(actions[idx], expected_actions[idx])

    for idx, _ in enumerate(policy_extras):
      keys = policy_extras[idx].keys()
      expected_policy_extras_keys = cast(
          Mapping[str, Any], expected_policy_extras[idx]
      ).keys()
      self.assertSameElements(keys, expected_policy_extras_keys)
      for key in keys:
        if policy_extras[idx][key].dtype == np.object_:
          # String.
          self.assertEqual(
              policy_extras[idx][key], expected_policy_extras[idx][key]
          )
        else:
          # Any other type.
          np.testing.assert_allclose(
              policy_extras[idx][key], expected_policy_extras[idx][key]
          )

  @parameterized.named_parameters(
      dict(
          testcase_name="wrong_observation",
          timestep=dm_env.TimeStep(
              step_type=np.asarray(dm_env.StepType.MID, dtype=np.uint8),
              reward=np.array(1.0, dtype=np.float32),
              discount=np.array(1.0, dtype=np.float32),
              observation={
                  "feature1": np.array(
                      [1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32
                  ),
                  "instruction": np.array("instruction", dtype=object),
              },
          ),
          action=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
          policy_extra={"extra1": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
      ),
      dict(
          testcase_name="wrong_action",
          timestep=dm_env.TimeStep(
              step_type=np.asarray(dm_env.StepType.MID, dtype=np.uint8),
              reward=np.array(1.0, dtype=np.float32),
              discount=np.array(1.0, dtype=np.float32),
              observation={
                  "feature1": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                  "instruction": np.array("instruction", dtype=object),
              },
          ),
          action=np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
          policy_extra={"extra1": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
      ),
      dict(
          testcase_name="wrong_policy_extra",
          timestep=dm_env.TimeStep(
              step_type=np.asarray(dm_env.StepType.MID, dtype=np.uint8),
              reward=np.array(1.0, dtype=np.float32),
              discount=np.array(1.0, dtype=np.float32),
              observation={
                  "feature1": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                  "instruction": np.array("instruction", dtype=object),
              },
          ),
          action=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
          policy_extra={"extra1": np.array([1.0, 2.0], dtype=np.float32)},
      ),
      dict(
          testcase_name="wrong_reward",
          timestep=dm_env.TimeStep(
              step_type=np.asarray(dm_env.StepType.MID, dtype=np.uint8),
              reward=np.array([1.0, 2.0], dtype=np.float32),
              discount=np.array(1.0, dtype=np.float32),
              observation={
                  "feature1": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                  "instruction": np.array("instruction", dtype=object),
              },
          ),
          action=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
          policy_extra={"extra1": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
      ),
      dict(
          testcase_name="wrong_discount",
          timestep=dm_env.TimeStep(
              step_type=np.asarray(dm_env.StepType.MID, dtype=np.uint8),
              reward=np.array(1.0, dtype=np.float32),
              discount={"discount1": np.array(1.0, dtype=np.float32)},
              observation={
                  "feature1": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                  "instruction": np.array("instruction", dtype=object),
              },
          ),
          action=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
          policy_extra={"extra1": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
      ),
  )
  def test_logger_validates_data(
      self,
      timestep: dm_env.TimeStep,
      action: gdmr_types.ActionType,
      policy_extra: Mapping[str, Any],
  ):
    timestep_spec = gdmr_types.TimeStepSpec(
        step_type=gdmr_types.STEP_TYPE_SPEC,
        reward=specs.Array(shape=(), dtype=np.float32),
        discount=specs.Array(shape=(), dtype=np.float32),
        observation={
            "instruction": specs.StringArray(shape=(), name="instruction"),
            "feature1": specs.Array(shape=(4,), dtype=np.float32),
        },
    )

    action_spec = specs.BoundedArray(
        shape=(5,),
        dtype=np.float32,
        minimum=np.array([-1.0, -2.0, -1.0, -2.0, 0.0], dtype=np.float32),
        maximum=np.array([1.0, 1.0, 2.0, 3.0, 1.0], dtype=np.float32),
    )

    policy_extra_spec = {
        "extra1": specs.Array(shape=(3,), dtype=np.float32),
    }

    logger = episodic_logger.EpisodicLogger.create(
        agent_id=_TEST_AGENT_ID,
        task_id=_TEST_TASK_ID,
        output_directory=self._episode_path.full_path,
        action_spec=action_spec,
        timestep_spec=timestep_spec,
        image_observation_keys=[],
        proprioceptive_observation_keys=["feature1"],
        policy_extra_spec=policy_extra_spec,
    )

    initial_timestep = self._generate_timestep(
        timestep_spec, dm_env.StepType.FIRST
    )
    logger.reset(initial_timestep)

    logger.record_action_and_next_timestep(
        action=action, next_timestep=timestep, policy_extra=policy_extra
    )
    # Note: the logger does not validate that the last timestep is LAST.
    with self.assertRaises(ValueError):
      logger.write()

  @parameterized.named_parameters(
      dict(
          testcase_name="wrong_observation",
          timestep=dm_env.TimeStep(
              step_type=np.asarray(dm_env.StepType.MID, dtype=np.uint8),
              reward=np.array(1.0, dtype=np.float32),
              discount=np.array(1.0, dtype=np.float32),
              observation={
                  "feature1": np.array(
                      [1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32
                  ),
                  "instruction": np.array("instruction", dtype=object),
              },
          ),
          action=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
          policy_extra={"extra1": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
      ),
      dict(
          testcase_name="wrong_action",
          timestep=dm_env.TimeStep(
              step_type=np.asarray(dm_env.StepType.MID, dtype=np.uint8),
              reward=np.array(1.0, dtype=np.float32),
              discount=np.array(1.0, dtype=np.float32),
              observation={
                  "feature1": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                  "instruction": np.array("instruction", dtype=object),
              },
          ),
          action=np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
          policy_extra={"extra1": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
      ),
      dict(
          testcase_name="wrong_policy_extra",
          timestep=dm_env.TimeStep(
              step_type=np.asarray(dm_env.StepType.MID, dtype=np.uint8),
              reward=np.array(1.0, dtype=np.float32),
              discount=np.array(1.0, dtype=np.float32),
              observation={
                  "feature1": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                  "instruction": np.array("instruction", dtype=object),
              },
          ),
          action=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
          policy_extra={"extra1": np.array([1.0, 2.0], dtype=np.float32)},
      ),
      dict(
          testcase_name="wrong_reward",
          timestep=dm_env.TimeStep(
              step_type=np.asarray(dm_env.StepType.MID, dtype=np.uint8),
              reward=np.array([1.0, 2.0], dtype=np.float32),
              discount=np.array(1.0, dtype=np.float32),
              observation={
                  "feature1": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                  "instruction": np.array("instruction", dtype=object),
              },
          ),
          action=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
          policy_extra={"extra1": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
      ),
      dict(
          testcase_name="wrong_discount",
          timestep=dm_env.TimeStep(
              step_type=np.asarray(dm_env.StepType.MID, dtype=np.uint8),
              reward=np.array(1.0, dtype=np.float32),
              discount={"discount1": np.array(1.0, dtype=np.float32)},
              observation={
                  "feature1": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                  "instruction": np.array("instruction", dtype=object),
              },
          ),
          action=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
          policy_extra={"extra1": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
      ),
  )
  def test_disabling_validation_does_not_raise_error(
      self,
      timestep: dm_env.TimeStep,
      action: gdmr_types.ActionType,
      policy_extra: Mapping[str, Any],
  ):
    timestep_spec = gdmr_types.TimeStepSpec(
        step_type=gdmr_types.STEP_TYPE_SPEC,
        reward=specs.Array(shape=(), dtype=np.float32),
        discount=specs.Array(shape=(), dtype=np.float32),
        observation={
            "instruction": specs.StringArray(shape=(), name="instruction"),
            "feature1": specs.Array(shape=(4,), dtype=np.float32),
            "feature2": specs.Array(shape=(3,), dtype=np.int32),
            "feature3": specs.Array(shape=(), dtype=np.float64),
        },
    )

    action_spec = specs.BoundedArray(
        shape=(5,),
        dtype=np.float32,
        minimum=np.array([-1.0, -2.0, -1.0, -2.0, 0.0], dtype=np.float32),
        maximum=np.array([1.0, 1.0, 2.0, 3.0, 1.0], dtype=np.float32),
    )

    policy_extra_spec = {
        "extra1": specs.Array(shape=(3,), dtype=np.float32),
        "extra2": specs.Array(shape=(4,), dtype=np.float64),
    }

    logger = episodic_logger.EpisodicLogger.create(
        agent_id=_TEST_AGENT_ID,
        task_id=_TEST_TASK_ID,
        output_directory=self._episode_path.full_path,
        action_spec=action_spec,
        timestep_spec=timestep_spec,
        image_observation_keys=[],
        proprioceptive_observation_keys=["feature1"],
        policy_extra_spec=policy_extra_spec,
        validate_data_with_spec=False,
    )

    initial_timestep = self._generate_timestep(
        timestep_spec, dm_env.StepType.FIRST
    )
    logger.reset(initial_timestep)

    logger.record_action_and_next_timestep(
        action=action, next_timestep=timestep, policy_extra=policy_extra
    )
    # Note: the logger does not validate that the last timestep is LAST.
    logger.write()

  def test_writing_multiple_episodes_can_be_read_correctly(self):
    timestep_spec = gdmr_types.TimeStepSpec(
        step_type=gdmr_types.STEP_TYPE_SPEC,
        reward=specs.Array(shape=(), dtype=np.float32),
        discount=specs.Array(shape=(), dtype=np.float32),
        observation={
            "instruction": specs.StringArray(shape=(), name="instruction"),
            "feature1": specs.Array(shape=(4,), dtype=np.float32),
        },
    )

    action_spec = specs.BoundedArray(
        shape=(5,),
        dtype=np.float32,
        minimum=np.array([-1.0, -2.0, -1.0, -2.0, 0.0], dtype=np.float32),
        maximum=np.array([1.0, 1.0, 2.0, 3.0, 1.0], dtype=np.float32),
    )

    logger = episodic_logger.EpisodicLogger.create(
        agent_id=_TEST_AGENT_ID,
        task_id=_TEST_TASK_ID,
        output_directory=self._episode_path.full_path,
        action_spec=action_spec,
        timestep_spec=timestep_spec,
        image_observation_keys=[],
        proprioceptive_observation_keys=["feature1"],
        policy_extra_spec={},
    )

    for _ in range(2):
      initial_timestep = self._generate_timestep(
          timestep_spec, dm_env.StepType.FIRST
      )
      logger.reset(initial_timestep)

      for _ in range(_DEFAULT_NUMBER_STEPS):
        next_timestep = self._generate_timestep(
            timestep_spec, dm_env.StepType.MID
        )
        action = specs_utils.valid_value_for_spec(action_spec)

        policy_extra = {}
        logger.record_action_and_next_timestep(
            action=action,
            next_timestep=next_timestep,
            policy_extra=policy_extra,
        )

      last_timestep = self._generate_timestep(
          timestep_spec, dm_env.StepType.LAST
      )

      action = specs_utils.valid_value_for_spec(action_spec)
      logger.record_action_and_next_timestep(
          action=action,
          next_timestep=last_timestep,
          policy_extra={},
      )

      logger.write()

    mcap_proto_data = mcap_parser_utils.read_proto_data(
        self._episode_path.full_path,
        constants.TIMESTEP_TOPIC_NAME,
        constants.ACTION_TOPIC_NAME,
        constants.POLICY_EXTRA_TOPIC_NAME,
    )
    timesteps_examples = mcap_proto_data.timesteps
    actions_examples = mcap_proto_data.actions
    policy_extras_examples = mcap_proto_data.policy_extra
    # The mcap reader does not order the files. In this test we only check that
    # the size of the data is correct without checking the actual data.
    self.assertLen(timesteps_examples, 2 * (_DEFAULT_NUMBER_STEPS + 2))
    self.assertLen(actions_examples, 2 * (_DEFAULT_NUMBER_STEPS + 2))
    self.assertLen(policy_extras_examples, 2 * (_DEFAULT_NUMBER_STEPS + 2))

  def test_data_is_cleared_between_episodes(self):
    # We test that the data is cleared between episodes by writing two episodes
    # and checking that the data in the second episode contains only the second
    # episode data.
    timestep_spec = gdmr_types.TimeStepSpec(
        step_type=gdmr_types.STEP_TYPE_SPEC,
        reward=specs.Array(shape=(), dtype=np.float32),
        discount=specs.Array(shape=(), dtype=np.float32),
        observation={
            "instruction": specs.StringArray(shape=(), name="instruction"),
            "feature1": specs.Array(shape=(4,), dtype=np.float32),
        },
    )

    action_spec = specs.BoundedArray(
        shape=(5,),
        dtype=np.float32,
        minimum=np.array([-1.0, -2.0, -1.0, -2.0, 0.0], dtype=np.float32),
        maximum=np.array([1.0, 1.0, 2.0, 3.0, 1.0], dtype=np.float32),
    )

    logger = episodic_logger.EpisodicLogger.create(
        agent_id=_TEST_AGENT_ID,
        task_id=_TEST_TASK_ID,
        output_directory=self._episode_path.full_path,
        action_spec=action_spec,
        timestep_spec=timestep_spec,
        image_observation_keys=[],
        proprioceptive_observation_keys=["feature1"],
        policy_extra_spec={},
    )

    initial_timestep = self._generate_timestep(
        timestep_spec, dm_env.StepType.FIRST
    )
    logger.reset(initial_timestep)

    for _ in range(_DEFAULT_NUMBER_STEPS):
      next_timestep = self._generate_timestep(
          timestep_spec, dm_env.StepType.MID
      )
      action = specs_utils.valid_value_for_spec(action_spec)

      policy_extra = {}
      logger.record_action_and_next_timestep(
          action=action,
          next_timestep=next_timestep,
          policy_extra=policy_extra,
      )

    last_timestep = self._generate_timestep(timestep_spec, dm_env.StepType.LAST)

    action = specs_utils.valid_value_for_spec(action_spec)
    logger.record_action_and_next_timestep(
        action=action,
        next_timestep=last_timestep,
        policy_extra={},
    )

    logger.write()

    mcap_proto_data = mcap_parser_utils.read_proto_data(
        self._episode_path.full_path,
        constants.TIMESTEP_TOPIC_NAME,
        constants.ACTION_TOPIC_NAME,
        constants.POLICY_EXTRA_TOPIC_NAME,
    )
    timesteps_examples = mcap_proto_data.timesteps
    actions_examples = mcap_proto_data.actions
    policy_extras_examples = mcap_proto_data.policy_extra

    self.assertLen(timesteps_examples, _DEFAULT_NUMBER_STEPS + 2)
    self.assertLen(actions_examples, _DEFAULT_NUMBER_STEPS + 2)
    self.assertLen(policy_extras_examples, _DEFAULT_NUMBER_STEPS + 2)

    shutil.rmtree(
        self._episode_path.full_path
    )  # Clean up directory so that we load only the second episode.
    # We now write a second episode and check that the data is cleared.

    expected_timesteps = []
    expected_actions = []

    initial_timestep = self._generate_timestep(
        timestep_spec, dm_env.StepType.FIRST
    )
    logger.reset(initial_timestep)
    expected_timesteps.append(initial_timestep)

    for _ in range(_DEFAULT_NUMBER_STEPS):
      next_timestep = self._generate_timestep(
          timestep_spec, dm_env.StepType.MID
      )
      action = specs_utils.valid_value_for_spec(action_spec)
      expected_actions.append(action)
      expected_timesteps.append(next_timestep)

      policy_extra = {}
      logger.record_action_and_next_timestep(
          action=action,
          next_timestep=next_timestep,
          policy_extra=policy_extra,
      )

    last_timestep = self._generate_timestep(timestep_spec, dm_env.StepType.LAST)

    action = specs_utils.valid_value_for_spec(action_spec)
    expected_actions.append(action)
    expected_timesteps.append(last_timestep)
    logger.record_action_and_next_timestep(
        action=action,
        next_timestep=last_timestep,
        policy_extra={},
    )
    # Append action another time for padding.
    expected_actions.append(action)

    logger.write()

    # Now check that the data is only for the second episode.
    mcap_proto_data = mcap_parser_utils.read_proto_data(
        self._episode_path.full_path,
        constants.TIMESTEP_TOPIC_NAME,
        constants.ACTION_TOPIC_NAME,
        constants.POLICY_EXTRA_TOPIC_NAME,
    )
    timesteps_examples = mcap_proto_data.timesteps
    actions_examples = mcap_proto_data.actions
    policy_extras_examples = mcap_proto_data.policy_extra

    timesteps, actions, policy_extras = (
        mcap_parser_utils.parse_examples_to_dm_env_types(
            timestep_spec,
            action_spec,
            {},
            timesteps_examples,
            actions_examples,
            policy_extras_examples,
            constants.STEP_TYPE_KEY,
            constants.OBSERVATION_KEY_PREFIX,
            constants.REWARD_KEY,
            constants.DISCOUNT_KEY,
            constants.ACTION_KEY_PREFIX,
            constants.POLICY_EXTRA_PREFIX,
        )
    )

    for idx, _ in enumerate(timesteps):
      self._assert_timestep_is_close(timesteps[idx], expected_timesteps[idx])

    for idx, _ in enumerate(actions):
      if isinstance(actions[idx], Mapping):
        keys = actions[idx].keys()
        expected_actions_keys = cast(
            Mapping[str, np.ndarray], expected_actions[idx]
        ).keys()
        self.assertSameElements(keys, expected_actions_keys)
        for key in keys:
          np.testing.assert_allclose(
              actions[idx][key], expected_actions[idx][key]
          )
      else:
        np.testing.assert_allclose(actions[idx], expected_actions[idx])

    expected_policy_extras = [{}] * (_DEFAULT_NUMBER_STEPS + 2)
    self.assertEqual(policy_extras, expected_policy_extras)

  def test_each_episode_is_written_with_a_unique_file_prefix(self):
    timestep_spec = gdmr_types.TimeStepSpec(
        step_type=gdmr_types.STEP_TYPE_SPEC,
        reward=specs.Array(shape=(), dtype=np.float32),
        discount=specs.Array(shape=(), dtype=np.float32),
        observation={
            "instruction": specs.StringArray(shape=(), name="instruction"),
            "feature1": specs.Array(shape=(4,), dtype=np.float32),
        },
    )

    action_spec = specs.BoundedArray(
        shape=(5,),
        dtype=np.float32,
        minimum=np.array([-1.0, -2.0, -1.0, -2.0, 0.0], dtype=np.float32),
        maximum=np.array([1.0, 1.0, 2.0, 3.0, 1.0], dtype=np.float32),
    )

    logger = episodic_logger.EpisodicLogger.create(
        agent_id=_TEST_AGENT_ID,
        task_id=_TEST_TASK_ID,
        output_directory=self._episode_path.full_path,
        action_spec=action_spec,
        timestep_spec=timestep_spec,
        image_observation_keys=[],
        proprioceptive_observation_keys=["feature1"],
        policy_extra_spec={},
    )

    for _ in range(2):
      initial_timestep = self._generate_timestep(
          timestep_spec, dm_env.StepType.FIRST
      )
      logger.reset(initial_timestep)

      for _ in range(_DEFAULT_NUMBER_STEPS):
        next_timestep = self._generate_timestep(
            timestep_spec, dm_env.StepType.MID
        )
        action = specs_utils.valid_value_for_spec(action_spec)

        policy_extra = {}
        logger.record_action_and_next_timestep(
            action=action,
            next_timestep=next_timestep,
            policy_extra=policy_extra,
        )

      last_timestep = self._generate_timestep(
          timestep_spec, dm_env.StepType.LAST
      )

      action = specs_utils.valid_value_for_spec(action_spec)
      logger.record_action_and_next_timestep(
          action=action,
          next_timestep=last_timestep,
          policy_extra={},
      )

      logger.write()

    episode_paths = glob.glob(
        os.path.join(self._episode_path.full_path, "*", "*", "*", "*.mcap")
    )

    file_name = set(
        os.path.basename(file_path).split(".")[0] for file_path in episode_paths
    )
    self.assertLen(file_name, 2, "Not all episodes have unique file prefixes")

  def test_writing_steps_over_shard_size_limit_creates_multiple_files(self):
    timestep_spec = gdmr_types.TimeStepSpec(
        step_type=gdmr_types.STEP_TYPE_SPEC,
        reward=specs.Array(shape=(), dtype=np.float32),
        discount=specs.Array(shape=(), dtype=np.float32),
        observation={
            "instruction": specs.StringArray(shape=(), name="instruction"),
            "feature1": specs.Array(shape=(4,), dtype=np.float32),
        },
    )

    action_spec = specs.BoundedArray(
        shape=(5,),
        dtype=np.float32,
        minimum=np.array([-1.0, -2.0, -1.0, -2.0, 0.0], dtype=np.float32),
        maximum=np.array([1.0, 1.0, 2.0, 3.0, 1.0], dtype=np.float32),
    )

    policy_extra_spec = {
        "extra1": specs.Array(shape=(3,), dtype=np.float32),
        "extra2": specs.Array(shape=(4,), dtype=np.float64),
    }

    logger = episodic_logger.EpisodicLogger.create(
        agent_id=_TEST_AGENT_ID,
        task_id=_TEST_TASK_ID,
        output_directory=self._episode_path.full_path,
        action_spec=action_spec,
        timestep_spec=timestep_spec,
        image_observation_keys=[],
        proprioceptive_observation_keys=["feature1"],
        policy_extra_spec=policy_extra_spec,
        file_shard_size_limit_bytes=50,
    )

    initial_timestep = self._generate_timestep(
        timestep_spec, dm_env.StepType.FIRST
    )
    logger.reset(initial_timestep)

    for _ in range(10):
      next_timestep = self._generate_timestep(
          timestep_spec, dm_env.StepType.MID
      )
      action = specs_utils.valid_value_for_spec(action_spec)
      policy_extra = specs_utils.valid_dict_value(policy_extra_spec)

      logger.record_action_and_next_timestep(
          action=action,
          next_timestep=next_timestep,
          policy_extra=policy_extra,
      )

    last_timestep = self._generate_timestep(timestep_spec, dm_env.StepType.LAST)
    action = specs_utils.valid_value_for_spec(action_spec)
    policy_extra = specs_utils.valid_dict_value(policy_extra_spec)

    logger.record_action_and_next_timestep(
        action=action,
        next_timestep=last_timestep,
        policy_extra=policy_extra,
    )

    logger.write()

    # Check the number of mcap files created. The number of files should be
    # greater than 1.
    episode_paths = glob.glob(
        os.path.join(self._episode_path.full_path, "*", "*", "*", "*.mcap")
    )
    self.assertGreater(len(episode_paths), 1)

  def test_shard_ordering_is_correct(self):
    timestep_spec = gdmr_types.TimeStepSpec(
        step_type=gdmr_types.STEP_TYPE_SPEC,
        reward=specs.Array(shape=(), dtype=np.float32),
        discount=specs.Array(shape=(), dtype=np.float32),
        observation={
            "instruction": specs.StringArray(shape=(), name="instruction"),
            "feature1": specs.Array(shape=(4,), dtype=np.float32),
        },
    )

    action_spec = specs.BoundedArray(
        shape=(5,),
        dtype=np.float32,
        minimum=np.array([-1.0, -2.0, -1.0, -2.0, 0.0], dtype=np.float32),
        maximum=np.array([1.0, 1.0, 2.0, 3.0, 1.0], dtype=np.float32),
    )

    policy_extra_spec = {
        "extra1": specs.Array(shape=(3,), dtype=np.float32),
        "extra2": specs.Array(shape=(4,), dtype=np.float64),
    }

    logger = episodic_logger.EpisodicLogger.create(
        agent_id=_TEST_AGENT_ID,
        task_id=_TEST_TASK_ID,
        output_directory=self._episode_path.full_path,
        action_spec=action_spec,
        timestep_spec=timestep_spec,
        image_observation_keys=[],
        proprioceptive_observation_keys=["feature1"],
        policy_extra_spec=policy_extra_spec,
        file_shard_size_limit_bytes=50,
    )

    expected_timesteps = []
    expected_actions = []
    expected_policy_extras = []

    initial_timestep = self._generate_timestep(
        timestep_spec, dm_env.StepType.FIRST
    )
    expected_timesteps.append(initial_timestep)
    logger.reset(initial_timestep)

    for _ in range(10):
      next_timestep = self._generate_timestep(
          timestep_spec, dm_env.StepType.MID
      )
      action = specs_utils.valid_value_for_spec(action_spec)
      policy_extra = specs_utils.valid_dict_value(policy_extra_spec)

      expected_timesteps.append(next_timestep)
      expected_actions.append(action)
      expected_policy_extras.append(policy_extra)

      logger.record_action_and_next_timestep(
          action=action,
          next_timestep=next_timestep,
          policy_extra=policy_extra,
      )

    last_timestep = self._generate_timestep(timestep_spec, dm_env.StepType.LAST)
    action = specs_utils.valid_value_for_spec(action_spec)
    policy_extra = specs_utils.valid_dict_value(policy_extra_spec)

    expected_timesteps.append(last_timestep)
    expected_actions.append(action)
    expected_policy_extras.append(policy_extra)

    logger.record_action_and_next_timestep(
        action=action,
        next_timestep=last_timestep,
        policy_extra=policy_extra,
    )

    # Append action and policy extra another time for padding.
    expected_actions.append(action)
    expected_policy_extras.append(policy_extra)

    logger.write()

    # Check that the date can be reconstructed in the correct order.
    mcap_proto_data = mcap_parser_utils.read_proto_data(
        self._episode_path.full_path,
        constants.TIMESTEP_TOPIC_NAME,
        constants.ACTION_TOPIC_NAME,
        constants.POLICY_EXTRA_TOPIC_NAME,
    )
    timesteps_examples = mcap_proto_data.timesteps
    actions_examples = mcap_proto_data.actions
    policy_extras_examples = mcap_proto_data.policy_extra

    timesteps, actions, policy_extras = (
        mcap_parser_utils.parse_examples_to_dm_env_types(
            timestep_spec,
            action_spec,
            policy_extra_spec,
            timesteps_examples,
            actions_examples,
            policy_extras_examples,
            constants.STEP_TYPE_KEY,
            constants.OBSERVATION_KEY_PREFIX,
            constants.REWARD_KEY,
            constants.DISCOUNT_KEY,
            constants.ACTION_KEY_PREFIX,
            constants.POLICY_EXTRA_PREFIX,
        )
    )

    for idx, _ in enumerate(timesteps):
      self._assert_timestep_is_close(timesteps[idx], expected_timesteps[idx])

    for idx, _ in enumerate(actions):
      if isinstance(actions[idx], Mapping):
        keys = actions[idx].keys()
        expected_actions_keys = cast(
            Mapping[str, np.ndarray], expected_actions[idx]
        ).keys()
        self.assertSameElements(keys, expected_actions_keys)
        for key in keys:
          np.testing.assert_allclose(
              actions[idx][key], expected_actions[idx][key]
          )
      else:
        np.testing.assert_allclose(actions[idx], expected_actions[idx])

    for idx, _ in enumerate(policy_extras):
      keys = policy_extras[idx].keys()
      expected_policy_extras_keys = cast(
          Mapping[str, Any], expected_policy_extras[idx]
      ).keys()
      self.assertSameElements(keys, expected_policy_extras_keys)
      for key in keys:
        np.testing.assert_allclose(
            policy_extras[idx][key], expected_policy_extras[idx][key]
        )

  def test_no_data_is_written_if_reset_or_record_action_is_not_called(self):
    timestep_spec = gdmr_types.TimeStepSpec(
        step_type=gdmr_types.STEP_TYPE_SPEC,
        reward=specs.Array(shape=(), dtype=np.float32),
        discount=specs.Array(shape=(), dtype=np.float32),
        observation={
            "instruction": specs.StringArray(shape=(), name="instruction"),
            "feature1": specs.Array(shape=(4,), dtype=np.float32),
        },
    )

    action_spec = specs.BoundedArray(
        shape=(5,),
        dtype=np.float32,
        minimum=np.array([-1.0, -2.0, -1.0, -2.0, 0.0], dtype=np.float32),
        maximum=np.array([1.0, 1.0, 2.0, 3.0, 1.0], dtype=np.float32),
    )

    policy_extra_spec = {
        "extra1": specs.Array(shape=(3,), dtype=np.float32),
        "extra2": specs.Array(shape=(4,), dtype=np.float64),
    }

    logger = episodic_logger.EpisodicLogger.create(
        agent_id=_TEST_AGENT_ID,
        task_id=_TEST_TASK_ID,
        output_directory=self._episode_path.full_path,
        action_spec=action_spec,
        timestep_spec=timestep_spec,
        image_observation_keys=[],
        proprioceptive_observation_keys=["feature1"],
        policy_extra_spec=policy_extra_spec,
    )
    logger.write()
    episode_paths = glob.glob(
        os.path.join(self._episode_path.full_path, "*", "*", "*", "*.mcap")
    )
    self.assertEmpty(episode_paths, "No mcap files should be written")

  @parameterized.named_parameters(
      dict(
          testcase_name="with_none_agent_id",
          agent_id=None,
          task_id=_TEST_TASK_ID,
      ),
      dict(
          testcase_name="with_empty_agent_id",
          agent_id="",
          task_id=_TEST_TASK_ID,
      ),
      dict(
          testcase_name="with_none_task_id",
          agent_id=_TEST_AGENT_ID,
          task_id=None,
      ),
      dict(
          testcase_name="with_empty_task_id",
          agent_id=_TEST_AGENT_ID,
          task_id="",
      ),
  )
  def test_create_with_invalid_ids_raises_assertion_error(
      self, agent_id, task_id
  ):
    """Tests that AssertionError is raised for invalid agent_id or task_id."""
    timestep_spec = gdmr_types.TimeStepSpec(
        step_type=gdmr_types.STEP_TYPE_SPEC,
        reward=specs.Array(shape=(), dtype=np.float32),
        discount=specs.Array(shape=(), dtype=np.float32),
        observation={
            "instruction": specs.StringArray(shape=(), name="instruction"),
            "feature1": specs.Array(shape=(4,), dtype=np.float32),
        },
    )
    action_spec = specs.BoundedArray(
        shape=(), dtype=np.float32, minimum=-np.inf, maximum=np.inf
    )
    with self.assertRaises(ValueError):
      episodic_logger.EpisodicLogger.create(
          agent_id=agent_id,
          task_id=task_id,
          output_directory=self._episode_path.full_path,
          action_spec=action_spec,
          timestep_spec=timestep_spec,
          proprioceptive_observation_keys=[],
          image_observation_keys=[],
          policy_extra_spec={},
      )

  def test_invalid_camera_name_raises_value_error(self):
    """Tests that ValueError is raised for invalid camera names."""
    timestep_spec = gdmr_types.TimeStepSpec(
        step_type=gdmr_types.STEP_TYPE_SPEC,
        reward=specs.Array(shape=(), dtype=np.float32),
        discount=specs.Array(shape=(), dtype=np.float32),
        observation={
            "instruction": specs.StringArray(shape=(), name="instruction"),
            "rgb_camera": specs.Array(shape=(64, 64, 3), dtype=np.uint8),
            "depth_camera": specs.Array(shape=(64, 64), dtype=np.uint16),
            _TEST_PROPRIO_KEY: specs.Array(shape=(4,), dtype=np.float32),
        },
    )
    action_spec = specs.BoundedArray(
        shape=(), dtype=np.float32, minimum=-np.inf, maximum=np.inf
    )
    policy_extra_spec = {}

    invalid_image_observation_keys = ["rgb_camera", "non_existent_camera"]

    with self.assertRaisesRegex(
        KeyError,
        "Image observation key non_existent_camera not found in observation"
        " spec.",
    ):
      episodic_logger.EpisodicLogger.create(
          agent_id=_TEST_AGENT_ID,
          task_id=_TEST_TASK_ID,
          output_directory=self._episode_path.full_path,
          action_spec=action_spec,
          timestep_spec=timestep_spec,
          proprioceptive_observation_keys=[_TEST_PROPRIO_KEY],
          image_observation_keys=invalid_image_observation_keys,
          policy_extra_spec=policy_extra_spec,
      )

  def test_camera_name_validation_fails_if_observation_spec_not_mapping(self):
    """Tests TypeError if observation_spec is not a Mapping and cameras are listed."""
    # Create a TimeStepSpec where observation is not a Mapping.
    timestep_spec_non_mapping_obs = gdmr_types.TimeStepSpec(
        step_type=gdmr_types.STEP_TYPE_SPEC,
        reward=specs.Array(shape=(), dtype=np.float32),
        discount=specs.Array(shape=(), dtype=np.float32),
        observation=specs.Array(shape=(64, 64, 3), dtype=np.uint8),
    )
    action_spec = specs.BoundedArray(
        shape=(), dtype=np.float32, minimum=-np.inf, maximum=np.inf
    )
    policy_extra_spec = {}
    # Camera names list must be non-empty for the TypeError to be raised.
    image_observation_keys_for_type_error = ["some_camera"]

    with self.assertRaisesRegex(
        TypeError,
        "Observation in timestep_spec must be a Mapping.",
    ):
      episodic_logger.EpisodicLogger.create(
          agent_id=_TEST_AGENT_ID,
          task_id=_TEST_TASK_ID,
          output_directory=self._episode_path.full_path,
          action_spec=action_spec,
          proprioceptive_observation_keys=["feature1"],
          timestep_spec=timestep_spec_non_mapping_obs,
          image_observation_keys=image_observation_keys_for_type_error,
          policy_extra_spec=policy_extra_spec,
      )

  def test_invalid_proprioceptive_observation_keys_not_in_observation_raises_key_error(
      self,
  ):
    """Tests KeyError if proprio_key is not in observation_spec."""
    timestep_spec = gdmr_types.TimeStepSpec(
        step_type=gdmr_types.STEP_TYPE_SPEC,
        reward=specs.Array(shape=(), dtype=np.float32),
        discount=specs.Array(shape=(), dtype=np.float32),
        observation={
            "instruction": specs.StringArray(shape=(), name="instruction"),
            "some_other_feature": specs.Array(shape=(4,), dtype=np.float32),
            # The _TEST_PROPRIO_KEY or any specific invalid key is missing here.
        },
    )
    action_spec = specs.BoundedArray(  # Valid action spec
        shape=(), dtype=np.float32, minimum=-np.inf, maximum=np.inf
    )
    invalid_proprio_key = "non_existent_proprio_key"

    with self.assertRaisesRegex(
        KeyError,
        f"Proprio key {invalid_proprio_key} not found in observation spec.",
    ):
      episodic_logger.EpisodicLogger.create(
          agent_id=_TEST_AGENT_ID,
          task_id=_TEST_TASK_ID,
          output_directory=self._episode_path.full_path,
          action_spec=action_spec,
          timestep_spec=timestep_spec,
          proprioceptive_observation_keys=[invalid_proprio_key],
          image_observation_keys=[],
          policy_extra_spec={},
      )

  def test_invalid_proprio_key_not_spec_array_raises_type_error(self):
    """Tests TypeError if proprio_key in spec is not a specs.Array."""
    proprio_key_to_test = "proprio_is_dict"
    timestep_spec = gdmr_types.TimeStepSpec(
        step_type=gdmr_types.STEP_TYPE_SPEC,
        reward=specs.Array(shape=(), dtype=np.float32),
        discount=specs.Array(shape=(), dtype=np.float32),
        observation={
            "instruction": specs.StringArray(shape=(), name="instruction"),
            proprio_key_to_test: {
                "sub_field": specs.Array(shape=(1,), dtype=np.float32)
            },  # Not a specs.Array
        },
    )
    action_spec = specs.BoundedArray(  # Valid action spec
        shape=(), dtype=np.float32, minimum=-np.inf, maximum=np.inf
    )

    with self.assertRaisesRegex(
        TypeError,
        f"Proprio data {proprio_key_to_test} must be a specs.Array in"
        " observation spec.",
    ):
      episodic_logger.EpisodicLogger.create(
          agent_id=_TEST_AGENT_ID,
          task_id=_TEST_TASK_ID,
          output_directory=self._episode_path.full_path,
          action_spec=action_spec,
          timestep_spec=timestep_spec,
          proprioceptive_observation_keys=[proprio_key_to_test],
          image_observation_keys=[],
          policy_extra_spec={},
      )

  def test_missing_instruction_in_observation_raises_key_error(self):
    """Tests KeyError if 'instruction' key is missing in observation_spec."""
    timestep_spec = gdmr_types.TimeStepSpec(
        step_type=gdmr_types.STEP_TYPE_SPEC,
        reward=specs.Array(shape=(), dtype=np.float32),
        discount=specs.Array(shape=(), dtype=np.float32),
        observation={
            # 'instruction' key is missing here.
            "some_other_feature": specs.Array(shape=(4,), dtype=np.float32),
        },
    )
    action_spec = specs.BoundedArray(  # Valid action spec
        shape=(), dtype=np.float32, minimum=-np.inf, maximum=np.inf
    )

    with self.assertRaisesRegex(
        KeyError,
        "'instruction' is required in timestep_spec.observation.",
    ):
      episodic_logger.EpisodicLogger.create(
          agent_id=_TEST_AGENT_ID,
          task_id=_TEST_TASK_ID,
          output_directory=self._episode_path.full_path,
          action_spec=action_spec,
          timestep_spec=timestep_spec,
          proprioceptive_observation_keys=[],
          image_observation_keys=[],
          policy_extra_spec={},
      )

  def _generate_timestep(
      self, timestep_spec: gdmr_types.TimeStepSpec, step_type: dm_env.StepType
  ) -> dm_env.TimeStep:
    return dm_env.TimeStep(
        step_type=np.asarray(step_type, dtype=np.uint8),
        reward=specs_utils.valid_value_for_spec(timestep_spec.reward),
        discount=specs_utils.valid_value_for_spec(timestep_spec.discount),
        observation=specs_utils.valid_dict_value(timestep_spec.observation),
    )

  def _assert_timestep_is_close(
      self, timestep: dm_env.TimeStep, expected_timestep: dm_env.TimeStep
  ) -> None:
    """Asserts that a timestep is close to an expected timestep."""
    # Check that the step type is the same.
    np.testing.assert_equal(timestep.step_type, expected_timestep.step_type)

    if isinstance(timestep.reward, Mapping):
      keys = timestep.reward.keys()
      self.assertSameElements(keys, expected_timestep.reward.keys())
      for key in keys:
        if timestep.reward[key].dtype == np.object_:
          self.assertEqual(timestep.reward[key], expected_timestep.reward[key])
        else:
          np.testing.assert_allclose(
              timestep.reward[key], expected_timestep.reward[key]
          )
    else:
      if (
          hasattr(timestep.reward, "dtype")
          and timestep.reward.dtype == np.object_
      ):
        self.assertEqual(timestep.reward, expected_timestep.reward)
      else:
        np.testing.assert_allclose(timestep.reward, expected_timestep.reward)

    if isinstance(timestep.discount, Mapping):
      keys = timestep.discount.keys()
      self.assertSameElements(keys, expected_timestep.discount.keys())
      for key in keys:
        if timestep.discount[key].dtype == np.object_:
          self.assertEqual(
              timestep.discount[key], expected_timestep.discount[key]
          )
        else:
          np.testing.assert_allclose(
              timestep.discount[key], expected_timestep.discount[key]
          )
    else:
      if (
          hasattr(timestep.discount, "dtype")
          and timestep.discount.dtype == np.object_
      ):
        self.assertEqual(timestep.discount, expected_timestep.discount)
      else:
        np.testing.assert_allclose(
            timestep.discount, expected_timestep.discount
        )

    keys = timestep.observation.keys()
    self.assertSameElements(keys, expected_timestep.observation.keys())
    for key in keys:
      if timestep.observation[key].dtype == np.object_:
        self.assertEqual(
            timestep.observation[key], expected_timestep.observation[key]
        )
      else:
        np.testing.assert_allclose(
            timestep.observation[key], expected_timestep.observation[key]
        )

  def test_camera_image_encoding_and_byte_handling(self):
    """Tests JPEG encoding for cameras and raw bytes for other arrays."""
    # Define all potential camera keys for spec and original_arrays
    obs_spec_dict = {
        "instruction": specs.StringArray(shape=(), name="instruction"),
        "rgb_cam": specs.Array(shape=(2, 2, 3), dtype=np.uint8),
        "rgba_cam": specs.Array(shape=(2, 2, 4), dtype=np.uint8),
        "gray_u8_cam": specs.Array(shape=(2, 2), dtype=np.uint8),
        "gray_u16_cam": specs.Array(shape=(2, 2), dtype=np.uint16),
        "non_cam_u8_array": specs.Array(shape=(2, 2, 3), dtype=np.uint8),
        "non_cam_u16_array": specs.Array(shape=(2, 2), dtype=np.uint16),
        "float_array": specs.Array(shape=(3,), dtype=np.float32),
    }
    timestep_spec = gdmr_types.TimeStepSpec(
        step_type=gdmr_types.STEP_TYPE_SPEC,
        reward=specs.Array(shape=(), dtype=np.float32),
        discount=specs.Array(shape=(), dtype=np.float32),
        observation=obs_spec_dict,
    )
    action_spec = specs.BoundedArray(
        shape=(), dtype=np.float32, minimum=-np.inf, maximum=np.inf
    )

    original_arrays = {
        "instruction": np.array("instruction", dtype=object),
        "rgb_cam": np.array(
            [[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]],
            dtype=np.uint8,
        ),
        "rgba_cam": np.array(
            [
                [[10, 20, 30, 255], [40, 50, 60, 255]],
                [[70, 80, 90, 255], [100, 110, 120, 255]],
            ],
            dtype=np.uint8,
        ),
        "gray_u8_cam": np.array([[10, 20], [30, 40]], dtype=np.uint8),
        "gray_u16_cam": np.array(
            [[1000, 2000], [3000, 4000]],
            dtype=np.uint16,
        ),
        "non_cam_u8_array": np.array(
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
            dtype=np.uint8,
        ),
        "non_cam_u16_array": np.array(
            [[100, 200], [300, 400]], dtype=np.uint16
        ),
        "float_array": np.array([0.1, 0.2, 0.3], dtype=np.float32),
    }

    # image_observation_keys for the logger: only types directly saveable as
    # JPEG by Pillow without conversion by the logger itself.
    logger_image_observation_keys = ["rgb_cam", "gray_u8_cam"]

    logger = episodic_logger.EpisodicLogger.create(
        agent_id=_TEST_AGENT_ID,
        task_id=_TEST_TASK_ID,
        output_directory=self._episode_path.full_path,
        action_spec=action_spec,
        image_observation_keys=logger_image_observation_keys,
        timestep_spec=timestep_spec,
        proprioceptive_observation_keys=["float_array"],
        policy_extra_spec={},
    )

    initial_timestep = dm_env.TimeStep(
        step_type=dm_env.StepType.FIRST,
        reward=np.array(0.0, dtype=np.float32),
        discount=np.array(1.0, dtype=np.float32),
        observation=original_arrays,
    )
    logger.reset(initial_timestep)
    last_timestep = dm_env.TimeStep(
        step_type=dm_env.StepType.LAST,
        reward=np.array(0.0, dtype=np.float32),
        discount=np.array(0.0, dtype=np.float32),
        observation=original_arrays,
    )
    logger.record_action_and_next_timestep(
        action=np.array(0.0, dtype=np.float32),
        next_timestep=last_timestep,
        policy_extra={},
    )
    logger.write()

    mcap_proto_data = mcap_parser_utils.read_proto_data(
        self._episode_path.full_path,
        constants.TIMESTEP_TOPIC_NAME,
        constants.ACTION_TOPIC_NAME,
        constants.POLICY_EXTRA_TOPIC_NAME,
    )
    timesteps_examples = mcap_proto_data.timesteps

    self.assertLen(timesteps_examples, 2)
    features = timesteps_examples[0].features.feature
    obs_prefix = constants.OBSERVATION_KEY_PREFIX

    # Test JPEG encoded camera observations
    # These are keys listed in logger_image_observation_keys and are expected
    # to be JPEGs.
    for key, arr in [
        ("rgb_cam", original_arrays["rgb_cam"]),
        ("gray_u8_cam", original_arrays["gray_u8_cam"]),
    ]:
      img = Image.fromarray(arr)
      if img.mode == "RGBA":  # JPEG doesn't support alpha
        img = img.convert("RGB")
      if img.mode == "I;16":  # Convert 16-bit grayscale to 8-bit for JPEG
        img = img.convert("L")

      with io.BytesIO() as output_stream:
        img.save(output_stream, format="JPEG")
        expected_jpeg_bytes = output_stream.getvalue()
      feature_key = f"{obs_prefix}/{key}"
      self.assertEqual(
          features[feature_key].bytes_list.value[0], expected_jpeg_bytes
      )

    # Test uint8/uint16 arrays not in logger_image_observation_keys, or those
    # that would fail direct JPEG conversion, are stored as raw bytes.
    for key, arr in [
        (
            "rgba_cam",
            original_arrays["rgba_cam"],
        ),  # Not in logger_image_observation_keys for this test
        (
            "gray_u16_cam",
            original_arrays["gray_u16_cam"],
        ),  # Not in logger_image_observation_keys for this test
        ("non_cam_u8_array", original_arrays["non_cam_u8_array"]),
        ("non_cam_u16_array", original_arrays["non_cam_u16_array"]),
    ]:
      feature_key = f"{obs_prefix}/{key}"
      self.assertEqual(features[feature_key].bytes_list.value[0], arr.tobytes())

    # Test other data types (e.g., float)
    arr_float = original_arrays["float_array"]
    feature_key_float = f"{obs_prefix}/float_array"
    np.testing.assert_allclose(
        features[feature_key_float].float_list.value, arr_float.flatten()
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="3d_uint16_camera_array",
          obs_key="cam2",
          array_data=np.array(
              [[[100, 200, 300], [400, 500, 600]]], dtype=np.uint16
          ),
          array_spec=specs.Array(shape=(1, 2, 3), dtype=np.uint16),
          expected_exception=TypeError,  # Pillow Image.fromarray fails
      ),
      dict(
          testcase_name="rgba_camera_direct_jpeg_save_fails_with_oserror",
          obs_key="rgba_cam_direct_save",
          array_data=np.array(
              [
                  [[10, 20, 30, 255], [40, 50, 60, 255]],
                  [[70, 80, 90, 255], [100, 110, 120, 255]],
              ],
              dtype=np.uint8,
          ),
          array_spec=specs.Array(shape=(2, 2, 4), dtype=np.uint8),
          expected_exception=OSError,  # Pillow img.save fails for RGBA as JPEG
      ),
      dict(
          testcase_name="gray_u16_camera_direct_jpeg_save_fails_with_oserror",
          obs_key="gray_u16_cam_direct_save",
          array_data=np.array([[1000, 2000], [3000, 4000]], dtype=np.uint16),
          array_spec=specs.Array(shape=(2, 2), dtype=np.uint16),
          expected_exception=OSError,  # Pillow img.save fails for I;16 as JPEG
      ),
  )
  def test_camera_encoding_fails_for_invalid_image_data(
      self, obs_key, array_data, array_spec, expected_exception
  ):
    """Tests that encoding fails for invalid camera data shapes/types."""
    image_observation_keys = [obs_key]
    timestep_spec = gdmr_types.TimeStepSpec(
        step_type=gdmr_types.STEP_TYPE_SPEC,
        reward=specs.Array(shape=(), dtype=np.float32),
        discount=specs.Array(shape=(), dtype=np.float32),
        observation={
            "instruction": specs.StringArray(shape=(), name="instruction"),
            obs_key: array_spec,
            _TEST_PROPRIO_KEY: specs.Array(shape=(5,), dtype=np.float32),
        },
    )
    action_spec = specs.BoundedArray(
        shape=(), dtype=np.float32, minimum=-np.inf, maximum=np.inf
    )

    logger = episodic_logger.EpisodicLogger.create(
        agent_id=_TEST_AGENT_ID,
        task_id=_TEST_TASK_ID,
        output_directory=self._episode_path.full_path,
        image_observation_keys=image_observation_keys,
        action_spec=action_spec,
        proprioceptive_observation_keys=[_TEST_PROPRIO_KEY],
        timestep_spec=timestep_spec,
        policy_extra_spec={},
    )

    initial_timestep = dm_env.TimeStep(
        step_type=dm_env.StepType.FIRST,
        reward=np.float32(0.0),
        discount=np.float32(1.0),
        observation={
            "instruction": np.array("instruction", dtype=object),
            obs_key: array_data,
            _TEST_PROPRIO_KEY: np.zeros((5,), dtype=np.float32),
        },
    )
    logger.reset(initial_timestep)

    # Add one more step to have something to write
    next_timestep = dm_env.TimeStep(
        step_type=dm_env.StepType.LAST,
        reward=np.float32(0.0),
        discount=np.float32(0.0),
        observation={
            "instruction": np.array("instruction", dtype=object),
            obs_key: array_data,
            _TEST_PROPRIO_KEY: np.zeros((5,), dtype=np.float32),
        },
    )
    logger.record_action_and_next_timestep(
        action=np.array(0.0, dtype=np.float32),
        next_timestep=next_timestep,
        policy_extra={},
    )

    with self.assertRaises(expected_exception):
      logger.write()


def _encode_to_jpeg(array: np.ndarray) -> bytes:
  img = Image.fromarray(array)
  with io.BytesIO() as output_stream:
    img.save(output_stream, format="JPEG")
    expected_jpeg_bytes = output_stream.getvalue()
  return expected_jpeg_bytes


if __name__ == "__main__":
  absltest.main()
