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

import json
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import dm_env
from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np

from safari_sdk.model import additional_observations_provider
from safari_sdk.model import constants
from safari_sdk.model import gemini_robotics_policy

FLAGS = flags.FLAGS
FLAGS.mark_as_parsed()


# TODO: We need to test that we encode the observation correctly.
class GeminiRoboticsPolicyTest(parameterized.TestCase):

  def test_initialize_policy(self):

    FLAGS.api_key = "mock_test_key"
    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_resource = mock.MagicMock()
      mock_resource.modelServing.return_value = mock.MagicMock()
      mock_build.return_value = mock_resource

      policy = gemini_robotics_policy.GeminiRoboticsPolicy(
          serve_id="test_serve_id",
          task_instruction_key="test_task_instruction",
          image_observation_keys=(
              "test_camera_1",
              "test_camera_2",
          ),
          proprioceptive_observation_keys=(
              "test_joint_1",
              "test_joint_2",
          ),
          inference_mode=constants.InferenceMode.SYNCHRONOUS,
      )
      self.assertIsInstance(
          policy._client, gemini_robotics_policy.genai_robotics.Client
      )

  def test_raise_error_if_step_spec_not_called(self):
    FLAGS.api_key = "mock_test_key"
    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_resource = mock.MagicMock()
      mock_resource.modelServing.return_value = mock.MagicMock()
      mock_build.return_value = mock_resource

      policy = gemini_robotics_policy.GeminiRoboticsPolicy(
          serve_id="test_serve_id",
          task_instruction_key="test_task_instruction",
          image_observation_keys=("test_camera_1",),
          proprioceptive_observation_keys=("test_joint_1",),
          min_replan_interval=3,
          inference_mode=constants.InferenceMode.SYNCHRONOUS,
      )

      with self.assertRaises(ValueError):
        policy.initial_state()

  @parameterized.named_parameters(
      dict(
          testcase_name="missing_task_instruction_key",
          timestep_spec=gdmr_types.TimeStepSpec(
              step_type=gdmr_types.STEP_TYPE_SPEC,
              reward={},
              discount={},
              observation={
                  "test_camera_1": specs.Array(
                      shape=(100, 100, 3), dtype=np.uint8
                  ),
                  "test_camera_2": specs.Array(
                      shape=(200, 200, 1), dtype=np.uint8
                  ),
                  "test_joint_1": specs.Array(shape=(1,), dtype=np.float32),
                  "test_joint_2": specs.Array(shape=(3,), dtype=np.float32),
              },
          ),
      ),
      dict(
          testcase_name="missing_image_observation_key",
          timestep_spec=gdmr_types.TimeStepSpec(
              step_type=gdmr_types.STEP_TYPE_SPEC,
              reward={},
              discount={},
              observation={
                  "test_camera_1": specs.Array(
                      shape=(100, 100, 3), dtype=np.uint8
                  ),
                  "test_joint_1": specs.Array(shape=(1,), dtype=np.float32),
                  "test_joint_2": specs.Array(shape=(3,), dtype=np.float32),
                  "test_instruction_key": specs.StringArray(()),
              },
          ),
      ),
      dict(
          testcase_name="missing_proprioceptive_observation_key",
          timestep_spec=gdmr_types.TimeStepSpec(
              step_type=gdmr_types.STEP_TYPE_SPEC,
              reward={},
              discount={},
              observation={
                  "test_camera_1": specs.Array(
                      shape=(100, 100, 3), dtype=np.uint8
                  ),
                  "test_camera_2": specs.Array(
                      shape=(200, 200, 1), dtype=np.uint8
                  ),
                  "test_joint_1": specs.Array(shape=(1,), dtype=np.float32),
                  "test_instruction_key": specs.StringArray(()),
              },
          ),
      ),
  )
  def test_spec_validates_timestep_spec(
      self, timestep_spec: gdmr_types.TimeStepSpec
  ):
    FLAGS.api_key = "mock_test_key"
    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_resource = mock.MagicMock()
      mock_resource.modelServing.return_value = mock.MagicMock()
      mock_build.return_value = mock_resource

      policy = gemini_robotics_policy.GeminiRoboticsPolicy(
          serve_id="test_serve_id",
          task_instruction_key="test_instruction_key",
          image_observation_keys=("test_camera_1", "test_camera_2"),
          proprioceptive_observation_keys=("test_joint_1", "test_joint_2"),
          min_replan_interval=3,
          inference_mode=constants.InferenceMode.SYNCHRONOUS,
      )

      # Action is size 2, with chunk of size 3.
      mock_query_model = mock.MagicMock(
          return_value=np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
      )
      policy._query_model = mock_query_model

      with self.assertRaises(ValueError):
        policy.step_spec(timestep_spec)

  def test_step_spec(self):
    FLAGS.api_key = "mock_test_key"
    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_resource = mock.MagicMock()
      mock_resource.modelServing.return_value = mock.MagicMock()
      mock_build.return_value = mock_resource

      policy = gemini_robotics_policy.GeminiRoboticsPolicy(
          serve_id="test_serve_id",
          task_instruction_key="test_instruction_key",
          image_observation_keys=("test_camera_1",),
          proprioceptive_observation_keys=("test_joint_1",),
          min_replan_interval=3,
          inference_mode=constants.InferenceMode.SYNCHRONOUS,
      )

      # Action is size 2, with chunk of size 3.
      mock_query_model = mock.MagicMock(
          return_value=np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
      )
      policy._query_model = mock_query_model

      timestep_spec = gdmr_types.TimeStepSpec(
          step_type=gdmr_types.STEP_TYPE_SPEC,
          reward={},
          discount={},
          observation={
              "test_camera_1": specs.Array(shape=(100, 100, 3), dtype=np.uint8),
              "test_joint_1": specs.Array(shape=(1,), dtype=np.float32),
              "test_instruction_key": specs.StringArray(()),
          },
      )
      step_spec = policy.step_spec(timestep_spec)
      (action_spec, extra_output_spec), policy_state_spec = step_spec
      self.assertEqual(
          action_spec,
          gdmr_types.UnboundedArraySpec(shape=(2,), dtype=np.float32),
      )
      self.assertEqual(extra_output_spec, {})
      self.assertEqual(
          policy_state_spec, specs.Array(shape=(), dtype=np.float32)
      )

  def test_step_spec_with_additional_observations(self):
    FLAGS.api_key = "mock_test_key"
    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_resource = mock.MagicMock()
      mock_resource.modelServing.return_value = mock.MagicMock()
      mock_build.return_value = mock_resource

      mock_provider_1 = mock.create_autospec(
          additional_observations_provider.AdditionalObservationsProvider
      )
      mock_provider_1.get_additional_observations_spec.return_value = {
          "additional_obs_1": specs.Array(shape=(1,), dtype=np.float32)
      }
      mock_provider_2 = mock.create_autospec(
          additional_observations_provider.AdditionalObservationsProvider
      )
      mock_provider_2.get_additional_observations_spec.return_value = {
          "additional_obs_2": specs.StringArray(shape=())
      }

      policy = gemini_robotics_policy.GeminiRoboticsPolicy(
          serve_id="test_serve_id",
          task_instruction_key="test_instruction_key",
          image_observation_keys=("test_camera_1",),
          proprioceptive_observation_keys=("test_joint_1",),
          min_replan_interval=3,
          inference_mode=constants.InferenceMode.SYNCHRONOUS,
          additional_observations_providers=[mock_provider_1, mock_provider_2],
      )

      # Action is size 2, with chunk of size 3.
      mock_query_model = mock.MagicMock(
          return_value=np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
      )
      policy._query_model = mock_query_model

      timestep_spec = gdmr_types.TimeStepSpec(
          step_type=gdmr_types.STEP_TYPE_SPEC,
          reward={},
          discount={},
          observation={
              "test_camera_1": specs.Array(shape=(100, 100, 3), dtype=np.uint8),
              "test_joint_1": specs.Array(shape=(1,), dtype=np.float32),
              "test_instruction_key": specs.StringArray(()),
          },
      )
      policy.step_spec(timestep_spec)

      timestep_spec_after_call = policy._timestep_spec
      self.assertIsNotNone(timestep_spec_after_call)
      self.assertIn("additional_obs_1", timestep_spec_after_call.observation)
      self.assertEqual(
          timestep_spec_after_call.observation["additional_obs_1"],
          specs.Array(shape=(1,), dtype=np.float32),
      )
      self.assertIn("additional_obs_2", timestep_spec_after_call.observation)
      self.assertEqual(
          timestep_spec_after_call.observation["additional_obs_2"],
          specs.StringArray(shape=()),
      )

  def test_step_spec_with_additional_observations_to_ensure_spec_is_added(self):
    FLAGS.api_key = "mock_test_key"
    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_resource = mock.MagicMock()
      mock_resource.modelServing.return_value = mock.MagicMock()
      mock_build.return_value = mock_resource

      mock_provider_1 = mock.create_autospec(
          additional_observations_provider.AdditionalObservationsProvider
      )
      mock_provider_1.get_additional_observations_spec.return_value = {
          "additional_obs_1": specs.Array(shape=(1,), dtype=np.float32)
      }
      mock_provider_2 = mock.create_autospec(
          additional_observations_provider.AdditionalObservationsProvider
      )
      mock_provider_2.get_additional_observations_spec.return_value = {
          "additional_obs_2": specs.StringArray(shape=())
      }

      policy = gemini_robotics_policy.GeminiRoboticsPolicy(
          serve_id="test_serve_id",
          task_instruction_key="test_instruction_key",
          image_observation_keys=("test_camera_1",),
          proprioceptive_observation_keys=("test_joint_1",),
          min_replan_interval=3,
          inference_mode=constants.InferenceMode.SYNCHRONOUS,
          additional_observations_providers=[mock_provider_1, mock_provider_2],
      )
      self.assertIn("additional_obs_1", policy._proprioceptive_observation_keys)
      self.assertIn("additional_obs_2", policy._string_observations_keys)

  def test_step_with_additional_observations_provider(self):
    FLAGS.api_key = "mock_test_key"
    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_resource = mock.MagicMock()
      mock_resource.modelServing.return_value = mock.MagicMock()
      mock_build.return_value = mock_resource

      mock_provider_1 = mock.create_autospec(
          additional_observations_provider.AdditionalObservationsProvider
      )
      mock_provider_1.get_additional_observations_spec.return_value = {
          "additional_obs_1": specs.Array(shape=(1,), dtype=np.float32)
      }
      mock_provider_1.get_additional_observations.return_value = {
          "additional_obs_1": np.array([42.0], dtype=np.float32)
      }
      mock_provider_2 = mock.create_autospec(
          additional_observations_provider.AdditionalObservationsProvider
      )
      mock_provider_2.get_additional_observations_spec.return_value = {
          "additional_obs_2": specs.StringArray(shape=())
      }
      mock_provider_2.get_additional_observations.return_value = {
          "additional_obs_2": np.array("hello", dtype=np.str_)
      }

      policy = gemini_robotics_policy.GeminiRoboticsPolicy(
          serve_id="test_serve_id",
          task_instruction_key="test_instruction_key",
          image_observation_keys=("test_camera_1",),
          proprioceptive_observation_keys=("test_joint_1",),
          min_replan_interval=3,
          inference_mode=constants.InferenceMode.SYNCHRONOUS,
          additional_observations_providers=[mock_provider_1, mock_provider_2],
      )

      returned_action = np.array([[1.0], [2.0], [3.0]])
      encoded_response = mock.MagicMock()
      encoded_response.text = json.dumps(
          {"action_chunk": returned_action.tolist()}
      )

      policy._client.models.generate_content = mock.MagicMock(
          return_value=encoded_response
      )

      timestep_spec = gdmr_types.TimeStepSpec(
          step_type=gdmr_types.STEP_TYPE_SPEC,
          reward={},
          discount={},
          observation={
              "test_camera_1": specs.Array(shape=(100, 100, 3), dtype=np.uint8),
              "test_joint_1": specs.Array(shape=(1,), dtype=np.float32),
              "test_instruction_key": specs.StringArray(()),
          },
      )
      policy.step_spec(timestep_spec)
      policy_state = policy.initial_state()
      mock_provider_1.reset.assert_called_once()
      mock_provider_2.reset.assert_called_once()

      observation = {
          "test_camera_1": np.zeros((100, 100, 3), dtype=np.uint8),
          "test_joint_1": np.array([0.0]),
          "test_instruction_key": np.array(
              "test_task_instruction", dtype=np.object_
          ),
      }

      # Spy on _query_model to check that the observation is updated.
      with mock.patch.object(
          policy, "_query_model", wraps=policy._query_model
      ) as mock_query_model:
        timestep = dm_env.transition(
            reward=0.0, discount=1.0, observation=observation
        )
        # First step, should trigger a query.
        (action, unused_extra), _ = policy.step(
            timestep,
            policy_state,
        )

        mock_provider_1.get_additional_observations.assert_called_once()
        # The first argument to get_additional_observations is the timestep,
        # and the second is should_replan, which is True on the first step.
        called_timestep, should_replan = (
            mock_provider_1.get_additional_observations.call_args[0]
        )
        self.assertEqual(called_timestep, timestep)
        self.assertTrue(should_replan)

        mock_provider_2.get_additional_observations.assert_called_once()
        called_timestep, should_replan = (
            mock_provider_2.get_additional_observations.call_args[0]
        )
        self.assertEqual(called_timestep, timestep)
        self.assertTrue(should_replan)

        mock_query_model.assert_called_once()
        observation_to_model = mock_query_model.call_args[0][0]

        self.assertIn("additional_obs_1", observation_to_model)
        np.testing.assert_array_equal(
            observation_to_model["additional_obs_1"], np.array([42.0])
        )
        self.assertIn("additional_obs_2", observation_to_model)
        np.testing.assert_array_equal(
            observation_to_model["additional_obs_2"],
            np.array("hello", dtype=np.str_),
        )

      self.assertEqual(action, [1.0])

  def test_step_policy(self):
    FLAGS.api_key = "mock_test_key"
    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_resource = mock.MagicMock()
      mock_resource.modelServing.return_value = mock.MagicMock()
      mock_build.return_value = mock_resource

      policy = gemini_robotics_policy.GeminiRoboticsPolicy(
          serve_id="test_serve_id",
          task_instruction_key="test_instruction_key",
          image_observation_keys=("test_camera_1",),
          proprioceptive_observation_keys=("test_joint_1",),
          min_replan_interval=3,
          inference_mode=constants.InferenceMode.SYNCHRONOUS,
      )

      returned_action = np.array([[1.0], [2.0], [3.0]])
      encoded_response = mock.MagicMock()
      encoded_response.text = json.dumps(
          {"action_chunk": returned_action.tolist()}
      )

      policy._client.models.generate_content = mock.MagicMock(
          return_value=encoded_response
      )

      timestep_spec = gdmr_types.TimeStepSpec(
          step_type=gdmr_types.STEP_TYPE_SPEC,
          reward={},
          discount={},
          observation={
              "test_camera_1": specs.Array(shape=(100, 100, 3), dtype=np.uint8),
              "test_joint_1": specs.Array(shape=(1,), dtype=np.float32),
              "test_instruction_key": specs.StringArray(()),
          },
      )
      policy.step_spec(timestep_spec)
      policy_state = policy.initial_state()

      observation = {
          "test_camera_1": np.zeros((100, 100, 3), dtype=np.uint8),
          "test_joint_1": np.array([0.0]),
          "test_instruction_key": np.array(
              "test_task_instruction", dtype=np.object_
          ),
      }

      expected_contents = {}
      expected_contents[constants.TASK_INSTRUCTION_ENCODED_OBS_KEY] = (
          "test_task_instruction"
      )
      expected_contents["test_joint_1"] = np.array([0.0]).tolist()
      expected_contents[
          f"{constants.IMAGE_ENCODED_OBS_PREFIX}test_camera_1"
      ] = 0

      policy._client.models.generate_content.reset_mock()
      # First step, should trigger a query.
      (action, unused_extra), policy_state = policy.step(
          dm_env.transition(reward=0.0, discount=1.0, observation=observation),
          policy_state,
      )

      policy._client.models.generate_content.assert_called_once()
      _, call_kwargs = policy._client.models.generate_content.call_args

      # The contents data structure is complex and direct assertion does not
      # work. Assert on the single components.
      self.assertEqual(call_kwargs["model"], "test_serve_id")
      call_contents = call_kwargs["contents"]
      self.assertLen(call_contents, 2)
      np.testing.assert_equal(
          call_contents[0], np.zeros((100, 100, 3), dtype=np.uint8)
      )
      self.assertEqual(call_contents[1], json.dumps(expected_contents))

      self.assertEqual(action, [1.0])
      policy._client.models.generate_content.reset_mock()

      # Second step, should not trigger a query.
      (action, unused_extra), policy_state = policy.step(
          dm_env.transition(reward=0.0, discount=1.0, observation=observation),
          policy_state,
      )
      self.assertEqual(action, [2.0])
      policy._client.models.generate_content.assert_not_called()

      # Third step, should not trigger a query.
      (action, unused_extra), policy_state = policy.step(
          dm_env.transition(reward=0.0, discount=1.0, observation=observation),
          policy_state,
      )
      policy._client.models.generate_content.assert_not_called()

      self.assertEqual(action, [3.0])
      # Fourth step, should trigger a query.
      (action, unused_extra), unused_policy_state = policy.step(
          dm_env.transition(reward=0.0, discount=1.0, observation=observation),
          policy_state,
      )
      self.assertEqual(action, [1.0])
      policy._client.models.generate_content.assert_called_once()

  def test_initialize_async_policy(self):
    FLAGS.api_key = "mock_test_key"
    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_resource = mock.MagicMock()
      mock_resource.modelServing.return_value = mock.MagicMock()
      mock_build.return_value = mock_resource

      policy = gemini_robotics_policy.GeminiRoboticsPolicy(
          serve_id="test_serve_id",
          task_instruction_key="test_instruction_key",
          image_observation_keys=(
              "test_camera_1",
              "test_camera_2",
          ),
          proprioceptive_observation_keys=(
              "test_joint_1",
              "test_joint_2",
          ),
          inference_mode=constants.InferenceMode.ASYNCHRONOUS,
      )
      self.assertIsInstance(
          policy._client, gemini_robotics_policy.genai_robotics.Client
      )

  def test_step_async_policy(self):
    FLAGS.api_key = "mock_test_key"
    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_resource = mock.MagicMock()
      mock_resource.modelServing.return_value = mock.MagicMock()
      mock_build.return_value = mock_resource

      policy = gemini_robotics_policy.GeminiRoboticsPolicy(
          serve_id="test_serve_id",
          task_instruction_key="test_instruction_key",
          image_observation_keys=("test_camera_1",),
          proprioceptive_observation_keys=("test_joint_1",),
          min_replan_interval=3,
          inference_mode=constants.InferenceMode.ASYNCHRONOUS,
      )

      returned_action = np.array([[1.0], [2.0], [3.0]])
      encoded_response = mock.MagicMock()
      encoded_response.text = json.dumps(
          {"action_chunk": returned_action.tolist()}
      )

      policy._client.models.generate_content = mock.MagicMock(
          return_value=encoded_response
      )

      timestep_spec = gdmr_types.TimeStepSpec(
          step_type=gdmr_types.STEP_TYPE_SPEC,
          reward={},
          discount={},
          observation={
              "test_camera_1": specs.Array(shape=(100, 100, 3), dtype=np.uint8),
              "test_joint_1": specs.Array(shape=(1,), dtype=np.float32),
              "test_instruction_key": specs.StringArray(()),
          },
      )

      policy.step_spec(timestep_spec)
      policy_state = policy.initial_state()

      observation = {
          "test_camera_1": np.zeros((100, 100, 3), dtype=np.uint8),
          "test_joint_1": np.array([0.0]),
          "test_instruction_key": np.array(
              "test_task_instruction", dtype=np.object_
          ),
      }

      expected_contents = {}
      expected_contents[constants.TASK_INSTRUCTION_ENCODED_OBS_KEY] = (
          "test_task_instruction"
      )
      expected_contents["test_joint_1"] = np.array([0.0]).tolist()
      expected_contents[
          f"{constants.IMAGE_ENCODED_OBS_PREFIX}test_camera_1"
      ] = 0

      policy._client.models.generate_content.reset_mock()

      # First step, should trigger a query.
      (action, unused_extra), policy_state = policy.step(
          dm_env.transition(reward=0.0, discount=1.0, observation=observation),
          policy_state,
      )

      policy._client.models.generate_content.assert_called_once()
      _, call_kwargs = policy._client.models.generate_content.call_args

      # The contents data structure is complex and direct assertion does not
      # work. Assert on the single components.
      self.assertEqual(call_kwargs["model"], "test_serve_id")
      call_contents = call_kwargs["contents"]
      self.assertLen(call_contents, 2)
      np.testing.assert_equal(
          call_contents[0], np.zeros((100, 100, 3), dtype=np.uint8)
      )
      self.assertEqual(call_contents[1], json.dumps(expected_contents))

      self.assertEqual(action, [1.0])
      policy._client.models.generate_content.reset_mock()

      # Second step, should not trigger a query.
      (action, unused_extra), policy_state = policy.step(
          dm_env.transition(reward=0.0, discount=1.0, observation=observation),
          policy_state,
      )
      self.assertEqual(action, [2.0])
      policy._client.models.generate_content.assert_not_called()

      # Third step, should not trigger a query.
      (action, unused_extra), policy_state = policy.step(
          dm_env.transition(reward=0.0, discount=1.0, observation=observation),
          policy_state,
      )
      self.assertEqual(action, [3.0])
      policy._client.models.generate_content.assert_not_called()

      # Fourth step, should trigger a query.
      (action, unused_extra), unused_policy_state = policy.step(
          dm_env.transition(reward=0.0, discount=1.0, observation=observation),
          policy_state,
      )
      self.assertEqual(action, [1.0])
      policy._client.models.generate_content.assert_called_once()


if __name__ == "__main__":
  absltest.main()
