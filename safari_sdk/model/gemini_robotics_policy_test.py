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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import dm_env
from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np

from safari_sdk.model import additional_observations_provider
from safari_sdk.model import constants
from safari_sdk.model import gemini_robotics_policy
from safari_sdk.model import remote_model_interface


class GeminiRoboticsPolicyTest(parameterized.TestCase):

  def test_raise_error_if_step_spec_not_called(self):

    with mock.patch.object(remote_model_interface, "RemoteModelInterface"):
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
    with mock.patch.object(
        remote_model_interface, "RemoteModelInterface", autospec=True
    ) as remote_model_mock_class:
      # We mock the class so we need to get the mock instance.
      remote_model_mock = remote_model_mock_class.return_value

      policy = gemini_robotics_policy.GeminiRoboticsPolicy(
          serve_id="test_serve_id",
          task_instruction_key="test_instruction_key",
          image_observation_keys=("test_camera_1", "test_camera_2"),
          proprioceptive_observation_keys=("test_joint_1", "test_joint_2"),
          min_replan_interval=3,
          inference_mode=constants.InferenceMode.SYNCHRONOUS,
      )

      # Action is size 2, with chunk of size 3.
      remote_model_mock.query_model.return_value = np.array(
          [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
      )

      with self.assertRaises(ValueError):
        policy.step_spec(timestep_spec)

  def test_step_spec(self):
    with mock.patch.object(
        remote_model_interface, "RemoteModelInterface", autospec=True
    ) as remote_model_mock_class:
      # We mock the class so we need to get the mock instance.
      remote_model_mock = remote_model_mock_class.return_value

      policy = gemini_robotics_policy.GeminiRoboticsPolicy(
          serve_id="test_serve_id",
          task_instruction_key="test_instruction_key",
          image_observation_keys=("test_camera_1",),
          proprioceptive_observation_keys=("test_joint_1",),
          min_replan_interval=3,
          inference_mode=constants.InferenceMode.SYNCHRONOUS,
      )

      # Action is size 2, with chunk of size 3.
      remote_model_mock.query_model.return_value = np.array(
          [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
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
    with mock.patch.object(
        remote_model_interface, "RemoteModelInterface", autospec=True
    ) as remote_model_mock_class:
      # We mock the class so we need to get the mock instance.
      remote_model_mock = remote_model_mock_class.return_value

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
      remote_model_mock.query_model.return_value = np.array(
          [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
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
    with mock.patch.object(
        remote_model_interface, "RemoteModelInterface", autospec=True
    ):

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
    with mock.patch.object(
        remote_model_interface, "RemoteModelInterface", autospec=True
    ) as remote_model_mock_class:
      # We mock the class so we need to get the mock instance.
      remote_model_mock = remote_model_mock_class.return_value

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

      remote_model_mock.query_model.return_value = np.array(
          [[1.0], [2.0], [3.0]]
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
    with mock.patch.object(
        remote_model_interface,
        "RemoteModelInterface",
        autospec=True,
    ) as remote_model_mock_class:
      # We mock the class so we need to get the mock instance.
      remote_model_mock = remote_model_mock_class.return_value

      policy = gemini_robotics_policy.GeminiRoboticsPolicy(
          serve_id="test_serve_id",
          task_instruction_key="test_instruction_key",
          image_observation_keys=("test_camera_1",),
          proprioceptive_observation_keys=("test_joint_1",),
          min_replan_interval=3,
          inference_mode=constants.InferenceMode.SYNCHRONOUS,
      )

      remote_model_mock.query_model.return_value = np.array(
          [[1.0], [2.0], [3.0]]
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

      # This resets the count of calls to query_model.
      remote_model_mock.query_model.reset_mock()

      # First step, should trigger a query.
      (action, unused_extra), policy_state = policy.step(
          dm_env.transition(reward=0.0, discount=1.0, observation=observation),
          policy_state,
      )

      remote_model_mock.query_model.assert_called_once()
      call_args, _ = remote_model_mock.query_model.call_args
      model_observation = call_args[0]

      np.testing.assert_equal(model_observation, observation)
      np.testing.assert_equal(action, [1.0])

      remote_model_mock.query_model.reset_mock()

      # Second step, should not trigger a query.
      (action, unused_extra), policy_state = policy.step(
          dm_env.transition(reward=0.0, discount=1.0, observation=observation),
          policy_state,
      )
      np.testing.assert_equal(action, [2.0])
      remote_model_mock.query_model.assert_not_called()

      # Third step, should not trigger a query.
      (action, unused_extra), policy_state = policy.step(
          dm_env.transition(reward=0.0, discount=1.0, observation=observation),
          policy_state,
      )
      remote_model_mock.query_model.assert_not_called()

      np.testing.assert_equal(action, [3.0])
      # Fourth step, should trigger a query.
      (action, unused_extra), unused_policy_state = policy.step(
          dm_env.transition(reward=0.0, discount=1.0, observation=observation),
          policy_state,
      )
      np.testing.assert_equal(action, [1.0])
      remote_model_mock.query_model.assert_called_once()

  def test_step_async_policy(self):
    with mock.patch.object(
        remote_model_interface, "RemoteModelInterface", autospec=True
    ) as remote_model_mock_class:
      # We mock the class so we need to get the mock instance.
      remote_model_mock = remote_model_mock_class.return_value

      policy = gemini_robotics_policy.GeminiRoboticsPolicy(
          serve_id="test_serve_id",
          task_instruction_key="test_instruction_key",
          image_observation_keys=("test_camera_1",),
          proprioceptive_observation_keys=("test_joint_1",),
          min_replan_interval=3,
          inference_mode=constants.InferenceMode.ASYNCHRONOUS,
      )

      # Action is size 2, with chunk of size 3.
      remote_model_mock.query_model.return_value = np.array(
          [[1.0], [2.0], [3.0]]
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

      remote_model_mock.query_model.reset_mock()

      # First step, should trigger a query.
      (action, unused_extra), policy_state = policy.step(
          dm_env.transition(reward=0.0, discount=1.0, observation=observation),
          policy_state,
      )

      remote_model_mock.query_model.assert_called_once()
      call_args, _ = remote_model_mock.query_model.call_args
      model_observation = call_args[0]

      np.testing.assert_equal(model_observation, observation)

      np.testing.assert_equal(action, [1.0])
      remote_model_mock.query_model.reset_mock()

      # Second step, should not trigger a query.
      (action, unused_extra), policy_state = policy.step(
          dm_env.transition(reward=0.0, discount=1.0, observation=observation),
          policy_state,
      )
      np.testing.assert_equal(action, [2.0])
      remote_model_mock.query_model.assert_not_called()

      # Third step, should not trigger a query.
      (action, unused_extra), policy_state = policy.step(
          dm_env.transition(reward=0.0, discount=1.0, observation=observation),
          policy_state,
      )
      np.testing.assert_equal(action, [3.0])
      remote_model_mock.query_model.assert_not_called()

      # Fourth step, should trigger a query.
      (action, unused_extra), unused_policy_state = policy.step(
          dm_env.transition(reward=0.0, discount=1.0, observation=observation),
          policy_state,
      )
      np.testing.assert_equal(action, [1.0])
      remote_model_mock.query_model.assert_called_once()

  def test_async_policy_increases_action_stall_count(self):
    with mock.patch.object(
        remote_model_interface, "RemoteModelInterface", autospec=True
    ) as remote_model_mock_class:
      # We mock the class so we need to get the mock instance.
      remote_model_mock = remote_model_mock_class.return_value

      policy = gemini_robotics_policy.GeminiRoboticsPolicy(
          serve_id="test_serve_id",
          task_instruction_key="test_instruction_key",
          image_observation_keys=("test_camera_1",),
          proprioceptive_observation_keys=("test_joint_1",),
          min_replan_interval=1,
          inference_mode=constants.InferenceMode.ASYNCHRONOUS,
      )

      # Action is size 1, with chunk of size 1.
      remote_model_mock.query_model.return_value = np.array([[1.0]])

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

      remote_model_mock.query_model.reset_mock()

      # First step, should trigger a query, not increase action_stall_count.
      policy.step(
          dm_env.restart(observation=observation),
          policy_state,
      )
      self.assertEqual(policy.episode_statistics.action_stall_count, 0)

      # Second step, should trigger a query and increase action_stall_count.
      policy.step(
          dm_env.transition(reward=0.0, discount=1.0, observation=observation),
          policy_state,
      )
      self.assertEqual(policy.episode_statistics.action_stall_count, 1)

  def test_model_action_not_2d_raises_error(self):
    with mock.patch.object(
        remote_model_interface, "RemoteModelInterface", autospec=True
    ) as remote_model_mock_class:
      # We mock the class so we need to get the mock instance.
      remote_model_mock = remote_model_mock_class.return_value

      policy = gemini_robotics_policy.GeminiRoboticsPolicy(
          serve_id="test_serve_id",
          task_instruction_key="test_instruction_key",
          image_observation_keys=("test_camera_1",),
          proprioceptive_observation_keys=("test_joint_1",),
          min_replan_interval=3,
          inference_mode=constants.InferenceMode.SYNCHRONOUS,
      )

      remote_model_mock.query_model.return_value = np.array([1.0, 2.0])

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
      with self.assertRaisesRegex(
          ValueError,
          "Action returned by the model must be a 2D array",
      ):
        policy.step_spec(timestep_spec)

  @parameterized.named_parameters(
      (
          "action_conditioning_chunk_length_none",
          None,
          [[2.0], [3.0], [4.0], [5.0]],
      ),
      (
          "action_conditioning_chunk_length_smaller_than_actions_left",
          1,
          [[2.0]],
      ),
  )
  def test_step_async_policy_with_action_conditioning_chunk_length(
      self, action_conditioning_chunk_length, expected_conditioning_chunk
  ):
    with mock.patch.object(
        remote_model_interface, "RemoteModelInterface", autospec=True
    ) as remote_model_mock_class:
      remote_model_mock = remote_model_mock_class.return_value
      policy = gemini_robotics_policy.GeminiRoboticsPolicy(
          serve_id="test_serve_id",
          task_instruction_key="test_instruction_key",
          image_observation_keys=("test_camera_1",),
          proprioceptive_observation_keys=("test_joint_1",),
          min_replan_interval=1,
          inference_mode=constants.InferenceMode.ASYNCHRONOUS,
          action_conditioning_chunk_length=action_conditioning_chunk_length,
      )

      # Action chunk is size 5.
      actions = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
      remote_model_mock.query_model.return_value = actions

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

      remote_model_mock.query_model.reset_mock()

      # First step.
      # Buffer is empty.
      # Should call query_model.
      # Returns [1, 2, 3].
      # Consumes [1]. Buffer: [2, 3].
      (action, unused_extra), policy_state = policy.step(
          dm_env.transition(reward=0.0, discount=1.0, observation=observation),
          policy_state,
      )
      remote_model_mock.query_model.assert_called_once()
      call_args, _ = remote_model_mock.query_model.call_args
      model_observation = call_args[0]
      np.testing.assert_equal(model_observation, observation)
      np.testing.assert_equal(action, [1.0])
      remote_model_mock.query_model.reset_mock()

      # Second step.
      # Buffer: [2, 3].
      # actions_left = 2.
      # min_replan = 1.
      # num_per_req = 3.
      # actions_executed_during_inference = 1
      # _should_replan logic:
      #  actions_left = self._model_output.shape[0] -> 2.
      #  3 - 2 = 1.
      #  1 >= 1. True.
      # Triggers new query (async).
      (action, unused_extra), _ = policy.step(
          dm_env.transition(reward=0.0, discount=1.0, observation=observation),
          policy_state,
      )
      # Action is the first element of the buffer: [2].
      np.testing.assert_equal(action, [2.0])
      # Wait for async execution by checking the future.
      # The policy should have submitted a future.
      self.assertIsNotNone(policy._future)
      policy._future.result()  # pytype: disable=attribute-error

      remote_model_mock.query_model.assert_called_once()
      call_args_2 = remote_model_mock.query_model.call_args_list[0]
      obs_2 = call_args_2[0][0]

      conditioning = obs_2.get(constants.CONDITIONING_ENCODED_OBS_KEY)
      self.assertIsNotNone(conditioning)

      # Conditioning should match expected.
      np.testing.assert_allclose(conditioning, expected_conditioning_chunk)

  def test_action_conditioning_chunk_length_validation(self):
    with mock.patch.object(
        remote_model_interface, "RemoteModelInterface", autospec=True
    ) as remote_model_mock_class:
      remote_model_mock = remote_model_mock_class.return_value

      # min_replan_interval=2, chunk_length=2. Sum = 4.
      # returns actions size 3. 3 <= 4. Should raise ValueError.
      policy = gemini_robotics_policy.GeminiRoboticsPolicy(
          serve_id="test_serve_id",
          task_instruction_key="test_instruction_key",
          image_observation_keys=("test_camera_1",),
          proprioceptive_observation_keys=("test_joint_1",),
          min_replan_interval=2,
          inference_mode=constants.InferenceMode.ASYNCHRONOUS,
          action_conditioning_chunk_length=2,
      )

      # Action is size 3.
      actions = np.array([[1.0], [2.0], [3.0]])
      remote_model_mock.query_model.return_value = actions

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

      with self.assertRaises(
          ValueError,
      ):
        policy.step_spec(timestep_spec)


if __name__ == "__main__":
  absltest.main()
