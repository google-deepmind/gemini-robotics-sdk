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

from absl import flags
from absl.testing import absltest
import numpy as np

from safari_sdk.model import gemini_robotics_policy

FLAGS = flags.FLAGS
FLAGS.mark_as_parsed()


class GeminiRoboticsPolicyTest(absltest.TestCase):

  def test_initialize_policy(self):

    FLAGS.api_key = "mock_test_key"
    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_resource = mock.MagicMock()
      mock_resource.modelServing.return_value = mock.MagicMock()
      mock_build.return_value = mock_resource

      policy = gemini_robotics_policy.GeminiRoboticsPolicy(
          serve_id="test_serve_id",
          task_instruction="test_task_instruction",
          cameras={
              "test_camera_1": (100, 100),
              "test_camera_2": (200, 200),
          },
          joints={
              "test_joint_1": 1,
              "test_joint_2": 2,
          },
      )
      self.assertIsInstance(
          policy._client, gemini_robotics_policy.genai_robotics.Client
      )

  def test_step_policy(self):
    FLAGS.api_key = "mock_test_key"
    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_resource = mock.MagicMock()
      mock_resource.modelServing.return_value = mock.MagicMock()
      mock_build.return_value = mock_resource

      policy = gemini_robotics_policy.GeminiRoboticsPolicy(
          serve_id="test_serve_id",
          task_instruction="test_task_instruction",
          cameras={
              "test_camera_1": (100, 100),
          },
          joints={
              "test_joint_1": 1,
          },
          min_replan_interval=3,
      )
      mock_query_model = mock.MagicMock(
          return_value=np.array([[1.0], [2.0], [3.0]])
      )
      policy._query_model = mock_query_model
      observation = {
          "test_camera_1": np.zeros((100, 100, 3), dtype=np.uint8),
          "test_joint_1": np.array([0.0]),
      }
      # First step, should trigger a query.
      action = policy.step(observation)
      self.assertEqual(action, [1.0])
      # Second step, should not trigger a query.
      action = policy.step(observation)
      self.assertEqual(action, [2.0])
      # Third step, should not trigger a query.
      action = policy.step(observation)
      self.assertEqual(action, [3.0])
      # Fourth step, should trigger a query.
      action = policy.step(observation)
      self.assertEqual(action, [1.0])

  def test_initialize_async_policy(self):
    FLAGS.api_key = "mock_test_key"
    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_resource = mock.MagicMock()
      mock_resource.modelServing.return_value = mock.MagicMock()
      mock_build.return_value = mock_resource

      policy = gemini_robotics_policy.GeminiRoboticsPolicy(
          serve_id="test_serve_id",
          task_instruction="test_task_instruction",
          cameras={
              "test_camera_1": (100, 100),
              "test_camera_2": (200, 200),
          },
          joints={
              "test_joint_1": 1,
              "test_joint_2": 2,
          },
          inference_mode=gemini_robotics_policy.InferenceMode.ASYNCHRONOUS,
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
          task_instruction="test_task_instruction",
          cameras={
              "test_camera_1": (100, 100),
          },
          joints={
              "test_joint_1": 1,
          },
          min_replan_interval=3,
          inference_mode=gemini_robotics_policy.InferenceMode.ASYNCHRONOUS,
      )
      mock_query_model = mock.MagicMock(
          return_value=np.array([[1.0], [2.0], [3.0]])
      )
      policy._query_model = mock_query_model
      observation = {
          "test_camera_1": np.zeros((100, 100, 3), dtype=np.uint8),
          "test_joint_1": np.array([0.0]),
      }
      # First step, should trigger a query.
      action = policy.step(observation)
      self.assertEqual(action, [1.0])
      # Second step, should not trigger a query.
      action = policy.step(observation)
      self.assertEqual(action, [2.0])
      # Third step, should not trigger a query.
      action = policy.step(observation)
      self.assertEqual(action, [3.0])
      # Fourth step, should trigger a query.
      action = policy.step(observation)
      self.assertEqual(action, [1.0])


if __name__ == "__main__":
  absltest.main()
