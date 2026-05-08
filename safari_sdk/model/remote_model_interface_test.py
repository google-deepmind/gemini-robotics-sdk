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
from dm_env import specs
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized
from safari_sdk.model import additional_observations_provider
from safari_sdk.model import constants
from safari_sdk.model import genai_robotics
from safari_sdk.model import observation_to_model_query_contents
from safari_sdk.model import remote_model_interface

FLAGS = flags.FLAGS
FLAGS.mark_as_parsed()


class RemoteModelInterfaceTest(parameterized.TestCase):

  def test_remote_model_is_queried(self):
    FLAGS.api_key = "mock_test_key"
    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_resource = mock.MagicMock()
      mock_resource.modelServing.return_value = mock.MagicMock()
      mock_build.return_value = mock_resource

      remote_model = remote_model_interface.RemoteModelInterface(
          serve_id="test_serve_id",
          robotics_api_connection=constants.RoboticsApiConnectionType.CLOUD,
          task_instruction_key="test_instruction_key",
          proprioceptive_observation_keys=("test_joint_1", "test_joint_2"),
          image_observation_keys=("test_camera_1", "test_camera_2"),
          image_compression_jpeg_quality=75,
      )

      returned_action = np.array([[1.0], [2.0], [3.0]])
      encoded_response = mock.MagicMock()
      encoded_response.text = json.dumps(
          {"action_chunk": returned_action.tolist()}
      )

      remote_model._client.models.generate_content = mock.MagicMock(
          return_value=encoded_response
      )

      observation = {
          "test_camera_1": np.zeros((100, 100, 3), dtype=np.uint8),
          "test_camera_2": np.zeros((75, 60, 3), dtype=np.uint8),
          "test_joint_1": np.array([0.0]),
          "test_joint_2": np.array([1.0, 2.0]),
          "test_instruction_key": np.array(
              "test_task_instruction", dtype=np.object_
          ),
      }

      expected_serialized_contents = observation_to_model_query_contents.observation_to_model_query_contents(
          observation=observation,
          string_observations_keys=("test_instruction_key",),
          task_instruction_key="test_instruction_key",
          proprioceptive_observation_keys=("test_joint_1", "test_joint_2"),
          image_observation_keys=("test_camera_1", "test_camera_2"),
      )

      returned_action = remote_model.query_model(observation)

      remote_model._client.models.generate_content.assert_called_once()
      _, call_kwargs = remote_model._client.models.generate_content.call_args

      # The contents data structure is complex and direct assertion does not
      # work. Assert on the single components.
      self.assertEqual(call_kwargs["model"], "test_serve_id")
      call_contents = call_kwargs["contents"]

      # The call should be the two images followed by the JSON serialized
      # observation dictionary.
      self.assertLen(call_contents, 3)
      np.testing.assert_equal(
          call_contents[0], np.zeros((100, 100, 3), dtype=np.uint8)
      )
      np.testing.assert_equal(
          call_contents[1], np.zeros((75, 60, 3), dtype=np.uint8)
      )

      # Compare the JSON serialized observation dictionary.
      self.assertEqual(call_contents[2], expected_serialized_contents[-1])

      np.testing.assert_equal(returned_action, np.array([[1.0], [2.0], [3.0]]))

  def test_cloud_genai_has_observations_updated(self):
    FLAGS.api_key = "mock_test_key"
    mock_build = self.enter_context(
        mock.patch("googleapiclient.discovery.build")
    )
    genai_robotics_mock = self.enter_context(
        mock.patch.object(
            genai_robotics, "update_robotics_content_to_genai_format"
        )
    )

    mock_resource = mock.MagicMock()
    mock_resource.modelServing.return_value = mock.MagicMock()
    mock_build.return_value = mock_resource

    remote_model = remote_model_interface.RemoteModelInterface(
        serve_id="test_serve_id",
        robotics_api_connection=constants.RoboticsApiConnectionType.CLOUD_GENAI,
        task_instruction_key="test_instruction_key",
        proprioceptive_observation_keys=("test_joint_1",),
        image_observation_keys=("test_camera_1",),
        image_compression_jpeg_quality=75,
    )

    returned_action = np.array([[1.0], [2.0], [3.0]])
    encoded_response = mock.MagicMock()
    encoded_response.text = json.dumps(
        {"action_chunk": returned_action.tolist()}
    )

    remote_model._client.models.generate_content = mock.MagicMock(
        return_value=encoded_response
    )

    observation = {
        "test_camera_1": np.zeros((100, 100, 3), dtype=np.uint8),
        "test_joint_1": np.array([0.0]),
        "test_instruction_key": np.array(
            "test_task_instruction", dtype=np.object_
        ),
    }

    expected_serialized_contents = (
        observation_to_model_query_contents.observation_to_model_query_contents(
            observation=observation,
            string_observations_keys=("test_instruction_key",),
            task_instruction_key="test_instruction_key",
            proprioceptive_observation_keys=("test_joint_1",),
            image_observation_keys=("test_camera_1",),
        )
    )

    remote_model.query_model(observation)

    genai_robotics_mock.assert_called_once()

    call_args, call_kwargs = genai_robotics_mock.call_args

    self.assertEqual(call_args[0], expected_serialized_contents)
    self.assertEqual(call_kwargs["image_compression_jpeg_quality"], 75)

  @parameterized.named_parameters(
      dict(
          testcase_name="non_single_object_response",
          response=mock.MagicMock(text=json.dumps([[1.0], [2.0], [3.0]])),
          assert_text=(
              "Response data does not have a single object as root object."
          ),
      ),
      dict(
          testcase_name="non_action_chunk_response",
          response=mock.MagicMock(
              text=json.dumps({"action": [[[1.0], [2.0], [3.0]]]})
          ),
          assert_text="Response JSON does not contain 'action_chunk'",
      ),
      dict(
          # Note that internally the code accepts 2 or 3 dimensions so fail for
          # 4.
          testcase_name="action_chunk_has_more_than_2_dimensions",
          response=mock.MagicMock(
              text=json.dumps({"action_chunk": [[[[1.0], [2.0], [3.0]]]]})
          ),
          assert_text="Action chunk has more than 2 dimensions",
      ),
  )
  def test_response_does_not_contain_action_chunk_raises_error(
      self, response: mock.MagicMock, assert_text: str
  ):
    FLAGS.api_key = "mock_test_key"
    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_resource = mock.MagicMock()
      mock_resource.modelServing.return_value = mock.MagicMock()
      mock_build.return_value = mock_resource

      remote_model = remote_model_interface.RemoteModelInterface(
          serve_id="test_serve_id",
          robotics_api_connection=constants.RoboticsApiConnectionType.CLOUD,
          task_instruction_key="test_instruction_key",
          proprioceptive_observation_keys=("test_joint_1",),
          image_observation_keys=("test_camera_1",),
          image_compression_jpeg_quality=75,
      )

      encoded_response = response

      remote_model._client.models.generate_content = mock.MagicMock(
          return_value=encoded_response
      )

      observation = {
          "test_camera_1": np.zeros((100, 100, 3), dtype=np.uint8),
          "test_joint_1": np.array([0.0]),
          "test_instruction_key": np.array(
              "test_task_instruction", dtype=np.object_
          ),
      }

      with self.assertRaisesRegex(ValueError, assert_text):
        remote_model.query_model(observation)

  def test_additional_observations_are_added(self):
    FLAGS.api_key = "mock_test_key"
    mock_build = self.enter_context(
        mock.patch("googleapiclient.discovery.build")
    )

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

    remote_model = remote_model_interface.RemoteModelInterface(
        serve_id="test_serve_id",
        robotics_api_connection=constants.RoboticsApiConnectionType.CLOUD,
        task_instruction_key="test_instruction_key",
        proprioceptive_observation_keys=("test_joint_1",),
        image_observation_keys=("test_camera_1",),
        image_compression_jpeg_quality=75,
        additional_observations_providers=[mock_provider_1, mock_provider_2],
    )

    self.assertIn(
        "additional_obs_1", remote_model._proprioceptive_observation_keys
    )
    self.assertIn("additional_obs_2", remote_model._string_observations_keys)

    returned_action = np.array([[1.0], [2.0], [3.0]])
    encoded_response = mock.MagicMock()
    encoded_response.text = json.dumps(
        {"action_chunk": returned_action.tolist()}
    )

    remote_model._client.models.generate_content = mock.MagicMock(
        return_value=encoded_response
    )

    observation = {
        "test_camera_1": np.zeros((100, 100, 3), dtype=np.uint8),
        "test_camera_2": np.zeros((75, 60, 3), dtype=np.uint8),
        "test_joint_1": np.array([0.0]),
        "test_joint_2": np.array([1.0, 2.0]),
        "test_instruction_key": np.array(
            "test_task_instruction", dtype=np.object_
        ),
        "additional_obs_1": np.array([1.0]),
        "additional_obs_2": np.array("test_string", dtype=np.object_),
    }

    expected_serialized_contents = (
        observation_to_model_query_contents.observation_to_model_query_contents(
            observation=observation,
            string_observations_keys=(
                "test_instruction_key",
                "additional_obs_2",
            ),
            task_instruction_key="test_instruction_key",
            proprioceptive_observation_keys=(
                "test_joint_1",
                "additional_obs_1",
            ),
            image_observation_keys=("test_camera_1",),
        )
    )

    remote_model.query_model(observation)

    remote_model._client.models.generate_content.assert_called_once()
    _, call_kwargs = remote_model._client.models.generate_content.call_args

    call_contents = call_kwargs["contents"]

    self.assertEqual(call_contents[-1], expected_serialized_contents[-1])

  @mock.patch.object(genai_robotics, "_check_server_compatibility")
  @mock.patch.object(genai_robotics, "_connect_to_grpc_json")
  @mock.patch.object(genai_robotics.grpc, "insecure_channel")
  def test_local_connection_passes_serve_id_as_grpc_url(
      self, mock_insecure_channel, mock_connect_json, mock_compat
  ):
    del mock_compat  # Not asserted; just prevents real server call.
    mock_channel = mock.MagicMock()
    mock_insecure_channel.return_value = mock_channel
    mock_connect_json.return_value = mock.MagicMock()
    remote_model = remote_model_interface.RemoteModelInterface(
        serve_id="grpc://10.0.0.5:10100",
        robotics_api_connection=constants.RoboticsApiConnectionType.LOCAL,
        task_instruction_key="test_instruction_key",
        proprioceptive_observation_keys=("test_joint_1",),
        image_observation_keys=("test_camera_1",),
        image_compression_jpeg_quality=75,
    )
    self.assertIsNotNone(remote_model)
    mock_insecure_channel.assert_called_once_with("10.0.0.5:10100")
    mock_connect_json.assert_called_once_with(
        mock_channel, "sample_actions_json_flat"
    )

if __name__ == "__main__":
  absltest.main()
