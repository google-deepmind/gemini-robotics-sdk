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

import base64
import json
from unittest import mock

from absl import flags
from absl.testing import absltest
from google.genai import types
import numpy as np
import tensorflow as tf

from safari_sdk.model import genai_robotics

FLAGS = flags.FLAGS
FLAGS.mark_as_parsed()


class GenaiRoboticsTest(absltest.TestCase):

  def test_robotics_api_create_client(self):
    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_service = mock.Mock()
      mock_build.return_value = mock_service
      FLAGS.api_key = "test_api_key"

      client = genai_robotics.Client(
          use_robotics_api=True,
      )
      self.assertIsNotNone(client)
      mock_build.assert_called_once_with(
          serviceName=genai_robotics.auth._DEFAULT_SERVICE_NAME,
          version=genai_robotics.auth._DEFAULT_VERSION,
          discoveryServiceUrl=(
              genai_robotics.auth._DEFAULT_DISCOVERY_SERVICE_URL
          ),
          developerKey="test_api_key",
          http=mock.ANY,
      )

  def test_robotics_api_generate_content(self):
    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_service = mock.Mock()
      mock_build.return_value = mock_service
      FLAGS.api_key = "test_api_key"

      client = genai_robotics.Client(
          use_robotics_api=True,
      )
      image = np.zeros((100, 100, 3), dtype=np.uint8)
      image_bytes = tf.io.encode_jpeg(image).numpy()
      expected_output = {"action_chunk": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}

      mock_cm_custom = mock_service.modelServing.return_value.cmCustom
      mock_cm_custom.return_value.execute.return_value = {
          "outputBytes": (
              base64.b64encode(
                  json.dumps(expected_output).encode("utf-8")
              ).decode("utf-8")
          ),
          "someOtherKey": "some_other_value",
      }

      obs = {
          "images/overhead_cam": 0,
          "task_instruction": "test_task_instruction",
          "joints_pos": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      }

      response = client.models.generate_content(
          model="test_model",
          contents=[
              types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
              json.dumps(obs),
          ],
      )
      self.assertEqual(response.text, json.dumps(expected_output))
      mock_cm_custom.assert_called_once()
      call_body = mock_cm_custom.call_args.kwargs["body"]
      self.assertEqual(call_body["modelId"], "test_model")
      self.assertEqual(call_body["methodName"], "sample_actions_json_flat")
      self.assertIsInstance(call_body["requestId"], int)
      query = json.loads(
          base64.b64decode(call_body["inputBytes"]).decode("utf-8")
      )
      self.assertEqual(
          query["images/overhead_cam"],
          base64.b64encode(image_bytes).decode("utf-8"),
      )
      self.assertEqual(query["task_instruction"], "test_task_instruction")
      self.assertEqual(query["joints_pos"], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

  def test_genai_create_client_via_auth_library(self):
    with mock.patch("google.genai.Client") as mock_genai_client:
      mock_client = mock.Mock()
      mock_genai_client.return_value = mock_client
      FLAGS.api_key = "test_api_key"

      client = genai_robotics.Client(
          use_robotics_api=False, project="test_project"
      )
      self.assertIsNotNone(client)
      mock_genai_client.assert_called_once_with(
          api_key="test_api_key", project="test_project"
      )

  def test_genai_create_client_via_param(self):
    with mock.patch("google.genai.Client") as mock_genai_client:
      mock_client = mock.Mock()
      mock_genai_client.return_value = mock_client
      FLAGS.api_key = None

      client = genai_robotics.Client(
          api_key="test_api_key", use_robotics_api=False, project="test_project"
      )
      self.assertIsNotNone(client)
      mock_genai_client.assert_called_once_with(
          api_key="test_api_key", project="test_project"
      )


if __name__ == "__main__":
  absltest.main()
