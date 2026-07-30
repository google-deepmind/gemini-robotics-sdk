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
import datetime
import json
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from google.genai import types  # pytype: disable=import-error
from googleapiclient.errors import HttpError
import grpc
import msgpack  # pytype: disable=import-error
import numpy as np
import tensorflow as tf

from safari_sdk.model import genai_robotics  # pytype: disable=import-error

FLAGS = flags.FLAGS
FLAGS.mark_as_parsed()


class FakeRpcError(grpc.RpcError):

  def __init__(self, code, details="gRPC error"):
    super().__init__()
    self._code = code
    self._details = details

  def code(self):
    return self._code

  def details(self):
    return self._details


class GenaiRoboticsTest(parameterized.TestCase):

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

  def test_robotics_api_generate_content_legacy_json(self):
    """Tests the legacy JSON protocol (server_version='unknown' / < 2.0.0)."""
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
          "backendRequestTime": "2024-05-01T12:00:00Z",
          "backendResponseTime": "2024-05-01T12:00:01Z",
          "someOtherKey": "some_other_value",
      }

      obs = {
          "images/overhead_cam": 0,
          "task_instruction": "test_task_instruction",
          "joints_pos": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      }

      config = types.GenerateContentConfig(
          http_options=types.HttpOptions(timeout=1500)
      )

      response = client.models.generate_content(
          model="test_model",
          contents=[
              types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
              json.dumps(obs),
          ],
          config=config,
      )
      self.assertEqual(response.text, json.dumps(expected_output))
      mock_cm_custom.assert_called_once()
      call_body = mock_cm_custom.call_args.kwargs["body"]
      self.assertEqual(call_body["modelId"], "test_model")
      self.assertEqual(call_body["methodName"], "sample_actions_json_flat")
      self.assertIsInstance(call_body["requestId"], int)
      self.assertEqual(call_body["modelOptions"]["timeout"]["seconds"], 1)
      self.assertEqual(call_body["modelOptions"]["timeout"]["nanos"], 500000000)
      # Legacy path uses JSON encoding.
      query = json.loads(
          base64.b64decode(call_body["inputBytes"]).decode("utf-8")
      )
      # Images are base64-encoded in the legacy JSON path.
      self.assertEqual(
          query["images/overhead_cam"],
          base64.b64encode(image_bytes).decode("utf-8"),
      )
      self.assertEqual(query["task_instruction"], "test_task_instruction")
      self.assertEqual(query["joints_pos"], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
      self.assertEqual(response.backend_request_time, "2024-05-01T12:00:00Z")
      self.assertEqual(response.backend_response_time, "2024-05-01T12:00:01Z")

  def test_robotics_api_generate_content_msgpack(self):
    """Tests the msgpack protocol (server_version >= 2.0.0 / grodv2)."""
    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_service = mock.Mock()
      mock_build.return_value = mock_service
      FLAGS.api_key = "test_api_key"

      client = genai_robotics.Client(
          use_robotics_api=True,
      )
      # Simulate a grodv2 server.
      client._server_version = "2.1.0"

      image = np.zeros((100, 100, 3), dtype=np.uint8)
      image_bytes = tf.io.encode_jpeg(image).numpy()
      expected_output = {"action_chunk": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}

      mock_cm_custom = mock_service.modelServing.return_value.cmCustom
      mock_cm_custom.return_value.execute.return_value = {
          "outputBytes": (
              base64.b64encode(msgpack.packb(expected_output)).decode("utf-8")
          ),
          "backendRequestTime": "2024-05-01T12:00:00Z",
          "backendResponseTime": "2024-05-01T12:00:01Z",
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
      # Msgpack path uses msgpack encoding.
      query = msgpack.unpackb(
          base64.b64decode(call_body["inputBytes"]), raw=False
      )
      # Images are raw bytes in the msgpack path.
      self.assertEqual(
          query["images/overhead_cam"],
          image_bytes,
      )
      self.assertEqual(query["task_instruction"], "test_task_instruction")
      self.assertEqual(query["joints_pos"], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

  def test_genai_create_client_via_auth_library(self):
    with mock.patch("google.genai.Client", autospec=True) as mock_genai_client:
      FLAGS.api_key = "test_api_key"

      client = genai_robotics.Client(
          robotics_api_connection=genai_robotics.constants.RoboticsApiConnectionType.CLOUD_GENAI,
          project="test_project",
      )
      self.assertIsNotNone(client)
      mock_genai_client.assert_called_once_with(
          api_key="test_api_key", project="test_project"
      )

  def test_genai_create_client_via_param(self):
    with mock.patch("google.genai.Client", autospec=True) as mock_genai_client:
      FLAGS.api_key = None

      client = genai_robotics.Client(
          robotics_api_connection=genai_robotics.constants.RoboticsApiConnectionType.CLOUD_GENAI,
          api_key="test_api_key",
          project="test_project",
      )
      self.assertIsNotNone(client)
      mock_genai_client.assert_called_once_with(
          api_key="test_api_key", project="test_project"
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="default",
          grpc_url=None,
          expected_url=genai_robotics._LOCAL_GRPC_URL,
      ),
      dict(
          testcase_name="custom",
          grpc_url="grpc://10.0.0.5:10100",
          expected_url="grpc://10.0.0.5:10100",
      ),
  )
  @mock.patch.object(genai_robotics, "_connect_to_grpc_json", autospec=True)
  def test_local_client_uses_grpc_url(
      self, mock_connect, grpc_url, expected_url
  ):
    del expected_url  # Validated by grpc.insecure_channel internally.

    def dummy_query(_):
      return ""

    mock_connect.return_value = mock.create_autospec(dummy_query)
    client = genai_robotics.Client(
        robotics_api_connection=genai_robotics.constants.RoboticsApiConnectionType.LOCAL,
        grpc_url=grpc_url,
        skip_version_check=True,
    )
    self.assertIsNotNone(client)
    mock_connect.assert_called_once()
    # Verify the channel was created from the expected URL by checking the
    # channel argument passed to _connect_to_grpc_json.
    call_args = mock_connect.call_args
    channel = call_args[0][0]
    self.assertIsNotNone(channel)

  @mock.patch.object(genai_robotics, "_connect_to_grpc_json", autospec=True)
  def test_local_client_populates_backend_times(self, mock_connect):
    mock_func = mock.create_autospec(lambda x: "")
    mock_func.return_value = '{"action_chunk": [0.1]}'
    mock_connect.return_value = mock_func

    client = genai_robotics.Client(
        robotics_api_connection=genai_robotics.constants.RoboticsApiConnectionType.LOCAL,
        skip_version_check=True,
    )
    obs = {"task_instruction": "test_task_instruction"}
    response = client.models.generate_content(
        model="test_model",
        contents=[json.dumps(obs)],
    )

    self.assertIsNotNone(response.backend_request_time)
    self.assertIsNotNone(response.backend_response_time)
    self.assertIsInstance(response.backend_request_time, str)
    self.assertIsInstance(response.backend_response_time, str)

    req_dt = datetime.datetime.fromisoformat(response.backend_request_time)
    res_dt = datetime.datetime.fromisoformat(response.backend_response_time)
    self.assertLessEqual(req_dt, res_dt)

  @parameterized.named_parameters(
      ("rate_limit_429", 429, "Rate limit exceeded"),
      ("service_unavailable_503", 503, "Service unavailable"),
      ("bad_request_400", 400, "Bad request"),
  )
  def test_robotics_api_http_error_propagation(self, status_code, reason):
    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_service = mock.Mock()
      mock_build.return_value = mock_service
      FLAGS.api_key = "test_api_key"

      client = genai_robotics.Client(
          robotics_api_connection=genai_robotics.constants.RoboticsApiConnectionType.CLOUD,
      )
      mock_cm_custom = mock_service.modelServing.return_value.cmCustom

      resp = mock.Mock()
      resp.status = status_code
      resp.reason = reason
      mock_cm_custom.return_value.execute.side_effect = HttpError(
          resp, b"Error content"
      )

      obs = {"task_instruction": "test_task"}
      with self.assertRaises(HttpError) as ctx:
        client.models.generate_content(
            model="test_model",
            contents=[json.dumps(obs)],
        )
      self.assertEqual(ctx.exception.resp.status, status_code)

  @parameterized.named_parameters(
      ("unavailable", grpc.StatusCode.UNAVAILABLE),
      ("deadline_exceeded", grpc.StatusCode.DEADLINE_EXCEEDED),
  )
  @mock.patch.object(genai_robotics, "_connect_to_grpc_json", autospec=True)
  def test_local_generate_content_grpc_error_propagation(
      self, status_code, mock_connect
  ):
    mock_func = mock.Mock(side_effect=FakeRpcError(status_code))
    mock_connect.return_value = mock_func

    client = genai_robotics.Client(
        robotics_api_connection=genai_robotics.constants.RoboticsApiConnectionType.LOCAL,
        skip_version_check=True,
    )
    obs = {"task_instruction": "test_task"}
    with self.assertRaises(grpc.RpcError):
      client.models.generate_content(
          model="test_model",
          contents=[json.dumps(obs)],
      )

  @parameterized.named_parameters(
      ("unimplemented", grpc.StatusCode.UNIMPLEMENTED),
      ("unavailable", grpc.StatusCode.UNAVAILABLE),
      ("deadline_exceeded", grpc.StatusCode.DEADLINE_EXCEEDED),
  )
  def test_check_server_compatibility_grpc_errors(self, status_code):
    mock_channel = mock.Mock()
    mock_stub = mock.Mock(side_effect=FakeRpcError(status_code))
    mock_channel.unary_unary.return_value = mock_stub

    res = genai_robotics._check_server_compatibility(mock_channel, "1.0.0")
    self.assertEqual(res, {"supported_protocols": ["msgpack", "json"]})


if __name__ == "__main__":
  absltest.main()
