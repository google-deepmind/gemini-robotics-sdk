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

"""Unit tests for bidi_remote_model_interface."""

from concurrent import futures
import datetime
import json
import time
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from dm_env import specs
import numpy as np

from safari_sdk.model import additional_observations_provider
from safari_sdk.model import bidi_remote_model_interface
from safari_sdk.model import bidi_session
from safari_sdk.model import constants
from safari_sdk.model import remote_model_interface
from safari_sdk.protos import modelserving_pb2


class RequestTimeoutTest(parameterized.TestCase):
  """Tests for request_timeout validation in BidiRemoteModelInterface."""

  @parameterized.parameters(0, 0.0, -1, -0.5)
  def test_interface_rejects_non_positive_timeout_at_construction(
      self, request_timeout
  ):
    with self.assertRaisesRegex(ValueError, "must be positive"):
      bidi_remote_model_interface.BidiRemoteModelInterface(
          serve_id="test_serve_id",
          robotics_api_connection=constants.RoboticsApiConnectionType.CLOUD,
          task_instruction_key="test_instruction_key",
          proprioceptive_observation_keys=("test_joint_1",),
          image_observation_keys=("test_camera_1",),
          image_compression_jpeg_quality=75,
          request_timeout=request_timeout,
      )

  def test_interface_requires_request_timeout(self):
    kwargs = {
        "serve_id": "test_serve_id",
        "robotics_api_connection": constants.RoboticsApiConnectionType.CLOUD,
        "task_instruction_key": "test_instruction_key",
        "proprioceptive_observation_keys": ("test_joint_1",),
        "image_observation_keys": ("test_camera_1",),
        "image_compression_jpeg_quality": 75,
    }
    with self.assertRaises(TypeError):
      getattr(bidi_remote_model_interface, "BidiRemoteModelInterface")(**kwargs)


class BidiRemoteModelInterfaceTest(parameterized.TestCase):

  def test_interface(self):
    bidi_model = bidi_remote_model_interface.BidiRemoteModelInterface(
        serve_id="test_serve_id",
        robotics_api_connection=constants.RoboticsApiConnectionType.CLOUD,
        task_instruction_key="test_instruction_key",
        proprioceptive_observation_keys=("test_joint_1",),
        image_observation_keys=("test_camera_1",),
        image_compression_jpeg_quality=75,
        request_timeout=30.0,
    )
    self.assertIsInstance(
        bidi_model, bidi_remote_model_interface.BidiRemoteModelInterface
    )
    self.assertNotIsInstance(
        bidi_model, remote_model_interface.RemoteModelInterface
    )
    self.assertFalse(hasattr(bidi_model, "_client"))

  def test_additional_observations_are_added(self):
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

    bidi_model = bidi_remote_model_interface.BidiRemoteModelInterface(
        serve_id="test_serve_id",
        robotics_api_connection=constants.RoboticsApiConnectionType.CLOUD,
        task_instruction_key="test_instruction_key",
        proprioceptive_observation_keys=("test_joint_1",),
        image_observation_keys=("test_camera_1",),
        image_compression_jpeg_quality=75,
        request_timeout=30.0,
        additional_observations_providers=[mock_provider_1, mock_provider_2],
    )

    self.assertIn(
        "additional_obs_1", bidi_model._proprioceptive_observation_keys
    )
    self.assertIn("additional_obs_2", bidi_model._string_observations_keys)

  def test_open_bidi_session(self):
    bidi_model = bidi_remote_model_interface.BidiRemoteModelInterface(
        serve_id="test_serve_id",
        robotics_api_connection=constants.RoboticsApiConnectionType.CLOUD,
        task_instruction_key="test_instruction_key",
        proprioceptive_observation_keys=("test_joint_1",),
        image_observation_keys=("test_camera_1",),
        image_compression_jpeg_quality=75,
        request_timeout=45,
    )
    mock_session = mock.MagicMock(spec=bidi_session.BidiSession)
    with mock.patch.object(
        bidi_session.BidiSession, "create", return_value=mock_session
    ) as mock_create:
      session = bidi_model._open_bidi_session()
      self.assertEqual(session, mock_session)
      mock_create.assert_called_once_with(
          connection_type=constants.RoboticsApiConnectionType.CLOUD,
          model_id="test_serve_id",
          timeout=45,
      )

  def test_query_model_without_reset_raises_runtime_error(self):
    """Verifies that calling query_model before reset() raises RuntimeError."""
    bidi_model = bidi_remote_model_interface.BidiRemoteModelInterface(
        serve_id="test_serve_id",
        robotics_api_connection=constants.RoboticsApiConnectionType.CLOUD,
        task_instruction_key="test_instruction_key",
        proprioceptive_observation_keys=("test_joint_1",),
        image_observation_keys=("test_camera_1",),
        image_compression_jpeg_quality=75,
        request_timeout=30.0,
    )
    obs = {
        "test_camera_1": np.zeros((100, 100, 3), dtype=np.uint8),
        "test_joint_1": np.array([0.0]),
        "test_instruction_key": np.array(
            "test_task_instruction", dtype=np.object_
        ),
    }
    with self.assertRaisesRegex(RuntimeError, "no active session"):
      bidi_model.query_model(obs)

  def test_query_model_bidi(self):
    bidi_model = bidi_remote_model_interface.BidiRemoteModelInterface(
        serve_id="test_serve_id",
        robotics_api_connection=constants.RoboticsApiConnectionType.CLOUD,
        task_instruction_key="test_instruction_key",
        proprioceptive_observation_keys=("test_joint_1",),
        image_observation_keys=("test_camera_1",),
        image_compression_jpeg_quality=75,
        request_timeout=30.0,
    )
    mock_session = mock.MagicMock()
    mock_session.use_msgpack = False
    mock_session.last_backend_request_time = datetime.datetime(
        2026, 5, 15, 1, 0, 0, tzinfo=datetime.timezone.utc
    )
    mock_session.last_backend_response_time = datetime.datetime(
        2026, 5, 15, 1, 0, 0, 50000, tzinfo=datetime.timezone.utc
    )
    mock_response = mock.MagicMock()
    mock_response.output_bytes = json.dumps(
        {"action_chunk": [[1.0, 2.0], [3.0, 4.0]]}
    ).encode("utf-8")
    mock_response.final_chunk = True
    mock_session.send_observation.return_value = 1
    mock_session.get_response.return_value = mock_response
    bidi_model._open_bidi_session = mock.MagicMock(return_value=mock_session)

    # Initialize streaming session explicitly via reset()
    bidi_model.reset()

    obs = {
        "test_camera_1": np.zeros((100, 100, 3), dtype=np.uint8),
        "test_joint_1": np.array([0.0]),
        "test_instruction_key": np.array(
            "test_task_instruction", dtype=np.object_
        ),
    }
    time_values = iter([100.0, 100.01, 100.02, 100.10])
    with mock.patch.object(time, "perf_counter", side_effect=time_values):
      action_chunk = bidi_model.query_model(obs).snapshot(
          wait_until_exhausted=True
      )

    bidi_model._open_bidi_session.assert_called_once()
    mock_session.start.assert_called_once()
    mock_session.send_observation.assert_called_once()
    mock_session.get_response.assert_called_once_with(1, timeout=30.0)
    np.testing.assert_array_equal(
        action_chunk, np.array([[1.0, 2.0], [3.0, 4.0]])
    )
    telemetry = bidi_model.get_telemetry()
    self.assertIsNotNone(telemetry)
    assert telemetry is not None
    self.assertEqual(
        telemetry.request_time,
        datetime.datetime(2026, 5, 15, 1, 0, 0, tzinfo=datetime.timezone.utc),
    )
    self.assertEqual(
        telemetry.response_time,
        datetime.datetime(
            2026, 5, 15, 1, 0, 0, 50000, tzinfo=datetime.timezone.utc
        ),
    )
    self.assertIsNotNone(telemetry.remote_inference_duration)
    assert telemetry.remote_inference_ms is not None
    self.assertAlmostEqual(telemetry.remote_inference_ms, 50.0, places=1)
    self.assertIsNotNone(telemetry.network_overhead_duration)
    assert telemetry.network_overhead_ms is not None
    self.assertAlmostEqual(telemetry.network_overhead_ms, 50.0, places=1)
    self.assertIsNotNone(telemetry.client_image_encode_duration)
    assert telemetry.client_image_encode_ms is not None
    self.assertAlmostEqual(telemetry.client_image_encode_ms, 10.0, places=1)

    # Test close
    bidi_model.close()
    mock_session.close.assert_called_once()
    self.assertIsNone(bidi_model._bidi_session)

  def test_query_model_partial_timestamps_sets_inference_time_none(self):
    bidi_model = bidi_remote_model_interface.BidiRemoteModelInterface(
        serve_id="test_serve_id",
        robotics_api_connection=constants.RoboticsApiConnectionType.CLOUD,
        task_instruction_key="test_instruction_key",
        proprioceptive_observation_keys=("test_joint_1",),
        image_observation_keys=("test_camera_1",),
        image_compression_jpeg_quality=75,
        request_timeout=30.0,
    )
    mock_session = mock.MagicMock()
    mock_session.use_msgpack = False
    mock_session.last_backend_request_time = None
    mock_session.last_backend_response_time = datetime.datetime(
        2026, 5, 15, 1, 0, 0, 50000, tzinfo=datetime.timezone.utc
    )
    mock_response = mock.MagicMock()
    mock_response.output_bytes = json.dumps(
        {"action_chunk": [[1.0, 2.0]]}
    ).encode("utf-8")
    mock_response.final_chunk = True
    mock_session.send_observation.return_value = 1
    mock_session.get_response.return_value = mock_response
    bidi_model._open_bidi_session = mock.MagicMock(return_value=mock_session)
    bidi_model.reset()

    obs = {
        "test_camera_1": np.zeros((10, 10, 3), dtype=np.uint8),
        "test_joint_1": np.array([0.0]),
        "test_instruction_key": np.array("task", dtype=np.object_),
    }
    bidi_model.query_model(obs).snapshot(wait_until_exhausted=True)
    telemetry = bidi_model.get_telemetry()
    self.assertIsNotNone(telemetry)
    assert telemetry is not None
    self.assertIsNone(telemetry.remote_inference_duration)
    self.assertIsNone(telemetry.remote_inference_ms)
    self.assertIsNone(telemetry.network_overhead_duration)
    self.assertIsNone(telemetry.network_overhead_ms)

  def test_destructor_closes_session(self):
    bidi_model = bidi_remote_model_interface.BidiRemoteModelInterface(
        serve_id="test_serve_id",
        robotics_api_connection=constants.RoboticsApiConnectionType.CLOUD,
        task_instruction_key="test_instruction_key",
        proprioceptive_observation_keys=("test_joint_1",),
        image_observation_keys=("test_camera_1",),
        image_compression_jpeg_quality=75,
        request_timeout=30.0,
    )
    mock_session = mock.MagicMock()
    bidi_model._bidi_session = mock_session
    bidi_model.__del__()
    mock_session.close.assert_called_once()
    self.assertIsNone(bidi_model._bidi_session)

  def test_concurrent_query_model_thread_safe(self):
    bidi_model = bidi_remote_model_interface.BidiRemoteModelInterface(
        serve_id="test_serve_id",
        robotics_api_connection=constants.RoboticsApiConnectionType.CLOUD,
        task_instruction_key="test_instruction_key",
        proprioceptive_observation_keys=("test_joint_1",),
        image_observation_keys=("test_camera_1",),
        image_compression_jpeg_quality=75,
        request_timeout=30.0,
    )
    mock_session = mock.MagicMock()
    mock_session.use_msgpack = False
    mock_session.last_backend_request_time = datetime.datetime(
        2026, 5, 15, 1, 0, 0, tzinfo=datetime.timezone.utc
    )
    mock_session.last_backend_response_time = datetime.datetime(
        2026, 5, 15, 1, 0, 0, 50000, tzinfo=datetime.timezone.utc
    )
    mock_response = mock.MagicMock()
    mock_response.output_bytes = json.dumps(
        {"action_chunk": [[1.0, 2.0]]}
    ).encode("utf-8")
    mock_response.final_chunk = True
    mock_session.send_observation.return_value = 1
    mock_session.get_response.return_value = mock_response

    bidi_model._open_bidi_session = mock.MagicMock(return_value=mock_session)
    bidi_model.reset()

    obs = {
        "test_camera_1": np.zeros((100, 100, 3), dtype=np.uint8),
        "test_joint_1": np.array([0.0]),
        "test_instruction_key": np.array(
            "test_task_instruction", dtype=np.object_
        ),
    }

    num_threads = 5
    with futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
      task_futures = [
          executor.submit(
              lambda: bidi_model.query_model(obs).snapshot(
                  wait_until_exhausted=True
              )
          )
          for _ in range(num_threads)
      ]
      results = [f.result() for f in task_futures]

    self.assertLen(results, num_threads)
    for res in results:
      np.testing.assert_array_equal(res, np.array([[1.0, 2.0]]))

    bidi_model._open_bidi_session.assert_called_once()
    mock_session.start.assert_called_once()
    self.assertEqual(mock_session.send_observation.call_count, num_threads)

  def test_persistent_session_across_queries(self):
    """Verifies that query_model reuses the persistent session across queries."""
    bidi_model = bidi_remote_model_interface.BidiRemoteModelInterface(
        serve_id="test_serve_id",
        robotics_api_connection=constants.RoboticsApiConnectionType.CLOUD,
        task_instruction_key="test_instruction_key",
        proprioceptive_observation_keys=("test_joint_1",),
        image_observation_keys=("test_camera_1",),
        image_compression_jpeg_quality=75,
        request_timeout=30.0,
    )
    mock_session = mock.MagicMock()
    mock_session.use_msgpack = False
    mock_session.last_backend_request_time = datetime.datetime(
        2026, 5, 15, 1, 0, 0, tzinfo=datetime.timezone.utc
    )
    mock_session.last_backend_response_time = datetime.datetime(
        2026, 5, 15, 1, 0, 0, 50000, tzinfo=datetime.timezone.utc
    )
    mock_response_1 = mock.MagicMock()
    mock_response_1.output_bytes = json.dumps(
        {"action_chunk": [[1.0, 2.0]]}
    ).encode("utf-8")
    mock_response_1.final_chunk = True
    mock_response_2 = mock.MagicMock()
    mock_response_2.output_bytes = json.dumps(
        {"action_chunk": [[10.0, 20.0]]}
    ).encode("utf-8")
    mock_response_2.final_chunk = True
    mock_session.send_observation.side_effect = [1, 2]
    mock_session.get_response.side_effect = [mock_response_1, mock_response_2]

    bidi_model._open_bidi_session = mock.MagicMock(return_value=mock_session)
    bidi_model.reset()

    obs = {
        "test_camera_1": np.zeros((100, 100, 3), dtype=np.uint8),
        "test_joint_1": np.array([0.0]),
        "test_instruction_key": np.array(
            "test_task_instruction", dtype=np.object_
        ),
    }

    # Query 1: streams over session initialized by reset()
    chunk_1 = bidi_model.query_model(obs).snapshot(wait_until_exhausted=True)
    np.testing.assert_array_equal(chunk_1, np.array([[1.0, 2.0]]))
    mock_session.start.assert_called_once()
    mock_session.send_observation.assert_called_once()

    # Query 2: reuses the persistent session without re-opening
    chunk_2 = bidi_model.query_model(obs).snapshot(wait_until_exhausted=True)
    np.testing.assert_array_equal(chunk_2, np.array([[10.0, 20.0]]))
    bidi_model._open_bidi_session.assert_called_once()
    self.assertEqual(mock_session.send_observation.call_count, 2)
    self.assertEqual(bidi_model._bidi_session, mock_session)

  def test_query_after_close_raises_error(self):
    """Verifies that calling query_model after close() raises RuntimeError."""
    bidi_model = bidi_remote_model_interface.BidiRemoteModelInterface(
        serve_id="test_serve_id",
        robotics_api_connection=constants.RoboticsApiConnectionType.CLOUD,
        task_instruction_key="test_instruction_key",
        proprioceptive_observation_keys=("test_joint_1",),
        image_observation_keys=("test_camera_1",),
        image_compression_jpeg_quality=75,
        request_timeout=30.0,
    )
    mock_session = mock.MagicMock()
    mock_session.use_msgpack = False
    mock_response = mock.MagicMock()
    mock_response.output_bytes = json.dumps(
        {"action_chunk": [[1.0, 2.0]]}
    ).encode("utf-8")
    mock_response.final_chunk = True
    mock_session.send_observation.return_value = 1
    mock_session.get_response.return_value = mock_response
    bidi_model._open_bidi_session = mock.MagicMock(return_value=mock_session)
    bidi_model.reset()

    obs = {
        "test_camera_1": np.zeros((100, 100, 3), dtype=np.uint8),
        "test_joint_1": np.array([0.0]),
        "test_instruction_key": np.array(
            "test_task_instruction", dtype=np.object_
        ),
    }

    # First query succeeds
    bidi_model.query_model(obs).snapshot(wait_until_exhausted=True)
    mock_session.start.assert_called_once()

    # Terminal close
    bidi_model.close()
    mock_session.close.assert_called_once()
    self.assertIsNone(bidi_model._bidi_session)

    # Subsequent query raises RuntimeError
    with self.assertRaisesRegex(RuntimeError, "has been closed"):
      bidi_model.query_model(obs)

  def test_reset_closes_previous_session_and_starts_new_session(self):
    """Verifies that reset() closes active session and immediately starts a new one."""
    bidi_model = bidi_remote_model_interface.BidiRemoteModelInterface(
        serve_id="test_serve_id",
        robotics_api_connection=constants.RoboticsApiConnectionType.CLOUD,
        task_instruction_key="test_instruction_key",
        proprioceptive_observation_keys=("test_joint_1",),
        image_observation_keys=("test_camera_1",),
        image_compression_jpeg_quality=75,
        request_timeout=30.0,
    )
    mock_session_1 = mock.MagicMock()
    mock_session_1.use_msgpack = False
    mock_response_1 = mock.MagicMock()
    mock_response_1.output_bytes = json.dumps(
        {"action_chunk": [[1.0, 2.0]]}
    ).encode("utf-8")
    mock_response_1.final_chunk = True
    mock_session_1.send_observation.return_value = 1
    mock_session_1.get_response.return_value = mock_response_1

    mock_session_2 = mock.MagicMock()
    mock_session_2.use_msgpack = False
    mock_response_2 = mock.MagicMock()
    mock_response_2.output_bytes = json.dumps(
        {"action_chunk": [[3.0, 4.0]]}
    ).encode("utf-8")
    mock_response_2.final_chunk = True
    mock_session_2.send_observation.return_value = 1
    mock_session_2.get_response.return_value = mock_response_2

    bidi_model._open_bidi_session = mock.MagicMock(
        side_effect=[mock_session_1, mock_session_2]
    )

    obs = {
        "test_camera_1": np.zeros((100, 100, 3), dtype=np.uint8),
        "test_joint_1": np.array([0.0]),
        "test_instruction_key": np.array(
            "test_task_instruction", dtype=np.object_
        ),
    }

    # reset() opens session 1
    bidi_model.reset()
    mock_session_1.start.assert_called_once()
    self.assertEqual(bidi_model._open_bidi_session.call_count, 1)

    # Query 1 streams over session 1
    chunk_1 = bidi_model.query_model(obs).snapshot(wait_until_exhausted=True)
    np.testing.assert_array_equal(chunk_1, np.array([[1.0, 2.0]]))

    # reset() closes session 1 and immediately starts fresh session 2
    bidi_model.reset()
    mock_session_1.close.assert_called_once()
    mock_session_2.start.assert_called_once()
    self.assertEqual(bidi_model._bidi_session, mock_session_2)
    self.assertEqual(bidi_model._open_bidi_session.call_count, 2)

    # Query 2 reuses the already active session 2
    chunk_2 = bidi_model.query_model(obs).snapshot(wait_until_exhausted=True)
    np.testing.assert_array_equal(chunk_2, np.array([[3.0, 4.0]]))
    self.assertEqual(bidi_model._open_bidi_session.call_count, 2)

  def test_query_model_flags_only_the_final_chunk(self):
    """`is_exhausted` must reflect the server's `final_chunk` flag."""
    bidi_model = bidi_remote_model_interface.BidiRemoteModelInterface(
        serve_id="test_serve_id",
        robotics_api_connection=constants.RoboticsApiConnectionType.CLOUD,
        task_instruction_key="test_instruction_key",
        proprioceptive_observation_keys=("test_joint_1",),
        image_observation_keys=("test_camera_1",),
        image_compression_jpeg_quality=75,
        request_timeout=30.0,
    )
    mock_session = mock.MagicMock()
    mock_session.use_msgpack = False
    mock_session.last_backend_request_time = None
    mock_session.last_backend_response_time = None
    first = modelserving_pb2.CmCustomBidiResponse(
        response_type=modelserving_pb2.CmCustomBidiResponse.ACTION_CHUNK,
        output_bytes=json.dumps({"action_chunk": [[1.0, 2.0]]}).encode("utf-8"),
        chunk_index=0,
        final_chunk=False,
    )
    last = modelserving_pb2.CmCustomBidiResponse(
        response_type=modelserving_pb2.CmCustomBidiResponse.ACTION_CHUNK,
        output_bytes=json.dumps({"action_chunk": [[3.0, 4.0]]}).encode("utf-8"),
        chunk_index=1,
        final_chunk=True,
    )
    mock_session.send_observation.return_value = 1
    mock_session.get_response.side_effect = [
        first,
        TimeoutError(),
        TimeoutError(),
        TimeoutError(),
        last,
    ]
    bidi_model._open_bidi_session = mock.MagicMock(return_value=mock_session)
    bidi_model.reset()

    obs = {
        "test_camera_1": np.zeros((10, 10, 3), dtype=np.uint8),
        "test_joint_1": np.array([0.0]),
        "test_instruction_key": np.array("task", dtype=np.object_),
    }
    horizon = bidi_model.query_model(obs)

    self.assertFalse(horizon.is_complete)
    self.assertFalse(horizon.is_exhausted)
    np.testing.assert_array_equal(horizon.snapshot(), np.array([[1.0, 2.0]]))
    self.assertTrue(horizon.is_complete)
    self.assertFalse(horizon.is_exhausted)
    np.testing.assert_array_equal(
        horizon.snapshot(), np.array([[1.0, 2.0], [3.0, 4.0]])
    )
    self.assertTrue(horizon.has_arrived_actions())
    self.assertEqual(horizon.available_actions_count, 2)
    horizon.take_next_action()
    self.assertTrue(horizon.has_arrived_actions())
    self.assertEqual(horizon.available_actions_count, 1)
    horizon.take_next_action()
    self.assertFalse(horizon.has_arrived_actions())
    self.assertEqual(horizon.available_actions_count, 0)
    self.assertTrue(horizon.is_exhausted)

  def test_bidi_model_telemetry_properties(self):
    telemetry = bidi_remote_model_interface.BidiModelTelemetry(
        request_time=datetime.datetime(
            2026, 5, 15, 1, 0, 0, tzinfo=datetime.timezone.utc
        ),
        response_time=datetime.datetime(
            2026, 5, 15, 1, 0, 0, 50000, tzinfo=datetime.timezone.utc
        ),
        remote_inference_duration=datetime.timedelta(milliseconds=50.0),
        network_overhead_duration=datetime.timedelta(milliseconds=25.5),
        client_image_encode_duration=datetime.timedelta(milliseconds=8.2),
        wire_transit_duration=datetime.timedelta(milliseconds=12.0),
        client_processing_duration=datetime.timedelta(milliseconds=6.0),
        rng_key=[42],
    )
    assert telemetry.remote_inference_ms is not None
    self.assertAlmostEqual(telemetry.remote_inference_ms, 50.0)
    assert telemetry.network_overhead_ms is not None
    self.assertAlmostEqual(telemetry.network_overhead_ms, 25.5)
    assert telemetry.client_image_encode_ms is not None
    self.assertAlmostEqual(telemetry.client_image_encode_ms, 8.2)
    assert telemetry.wire_transit_ms is not None
    self.assertAlmostEqual(telemetry.wire_transit_ms, 12.0)
    assert telemetry.client_processing_ms is not None
    self.assertAlmostEqual(telemetry.client_processing_ms, 6.0)
    self.assertEqual(telemetry.rng_key, [42])

  def test_bidi_model_telemetry_defaults(self):
    telemetry = bidi_remote_model_interface.BidiModelTelemetry()
    self.assertIsNone(telemetry.remote_inference_ms)
    self.assertIsNone(telemetry.network_overhead_ms)
    self.assertIsNone(telemetry.client_image_encode_ms)
    self.assertIsNone(telemetry.wire_transit_ms)
    self.assertIsNone(telemetry.client_processing_ms)
    self.assertIsNone(telemetry.rng_key)

  def _make_bidi_model(self):
    return bidi_remote_model_interface.BidiRemoteModelInterface(
        serve_id="test_serve_id",
        robotics_api_connection=constants.RoboticsApiConnectionType.CLOUD,
        task_instruction_key="test_instruction_key",
        proprioceptive_observation_keys=("test_joint_1",),
        image_observation_keys=("test_camera_1",),
        image_compression_jpeg_quality=75,
        request_timeout=30.0,
    )

  def _make_obs(self):
    return {
        "test_camera_1": np.zeros((10, 10, 3), dtype=np.uint8),
        "test_joint_1": np.array([0.0]),
        "test_instruction_key": np.array("task", dtype=np.object_),
    }

  def test_query_model_yields_every_chunk_in_order(self):
    bidi_model = self._make_bidi_model()
    mock_session = mock.create_autospec(
        bidi_session.BidiSession, instance=True, spec_set=True
    )
    mock_session.use_msgpack = False

    def make_response(action_chunk, final_chunk=False):
      return modelserving_pb2.CmCustomBidiResponse(
          output_bytes=json.dumps({"action_chunk": action_chunk}).encode(
              "utf-8"
          ),
          final_chunk=final_chunk,
      )

    mock_session.send_observation.return_value = 7
    mock_session.get_response.side_effect = [
        make_response([[1.0, 2.0]]),
        make_response([[3.0, 4.0]]),
        make_response([[5.0, 6.0]], final_chunk=True),
    ]
    bidi_model._open_bidi_session = mock.create_autospec(
        bidi_model._open_bidi_session, return_value=mock_session
    )
    bidi_model.reset()

    horizon = bidi_model.query_model(self._make_obs())

    np.testing.assert_array_equal(
        horizon.snapshot(wait_until_exhausted=True),
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    )
    # Chunks must be retrieved against the id that was sent.
    mock_session.get_response.assert_called_with(7, timeout=30.0)

  def test_query_model_sends_observation_eagerly(self):
    """The observation must be in flight before the caller reads any actions."""
    bidi_model = self._make_bidi_model()
    mock_session = mock.create_autospec(
        bidi_session.BidiSession, instance=True, spec_set=True
    )
    mock_session.use_msgpack = False
    mock_response = modelserving_pb2.CmCustomBidiResponse(
        output_bytes=json.dumps({"action_chunk": [[1.0, 2.0]]}).encode("utf-8"),
        final_chunk=True,
    )
    reads = []

    def get_response(request_id, timeout=None):
      del request_id, timeout
      if reads:
        raise TimeoutError()
      reads.append(mock_response)
      return mock_response

    mock_session.send_observation.return_value = 1
    mock_session.get_response.side_effect = get_response
    bidi_model._open_bidi_session = mock.create_autospec(
        bidi_model._open_bidi_session, return_value=mock_session
    )
    bidi_model.reset()

    horizon = bidi_model.query_model(self._make_obs())

    # Not a single read yet, but the send must already have happened.
    mock_session.send_observation.assert_called_once()
    # Opening the response stream reads nothing from it: retrieval stays lazy.
    self.assertEmpty(reads)

    np.testing.assert_array_equal(
        horizon.take_next_action(), np.array([1.0, 2.0])
    )
    self.assertLen(reads, 1)


if __name__ == "__main__":
  absltest.main()
