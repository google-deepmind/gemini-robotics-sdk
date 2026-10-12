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

import datetime
import json
import queue
import threading
import time
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import grpc

from safari_sdk.model import bidi_session
from safari_sdk.model import constants
from safari_sdk.protos import modelserving_pb2


class BidiSessionTest(parameterized.TestCase):

  _RESPONSE_TYPE = modelserving_pb2.CmCustomBidiResponse.ResponseType

  def test_bidi_session_lifecycle(self):
    mock_channel = mock.Mock()
    mock_stub = mock.Mock()

    setup_ack = modelserving_pb2.CmCustomBidiResponse(
        response_type=self._RESPONSE_TYPE.SETUP_ACK
    )
    action_payload = json.dumps({"action_chunk": [[0.1, 0.2]]}).encode("utf-8")
    action_resp = modelserving_pb2.CmCustomBidiResponse(
        response_type=self._RESPONSE_TYPE.ACTION_CHUNK,
        output_bytes=action_payload,
        request_id=1,
    )
    action_resp.backend_request_time.FromJsonString("2026-05-15T01:00:00Z")
    action_resp.backend_response_time.FromJsonString("2026-05-15T01:00:01Z")

    resp_queue = queue.Queue()
    resp_queue.put(setup_ack)

    stop_event = threading.Event()

    def mock_stream(requests, metadata=None):
      del requests, metadata
      while not stop_event.is_set():
        try:
          item = resp_queue.get(timeout=0.05)
          yield item
        except queue.Empty:
          continue

    mock_stub.CmCustomBidi.side_effect = mock_stream

    with mock.patch.object(
        bidi_session,
        "_ModelServingStub",
        return_value=mock_stub,
    ):
      session = bidi_session.BidiSession(
          channel=mock_channel,
          model_id="test_model",
          method_name="sample_actions_json_flat",
      )
      self.assertEqual(session.state, bidi_session.SessionState.CREATED)
      session.start()
      self.assertEqual(session.state, bidi_session.SessionState.ACTIVE)
      self.assertTrue(session.is_active)
      self.assertFalse(session.is_closed)

      request_id = session.send_observation(b"{}")
      self.assertEqual(request_id, 1)
      # The server can only answer a request that was actually sent.
      resp_queue.put(action_resp)

      resp = session.get_response(request_id, timeout=2.0)
      self.assertEqual(resp.output_bytes, action_payload)
      self.assertEqual(
          session.last_backend_request_time,
          datetime.datetime(2026, 5, 15, 1, 0, 0, tzinfo=datetime.timezone.utc),
      )
      self.assertEqual(
          session.last_backend_response_time,
          datetime.datetime(2026, 5, 15, 1, 0, 1, tzinfo=datetime.timezone.utc),
      )

      stop_event.set()
      session.close()
      self.assertEqual(session.state, bidi_session.SessionState.CLOSED)
      self.assertFalse(session.is_active)
      self.assertTrue(session.is_closed)

  def test_bidi_session_server_error(self):
    mock_channel = mock.Mock()
    mock_stub = mock.Mock()

    error_resp = modelserving_pb2.CmCustomBidiResponse(
        response_type=self._RESPONSE_TYPE.ERROR,
    )
    error_resp.status.message = "Failed to load model"

    mock_stub.CmCustomBidi.return_value = [error_resp]

    with mock.patch.object(
        bidi_session,
        "_ModelServingStub",
        return_value=mock_stub,
    ):
      session = bidi_session.BidiSession(
          channel=mock_channel,
          model_id="test_model",
      )
      with self.assertRaisesRegex(RuntimeError, "Failed to load model"):
        session.start()

  def test_bidi_session_handshake_timeout(self):
    mock_channel = mock.Mock()
    mock_stub = mock.Mock()

    def mock_hang(*args, **kwargs):
      del args, kwargs
      time.sleep(0.5)
      yield from ()

    mock_stub.CmCustomBidi.side_effect = mock_hang

    with mock.patch.object(
        bidi_session,
        "_ModelServingStub",
        return_value=mock_stub,
    ):
      session = bidi_session.BidiSession(
          channel=mock_channel,
          model_id="test_model",
      )
      with self.assertRaises(TimeoutError):
        session.start(handshake_timeout=0.01)

  def test_bidi_session_stream_terminated_before_ack(self):
    mock_channel = mock.Mock()
    mock_stub = mock.Mock()
    mock_stub.CmCustomBidi.return_value = []

    with mock.patch.object(
        bidi_session,
        "_ModelServingStub",
        return_value=mock_stub,
    ):
      session = bidi_session.BidiSession(
          channel=mock_channel,
          model_id="test_model",
      )
      with self.assertRaisesRegex(
          ConnectionError, "Server closed connection before SETUP_ACK"
      ):
        session.start(handshake_timeout=0.5)

  def test_send_observation(self):
    mock_channel = mock.Mock()
    mock_stub = mock.Mock()
    setup_ack = modelserving_pb2.CmCustomBidiResponse(
        response_type=self._RESPONSE_TYPE.SETUP_ACK
    )
    resp_queue = queue.Queue()
    resp_queue.put(setup_ack)
    stop_event = threading.Event()

    def mock_stream(requests, metadata=None):
      del requests, metadata
      while not stop_event.is_set():
        try:
          item = resp_queue.get(timeout=0.05)
          yield item
        except queue.Empty:
          continue

    mock_stub.CmCustomBidi.side_effect = mock_stream

    with mock.patch.object(
        bidi_session,
        "_ModelServingStub",
        return_value=mock_stub,
    ):
      session = bidi_session.BidiSession(
          channel=mock_channel,
          model_id="test_model",
      )
      session.start()

      payload = b'{"instruction": "do task"}'
      request_id = session.send_observation(payload)

      # First frame in _send_queue is setup frame from start()
      setup_req = session._send_queue.get(timeout=1.0)
      self.assertIsNotNone(setup_req)
      assert setup_req is not None
      self.assertTrue(setup_req.HasField("setup"))
      self.assertEqual(setup_req.setup.sax_config.model_id, "test_model")

      # Second frame in _send_queue is observation frame from send_observation()
      obs_req = session._send_queue.get(timeout=1.0)
      self.assertIsNotNone(obs_req)
      assert obs_req is not None
      self.assertEqual(obs_req.observation.request_id, request_id)
      self.assertEqual(obs_req.observation.input_bytes, payload)

      stop_event.set()
      session.close()

  def test_send_observation_when_not_active(self):
    mock_channel = mock.Mock()
    session = bidi_session.BidiSession(
        channel=mock_channel,
        model_id="test_model",
    )
    with self.assertRaisesRegex(RuntimeError, "not active"):
      session.send_observation(b"test_payload")

  def test_get_response_uses_default_timeout(self):
    mock_channel = mock.Mock()
    mock_stub = mock.Mock()
    setup_ack = modelserving_pb2.CmCustomBidiResponse(
        response_type=self._RESPONSE_TYPE.SETUP_ACK
    )

    def mock_stream(requests, metadata=None):
      del requests, metadata
      yield setup_ack
      time.sleep(1.0)

    mock_stub.CmCustomBidi.side_effect = mock_stream

    with mock.patch.object(
        bidi_session,
        "_ModelServingStub",
        return_value=mock_stub,
    ):
      session = bidi_session.BidiSession(
          channel=mock_channel,
          model_id="test_model",
          timeout=0.05,
      )
      session.start()
      request_id = session.send_observation(b"{}")
      start_t = time.perf_counter()
      with self.assertRaises(TimeoutError):
        session.get_response(request_id)
      elapsed = time.perf_counter() - start_t
      self.assertLess(elapsed, 0.5)
      session.close()

  def test_get_response_fails_fast_on_stream_error(self):
    mock_channel = mock.Mock()
    mock_stub = mock.Mock()
    setup_ack = modelserving_pb2.CmCustomBidiResponse(
        response_type=self._RESPONSE_TYPE.SETUP_ACK
    )
    # No request_id, so no caller owns this error and it fails the session.
    error_resp = modelserving_pb2.CmCustomBidiResponse(
        response_type=self._RESPONSE_TYPE.ERROR,
    )
    error_resp.status.message = "Backend crashed"

    def mock_stream(requests, metadata=None):
      del requests, metadata
      yield setup_ack
      time.sleep(0.05)
      yield error_resp

    mock_stub.CmCustomBidi.side_effect = mock_stream

    with mock.patch.object(
        bidi_session,
        "_ModelServingStub",
        return_value=mock_stub,
    ):
      session = bidi_session.BidiSession(
          channel=mock_channel,
          model_id="test_model",
      )
      session.start()
      request_id = session.send_observation(b"{}")
      start_t = time.perf_counter()
      with self.assertRaisesRegex(RuntimeError, "Backend crashed"):
        session.get_response(request_id, timeout=10.0)
      elapsed = time.perf_counter() - start_t
      self.assertLess(elapsed, 1.0)
      self.assertEqual(session.state, bidi_session.SessionState.CLOSED)
      self.assertTrue(session.is_closed)
      session.close()

  def test_cannot_restart_closed_session(self):
    mock_channel = mock.Mock()
    mock_stub = mock.Mock()
    setup_ack = modelserving_pb2.CmCustomBidiResponse(
        response_type=self._RESPONSE_TYPE.SETUP_ACK
    )
    stop_event = threading.Event()

    def stream_generator():
      yield setup_ack
      stop_event.wait()

    mock_stub.CmCustomBidi.return_value = stream_generator()

    with mock.patch.object(
        bidi_session,
        "_ModelServingStub",
        return_value=mock_stub,
    ):
      session = bidi_session.BidiSession(
          channel=mock_channel,
          model_id="test_model",
      )
      self.assertEqual(session.state, bidi_session.SessionState.CREATED)
      self.assertFalse(session.is_closed)
      session.start()
      self.assertEqual(session.state, bidi_session.SessionState.ACTIVE)
      self.assertFalse(session.is_closed)
      self.assertIsNotNone(session._recv_thread)
      assert session._recv_thread is not None
      self.assertTrue(session._recv_thread.daemon)
      stop_event.set()
      session.close()
      self.assertEqual(session.state, bidi_session.SessionState.CLOSED)
      self.assertTrue(session.is_closed)
      with self.assertRaisesRegex(RuntimeError, "Cannot restart a closed"):
        session.start()

  def test_close_is_idempotent(self):
    mock_channel = mock.Mock()
    session = bidi_session.BidiSession(
        channel=mock_channel,
        model_id="test_model",
    )
    self.assertEqual(session.state, bidi_session.SessionState.CREATED)
    session.close()
    self.assertEqual(session.state, bidi_session.SessionState.CLOSED)
    self.assertIsNone(session._send_queue.get_nowait())
    # Calling close again should be a no-op
    session.close()
    self.assertEqual(session.state, bidi_session.SessionState.CLOSED)

  def test_create_local_unsupported(self):
    with self.assertRaisesRegex(
        ValueError, "Bidi streaming is not supported for connection type"
    ):
      bidi_session.BidiSession.create(
          connection_type=constants.RoboticsApiConnectionType.LOCAL,
          model_id="local_model",
          timeout=15.0,
      )

  def test_create_cloud_loas2(self):
    mock_channel = mock.Mock()
    mock_loas2_mod = mock.Mock()
    with (
        mock.patch.dict(
            "os.environ",
            {"ROBOTICS_DEVELOPER_GRPC_ADDRESS": "dns:///127.0.0.1:443"},
        ),
        mock.patch.object(bidi_session, "loas2", mock_loas2_mod),
        mock.patch.object(
            grpc, "secure_channel", return_value=mock_channel
        ) as mock_secure,
        mock.patch.object(
            bidi_session.auth, "get_api_key", return_value="test_key"
        ),
        mock.patch.object(bidi_session, "_ModelServingStub"),
    ):
      session = bidi_session.BidiSession.create(
          connection_type=constants.RoboticsApiConnectionType.CLOUD,
          model_id="cloud_model",
      )
      mock_loas2_mod.loas2_channel_credentials.assert_called_once()
      mock_secure.assert_called_once()
      self.assertEqual(session._model_id, "cloud_model")
      self.assertEqual(session._api_key, "test_key")

  def test_create_cloud_ssl(self):
    mock_channel = mock.Mock()
    with (
        mock.patch.dict(
            "os.environ",
            {
                "ROBOTICS_DEVELOPER_GRPC_ADDRESS": (
                    "dns:///roboticsdeveloper.googleapis.com:443"
                )
            },
        ),
        mock.patch.object(grpc, "ssl_channel_credentials") as mock_ssl,
        mock.patch.object(
            grpc, "secure_channel", return_value=mock_channel
        ) as mock_secure,
        mock.patch.object(
            bidi_session.auth, "get_api_key", return_value="custom_key"
        ),
        mock.patch.object(bidi_session, "_ModelServingStub"),
    ):
      session = bidi_session.BidiSession.create(
          connection_type=constants.RoboticsApiConnectionType.CLOUD,
          model_id="cloud_model",
          api_key="explicit_key",
      )
      mock_ssl.assert_called_once()
      mock_secure.assert_called_once_with(
          "dns:///roboticsdeveloper.googleapis.com:443",
          mock_ssl.return_value,
          options=[("grpc.service_config_disable_resolution", 1)],
      )
      self.assertEqual(session._model_id, "cloud_model")
      self.assertEqual(session._api_key, "explicit_key")

  def test_create_unsupported_connection(self):
    with self.assertRaisesRegex(ValueError, "Bidi streaming is not supported"):
      bidi_session.BidiSession.create(
          connection_type=constants.RoboticsApiConnectionType.CLOUD_GENAI,
          model_id="unsupported",
      )


class ChunkReorderBufferTest(parameterized.TestCase):
  """Tests for the pure, single-threaded chunk reordering logic."""

  _RESPONSE_TYPE = modelserving_pb2.CmCustomBidiResponse.ResponseType

  def _chunk(
      self,
      chunk_index: int,
      final_chunk: bool = False,
      output_bytes: bytes = b"payload",
  ) -> modelserving_pb2.CmCustomBidiResponse:
    return modelserving_pb2.CmCustomBidiResponse(
        response_type=self._RESPONSE_TYPE.ACTION_CHUNK,
        request_id=1,
        chunk_index=chunk_index,
        final_chunk=final_chunk,
        output_bytes=output_bytes,
    )

  def _payloads(self, responses) -> list[bytes]:
    return [response.output_bytes for response in responses]

  def test_in_order_chunks_are_released_immediately(self):
    buffer = bidi_session._ChunkReorderBuffer(request_id=1)

    buffer.put(self._chunk(0, output_bytes=b"a"))
    self.assertEqual(self._payloads(buffer.drain_ready()), [b"a"])
    buffer.put(self._chunk(1, output_bytes=b"b"))
    self.assertEqual(self._payloads(buffer.drain_ready()), [b"b"])

  def test_out_of_order_chunks_are_reordered(self):
    buffer = bidi_session._ChunkReorderBuffer(request_id=1)

    # 2 and 1 arrive first and must be held back until 0 fills the gap.
    buffer.put(self._chunk(2, final_chunk=True, output_bytes=b"c"))
    self.assertEmpty(buffer.drain_ready())
    buffer.put(self._chunk(1, output_bytes=b"b"))
    self.assertEmpty(buffer.drain_ready())

    buffer.put(self._chunk(0, output_bytes=b"a"))
    self.assertEqual(self._payloads(buffer.drain_ready()), [b"a", b"b", b"c"])

  def test_final_chunk_arriving_first_is_held_back(self):
    buffer = bidi_session._ChunkReorderBuffer(request_id=1)

    # The final chunk may arrive while lower indices are still in flight; it
    # must not be released, and must not close the request either.
    buffer.put(self._chunk(1, final_chunk=True, output_bytes=b"b"))
    self.assertEmpty(buffer.drain_ready())

    buffer.put(self._chunk(0, output_bytes=b"a"))
    released = buffer.drain_ready()
    self.assertEqual(self._payloads(released), [b"a", b"b"])
    self.assertTrue(released[-1].final_chunk)

  def test_duplicate_chunk_is_ignored(self):
    buffer = bidi_session._ChunkReorderBuffer(request_id=1)

    self.assertFalse(buffer.put(self._chunk(0, output_bytes=b"original")))
    self.assertTrue(buffer.put(self._chunk(0, output_bytes=b"buffered again")))
    buffer.drain_ready()
    # A retransmission of an index that was already released is also ignored.
    self.assertTrue(buffer.put(self._chunk(0, output_bytes=b"released")))
    self.assertEmpty(buffer.drain_ready())

  def test_drained_count_tracks_released_prefix(self):
    buffer = bidi_session._ChunkReorderBuffer(request_id=1)

    buffer.put(self._chunk(1))
    buffer.drain_ready()
    self.assertEqual(buffer.drained_count, 0)  # Still waiting on the gap.

    buffer.put(self._chunk(0))
    buffer.drain_ready()
    self.assertEqual(buffer.drained_count, 2)


class ResponseRoutingTest(parameterized.TestCase):
  """Tests for request lifecycle and response routing.

  These drive `_registry.put()` directly instead of going through a mock gRPC
  stream, so arrival order, lifetime and eviction can be exercised
  deterministically without relying on thread timing.
  """

  _RESPONSE_TYPE = modelserving_pb2.CmCustomBidiResponse.ResponseType

  def _make_session(self, **kwargs) -> bidi_session.BidiSession:
    with mock.patch.object(
        bidi_session, "_ModelServingStub", return_value=mock.Mock()
    ):
      session = bidi_session.BidiSession(
          channel=mock.Mock(), model_id="test_model", **kwargs
      )
    session._state = bidi_session.SessionState.ACTIVE
    return session

  def _make_chunk(
      self,
      request_id: int,
      chunk_index: int,
      final_chunk: bool,
      output_bytes: bytes = b"payload",
  ) -> modelserving_pb2.CmCustomBidiResponse:
    return modelserving_pb2.CmCustomBidiResponse(
        response_type=self._RESPONSE_TYPE.ACTION_CHUNK,
        request_id=request_id,
        chunk_index=chunk_index,
        final_chunk=final_chunk,
        output_bytes=output_bytes,
    )

  def test_request_ids_are_monotonic(self):
    session = self._make_session()
    ids = [session.send_observation(b"{}") for _ in range(3)]
    self.assertEqual(ids, [1, 2, 3])

  def test_in_order_multi_chunk_retrieval(self):
    session = self._make_session()
    request_id = session.send_observation(b"{}")
    session._registry.put(self._make_chunk(request_id, 0, False, b"first"))
    session._registry.put(self._make_chunk(request_id, 1, True, b"second"))

    self.assertEqual(
        session.get_response(request_id, timeout=1.0).output_bytes, b"first"
    )
    self.assertEqual(
        session.get_response(request_id, timeout=1.0).output_bytes, b"second"
    )

  def test_out_of_order_chunks_are_delivered_in_order(self):
    session = self._make_session()
    request_id = session.send_observation(b"{}")
    for chunk_index, payload in ((2, b"c"), (0, b"a"), (1, b"b")):
      session._registry.put(
          self._make_chunk(request_id, chunk_index, chunk_index == 2, payload)
      )

    payloads = [
        session.get_response(request_id, timeout=1.0).output_bytes
        for _ in range(3)
    ]
    self.assertEqual(payloads, [b"a", b"b", b"c"])
    self.assertFalse(session.is_closed)

  def test_gap_blocks_until_filled(self):
    session = self._make_session()
    request_id = session.send_observation(b"{}")
    # Chunk 1 is deliverable only once chunk 0 shows up.
    session._registry.put(self._make_chunk(request_id, 1, True, b"b"))
    with self.assertRaisesRegex(TimeoutError, "chunk 0"):
      session.get_response(request_id, timeout=0.05)

    session._registry.put(self._make_chunk(request_id, 0, False, b"a"))
    self.assertEqual(
        session.get_response(request_id, timeout=1.0).output_bytes, b"a"
    )
    self.assertEqual(
        session.get_response(request_id, timeout=1.0).output_bytes, b"b"
    )

  def test_duplicate_chunk_is_ignored_without_failing_the_session(self):
    session = self._make_session()
    request_id = session.send_observation(b"{}")
    session._registry.put(self._make_chunk(request_id, 0, False, b"original"))
    session._registry.put(self._make_chunk(request_id, 0, False, b"duplicate"))

    self.assertEqual(
        session.get_response(request_id, timeout=1.0).output_bytes, b"original"
    )
    self.assertEqual(session.response_stats["dropped_duplicate_chunks"], 1)
    self.assertFalse(session.is_closed)

  def test_concurrent_requests_do_not_interfere(self):
    session = self._make_session()
    first = session.send_observation(b"{}")
    second = session.send_observation(b"{}")

    # The two requests interleave on the wire.
    session._registry.put(self._make_chunk(second, 0, False, b"second-0"))
    session._registry.put(self._make_chunk(first, 0, False, b"first-0"))
    session._registry.put(self._make_chunk(second, 1, True, b"second-1"))
    session._registry.put(self._make_chunk(first, 1, True, b"first-1"))

    self.assertEqual(
        [
            session.get_response(second, timeout=1.0).output_bytes,
            session.get_response(second, timeout=1.0).output_bytes,
        ],
        [b"second-0", b"second-1"],
    )
    # Draining the newer request must not have disturbed the older one.
    self.assertEqual(
        [
            session.get_response(first, timeout=1.0).output_bytes,
            session.get_response(first, timeout=1.0).output_bytes,
        ],
        [b"first-0", b"first-1"],
    )
    self.assertEqual(session.response_stats["dropped_after_close"], 0)

  def test_response_after_completion_is_dropped(self):
    session = self._make_session()
    request_id = session.send_observation(b"{}")
    session._registry.put(self._make_chunk(request_id, 0, True))
    session.get_response(request_id, timeout=1.0)

    session._registry.put(self._make_chunk(request_id, 1, True))
    self.assertEqual(session.response_stats["dropped_after_close"], 1)
    self.assertEqual(session.response_stats["dropped_unknown"], 0)

  def test_response_for_never_sent_request_is_reported(self):
    session = self._make_session()
    session._registry.put(self._make_chunk(42, 0, True))

    self.assertEqual(session.response_stats["dropped_unknown"], 1)
    self.assertEqual(session.response_stats["dropped_after_close"], 0)

  def test_get_response_after_final_chunk_raises(self):
    session = self._make_session()
    request_id = session.send_observation(b"{}")
    session._registry.put(self._make_chunk(request_id, 0, True))
    session.get_response(request_id, timeout=1.0)

    with self.assertRaisesRegex(RuntimeError, "no longer live"):
      session.get_response(request_id, timeout=1.0)

  def test_get_response_for_never_sent_request_raises(self):
    session = self._make_session()
    with self.assertRaisesRegex(RuntimeError, "was never sent"):
      session.get_response(7, timeout=1.0)

  def test_leaked_request_is_evicted_and_its_waiter_fails_fast(self):
    session = self._make_session(max_live_requests=2)
    leaked = session.send_observation(b"{}")
    waiter_error = []

    def wait_for_evicted():
      try:
        session.get_response(leaked, timeout=10.0)
      except Exception as e:  # pylint: disable=broad-except
        waiter_error.append(e)

    waiter = threading.Thread(target=wait_for_evicted)
    waiter.start()

    start_t = time.perf_counter()
    for _ in range(2):  # Two more requests push the leaked one out.
      session.send_observation(b"{}")
    waiter.join(timeout=5.0)

    self.assertLess(time.perf_counter() - start_t, 2.0)
    self.assertLen(waiter_error, 1)
    self.assertIn("evicted", str(waiter_error[0]))
    self.assertEqual(session.response_stats["evicted_requests"], 1)

  def test_request_scoped_error_raises_for_that_request_only(self):
    session = self._make_session()
    failing = session.send_observation(b"{}")
    healthy = session.send_observation(b"{}")
    error_resp = modelserving_pb2.CmCustomBidiResponse(
        response_type=self._RESPONSE_TYPE.ERROR,
        request_id=failing,
    )
    error_resp.status.message = "model exploded"
    session._registry.put(error_resp)
    session._registry.put(self._make_chunk(healthy, 0, True, b"fine"))

    with self.assertRaisesRegex(RuntimeError, "model exploded"):
      session.get_response(failing, timeout=1.0)
    # Neither the session nor the other request is affected.
    self.assertFalse(session.is_closed)
    self.assertEqual(
        session.get_response(healthy, timeout=1.0).output_bytes, b"fine"
    )

  def test_request_scoped_error_supersedes_pending_chunks(self):
    session = self._make_session()
    request_id = session.send_observation(b"{}")
    # A chunk is still stuck behind a gap when the request fails.
    session._registry.put(self._make_chunk(request_id, 1, True, b"never"))
    error_resp = modelserving_pb2.CmCustomBidiResponse(
        response_type=self._RESPONSE_TYPE.ERROR,
        request_id=request_id,
    )
    error_resp.status.message = "model exploded"
    session._registry.put(error_resp)

    with self.assertRaisesRegex(RuntimeError, "model exploded"):
      session.get_response(request_id, timeout=1.0)

  def test_close_wakes_blocked_waiter(self):
    session = self._make_session()
    request_id = session.send_observation(b"{}")

    def close_soon():
      time.sleep(0.05)
      session.close()

    closer = threading.Thread(target=close_soon)
    closer.start()
    start_t = time.perf_counter()
    with self.assertRaisesRegex(RuntimeError, "closed while waiting"):
      session.get_response(request_id, timeout=10.0)
    self.assertLess(time.perf_counter() - start_t, 2.0)
    closer.join()

  def test_send_after_close_returns_failed_stream(self):
    session = self._make_session()
    session.close()
    session._state = bidi_session.SessionState.ACTIVE  # Force the race.

    request_id = session.send_observation(b"{}")
    with self.assertRaisesRegex(RuntimeError, "closed while waiting"):
      session.get_response(request_id, timeout=10.0)

  def test_late_chunk_unblocks_waiter(self):
    session = self._make_session()
    request_id = session.send_observation(b"{}")

    def deliver_soon():
      time.sleep(0.05)
      session._registry.put(self._make_chunk(request_id, 0, True, b"late"))

    deliverer = threading.Thread(target=deliver_soon)
    deliverer.start()
    response = session.get_response(request_id, timeout=10.0)
    self.assertEqual(response.output_bytes, b"late")
    deliverer.join()


if __name__ == "__main__":
  absltest.main()
