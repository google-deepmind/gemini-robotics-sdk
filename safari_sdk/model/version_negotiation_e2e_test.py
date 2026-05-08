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

"""E2E test for get_server_info version negotiation.

Starts a real gRPC server, connects a real client, and verifies the
full handshake flow including protocol negotiation and error cases.
"""

from concurrent import futures
import json
import warnings

import grpc

from absl.testing import absltest
from safari_sdk.model import genai_robotics


def _start_server_with_info(port: int, server_info: dict[str, object]):
  """Starts a minimal gRPC server with get_server_info + sample_actions_json_flat.

  Args:
    port: The port number to listen on.
    server_info: The dict to return from get_server_info (serialized as JSON).

  Returns:
    The grpc.Server instance (already started).
  """
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))

  def handle_get_server_info(request: bytes, context: grpc.ServicerContext):  # pylint: disable=unused-argument
    del request, context
    return json.dumps(server_info).encode('utf-8')

  def handle_query(request: bytes, context: grpc.ServicerContext):  # pylint: disable=unused-argument
    del context
    # Echo the request back as a valid JSON response.
    return json.dumps({'action_chunk': [0.0], 'dtype': 'float32'}).encode(
        'utf-8'
    )

  info_handler = grpc.unary_unary_rpc_method_handler(
      handle_get_server_info,
      request_deserializer=lambda x: x,
      response_serializer=lambda x: x,
  )
  query_handler = grpc.unary_unary_rpc_method_handler(
      handle_query,
      request_deserializer=lambda x: x.decode('utf-8'),
      response_serializer=lambda x: x,
  )
  generic_handler = grpc.method_handlers_generic_handler(
      'gemini_robotics',
      {
          'get_server_info': info_handler,
          'sample_actions_json_flat': query_handler,
      },
  )
  server.add_generic_rpc_handlers((generic_handler,))
  server.add_insecure_port(f'[::]:{port}')
  server.start()
  return server


def _start_server_without_info(port: int):
  """Starts a gRPC server WITHOUT get_server_info (simulates legacy server).

  Args:
    port: The port number to listen on.

  Returns:
    The grpc.Server instance (already started).
  """
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))

  def handle_query(request: bytes, context: grpc.ServicerContext):  # pylint: disable=unused-argument
    del context
    return json.dumps({'action_chunk': [0.0], 'dtype': 'float32'}).encode(
        'utf-8'
    )

  query_handler = grpc.unary_unary_rpc_method_handler(
      handle_query,
      request_deserializer=lambda x: x.decode('utf-8'),
      response_serializer=lambda x: x,
  )
  generic_handler = grpc.method_handlers_generic_handler(
      'gemini_robotics',
      {
          'sample_actions_json_flat': query_handler,
      },
  )
  server.add_generic_rpc_handlers((generic_handler,))
  server.add_insecure_port(f'[::]:{port}')
  server.start()
  return server


class VersionNegotiationE2ETest(absltest.TestCase):
  """E2E tests for client-server version negotiation."""

  def test_new_server_returns_info(self):
    """Client connects to a new server and gets version info."""
    port = 50051
    server_info = {
        'server_version': '1.3.0',
        'supported_protocols': ['json'],
        'min_client_version': '0.0.0',
    }
    server = _start_server_with_info(port, server_info)
    try:
      channel = grpc.insecure_channel(f'localhost:{port}')
      result = genai_robotics._check_server_compatibility(
          channel, '2.122.0', timeout=5.0
      )
      self.assertEqual(result['server_version'], '1.3.0')
      self.assertEqual(result['supported_protocols'], ['json'])
      self.assertEqual(result['min_client_version'], '0.0.0')
    finally:
      server.stop(grace=0)

  def test_legacy_server_unimplemented_fallback(self):
    """Client falls back to JSON when server lacks get_server_info."""
    port = 50052
    server = _start_server_without_info(port)
    try:
      channel = grpc.insecure_channel(f'localhost:{port}')
      with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        result = genai_robotics._check_server_compatibility(
            channel, '2.122.0', timeout=5.0
        )
      self.assertEqual(result['supported_protocols'], ['json'])
      self.assertEqual(result['server_version'], 'unknown')
      # Verify a warning was issued.
      self.assertTrue(
          any('UNIMPLEMENTED' in str(warning.message)
              or 'get_server_info' in str(warning.message)
              for warning in w),
          f'Expected UNIMPLEMENTED warning, got: {[str(x.message) for x in w]}',
      )
    finally:
      server.stop(grace=0)

  def test_min_client_version_rejection(self):
    """Server rejects a client that is too old."""
    port = 50053
    server_info = {
        'server_version': '2.0.0',
        'supported_protocols': ['msgpack'],
        'min_client_version': '3.0.0',
    }
    server = _start_server_with_info(port, server_info)
    try:
      channel = grpc.insecure_channel(f'localhost:{port}')
      with self.assertRaises(RuntimeError) as cm:
        genai_robotics._check_server_compatibility(
            channel, '2.122.0', timeout=5.0
        )
      self.assertIn('too old', str(cm.exception))
      self.assertIn('>= 3.0.0', str(cm.exception))
    finally:
      server.stop(grace=0)

  def test_server_too_old_warns(self):
    """Client warns when server version is below _MIN_SERVER_VERSION."""
    port = 50056
    server_info = {
        'server_version': '0.1.0',
        'supported_protocols': ['json'],
        'min_client_version': '0.0.0',
    }
    server = _start_server_with_info(port, server_info)
    try:
      channel = grpc.insecure_channel(f'localhost:{port}')
      # Temporarily set a high min server version to trigger the warning.
      original = genai_robotics._MIN_SERVER_VERSION
      genai_robotics._MIN_SERVER_VERSION = '1.0.0'
      try:
        with warnings.catch_warnings(record=True) as w:
          warnings.simplefilter('always')
          result = genai_robotics._check_server_compatibility(
              channel, '2.122.0', timeout=5.0
          )
        self.assertEqual(result['server_version'], '0.1.0')
        self.assertTrue(
            any('older than the minimum' in str(warning.message)
                for warning in w),
            f'Expected server-too-old warning, got: '
            f'{[str(x.message) for x in w]}',
        )
      finally:
        genai_robotics._MIN_SERVER_VERSION = original
    finally:
      server.stop(grace=0)

  def test_unreachable_server_graceful_fallback(self):
    """Client falls back to JSON when server is unreachable."""
    channel = grpc.insecure_channel('localhost:59999')  # No server here.
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      result = genai_robotics._check_server_compatibility(
          channel, '2.122.0', timeout=1.0
      )
    self.assertEqual(result['supported_protocols'], ['json'])
    self.assertEqual(result['server_version'], 'unknown')
    self.assertTrue(
        any('UNAVAILABLE' in str(warning.message) for warning in w),
        f'Expected UNAVAILABLE warning, got: {[str(x.message) for x in w]}',
    )

  def test_skip_version_check_bypasses_handshake(self):
    """skip_version_check=True skips the handshake entirely."""
    port = 50054
    server = _start_server_without_info(port)
    try:
      # If version check was NOT skipped, this would warn. We verify no
      # warning is emitted.
      with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        client = genai_robotics.Client(
            robotics_api_connection=genai_robotics.constants.RoboticsApiConnectionType.LOCAL,
            grpc_url=f'grpc://localhost:{port}',
            skip_version_check=True,
        )
      self.assertIsNotNone(client)
      safari_warnings = [
          x for x in w if 'get_server_info' in str(x.message)
      ]
      self.assertEmpty(
          safari_warnings,
          'Expected no version check warnings with skip_version_check=True',
      )
    finally:
      server.stop(grace=0)

  def test_full_e2e_handshake_then_query(self):
    """Full e2e: handshake + inference query on the same channel."""
    port = 50055
    server_info = {
        'server_version': '1.3.0',
        'supported_protocols': ['json'],
        'min_client_version': '0.0.0',
    }
    server = _start_server_with_info(port, server_info)
    try:
      channel = grpc.insecure_channel(f'localhost:{port}')
      # Handshake succeeds.
      result = genai_robotics._check_server_compatibility(
          channel, '2.122.0', timeout=5.0
      )
      self.assertEqual(result['server_version'], '1.3.0')

      # Query on the same channel works.
      query_fn = genai_robotics._connect_to_grpc_json(
          channel, 'sample_actions_json_flat'
      )
      response = query_fn({'task_instruction': 'test', 'joints_pos': [0.0]})
      parsed = json.loads(response)
      self.assertIn('action_chunk', parsed)
    finally:
      server.stop(grace=0)

  def test_version_lt_helper(self):
    """Unit tests for _version_lt embedded in the e2e test file."""
    self.assertTrue(genai_robotics._version_lt('1.0.0', '2.0.0'))
    self.assertTrue(genai_robotics._version_lt('2.103.0', '2.122.0'))
    self.assertFalse(genai_robotics._version_lt('2.122.0', '2.122.0'))
    self.assertFalse(genai_robotics._version_lt('3.0.0', '2.0.0'))
    # Invalid versions return False.
    self.assertFalse(genai_robotics._version_lt('not_a_version', '2.0.0'))
    self.assertFalse(genai_robotics._version_lt('2.0.0', 'garbage'))
    # dev/unknown strings.
    self.assertFalse(genai_robotics._version_lt('dev', '2.0.0'))
    self.assertFalse(genai_robotics._version_lt('0.0.0', '0.0.0'))


if __name__ == '__main__':
  absltest.main()
