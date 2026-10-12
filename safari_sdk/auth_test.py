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

"""Unit tests for auth.py."""

import os
import socket
import tempfile
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
import httplib2

from safari_sdk import auth
from safari_sdk import utils


class AuthQoSTest(absltest.TestCase):

  def test_apply_dscp_marking_success(self):
    mock_sock = mock.MagicMock(spec=socket.socket)
    auth._apply_dscp_marking(mock_sock, dscp=32)

    # DSCP 32 shifted by 2 is 128 (0x80)
    expected_val = 32 << 2
    mock_sock.setsockopt.assert_any_call(
        socket.IPPROTO_IPV6, socket.IPV6_TCLASS, expected_val
    )
    mock_sock.setsockopt.assert_any_call(
        socket.IPPROTO_IP, socket.IP_TOS, expected_val
    )

  def test_apply_dscp_marking_none_socket(self):
    # Should safely return without error when sock is None
    auth._apply_dscp_marking(None, dscp=32)

  def test_apply_dscp_marking_disabled_when_negative(self):
    mock_sock = mock.MagicMock(spec=socket.socket)
    auth._apply_dscp_marking(mock_sock, dscp=-1)
    mock_sock.setsockopt.assert_not_called()

  def test_apply_dscp_marking_partial_oserror_still_succeeds(self):
    mock_sock = mock.MagicMock(spec=socket.socket)
    # First call (IPV6_TCLASS) raises OSError, second call (IP_TOS) succeeds
    mock_sock.setsockopt.side_effect = [OSError("IPv6 not supported"), None]

    with (
        mock.patch.object(auth.logging, "info") as mock_info,
        mock.patch.object(auth.logging, "warning") as mock_warning,
    ):
      auth._apply_dscp_marking(mock_sock, dscp=16)
      mock_info.assert_called_once()
      mock_warning.assert_not_called()

  def test_apply_dscp_marking_all_oserror_logs_warning(self):
    mock_sock = mock.MagicMock(spec=socket.socket)
    mock_sock.setsockopt.side_effect = OSError("Operation not permitted")

    with mock.patch.object(auth.logging, "warning") as mock_warning:
      auth._apply_dscp_marking(mock_sock, dscp=16)
      mock_warning.assert_called_once()

  def test_qos_https_connection_marks_socket_on_connect(self):
    mock_sock = mock.MagicMock(spec=socket.socket)
    auth._QoSHttp(timeout=10, dscp=32)
    conn = auth._QoSHTTPSConnection("roboticsdeveloper.googleapis.com", 443)
    conn.sock = mock_sock

    with mock.patch.object(
        httplib2.HTTPSConnectionWithTimeout, "connect"
    ) as mock_super_connect:
      conn.connect()
      mock_super_connect.assert_called_once()
      mock_sock.setsockopt.assert_any_call(
          socket.IPPROTO_IPV6, socket.IPV6_TCLASS, 128
      )
      mock_sock.setsockopt.assert_any_call(
          socket.IPPROTO_IP, socket.IP_TOS, 128
      )

  def test_qos_https_connection_no_socket_on_connect(self):
    auth._QoSHttp(timeout=10, dscp=32)
    conn = auth._QoSHTTPSConnection("roboticsdeveloper.googleapis.com", 443)
    conn.sock = None

    with mock.patch.object(
        httplib2.HTTPSConnectionWithTimeout, "connect"
    ) as mock_super_connect, mock.patch.object(
        auth, "_apply_dscp_marking"
    ) as mock_apply_dscp:
      conn.connect()
      mock_super_connect.assert_called_once()
      mock_apply_dscp.assert_not_called()

  def test_qos_http_uses_qos_https_connection_for_https(self):
    http = auth._QoSHttp(timeout=10, dscp=32)
    with mock.patch.object(httplib2.Http, "request") as mock_super_request:
      http.request("https://roboticsdeveloper.googleapis.com/test")
      mock_super_request.assert_called_once()
      _, kwargs = mock_super_request.call_args
      self.assertEqual(kwargs["connection_type"], auth._QoSHTTPSConnection)

  def test_qos_http_does_not_override_connection_for_http(self):
    http = auth._QoSHttp(timeout=10, dscp=32)
    with mock.patch.object(httplib2.Http, "request") as mock_super_request:
      http.request("http://roboticsdeveloper.googleapis.com/test")
      mock_super_request.assert_called_once()
      _, kwargs = mock_super_request.call_args
      self.assertIsNone(kwargs["connection_type"])

  def test_qos_http_preserves_explicit_connection_type(self):
    http = auth._QoSHttp(timeout=10, dscp=32)
    custom_conn_type = httplib2.HTTPSConnectionWithTimeout
    with mock.patch.object(httplib2.Http, "request") as mock_super_request:
      http.request(
          "https://roboticsdeveloper.googleapis.com/test",
          connection_type=custom_conn_type,
      )
      mock_super_request.assert_called_once()
      _, kwargs = mock_super_request.call_args
      self.assertIs(kwargs["connection_type"], custom_conn_type)


class AuthApiKeyAndServiceTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.dir1 = tempfile.TemporaryDirectory()
    self.dir2 = tempfile.TemporaryDirectory()
    self.addCleanup(self.dir1.cleanup)
    self.addCleanup(self.dir2.cleanup)
    self.dirs_patcher = mock.patch.object(
        utils, "_CONFIG_DIRECTORIES", [self.dir1.name, self.dir2.name]
    )
    self.dirs_patcher.start()
    self.addCleanup(self.dirs_patcher.stop)
    env_patcher = mock.patch.dict(os.environ)
    env_patcher.start()
    self.addCleanup(env_patcher.stop)
    os.environ.pop("API_KEY_SAFARI", None)
    os.environ.pop("API_KEY_GEMINI", None)

  def _write_key_file(self, directory: str, filename: str, content: str) -> str:
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as f:
      f.write(content)
    return path

  @flagsaver.flagsaver(api_key="flag_api_key", api_key_gemini=None)
  def test_get_api_key_default_from_flag_takes_precedence_over_file(self):
    self._write_key_file(self.dir1.name, "API_KEY", "file_api_key")
    self.assertEqual(auth.get_api_key(), "flag_api_key")
    self.assertEqual(auth.get_api_key("api_key"), "flag_api_key")

  @flagsaver.flagsaver(api_key=None, api_key_gemini=None)
  def test_get_api_key_from_first_directory_takes_precedence(self):
    self._write_key_file(self.dir1.name, "API_KEY", "dir1_key\n")
    self._write_key_file(self.dir2.name, "API_KEY", "dir2_key\n")
    self.assertEqual(auth.get_api_key(), "dir1_key")

  @flagsaver.flagsaver(api_key=None, api_key_gemini=None)
  def test_get_api_key_skips_empty_file_and_reads_second_directory(self):
    self._write_key_file(self.dir1.name, "API_KEY", "")
    self._write_key_file(self.dir2.name, "API_KEY", "dir2_valid_key")
    self.assertEqual(auth.get_api_key(), "dir2_valid_key")

  @flagsaver.flagsaver(api_key=None, api_key_gemini=None)
  def test_get_api_key_skips_whitespace_only_file_and_reads_second_directory(
      self,
  ):
    self._write_key_file(self.dir1.name, "API_KEY", "   \n\t  ")
    self._write_key_file(self.dir2.name, "API_KEY", "dir2_valid_key")
    self.assertEqual(auth.get_api_key(), "dir2_valid_key")

  @flagsaver.flagsaver(api_key=None, api_key_gemini=None)
  def test_get_api_key_returns_none_when_file_is_empty(self):
    self._write_key_file(self.dir1.name, "API_KEY", "")
    self._write_key_file(self.dir2.name, "API_KEY", "")
    self.assertIsNone(auth.get_api_key())
    with self.assertRaisesRegex(ValueError, auth._ERROR_NO_API_KEY_PROVIDED):
      auth.get_service()

  @flagsaver.flagsaver(api_key=None, api_key_gemini=None)
  def test_get_api_key_returns_none_when_file_is_whitespace_only(self):
    self._write_key_file(self.dir1.name, "API_KEY", "   \t\n  ")
    self._write_key_file(self.dir2.name, "API_KEY", " ")
    self.assertIsNone(auth.get_api_key())
    with self.assertRaisesRegex(ValueError, auth._ERROR_NO_API_KEY_PROVIDED):
      auth.get_service()

  @flagsaver.flagsaver(api_key="   \t\n ", api_key_gemini=None)
  def test_get_api_key_returns_none_when_flag_is_whitespace_only(self):
    self.assertIsNone(auth.get_api_key())
    with self.assertRaisesRegex(ValueError, auth._ERROR_NO_API_KEY_PROVIDED):
      auth.get_service()

  @flagsaver.flagsaver(api_key="   \t\n ", api_key_gemini=None)
  def test_get_api_key_whitespace_only_flag_falls_back_to_file(self):
    self._write_key_file(self.dir1.name, "API_KEY", "valid_file_key")
    self.assertEqual(auth.get_api_key(), "valid_file_key")

  @flagsaver.flagsaver(api_key="", api_key_gemini=None)
  def test_get_api_key_empty_string_flag_returns_none_without_file(self):
    self.assertIsNone(auth.get_api_key())
    with self.assertRaisesRegex(ValueError, auth._ERROR_NO_API_KEY_PROVIDED):
      auth.get_service()

  @flagsaver.flagsaver(api_key=None, api_key_gemini=None)
  def test_get_api_key_returns_none_when_not_found(self):
    self.assertIsNone(auth.get_api_key())
    self.assertIsNone(auth.get_api_key("api_key_gemini"))

  @flagsaver.flagsaver(api_key="default_key", api_key_gemini=None)
  def test_get_api_key_gemini_falls_back_to_default_api_key(self):
    self.assertEqual(auth.get_api_key("api_key_gemini"), "default_key")

  @flagsaver.flagsaver(api_key="default_key", api_key_gemini="gemini_flag_key")
  def test_get_api_key_gemini_uses_own_flag_before_fallback(self):
    self.assertEqual(auth.get_api_key("api_key_gemini"), "gemini_flag_key")

  @flagsaver.flagsaver(api_key="default_key", api_key_gemini=None)
  def test_get_api_key_gemini_uses_own_file_before_fallback(self):
    self._write_key_file(self.dir1.name, "API_KEY_GEMINI", "gemini_file_key")
    self.assertEqual(auth.get_api_key("api_key_gemini"), "gemini_file_key")

  @flagsaver.flagsaver(api_key=None, api_key_gemini=None)
  def test_get_api_key_env_var_takes_precedence_over_file(self):
    os.environ["API_KEY_SAFARI"] = " env_api_key\n"
    os.environ["API_KEY_GEMINI"] = "env_gemini_key"
    self._write_key_file(self.dir1.name, "API_KEY_SAFARI", "file_api_key")
    self._write_key_file(self.dir1.name, "API_KEY_GEMINI", "file_gemini_key")
    self.assertEqual(auth.get_api_key(), "env_api_key")
    self.assertEqual(auth.get_api_key("api_key_gemini"), "env_gemini_key")

  @flagsaver.flagsaver(
      api_key_safari="flag_api_key", api_key_gemini="flag_gemini_key"
  )
  def test_get_api_key_flag_takes_precedence_over_env_var(self):
    os.environ["API_KEY_SAFARI"] = "env_api_key"
    os.environ["API_KEY_GEMINI"] = "env_gemini_key"
    self.assertEqual(auth.get_api_key(), "flag_api_key")
    self.assertEqual(auth.get_api_key("api_key_gemini"), "flag_gemini_key")

  @flagsaver.flagsaver(api_key=None, api_key_gemini=None)
  def test_get_api_key_default_api_key_has_no_env_var(self):
    os.environ["API_KEY_SAFARI"] = "env_api_key"
    self.assertIsNone(auth.get_api_key("api_key"))

  @flagsaver.flagsaver(api_key="default_key", api_key_safari=None)
  def test_get_api_key_safari_falls_back_to_default_api_key(self):
    self.assertEqual(auth.get_api_key(), "default_key")
    self.assertEqual(auth.get_api_key("api_key_safari"), "default_key")

  @flagsaver.flagsaver(api_key="default_key", api_key_safari="safari_flag_key")
  def test_get_api_key_safari_uses_own_flag_before_fallback(self):
    self.assertEqual(auth.get_api_key(), "safari_flag_key")
    self.assertEqual(auth.get_api_key("api_key"), "default_key")

  @flagsaver.flagsaver(api_key="default_key", api_key_safari=None)
  def test_get_api_key_safari_uses_own_file_before_fallback(self):
    self._write_key_file(self.dir1.name, "API_KEY_SAFARI", "safari_file_key")
    self.assertEqual(auth.get_api_key(), "safari_file_key")

  def test_get_api_key_unregistered_key_raises_value_error(self):
    with self.assertRaisesRegex(
        ValueError,
        r"Key name \[unregistered_key\] is not registered",
    ):
      auth.get_api_key("unregistered_key")

  def test_register_custom_key_duplicate_raises_value_error(self):
    with self.assertRaisesRegex(
        ValueError,
        r"Key name \[api_key\] is already registered",
    ):
      auth.register_custom_key(
          key_name="api_key",
          key_file_name="CUSTOM_FILE",
          fallback_to_default_api_key=False,
      )

  def _cleanup_custom_key(self, key_name: str):
    auth._API_KEYS_REGISTRY.pop(key_name, None)
    auth._FALLBACK_TO_DEFAULT_API_KEY.discard(key_name)
    if key_name in flags.FLAGS:
      delattr(flags.FLAGS, key_name)

  @flagsaver.flagsaver(api_key="default_fallback_key")
  def test_register_custom_key_with_fallback(self):
    key_name = "test_custom_key_with_fallback"
    self.addCleanup(self._cleanup_custom_key, key_name)

    auth.register_custom_key(
        key_name=key_name,
        key_file_name="CUSTOM_KEY_FB",
        fallback_to_default_api_key=True,
    )

    # 1. Falls back to default api_key when neither flag nor file is set
    self.assertEqual(auth.get_api_key(key_name), "default_fallback_key")

    # 2. File overrides fallback
    self._write_key_file(self.dir2.name, "CUSTOM_KEY_FB", "custom_from_file")
    self.assertEqual(auth.get_api_key(key_name), "custom_from_file")

    # 3. Flag overrides file
    with flagsaver.flagsaver(**{key_name: "custom_from_flag"}):
      self.assertEqual(auth.get_api_key(key_name), "custom_from_flag")

  @flagsaver.flagsaver(api_key="default_fallback_key")
  def test_register_custom_key_without_fallback(self):
    key_name = "test_custom_key_no_fallback"
    self.addCleanup(self._cleanup_custom_key, key_name)

    auth.register_custom_key(
        key_name=key_name,
        key_file_name="CUSTOM_KEY_NO_FB",
        fallback_to_default_api_key=False,
    )

    # Does not fall back to default api_key when not set
    self.assertIsNone(auth.get_api_key(key_name))

  @flagsaver.flagsaver(api_key=None)
  def test_register_custom_key_with_env_vars(self):
    key_name = "test_custom_key_env_vars"
    self.addCleanup(self._cleanup_custom_key, key_name)

    auth.register_custom_key(
        key_name=key_name,
        key_file_name="CUSTOM_KEY_ENV",
        fallback_to_default_api_key=False,
        key_env_vars=("CUSTOM_ENV_A", "CUSTOM_ENV_B"),
    )
    self._write_key_file(self.dir1.name, "CUSTOM_KEY_ENV", "custom_from_file")

    # 1. Env vars override file, in priority order
    os.environ["CUSTOM_ENV_B"] = "custom_from_env_b"
    self.assertEqual(auth.get_api_key(key_name), "custom_from_env_b")
    os.environ["CUSTOM_ENV_A"] = "custom_from_env_a"
    self.assertEqual(auth.get_api_key(key_name), "custom_from_env_a")

    # 2. Flag overrides env vars
    with flagsaver.flagsaver(**{key_name: "custom_from_flag"}):
      self.assertEqual(auth.get_api_key(key_name), "custom_from_flag")

  @flagsaver.flagsaver(api_key=None)
  def test_get_service_raises_value_error_when_no_api_key(self):
    with self.assertRaisesRegex(ValueError, auth._ERROR_NO_API_KEY_PROVIDED):
      auth.get_service()

  @mock.patch("googleapiclient.discovery.build")
  def test_get_service_with_dscp(self, mock_build):
    with flagsaver.flagsaver(api_key="test_key"):
      service = auth.get_service(timeout=10, dscp=32)
      self.assertEqual(service, mock_build.return_value)
      mock_build.assert_called_once()
      _, kwargs = mock_build.call_args
      self.assertEqual(kwargs["serviceName"], auth._DEFAULT_SERVICE_NAME)
      self.assertEqual(kwargs["version"], auth._DEFAULT_VERSION)
      self.assertEqual(
          kwargs["discoveryServiceUrl"], auth._DEFAULT_DISCOVERY_SERVICE_URL
      )
      self.assertEqual(kwargs["developerKey"], "test_key")
      self.assertIsInstance(kwargs["http"], auth._QoSHttp)
      self.assertEqual(kwargs["http"].timeout, 10)
      self.assertEqual(auth._QoSHTTPSConnection.dscp, 32)

  @mock.patch("googleapiclient.discovery.build")
  def test_get_service_default_without_dscp(self, mock_build):
    with flagsaver.flagsaver(api_key="test_key"):
      service = auth.get_service()
      self.assertEqual(service, mock_build.return_value)
      mock_build.assert_called_once()
      _, kwargs = mock_build.call_args
      self.assertNotIsInstance(kwargs["http"], auth._QoSHttp)
      self.assertIsInstance(kwargs["http"], httplib2.Http)
      self.assertEqual(kwargs["http"].timeout, auth._DEFAULT_TIMEOUT)


if __name__ == "__main__":
  absltest.main()
