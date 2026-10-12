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

"""Helper functions for authentication and connecting to discovery services.

This module provides helper functions to build discovery services and manage
API keys (e.g. default Safari API key, Gemini API key, or custom keys).

API keys can be provided via flags, environment variables, or stored in files
under search directories:
  1. By flag (e.g., --api_key_safari="...", --api_key_gemini="...")
  2. By environment variable (e.g., $API_KEY_SAFARI, $API_KEY_GEMINI)
  3. By file in one of these directories:
       - $SAFARI_CONFIG/<KEY_FILE_NAME> (if the env var is set)
       - $HOME/.config/safari_sdk/<KEY_FILE_NAME>
       - /opt/safari_sdk/<KEY_FILE_NAME>

Pre-registered keys include:
  - "api_key" (flag: --api_key, env var: None, file: API_KEY, fallback: None)
  - "api_key_safari" (flag: --api_key_safari, env var: API_KEY_SAFARI, file:
    API_KEY_SAFARI, fallback: "api_key")
  - "api_key_gemini" (flag: --api_key_gemini, env var: API_KEY_GEMINI, file:
    API_KEY_GEMINI, fallback: "api_key")

The resolution order for retrieving an API key is:
  1. By flag (e.g., "--<key_name>")
  2. By environment variable (e.g., "$API_KEY_SAFARI"; custom keys may set
     their own with `key_env_vars`)
  3. By file path ($SAFARI_CONFIG/<key_file_name> if set, then
     $HOME/.config/safari_sdk/<key_file_name>, then
     /opt/safari_sdk/<key_file_name>)
  4. By fallback to the default "api_key" (if enabled for the key)

Examples of using this module:

  from safari_sdk import auth

  # 1. Build and return the discovery.Resource object (uses "api_key_safari"):
  service = auth.get_service()

  # 2. Retrieve an API key:
  safari_key = auth.get_api_key()  # "api_key_safari", falling back to "api_key"
  gemini_key = auth.get_api_key("api_key_gemini")

  # 3. Register and use a custom API key:
  auth.register_custom_key(
      key_name="my_custom_key",
      key_file_name="MY_CUSTOM_KEY",
      key_env_vars=("MY_CUSTOM_KEY",),
      fallback_to_default_api_key=True,
  )
  custom_key = auth.get_api_key("my_custom_key")
"""

from collections.abc import MutableMapping
from collections.abc import Sequence
import socket
from typing import Any

from absl import flags
from absl import logging
from googleapiclient import discovery
import httplib2

from safari_sdk import utils


# Pre-registered API key flags.
_API_KEY = flags.DEFINE_string(
    name="api_key",
    default=None,
    help="API key to use for the Safari API.",
)
_API_KEY_SAFARI = flags.DEFINE_string(
    name="api_key_safari",
    default=None,
    help="API key to use for the Safari API (equivalent to --api_key).",
)
_API_KEY_GEMINI = flags.DEFINE_string(
    name="api_key_gemini",
    default=None,
    help="API key to use for the Gemini API.",
)

# Maps each API key name to the arguments for `utils.get_config_value()`, so
# that the inner dictionary can be passed to that function via **kwargs.
_API_KEYS_REGISTRY: dict[str, dict[str, Any]] = {
    "api_key": {
        "flag": _API_KEY,
        "file_name": "API_KEY",
    },
    "api_key_safari": {
        "flag": _API_KEY_SAFARI,
        "env_vars": ("API_KEY_SAFARI",),
        "file_name": "API_KEY_SAFARI",
    },
    "api_key_gemini": {
        "flag": _API_KEY_GEMINI,
        "env_vars": ("API_KEY_GEMINI",),
        "file_name": "API_KEY_GEMINI",
    },
}

# Names of API keys which fall back to the default "api_key" if not found.
_FALLBACK_TO_DEFAULT_API_KEY = {"api_key_gemini", "api_key_safari"}

# Default service name, version, and discovery service URL for connection API.
_DEFAULT_SERVICE_NAME = "roboticsdeveloper"
_DEFAULT_VERSION = "v1"
_DEFAULT_DISCOVERY_SERVICE_URL = (
    "https://roboticsdeveloper.googleapis.com/$discovery/rest?version=v1"
)
_DEFAULT_TIMEOUT = 30  # 30 seconds

# Error message.
_ERROR_NO_API_KEY_PROVIDED = (
    "Auth: No API key provided by flag, env var, or file."
)
_ERROR_NO_API_KEY_PROVIDED_IN_FILE = (
    "Auth: No API key provided in file:"
)


def _apply_dscp_marking(sock: socket.socket | None, dscp: int) -> None:
  """Applies DSCP / QoS marking to a socket."""
  if sock is None or dscp < 0:
    return
  val = dscp << 2
  applied = False
  # Try setting IPv6 Traffic Class
  if hasattr(socket, "IPV6_TCLASS"):
    try:
      sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_TCLASS, val)
      applied = True
    except OSError:
      pass
  # Try setting IPv4 Type of Service
  if hasattr(socket, "IP_TOS"):
    try:
      sock.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, val)
      applied = True
    except OSError:
      pass
  if applied:
    logging.info(
        "Applied DSCP %d (TOS 0x%02x) QoS marking to socket.", dscp, val
    )
  else:
    logging.warning("Failed to apply DSCP %d QoS marking to socket.", dscp)


class _QoSHTTPSConnection(httplib2.HTTPSConnectionWithTimeout):
  """HTTPS connection with QoS/DSCP socket marking."""

  # Set by `_QoSHttp`. httplib2 constructs the connection itself with a fixed
  # signature, so the value cannot be passed to `__init__`.
  dscp: int = 0

  def connect(self) -> None:
    """See base class."""
    super().connect()
    if self.sock is not None:
      _apply_dscp_marking(self.sock, self.dscp)


class _QoSHttp(httplib2.Http):
  """An Http client that applies QoS (DSCP) marking to outgoing sockets."""

  def __init__(
      self,
      *args,
      dscp: int | None = None,
      **kwargs,
  ):
    """Initializes the instance.

    Args:
      *args: Positional arguments forwarded to `httplib2.Http`.
      dscp: Optional DSCP marking value to apply to outgoing HTTPS sockets. If
        None, no DSCP socket marking is applied.
      **kwargs: Keyword arguments forwarded to `httplib2.Http`.
    """
    super().__init__(*args, **kwargs)
    self._dscp = dscp
    if dscp is not None:
      _QoSHTTPSConnection.dscp = dscp

  def request(
      self,
      uri: str,
      method: str = "GET",
      body: str | bytes | None = None,
      headers: MutableMapping[str, str] | None = None,
      redirections: int = httplib2.DEFAULT_MAX_REDIRECTS,
      connection_type: type[Any] | None = None,
  ) -> tuple[httplib2.Response, bytes]:
    """See base class."""
    if (
        self._dscp is not None
        and connection_type is None
        and not uri.startswith("http://")
    ):
      connection_type = _QoSHTTPSConnection
    return super().request(
        uri,
        method=method,
        body=body,
        headers=headers,
        redirections=redirections,
        connection_type=connection_type,
    )


def _build_service(
    api_key: str,
    timeout: int,
    dscp: int | None = None,
) -> discovery.Resource:
  """Builds the service."""
  if dscp is not None:
    logging.info("Configuring Robotics Developer API client with DSCP %d", dscp)
    http = _QoSHttp(timeout=timeout, dscp=dscp)
  else:
    http = httplib2.Http(timeout=timeout)
  return discovery.build(
      serviceName=_DEFAULT_SERVICE_NAME,
      version=_DEFAULT_VERSION,
      discoveryServiceUrl=_DEFAULT_DISCOVERY_SERVICE_URL,
      developerKey=api_key,
      http=http,
  )


def register_custom_key(
    key_name: str,
    key_file_name: str,
    key_env_vars: Sequence[str] = (),
    fallback_to_default_api_key: bool = False,
) -> None:
  """Registers a custom API key.

  Registration of any new custom API key should occur at module import time if
  you expect an user to set the key via command-line flag. This means this
  registration should be called before app.run() in the global scope of your
  module.

  Args:
    key_name: The name of the API key.
    key_file_name: The name of the file containing the API key.
    key_env_vars: Optional environment variable names to check for the API key,
      in priority order, after the flag and before the file.
    fallback_to_default_api_key: Whether to fallback to the default API key if
      the custom API key is not found.

  Raises:
    ValueError: If the key name is already registered in the API keys registry.
  """
  if key_name in _API_KEYS_REGISTRY:
    raise ValueError(
        f"Key name [{key_name}] is already registered in the API keys registry."
    )
  _new_key = flags.DEFINE_string(
      name=key_name,
      default=None,
      help=f"API key to use for the {key_name} key in the Safari API.",
  )
  _API_KEYS_REGISTRY[key_name] = {
      "flag": _new_key,
      "env_vars": tuple(key_env_vars),
      "file_name": key_file_name,
  }
  if fallback_to_default_api_key:
    _FALLBACK_TO_DEFAULT_API_KEY.add(key_name)


def get_api_key(key_name: str = "api_key_safari") -> str | None:
  """Gets specified API key from flags, env vars, or fixed file locations.

  Args:
    key_name: The name of the API key to retrieve. If not specified,
      "api_key_safari" will be used, with a fallback to "api_key".

  Returns:
    The API key if found, otherwise None.

  Raises:
    ValueError: If the key name is not registered in the API keys registry.
  """
  if key_name not in _API_KEYS_REGISTRY:
    raise ValueError(
        f"Key name [{key_name}] is not registered in the API keys registry."
    )

  api_key = utils.get_config_value(**_API_KEYS_REGISTRY[key_name])
  if api_key:
    return api_key

  if key_name in _FALLBACK_TO_DEFAULT_API_KEY:
    logging.info(
        "Key name [%s] not found, falling back to default API key [api_key].",
        key_name,
    )
    return get_api_key("api_key")

  # No API key found by flag, env var, or file.
  return None


def get_service(
    timeout: int = _DEFAULT_TIMEOUT,
    dscp: int | None = None,
) -> discovery.Resource:
  """Gets a built discovery service based on flags, env vars, or files.

  The API key is the "api_key_safari" key, falling back to "api_key" (see
  `get_api_key()`). The order of resolution precedence is:
  1. By flag, "--api_key_safari"
  2. By environment variable, "$API_KEY_SAFARI"
  3. By file, "API_KEY_SAFARI" in the config directories ($SAFARI_CONFIG/ if
     set, $HOME/.config/safari_sdk/, /opt/safari_sdk/)
  4. By flag, "--api_key"
  5. By file, "API_KEY" in the config directories

  Args:
    timeout: Optional. The timeout in seconds for the HTTP request.
    dscp: Optional. Differentiated Services Code Point (DSCP) value (0-63) to
      apply QoS marking to outgoing sockets. If None, standard Best Effort
      traffic is used.

  Returns:
    The service as a discovery.Resource object.

  Raises:
    ValueError: If no API key is provided by flag, env var, or file.
  """
  api_key = get_api_key()
  if api_key is None:
    raise ValueError(_ERROR_NO_API_KEY_PROVIDED)

  return _build_service(
      api_key=api_key,
      timeout=timeout,
      dscp=dscp,
  )
