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

"""Helper functions for connecting to a discovery service endpoint.

This module provides a helper function to build a discovery service with a given
API key. The API key can be provided via two methods.

  1. By flag, --api_key="your_api_key"
  2. By file at one of these locations:
        $HOME/.config/safari_sdk/API_KEY
        /opt/safari_sdk/API_KEY

The resolution order on which API key value will be used is:

  1. By flag, "--api_key"
  2. By path, "$HOME/.config/safari_sdk/API_KEY"
  3. By path, "/opt/safari_sdk/API_KEY"

Here is an example of using this module:

  from safari_sdk import auth

  # This returns a discovery.Resource object.
  service = auth.get_service()
"""

import os
from absl import flags

from googleapiclient import discovery

import httplib2

# Flag to manually specify the API key.
_API_KEY = flags.DEFINE_string(
    name="api_key",
    default=None,
    help="API key to use for the Safari API.",
)

# Fixed paths to search for the API key file.
_API_KEY_FILE_PATHS = [
    os.path.join(os.path.expanduser("~"), ".config/safari_sdk/API_KEY"),
    "/opt/safari_sdk/API_KEY",
]

# Default service name, version, and discovery service URL for connection API.
_DEFAULT_SERVICE_NAME = "roboticsdeveloper"
_DEFAULT_VERSION = "v1"
_DEFAULT_DISCOVERY_SERVICE_URL = (
    "https://roboticsdeveloper.googleapis.com/$discovery/rest?version=v1"
)

# Error message.
_ERROR_NO_API_KEY_PROVIDED = (
    "Auth: No API key provided by flag or file."
)
_ERROR_NO_API_KEY_PROVIDED_IN_FILE = (
    "Auth: No API key provided in file:"
)


def _extract_api_key_from_file(file_path: str) -> str:
  """Extracts API key from file."""
  with open(file_path, "r") as f:
    return f.read().strip()


def _build_service(api_key: str) -> discovery.Resource:
  """Builds the service."""
  http = httplib2.Http(timeout=900)  # 15 minutes
  return discovery.build(
      serviceName=_DEFAULT_SERVICE_NAME,
      version=_DEFAULT_VERSION,
      discoveryServiceUrl=_DEFAULT_DISCOVERY_SERVICE_URL,
      developerKey=api_key,
      http=http,
  )


def get_service() -> discovery.Resource:
  """Gets a built discovery service based on flags or fixed file locations.

  The order of resolution precedence for the API key is:
  1. By flag, "--api_key"
  2. By path, "$HOME/.config/safari_sdk/API_KEY"
  3. By path, "/opt/safari_sdk/API_KEY"

  Returns:
    The service as a discovery.Resource object.

  Raises:
    ValueError: If no API key is provided by flag or file.
  """
  if _API_KEY.value:
    return _build_service(api_key=_API_KEY.value)

  for file_path in _API_KEY_FILE_PATHS:
    if os.path.isfile(file_path):
      api_key_from_file = _extract_api_key_from_file(file_path)
      if not api_key_from_file:
        raise ValueError(f"{_ERROR_NO_API_KEY_PROVIDED_IN_FILE} {file_path}")
      return _build_service(api_key=api_key_from_file)

  raise ValueError(_ERROR_NO_API_KEY_PROVIDED)


def get_api_key() -> str | None:
  """Gets the API key based on flags or fixed file locations."""
  if _API_KEY.value:
    return _API_KEY.value

  for file_path in _API_KEY_FILE_PATHS:
    if os.path.isfile(file_path):
      api_key_from_file = _extract_api_key_from_file(file_path)
      if api_key_from_file:
        return api_key_from_file

  # No API key found by flag or file.
  return None
