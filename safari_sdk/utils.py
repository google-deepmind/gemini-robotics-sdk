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

"""Common utility functions for Safari SDK."""

import os

# Environment variable name to extract the robot ID.
_DEFAULT_ROBOT_ID_IN_SYSTEM_ENV = "GA_ROBOT_ID"


def get_system_env_variable(var_name: str) -> str:
  """Gets the requested system environment variable."""
  return os.environ.get(var_name, "")


def get_robot_id_from_system_env(
    var_name: str = _DEFAULT_ROBOT_ID_IN_SYSTEM_ENV,
) -> str:
  """Gets the robot ID from the system environment variable."""
  return get_system_env_variable(var_name=var_name)
