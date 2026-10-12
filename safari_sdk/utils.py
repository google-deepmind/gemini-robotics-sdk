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

from collections.abc import Sequence
import os

from absl import flags

# Name of the absl flag holding the robot ID.
#
# Many binaries that (transitively) import this module already define their own
# `--robot_id` flag, so defining it here would raise DuplicateFlagError. Until
# then, `get_robot_id()` only reads the flag if some other module defined it.
# TODO: Define flag here once existing definitions are migrated.
_ROBOT_ID_FLAG_NAME = "robot_id"

# Arguments to `get_config_value()` for looking up the robot ID: the
# environment variable names to check, in priority order, and the name of the
# file in a config directory containing the robot ID.
_ROBOT_ID_CONFIG = {
    "env_vars": ("GA_ROBOT_ID", "SAFARI_ROBOT_ID"),
    "file_name": "ROBOT_ID",
}

# Directories to search for config files, in priority order. Entries may start
# with "~" and may be empty (e.g. if $SAFARI_CONFIG is unset).
_CONFIG_DIRECTORIES = [
    os.environ.get("SAFARI_CONFIG", ""),
    "~/.config/safari_sdk/",
    "/opt/safari_sdk/",  # Deprecated: prefer the user config directory.
]


def _get_value_from_config_file(file_name: str) -> str | None:
  """Reads a value from a file in the Safari SDK config directories.

  Checks "<config_dir>/<file_name>" for each config directory, in order:
    1. $SAFARI_CONFIG (if set)
    2. $HOME/.config/safari_sdk/
    3. /opt/safari_sdk/

  Args:
    file_name: The name of the file to read.

  Returns:
    The stripped contents of the first non-empty file found, otherwise None.
  """
  for dir_path in _CONFIG_DIRECTORIES:
    if not dir_path:
      continue
    full_file_path = os.path.join(os.path.expanduser(dir_path), file_name)
    if not os.path.isfile(full_file_path):
      continue
    with open(full_file_path, "r") as f:
      value = f.read().strip()
    if value:
      return value
  return None


def get_config_value(
    flag: flags.FlagHolder[str | None] | None = None,
    env_vars: Sequence[str] = (),
    file_name: str | None = None,
    required: bool = False,
) -> str | None:
  """Gets a config value from a flag, the environment, or a config file.

  Each source is checked in the order below, and values are stripped of
  surrounding whitespace. Empty values are skipped.
    1. The value of `flag`, if provided.
    2. Each environment variable in `env_vars`, in order.
    3. "<config_dir>/<file_name>", if `file_name` is provided (see
       `_get_value_from_config_file()`).

  Args:
    flag: Optional string flag to check first.
    env_vars: Environment variable names to check, in priority order.
    file_name: Optional name of a file in the config directories to check.
    required: If True, raise an error instead of returning None.

  Returns:
    The first non-empty value found, otherwise None.

  Raises:
    ValueError: If `required` is True and no value is found.
  """
  if flag is not None:
    value = (flag.value or "").strip()
    if value:
      return value

  for var_name in env_vars:
    value = os.environ.get(var_name, "").strip()
    if value:
      return value

  if file_name is not None:
    value = _get_value_from_config_file(file_name)
    if value is not None:
      return value

  if required:
    sources = []
    if flag is not None:
      sources.append(f"the --{flag.name} flag")
    if env_vars:
      env_var_names = " or ".join(f"${var_name}" for var_name in env_vars)
      sources.append(f"the {env_var_names} environment variable")
    if file_name is not None:
      sources.append(f"a {file_name} file in a Safari SDK config directory")
    raise ValueError(f"No value found. Set {', or '.join(sources)}.")

  return None


def _get_robot_id_from_flag() -> str | None:
  """Returns --robot_id if some module defined it as a string flag."""
  if _ROBOT_ID_FLAG_NAME not in flags.FLAGS:
    return None
  value = flags.FLAGS[_ROBOT_ID_FLAG_NAME].value
  if not isinstance(value, str):
    return None
  return value.strip()


def get_robot_id() -> str | None:
  """Gets the robot ID from a flag, the environment, or a config file.

  The resolution order is:
    1. By flag, "--robot_id" (only if defined by another module; see the TODO
       on `_ROBOT_ID_FLAG_NAME`)
    2. By environment variable, $GA_ROBOT_ID then $SAFARI_ROBOT_ID
    3. By file, "<config_dir>/ROBOT_ID" (see `_get_value_from_config_file()`)

  Returns:
    The robot ID if found, otherwise None.
  """
  return _get_robot_id_from_flag() or get_config_value(**_ROBOT_ID_CONFIG)


def get_required_robot_id() -> str:
  """Gets the robot ID, raising an error if it is not set.

  Returns:
    The robot ID.

  Raises:
    ValueError: If no robot ID is found.
  """
  robot_id = _get_robot_id_from_flag() or get_config_value(
      **_ROBOT_ID_CONFIG, required=True
  )
  assert robot_id is not None  # `required=True` raises instead.
  return robot_id


def get_robot_id_from_system_env() -> str | None:
  """Gets the robot ID from the system environment variables.

  Deprecated: use `get_robot_id()` instead.

  Returns:
    The first non-empty value found, otherwise None.
  """
  return get_config_value(env_vars=_ROBOT_ID_CONFIG["env_vars"])
