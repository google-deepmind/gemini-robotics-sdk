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

"""Extracts a string value from an observation."""

import numpy as np

_UTF_8_ENCODING = 'utf-8'


def get_string_value_from_observation(
    string_obs: str | bytes | np.ndarray,
    string_obs_name: str,
) -> str:
  """Returns a string value from the observation for a given key.

  Args:
    string_obs: The string observation value to be converted to a
      string. It can be a string, bytes, or a numpy array.
    string_obs_name: The key of the string observation in the
      observation.

  Returns:
    The string value of the string observation.

  Raises:
    ValueError: If the string observation is not a string, bytes, or numpy
    array.
  """
  def _maybe_decode_bytes(value: str | bytes) -> str:
    """Decodes a bytes value to a string if it is not already a string."""
    if isinstance(value, bytes):
      return value.decode(_UTF_8_ENCODING)
    elif isinstance(value, str):
      return value
    else:
      raise ValueError(
          f'Value {value} for key {string_obs_name} is not a string or'
          ' bytes.'
      )

  # Handles different ways the a string observation can be stored and encoded
  # in the observation.
  if isinstance(string_obs, np.ndarray):
    # The string observation is stored as a numpy array. It might still be a
    # scalar list or a list of lists, e.g. np.array('task_instruction') vs
    # np.array(['task_instruction']), so we distinguish between these cases by
    # checking the ndim.
    if string_obs.ndim == 0:
      # Scalar list (e.g. np.array('task_instruction')).
      string_value = string_obs.item()
    else:
      # List of lists (e.g. np.array(['task_instruction'])).
      string_value = string_obs.tolist()[0]
    # If needed, decode the bytes to a string.
    return _maybe_decode_bytes(string_value)
  elif (
      isinstance(string_obs, str)
      or isinstance(string_obs, bytes)
  ):
    # The string observation is stored as a string or bytes.
    # We decode the bytes to a string.
    return _maybe_decode_bytes(string_obs)
  else:
    raise ValueError(
        f'Observation for key {string_obs_name} is not a string,'
        ' bytes, or numpy array.'
    )
