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

"""Converts observations to contents for a Gemini Robotics model query."""

from collections.abc import Mapping, Sequence
import json
from typing import Any

import jax
import numpy as np
import tensorflow as tf

from safari_sdk.model import constants

_UTF_8_ENCODING = 'utf-8'


def observation_to_model_query_contents(
    *,
    observation: Mapping[str, Any],
    string_observations_keys: Sequence[str],
    task_instruction_key: str,
    proprioceptive_observation_keys: Sequence[str],
    image_observation_keys: Sequence[str],
) -> list[Any]:
  """Encodes the observation as a GenerateRequest."""
  images, encoded_observation = _encode_observations_and_pack_images(
      observation,
      string_observations_keys,
      task_instruction_key,
      proprioceptive_observation_keys,
      image_observation_keys,
  )
  return [
      *images,
      json.dumps(encoded_observation),
  ]


def _encode_observations_and_pack_images(
    observation: Mapping[str, Any],
    string_observations_keys: Sequence[str],
    task_instruction_key: str,
    proprioceptive_observation_keys: Sequence[str],
    image_observation_keys: Sequence[str],
) -> tuple[list[Any], Mapping[str, Any]]:
  """Encodes the observation as a GenerateRequest."""
  encoded_observation = {}

  if constants.CONDITIONING_ENCODED_OBS_KEY in observation:
    encoded_observation[constants.CONDITIONING_ENCODED_OBS_KEY] = observation[
        constants.CONDITIONING_ENCODED_OBS_KEY
    ]

  if constants.RNG_KEY_ENCODED_OBS_KEY in observation:
    rng_key = observation[constants.RNG_KEY_ENCODED_OBS_KEY]
    encoded_observation[constants.RNG_KEY_ENCODED_OBS_KEY] = rng_key.tolist()

  # Encode the task instruction as plain string.
  for obs_name in string_observations_keys:
    plain_str = _get_string_value_from_observation(
        string_obs=observation[obs_name],
        string_obs_name=obs_name,
    )
    if obs_name == task_instruction_key:
      encoded_observation[constants.TASK_INSTRUCTION_ENCODED_OBS_KEY] = (
          plain_str
      )
    else:
      encoded_observation[obs_name] = plain_str

  for obs_name in proprioceptive_observation_keys:
    proprio_obs = observation[obs_name]
    # Tolerate common mistake of having an extra batch dimension.
    if proprio_obs.ndim == 2:
      proprio_obs = proprio_obs[0]
    if proprio_obs.ndim != 1:
      raise ValueError(
          f'Observation {obs_name} has {proprio_obs.ndim} dimensions, but'
          ' should be 1.'
      )
    encoded_observation[obs_name] = proprio_obs.tolist()

  images = []
  for i, image_obs_name in enumerate(image_observation_keys):
    encoded_observation[
        f'{constants.IMAGE_ENCODED_OBS_PREFIX}{image_obs_name}'
    ] = i
    image = observation[image_obs_name]

    if isinstance(image, (np.ndarray, tf.Tensor, jax.Array)):
      # First convert to numpy array if it is a JAX array or Tensor. This
      # ensures that when we eventually coerce to image bytes, they are in a
      # format that can be handled.
      if isinstance(image, jax.Array):
        image = np.asarray(image)
      elif isinstance(image, tf.Tensor):
        image = image.numpy()
      # Tolerate common mistake of having an extra batch dimension.
      image_dim = image.ndim
      if image_dim == 4:
        image = image[0]
      if image.ndim != 3:
        raise ValueError(
            f'Image {image_obs_name} has {image_dim} dimensions, but should'
            ' be 3.'
        )
    elif isinstance(image, bytes):
      pass  # can directly take encoded image bytes.
    else:
      raise ValueError(
          f'Image {image_obs_name} is of type {type(image)}, but should be'
          ' np.ndarray, tf.Tensor or bytes.'
      )
    images.append(image)
  return images, encoded_observation


def _get_string_value_from_observation(
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
