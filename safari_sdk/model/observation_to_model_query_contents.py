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

import numpy as np
import tensorflow as tf

from safari_sdk.model import constants


def observation_to_model_query_contents(
    observation: Mapping[str, Any],
    model_output: np.ndarray,
    string_observations_keys: Sequence[str],
    task_instruction_key: str,
    proprioceptive_observation_keys: Sequence[str],
    image_observation_keys: Sequence[str],
    inference_mode: constants.InferenceMode,
) -> list[Any]:
  """Encodes the observation as a GenerateRequest."""
  images, encoded_observation = _observation_to_model_query_contents(
      observation,
      model_output,
      string_observations_keys,
      task_instruction_key,
      proprioceptive_observation_keys,
      image_observation_keys,
      inference_mode,
  )
  return [
      *images,
      json.dumps(encoded_observation),
  ]


# Build this function to be able to test it.
def _observation_to_model_query_contents(
    observation: Mapping[str, Any],
    model_output: np.ndarray,
    string_observations_keys: Sequence[str],
    task_instruction_key: str,
    proprioceptive_observation_keys: Sequence[str],
    image_observation_keys: Sequence[str],
    inference_mode: constants.InferenceMode,
) -> tuple[list[Any], Mapping[str, Any]]:
  """Encodes the observation as a GenerateRequest."""
  encoded_observation = {}
  # Conditioning on what the model has left to output.
  if (
      inference_mode == constants.InferenceMode.ASYNCHRONOUS
      and model_output.size > 0
  ):
    encoded_observation[constants.CONDITIONING_ENCODED_OBS_KEY] = (
        model_output.tolist()
    )

  # Encode the task instruction as plain string.
  for obs_name in string_observations_keys:
    plain_str = np.array_str(observation[obs_name])
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
    if isinstance(image, (np.ndarray, tf.Tensor)):
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
