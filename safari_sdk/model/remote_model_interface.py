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

"""ModelInterface implementation for querying remote models."""

from collections.abc import Mapping, Sequence
import datetime
import json
import time
from typing import Any

from dm_env import specs
import jax
import numpy as np

from safari_sdk.model import additional_observations_provider
from safari_sdk.model import constants
from safari_sdk.model import genai_robotics
from safari_sdk.model import model_interface
from safari_sdk.model import observation_to_model_query_contents


class RemoteModelInterface(model_interface.ModelInterface):
  """Model interface object that queries a remote model."""

  def __init__(
      self,
      serve_id: str,
      robotics_api_connection: constants.RoboticsApiConnectionType,
      task_instruction_key: str,
      proprioceptive_observation_keys: Sequence[str],
      image_observation_keys: Sequence[str],
      image_compression_jpeg_quality: int,
      num_of_retries: int = 1,
      method_name: str = "sample_actions_json_flat",
      additional_observations_providers: Sequence[
          additional_observations_provider.AdditionalObservationsProvider
      ] = (),
  ):
    """Initializes the remote model interface.

    Args:
      serve_id: The serve ID to use for the connecting to the model.
      robotics_api_connection: Connection type for the Robotics API.
      task_instruction_key: The key of the task instruction in the observation.
      proprioceptive_observation_keys: The list of observation keys that are
        related to proprioceptive sensors (e.g. joints).
      image_observation_keys: A list of observation keys that are related to
        images.
      image_compression_jpeg_quality: The JPEG quality for encoding images.
      num_of_retries: The number of retries for inference calls to the server
        when the connection is CLOUD.
      method_name: The method name to call on the robotics API.
      additional_observations_providers: A sequence of providers for additional
        observations.
    """

    self._string_observations_keys = [task_instruction_key]
    self._task_instruction_key = task_instruction_key
    self._image_observation_keys = list(image_observation_keys)
    self._proprioceptive_observation_keys = list(
        proprioceptive_observation_keys
    )
    self._serve_id = serve_id
    self._robotics_api_connection = robotics_api_connection
    self._image_compression_jpeg_quality = image_compression_jpeg_quality

    # Go through the additional observation observations spec and
    # augment the image, string and proprioceptive keys. This is necessary to
    # ensure that the additional observations are serialized and sent to the
    # model.
    for provider in additional_observations_providers:
      additional_specs = provider.get_additional_observations_spec()
      for key, spec in additional_specs.items():
        if isinstance(spec, specs.StringArray):
          self._string_observations_keys.append(key)
        elif isinstance(spec, specs.Array):
          if len(spec.shape) == 3:
            self._image_observation_keys.append(key)
          elif len(spec.shape) == 1 or len(spec.shape) == 2:
            self._proprioceptive_observation_keys.append(key)

    grpc_url = None
    if robotics_api_connection == constants.RoboticsApiConnectionType.LOCAL:
      # Only use serve_id as grpc_url if it looks like a URL or host:port.
      # Dummy IDs like 'gemini_robotics_on_device' will be ignored.
      if serve_id and (serve_id.startswith("grpc://") or ":" in serve_id):
        grpc_url = serve_id

    self._client = genai_robotics.Client(
        robotics_api_connection=robotics_api_connection,
        num_retries=num_of_retries,
        grpc_url=grpc_url,
        method_name=method_name,
    )
    self._last_remote_inference_time_ms = None
    self._last_network_overhead_ms = None

  def query_model(
      self,
      model_input: Mapping[str, np.ndarray],
      *,
      rng_key: jax.Array | None = None,
  ) -> np.ndarray:
    """Queries the model with the given observation."""
    del rng_key  # Unused.
    # Serialize the observation to the format expected by the transport.
    serialized_contents = observation_to_model_query_contents.observation_to_model_query_contents(
        observation=model_input,
        string_observations_keys=self._string_observations_keys,
        task_instruction_key=self._task_instruction_key,
        proprioceptive_observation_keys=self._proprioceptive_observation_keys,
        image_observation_keys=self._image_observation_keys,
    )
    if (
        self._robotics_api_connection
        == constants.RoboticsApiConnectionType.CLOUD_GENAI
    ):
      serialized_contents = genai_robotics.update_robotics_content_to_genai_format(
          serialized_contents,
          image_compression_jpeg_quality=self._image_compression_jpeg_quality,
      )

    start_time_sec = time.perf_counter()
    response = self._client.models.generate_content(
        model=self._serve_id,
        contents=serialized_contents,
    )
    end_time_sec = time.perf_counter()
    client_round_trip_ms = (end_time_sec - start_time_sec) * 1000.0

    # Calculate remote inference time
    self._update_latency_metrics(response, client_round_trip_ms)

    # Parse the response text (assuming its JSON containing the action)
    if response.text:
      response_data = json.loads(response.text)
    elif response.candidates:
      response_data = json.loads(
          response.candidates[0].content.parts[0].inline_data.data
      )
    else:
      raise ValueError("Response does not contain text or candidates.")

    if not isinstance(response_data, dict):
      raise ValueError(
          "Response data does not have a single object as root object."
      )

    # Assuming the structure is {'action_chunk': [...]}
    action_chunk = response_data.get(constants.ACTION_CHUNK_RESPONSE_KEY)
    if action_chunk is None:
      raise ValueError(
          "Response JSON does not contain"
          f" '{constants.ACTION_CHUNK_RESPONSE_KEY}'"
      )
    action_dtype = response_data.get(constants.DTYPE_RESPONSE_KEY) or np.float64
    action_chunk = np.array(action_chunk, dtype=action_dtype)
    if action_chunk.ndim != 2:
      raise ValueError(
          "Action chunk has more than 2 dimensions:"
          f" {action_chunk.shape}. Please make sure the model is configured to"
          " output a 2D array."
      )

    return action_chunk

  def _update_latency_metrics(
      self, response: Any, client_round_trip_ms: float
  ) -> None:
    """Calculates and updates remote inference time and network overhead."""
    backend_req_time = getattr(response, "backend_request_time", None)
    backend_res_time = getattr(response, "backend_response_time", None)

    if isinstance(backend_req_time, str) and isinstance(backend_res_time, str):
      try:
        req_dt = datetime.datetime.fromisoformat(
            backend_req_time.replace("Z", "+00:00")
        )
        res_dt = datetime.datetime.fromisoformat(
            backend_res_time.replace("Z", "+00:00")
        )
        self._last_remote_inference_time_ms = (
            res_dt - req_dt
        ).total_seconds() * 1000.0
        self._last_network_overhead_ms = max(
            0.0, client_round_trip_ms - self._last_remote_inference_time_ms
        )
      except (ValueError, TypeError):
        self._last_remote_inference_time_ms = None
        self._last_network_overhead_ms = None
    else:
      self._last_remote_inference_time_ms = None
      self._last_network_overhead_ms = None

  @property
  def last_remote_inference_time_ms(self) -> float | None:
    return self._last_remote_inference_time_ms

  @property
  def last_network_overhead_ms(self) -> float | None:
    return self._last_network_overhead_ms
