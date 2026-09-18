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
import logging
import time
from typing import Any

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
      request_timeout: int | None = None,
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
      request_timeout: Timeout in seconds for remote model inference requests.
    """

    self._serve_id = serve_id
    self._robotics_api_connection = robotics_api_connection
    self._image_compression_jpeg_quality = image_compression_jpeg_quality

    self._observation_keys = (
        observation_to_model_query_contents.resolve_observation_keys(
            task_instruction_key=task_instruction_key,
            proprioceptive_observation_keys=(proprioceptive_observation_keys),
            image_observation_keys=image_observation_keys,
            additional_observations_providers=(
                additional_observations_providers
            ),
        )
    )
    self._task_instruction_key = self._observation_keys.task_instruction_key
    self._string_observations_keys = list(self._observation_keys.string_keys)
    self._image_observation_keys = list(self._observation_keys.image_keys)
    self._proprioceptive_observation_keys = list(
        self._observation_keys.proprioceptive_keys
    )

    grpc_url = None
    if robotics_api_connection == constants.RoboticsApiConnectionType.LOCAL:
      # Only use serve_id as grpc_url if it looks like a URL or host:port.
      # Dummy IDs like 'gemini_robotics_on_device' will be ignored.
      if serve_id and (serve_id.startswith("grpc://") or ":" in serve_id):
        grpc_url = serve_id

    self._grpc_url = grpc_url
    self._client = genai_robotics.Client(
        robotics_api_connection=robotics_api_connection,
        num_retries=num_of_retries,
        grpc_url=grpc_url,
        method_name=method_name,
        timeout=request_timeout,
    )
    self._method_name = method_name
    self._request_timeout = request_timeout
    self._last_remote_inference_time_ms = None
    self._last_network_overhead_ms = None
    self._last_client_image_encode_ms: float | None = None
    self._last_wire_transit_ms: float | None = None
    self._last_client_processing_ms: float | None = None
    self._last_rng_key: list[int] | None = None
    self._server_ping_ms: float | None = None
    self.ping_server()

  def close(self) -> None:
    """Cleans up resources associated with this model interface."""

  def reset(self) -> None:
    """Resets resources associated with this model interface between episodes."""

  def query_model(
      self,
      model_input: Mapping[str, np.ndarray],
      *,
      rng_key: jax.Array | None = None,
  ) -> np.ndarray:
    """Queries the model with the given observation."""
    del rng_key  # Unused.
    serialized_contents = (
        observation_to_model_query_contents.observation_to_model_query_contents(
            observation=model_input,
            string_observations_keys=self._string_observations_keys,
            task_instruction_key=self._task_instruction_key,
            proprioceptive_observation_keys=(
                self._proprioceptive_observation_keys
            ),
            image_observation_keys=self._image_observation_keys,
        )
    )

    # Serialize the observation to the format expected by the transport.
    if self._robotics_api_connection in (
        constants.RoboticsApiConnectionType.CLOUD_GENAI,
        constants.RoboticsApiConnectionType.LOCAL,
    ):
      serialized_contents = (
          genai_robotics.update_robotics_content_to_genai_format(
              serialized_contents,
              image_compression_jpeg_quality=(
                  self._image_compression_jpeg_quality
              ),
          )
      )

    start_time_sec = time.perf_counter()
    response = self._client.models.generate_content(
        model=self._serve_id,
        contents=serialized_contents,
    )
    end_time_sec = time.perf_counter()
    client_round_trip_ms = (end_time_sec - start_time_sec) * 1000.0

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

    self._last_rng_key = response_data.get(constants.RNG_KEY_RESPONSE_KEY)

    # Calculate remote inference time, wire transit, and client processing
    self._update_latency_metrics(response, client_round_trip_ms)

    return action_chunk

  def _update_latency_metrics(
      self, response: Any, client_round_trip_ms: float
  ) -> None:
    """Calculates and updates remote inference time, wire transit, and overhead."""
    backend_req_time = getattr(response, "backend_request_time", None)
    backend_res_time = getattr(response, "backend_response_time", None)
    self._last_client_image_encode_ms = getattr(
        response, "client_image_encode_ms", None
    )
    client_rpc_ms = getattr(response, "client_rpc_ms", None)

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
        if client_rpc_ms is not None:
          self._last_wire_transit_ms = max(
              0.0, client_rpc_ms - self._last_remote_inference_time_ms
          )
          self._last_client_processing_ms = max(
              0.0, client_round_trip_ms - client_rpc_ms
          )
        else:
          self._last_wire_transit_ms = None
          self._last_client_processing_ms = None
      except (ValueError, TypeError):
        self._last_remote_inference_time_ms = None
        self._last_network_overhead_ms = None
        self._last_wire_transit_ms = None
        self._last_client_processing_ms = None
    else:
      self._last_remote_inference_time_ms = None
      self._last_network_overhead_ms = None
      self._last_wire_transit_ms = None
      self._last_client_processing_ms = None

  @property
  def last_remote_inference_time_ms(self) -> float | None:
    """Remote model execution time in ms (backendResponseTime - backendRequestTime)."""
    return self._last_remote_inference_time_ms

  @property
  def last_wire_transit_ms(self) -> float | None:
    """Pure network flight time in ms across wire (client_rpc_ms - remote_inference_ms)."""
    return self._last_wire_transit_ms

  @property
  def last_client_processing_ms(self) -> float | None:
    """Client-side CPU processing time in ms (client_round_trip_ms - client_rpc_ms)."""
    return self._last_client_processing_ms

  @property
  def last_client_image_encode_ms(self) -> float | None:
    """Client CPU time in ms spent JPEG encoding all camera image feeds."""
    return self._last_client_image_encode_ms

  @property
  def last_network_overhead_ms(self) -> float | None:
    """Total non-server overhead in ms (client round-trip - remote inference)."""
    return self._last_network_overhead_ms

  @property
  def last_rng_key(self) -> list[int] | None:
    """Next PRNG seed key returned by the model server, if available."""
    return self._last_rng_key

  def ping_server(self) -> float | None:
    """Measures and caches the round-trip ping time to the server in ms."""
    try:
      self._server_ping_ms = self._client.ping()
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning("Failed to ping server: %s", e)
      self._server_ping_ms = None
    return self._server_ping_ms

  @property
  def server_ping_ms(self) -> float | None:
    """Baseline ping round-trip time in ms to server measured at start."""
    return self._server_ping_ms
