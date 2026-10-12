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

"""Remote model interface for querying models using bidirectional streaming."""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import datetime
import threading
import time

import jax
import numpy as np

from safari_sdk.model import additional_observations_provider
from safari_sdk.model import bidi_session
from safari_sdk.model import constants
from safari_sdk.model import observation_to_model_query_contents
from safari_sdk.model import payload_codec


@dataclasses.dataclass(frozen=True, kw_only=True)
class BidiModelTelemetry:
  """Telemetry metrics for a bidirectional model query.

  Attributes:
    request_time: Timestamp when the observation request was sent / processed by
      the backend.
    response_time: Timestamp when the model response was received / emitted by
      the backend.
    remote_inference_duration: Duration of model execution on the server
      backend.
    network_overhead_duration: Duration of client round-trip transit minus
      remote inference duration.
    client_image_encode_duration: Duration spent compressing camera observation
      images on client.
    wire_transit_duration: Physical wire transit duration excluding server
      compute, if measured.
    client_processing_duration: Client CPU workstation processing time, if
      measured.
    rng_key: Random number generator key returned by the model backend.
  """

  # Wall-clock timestamps (UTC)
  request_time: datetime.datetime | None = None
  response_time: datetime.datetime | None = None

  # Latency durations
  remote_inference_duration: datetime.timedelta | None = None
  network_overhead_duration: datetime.timedelta | None = None
  client_image_encode_duration: datetime.timedelta | None = None
  wire_transit_duration: datetime.timedelta | None = None
  client_processing_duration: datetime.timedelta | None = None

  # Metadata
  rng_key: list[int] | None = None

  @property
  def remote_inference_ms(self) -> float | None:
    """Server-side model inference time in milliseconds."""
    if self.remote_inference_duration is not None:
      return self.remote_inference_duration.total_seconds() * 1000.0
    return None

  @property
  def network_overhead_ms(self) -> float | None:
    """Network overhead in milliseconds."""
    if self.network_overhead_duration is not None:
      return self.network_overhead_duration.total_seconds() * 1000.0
    return None

  @property
  def client_image_encode_ms(self) -> float | None:
    """Client image compression time in milliseconds."""
    if self.client_image_encode_duration is not None:
      return self.client_image_encode_duration.total_seconds() * 1000.0
    return None

  @property
  def wire_transit_ms(self) -> float | None:
    """Physical wire transit in milliseconds."""
    if self.wire_transit_duration is not None:
      return self.wire_transit_duration.total_seconds() * 1000.0
    return None

  @property
  def client_processing_ms(self) -> float | None:
    """Client workstation processing time in milliseconds."""
    if self.client_processing_duration is not None:
      return self.client_processing_duration.total_seconds() * 1000.0
    return None


@dataclasses.dataclass(frozen=True)
class ActionChunk:
  """One chunk of a streamed response.

  Attributes:
    actions: The actions this chunk carries, shaped [chunk_size, action_dim].
    is_final: Whether the server marked this the last chunk of the response.
      Taken from `CmCustomBidiResponse.final_chunk`, so a consumer learns the
      response is over from the chunk in hand. The alternative -- asking the
      iterator for one more chunk and reading `StopIteration` as the signal --
      only answers promptly when the response really is over; when it is not, it
      blocks for as long as the next chunk takes to arrive, and the caller
      cannot tell the two cases apart before making the call.
  """

  actions: np.ndarray
  is_final: bool


class ActionHorizon:
  """Owns the streamed action horizon for a single model query.

  Actions in an `ActionHorizon` stream in progressively from the model rather
  than arriving as a complete batch upfront. At construction time, the full set
  of actions does not exist yet: the system cannot know in advance how many
  total
  actions the model will generate, nor when individual action chunks will arrive
  over the wire. Instead, actions accumulate dynamically in the buffer as chunks
  land.

  Chunks from the underlying response stream are folded into the horizon on
  demand:
  - State queries (`is_complete`, `is_exhausted`, `has_arrived_actions`,
    `available_actions_count`, `snapshot`) non-blockingly drain any chunks
    already ready in the session
    (by calling `read_chunk(blocking=False)` and stopping on `TimeoutError`).
  - `take_next_action()` returns the action at the current execution cursor and
    advances the cursor, blocking (`read_chunk(blocking=True)`) for the next
    chunk from the stream only when no buffered actions remain.
  """

  def __init__(
      self,
      read_chunk: Callable[..., ActionChunk],
  ):
    self._read_chunk = read_chunk
    self._horizon = np.array([])
    self._next_index = 0
    self._chunks_received = 0
    self._is_complete = False

  def _fold_chunk(self, chunk: ActionChunk) -> None:
    self._chunks_received += 1
    self._is_complete = chunk.is_final
    if self._horizon.size == 0:
      self._horizon = chunk.actions.copy()
    else:
      self._horizon = np.concatenate([self._horizon, chunk.actions], axis=0)

  def _drain_ready(self) -> None:
    while not self._is_complete:
      try:
        chunk = self._read_chunk(blocking=False)
      except TimeoutError:
        break
      self._fold_chunk(chunk)

  @property
  def is_complete(self) -> bool:
    """Whether all response chunks for this query have been received."""
    self._drain_ready()
    return self._is_complete

  @property
  def is_exhausted(self) -> bool:
    """Whether all chunks have arrived and all actions have been executed."""
    self._drain_ready()
    return self._is_complete and (
        self._horizon.shape[0] - self._next_index <= 0
    )

  @property
  def chunks_received(self) -> int:
    """Number of response chunks folded into this horizon so far."""
    self._drain_ready()
    return self._chunks_received

  def actions_consumed(self) -> int:
    """Returns how far the execution cursor has advanced into this horizon."""
    return self._next_index

  def advance_cursor(self, steps: int = 1) -> None:
    """Advances the execution cursor by `steps` while a newer query is in flight."""
    self._next_index += steps

  def take_next_action(self) -> np.ndarray:
    """Returns the next action and advances the execution cursor.

    Blocks waiting for the next streamed chunk only if no buffered actions
    remain.

    Returns:
      The action to execute now.

    Raises:
      ValueError: The response stream ended without providing an actionable
        chunk past the current execution cursor.
      TimeoutError: Waiting for the next chunk timed out.
    """
    self._drain_ready()
    while self._horizon.shape[0] - self._next_index <= 0:
      if self._is_complete:
        raise ValueError(
            "Bidi model ended its response stream with no action left to"
            f" execute after {self._chunks_received} chunk(s)."
        )
      self._fold_chunk(self._read_chunk(blocking=True))
      self._drain_ready()

    action = self._horizon[self._next_index]
    self._next_index += 1
    return action

  def has_arrived_actions(self) -> bool:
    """Returns whether there are arrived actions ready to execute past the cursor."""
    self._drain_ready()
    return (self._horizon.shape[0] - self._next_index) > 0

  @property
  def available_actions_count(self) -> int:
    """Returns the number of buffered actions currently available to execute."""
    self._drain_ready()
    return max(0, self._horizon.shape[0] - self._next_index)

  def snapshot(self, *, wait_until_exhausted: bool = False) -> np.ndarray:
    """Returns a copy of the whole horizon, executed part included.

    Args:
      wait_until_exhausted: If True, blocks until all chunks of the response
        have been received before returning the copy.
    """
    if wait_until_exhausted:
      while not self._is_complete:
        self._fold_chunk(self._read_chunk(blocking=True))
    else:
      self._drain_ready()
    return self._horizon.copy()


class BidiRemoteModelInterface:
  """Remote model interface that uses persistent bidirectional gRPC streaming."""

  def __init__(
      self,
      serve_id: str,
      robotics_api_connection: constants.RoboticsApiConnectionType,
      task_instruction_key: str,
      proprioceptive_observation_keys: Sequence[str],
      image_observation_keys: Sequence[str],
      image_compression_jpeg_quality: int,
      request_timeout: float,
      num_of_retries: int = 1,
      additional_observations_providers: Sequence[
          additional_observations_provider.AdditionalObservationsProvider
      ] = (),
  ):
    """Initializes the bidirectional streaming remote model interface.

    Args:
      serve_id: The serve ID to use for connecting to the model.
      robotics_api_connection: Connection type for the Robotics API.
      task_instruction_key: The key of the task instruction in the observation.
      proprioceptive_observation_keys: The list of observation keys that are
        related to proprioceptive sensors (e.g. joints).
      image_observation_keys: A list of observation keys that are related to
        images.
      image_compression_jpeg_quality: The JPEG quality for encoding images.
      request_timeout: Seconds to wait for each individual response chunk. Note
        this bounds a single chunk, not the whole request: a response of N
        chunks may legitimately take up to N times this long.
      num_of_retries: The number of retries (reserved for reconnection
        policies).
      additional_observations_providers: A sequence of providers for additional
        observations.

    Raises:
      ValueError: `request_timeout` is not positive.
    """
    del num_of_retries  # Reserved.
    if request_timeout <= 0:
      raise ValueError(
          f"request_timeout must be positive, got {request_timeout}."
      )
    self._serve_id = serve_id
    self._robotics_api_connection = robotics_api_connection
    self._image_compression_jpeg_quality = image_compression_jpeg_quality
    self._chunk_timeout_s = float(request_timeout)

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

    self._bidi_session: bidi_session.BidiSession | None = None
    self._lock = threading.Lock()
    self._closed = False

    self._last_telemetry: BidiModelTelemetry | None = None

  def get_telemetry(self) -> BidiModelTelemetry | None:
    return self._last_telemetry

  def reset(self) -> None:
    """Resets the streaming session by closing the current session and opening a fresh one."""
    with self._lock:
      if self._closed:
        raise RuntimeError(
            "Cannot reset model: BidiRemoteModelInterface has been closed."
        )
      if self._bidi_session is not None:
        self._bidi_session.close()
        self._bidi_session = None
      session = self._open_bidi_session()
      session.start()
      self._bidi_session = session

  def close(self) -> None:
    """Cleans up resources and closes any active streaming session."""
    with self._lock:
      self._closed = True
      if self._bidi_session is not None:
        self._bidi_session.close()
        self._bidi_session = None

  def __del__(self) -> None:
    try:
      self.close()
    except Exception:  # pylint: disable=broad-except
      pass

  def _open_bidi_session(self) -> bidi_session.BidiSession:
    """Opens a persistent bidirectional streaming session for this model."""
    return bidi_session.BidiSession.create(
        connection_type=self._robotics_api_connection,
        model_id=self._serve_id,
        timeout=self._chunk_timeout_s,
    )

  def _decode_action_chunk(
      self,
      output_bytes: bytes,
      use_msgpack: bool = False,
  ) -> tuple[np.ndarray, list[int] | None]:
    """Decodes action chunk and RNG key from output bytes."""
    decoded = payload_codec.decode_action_chunk_payload(
        output_bytes, use_msgpack=use_msgpack
    )
    rng_key = decoded.metadata.get(constants.RNG_KEY_RESPONSE_KEY)
    return decoded.action_chunk, rng_key

  def query_model(
      self,
      model_input: Mapping[str, np.ndarray],
      *,
      rng_key: jax.Array | None = None,
  ) -> ActionHorizon:
    """Queries the model using bidirectional streaming with the given observation.

    The observation is sent before this method returns; the returned
    `ActionHorizon` only drives retrieval. This keeps the send off the
    consumer's critical path and guarantees the request is in flight even if the
    caller defers reading actions.

    Args:
      model_input: The observation to query the model with.
      rng_key: Unused.

    Returns:
      The `ActionHorizon` assembling the streamed action chunks for this query.

    Raises:
      RuntimeError: The interface is closed or has no active session.
    """
    del rng_key  # Unused.
    if self._closed:
      raise RuntimeError(
          "Cannot query model: BidiRemoteModelInterface has been closed."
      )
    if self._bidi_session is None:
      raise RuntimeError(
          "Cannot query model: BidiRemoteModelInterface has no active session. "
          "Call reset() before querying."
      )
    session = self._bidi_session

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

    start_time_sec = time.perf_counter()
    encoded = payload_codec.encode_observation_payload(
        serialized_contents,
        image_compression_jpeg_quality=self._image_compression_jpeg_quality,
        use_msgpack=session.use_msgpack,
    )
    client_image_encode_duration = (
        datetime.timedelta(milliseconds=encoded.encode_time_ms)
        if encoded.encode_time_ms is not None
        else None
    )

    request_id = session.send_observation(encoded.payload_bytes)

    def read_chunk(blocking: bool = True) -> ActionChunk:
      response = session.get_response(
          request_id, timeout=self._chunk_timeout_s if blocking else 0.0
      )
      action_chunk, resp_rng_key = self._decode_action_chunk(
          response.output_bytes,
          use_msgpack=session.use_msgpack,
      )
      self._last_telemetry = self._build_telemetry(
          session=session,
          start_time_sec=start_time_sec,
          client_image_encode_duration=client_image_encode_duration,
          rng_key=resp_rng_key,
      )
      return ActionChunk(actions=action_chunk, is_final=response.final_chunk)

    return ActionHorizon(read_chunk=read_chunk)

  def _build_telemetry(
      self,
      session: bidi_session.BidiSession,
      start_time_sec: float,
      client_image_encode_duration: datetime.timedelta | None,
      rng_key: list[int] | None,
  ) -> BidiModelTelemetry:
    """Builds telemetry for the chunk that was just received.

    Args:
      session: The session the chunk arrived on.
      start_time_sec: `time.perf_counter()` reading taken before encoding.
      client_image_encode_duration: Time spent JPEG-encoding the observation.
      rng_key: RNG key reported by the backend for this chunk.

    Returns:
      Telemetry describing the round trip up to and including this chunk.
    """
    end_time_sec = time.perf_counter()
    req_dt = session.last_backend_request_time
    res_dt = session.last_backend_response_time
    remote_inference_duration = (
        res_dt - req_dt if req_dt is not None and res_dt is not None else None
    )
    client_round_trip_duration = datetime.timedelta(
        seconds=end_time_sec - start_time_sec
    )
    network_overhead_duration = (
        client_round_trip_duration - remote_inference_duration
        if remote_inference_duration is not None
        else None
    )
    return BidiModelTelemetry(
        request_time=req_dt,
        response_time=res_dt,
        remote_inference_duration=remote_inference_duration,
        network_overhead_duration=network_overhead_duration,
        client_image_encode_duration=client_image_encode_duration,
        rng_key=rng_key,
    )
