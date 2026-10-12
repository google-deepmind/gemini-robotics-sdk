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

"""Bidirectional streaming session for Robotics Developer API."""

import collections
from collections.abc import Iterator
import datetime
import enum
import os
import queue
import threading
import types
from typing import Any, TypeAlias

from absl import logging
import grpc

from safari_sdk import auth
from safari_sdk.model import constants
from safari_sdk.protos import modelserving_pb2

loas2: Any = None


class _ModelServingStub:
  """gRPC stub for ModelServing.CmCustomBidi."""

  def __init__(self, channel: grpc.Channel):
    self.CmCustomBidi = channel.stream_stream(  # pylint: disable=invalid-name
        "/google.robotics.developer.v1.ModelServing/CmCustomBidi",
        request_serializer=modelserving_pb2.CmCustomBidiRequest.SerializeToString,
        response_deserializer=modelserving_pb2.CmCustomBidiResponse.FromString,
    )


_ResponseType = modelserving_pb2.CmCustomBidiResponse.ResponseType
_RequestId: TypeAlias = int
_ChunkIndex: TypeAlias = int

# Maximum number of requests that may be live (opened but not yet closed) at
# once. Exceeding it means a caller walked away from a request without
# draining or closing it; the oldest one is evicted so memory stays bounded.
_DEFAULT_MAX_LIVE_REQUESTS = 8

# Bound on the initial SETUP/SETUP_ACK handshake. Unrelated to the per-chunk
# response deadline: this covers connection establishment once per session,
# not steady-state streaming.
DEFAULT_HANDSHAKE_TIMEOUT_SEC = 30.0


def _error_from(
    response: modelserving_pb2.CmCustomBidiResponse,
) -> RuntimeError:
  """Returns the exception to raise to the caller of a failed request.

  Args:
    response: An ERROR response scoped to a single request_id.
  """
  message = (
      response.status.message
      if response.HasField("status")
      else "Server returned ERROR response"
  )
  return RuntimeError(
      f"Server error for request_id={response.request_id}: {message}"
  )


class _ChunkReorderBuffer:
  """Reorders the chunks of a single request by `chunk_index`.

  The server may deliver chunks out of order (see `chunk_index` in
  modelserving.proto), so a chunk is held here until the contiguous prefix
  `0..drained_count` is complete and can be released in order.
  Duplicates and late retransmissions are ignored rather than treated as
  errors.

  Not thread-safe by design: only the session's read loop ever touches it.
  """

  def __init__(self, request_id: _RequestId):
    """Initializes the instance.

    Args:
      request_id: The request whose chunks this buffer reorders.
    """
    self.request_id = request_id
    self._pending: dict[_ChunkIndex, modelserving_pb2.CmCustomBidiResponse] = {}
    self._drained_count: int = 0
    self._final_index: _ChunkIndex | None = None

  def put(self, response: modelserving_pb2.CmCustomBidiResponse) -> bool:
    """Buffers `response` until its turn comes.

    Args:
      response: The chunk to buffer.

    Returns:
      True if the chunk was ignored because that index was already delivered
      or already buffered.
    """
    chunk_index = response.chunk_index
    if chunk_index < self._drained_count:
      return True
    already_buffered = chunk_index in self._pending
    self._pending[chunk_index] = response
    if response.final_chunk:
      self._final_index = chunk_index
    return already_buffered

  def drain_ready(self) -> list[modelserving_pb2.CmCustomBidiResponse]:
    """Removes and returns the longest prefix that is now deliverable."""
    ready = []
    while (
        response := self._pending.pop(self._drained_count, None)
    ) is not None:
      self._drained_count += 1
      ready.append(response)
    return ready

  @property
  def drained_count(self) -> int:
    """The number of contiguous chunks already drained from this buffer."""
    return self._drained_count


class _ChunkSequencer:
  """Sequences out-of-order response chunks into an in-order stream.

  The producer (the session's read loop) pushes chunks in through `put()`.
  They are reordered there and enqueued only once their turn comes, so the
  consumer side is a plain FIFO and needs no condition variable of its own.
  Terminal conditions travel through the same queue as a sentinel exception,
  which wakes a blocked consumer immediately.

  Lifetime: opened by `send_observation()`, unregistered by the consumer once
  it is done with the request. Unregistering does not destroy the sequencer, so
  chunks that were already enqueued remain consumable.
  """

  def __init__(self, request_id: _RequestId):
    """Initializes the instance.

    Args:
      request_id: The request whose chunks this sequencer orders.
    """
    self.request_id = request_id
    self._reorder = _ChunkReorderBuffer(request_id)
    self._ready: queue.SimpleQueue[Any] = queue.SimpleQueue()

  # --- Producer side: called only from the session's read loop. ---

  def put(self, response: modelserving_pb2.CmCustomBidiResponse) -> bool:
    """Buffers `response` and releases every chunk that is now in order.

    Args:
      response: The chunk that just arrived.

    Returns:
      True if the chunk was ignored as a duplicate.
    """
    ignored = self._reorder.put(response)
    for ready in self._reorder.drain_ready():
      self._ready.put(ready)
    return ignored

  # --- Safe to call from any thread. ---

  def terminate(self, error: Exception) -> None:
    """Wakes the consumer with `error` instead of another chunk.

    Args:
      error: The exception subsequent and in-flight `next_chunk()` calls raise.
    """
    self._ready.put(error)

  # --- Consumer side. ---

  def next_chunk(
      self, timeout: float | None = None
  ) -> modelserving_pb2.CmCustomBidiResponse:
    """Returns the next chunk in `chunk_index` order, blocking if needed.

    Args:
      timeout: Seconds to wait for this single chunk. `None` waits forever.

    Returns:
      The next chunk for this request.

    Raises:
      TimeoutError: No chunk became deliverable within `timeout`. Note a chunk
        may have arrived but still not be deliverable, because a lower
        `chunk_index` is missing.
      Exception: The terminal error recorded via `terminate()`.
    """
    try:
      item = self._ready.get(
          timeout=None if timeout is None else max(timeout, 0.0)
      )
    except queue.Empty as e:
      raise TimeoutError(
          f"Timed out waiting for chunk {self._reorder.drained_count}"
          f" of request_id={self.request_id}."
      ) from e
    if isinstance(item, Exception):
      self._ready.put(item)  # Terminal states are sticky.
      raise item
    return item


class _RequestRegistry:
  """Routes server responses to the sequencer of the request they belong to.

  Owns the lifecycle of every in-flight request: it allocates request_ids,
  registers a `_ChunkSequencer` for each, and retires them when the consumer is
  done, when the server fails them, or when the session ends. Because the
  lifecycle is explicit, the registry never has to guess whether a request is
  still wanted, which is what lets it hold a single small lock.

  Thread-safety: Fully thread-safe. All public methods can be called
  concurrently from arbitrary threads (e.g., background receiver feeding
  `put()`, multiple caller threads issuing or closing requests). Internal
  locks protect only in-memory lookups and never block across I/O or waiting.
  """

  def __init__(self, max_live_requests: int = _DEFAULT_MAX_LIVE_REQUESTS):
    """Initializes the instance.

    Args:
      max_live_requests: Maximum number of requests that may be open at once
        before the oldest is evicted.
    """
    self._lock = threading.Lock()
    self._max_live_requests = max_live_requests

    # Insertion-ordered so that the oldest leaked request is evicted first.
    self._sequencers: collections.OrderedDict[_RequestId, _ChunkSequencer] = (
        collections.OrderedDict()
    )
    self._last_issued_id: _RequestId = 0  # 0 is reserved as "unset".
    self._terminal: Exception | None = None

    # Advisory counters. Everything but `_evicted_requests` is written only by
    # the read loop, so reads may lag by one response.
    self._dropped_duplicate_chunks = 0
    self._dropped_after_close = 0
    self._dropped_unknown = 0
    self._evicted_requests = 0

  @property
  def stats(self) -> dict[str, int]:
    """Counters describing routing health.

    A healthy session reports zeros. `dropped_unknown` means the server named
    a request that was never sent; `evicted_requests` means a caller leaked a
    request without closing it.
    """
    with self._lock:
      return {
          "dropped_duplicate_chunks": self._dropped_duplicate_chunks,
          "dropped_after_close": self._dropped_after_close,
          "dropped_unknown": self._dropped_unknown,
          "evicted_requests": self._evicted_requests,
          "live_requests": len(self._sequencers),
      }

  def open(self) -> _ChunkSequencer:
    """Allocates the next request_id and registers its sequencer.

    Allocation and registration happen together so that request_ids are
    monotonic by construction and a response can never arrive before the
    sequencer that must receive it exists.

    Returns:
      The sequencer that will deliver this request's responses. If the session
      has already ended, the sequencer is returned pre-terminated so its
      consumer fails immediately instead of waiting out its timeout.
    """
    evicted = None
    with self._lock:
      self._last_issued_id += 1
      sequencer = _ChunkSequencer(self._last_issued_id)
      if self._terminal is not None:
        sequencer.terminate(self._terminal)
        return sequencer
      self._sequencers[sequencer.request_id] = sequencer
      if len(self._sequencers) > self._max_live_requests:
        _, evicted = self._sequencers.popitem(last=False)
        self._evicted_requests += 1

    if evicted is not None:
      logging.error(
          "Evicting request_id=%d: it was never closed and the registry is"
          " limited to %d live requests. This is a caller-side leak.",
          evicted.request_id,
          self._max_live_requests,
      )
      evicted.terminate(
          RuntimeError(
              f"request_id={evicted.request_id} was evicted because it was"
              " never closed."
          )
      )
    return sequencer

  def close(self, request_id: _RequestId) -> None:
    """Stops routing responses to `request_id`.

    Already-enqueued chunks stay consumable through the sequencer the caller
    still holds. Safe to call more than once.

    Args:
      request_id: The request to retire.
    """
    with self._lock:
      self._sequencers.pop(request_id, None)

  def lookup(self, request_id: _RequestId) -> _ChunkSequencer:
    """Returns the live sequencer for `request_id`.

    Args:
      request_id: The request to resolve.

    Returns:
      The registered sequencer.

    Raises:
      RuntimeError: `request_id` is not live, either because it finished, was
        abandoned or evicted, or because it was never sent.
      Exception: The session ended and `request_id` was one of its requests.
    """
    with self._lock:
      sequencer = self._sequencers.get(request_id)
      if sequencer is not None:
        return sequencer
      if request_id > self._last_issued_id:
        raise RuntimeError(f"request_id={request_id} was never sent.")
      if self._terminal is not None:
        # The session ended; that is the more useful reason to report.
        raise self._terminal
      raise RuntimeError(
          f"request_id={request_id} is no longer live: it already delivered"
          " its final chunk, or it was abandoned or evicted."
      )

  def put(self, response: modelserving_pb2.CmCustomBidiResponse) -> None:
    """Routes `response` to its request's sequencer.

    ERROR responses are handled out of band: they poison their request rather
    than taking a slot in the reorder buffer, whose keys are chunk indices
    they do not carry. The sequencer stays registered so that the consumer
    still resolves it and sees why its request failed; it is retired on the
    consumer's way out like any other.

    Args:
      response: The response that just arrived on the stream.
    """
    request_id = response.request_id

    with self._lock:
      sequencer = self._sequencers.get(request_id)
      if sequencer is None:
        # Nobody is waiting for this response. An id we already issued is
        # benign: that request finished, was abandoned, or was evicted.
        if request_id <= self._last_issued_id:
          self._dropped_after_close += 1
        else:
          # An id we never issued means the server invented it, which is a bug.
          self._dropped_unknown += 1
          logging.error(
              "Dropping response for request_id=%d, which was never sent"
              " (highest issued id is %d). The server is naming requests we"
              " do not know.",
              request_id,
              self._last_issued_id,
          )
        return

    # Outside the lock: neither call blocks, and only the read loop reaches
    # the reorder buffer, so it needs no synchronization of its own.
    if response.response_type == _ResponseType.ERROR:
      sequencer.terminate(_error_from(response))
    elif sequencer.put(response):
      self._dropped_duplicate_chunks += 1
      logging.warning(
          "Ignoring duplicate chunk_index=%d for request_id=%d.",
          response.chunk_index,
          request_id,
      )

  def terminate_all(self, error: Exception) -> None:
    """Retires every live request and wakes its consumer with `error`.

    Requests opened afterwards are handed a pre-terminated sequencer.

    Args:
      error: The exception blocked and subsequent consumers raise.
    """
    with self._lock:
      if self._terminal is None:
        self._terminal = error
      terminal, live = self._terminal, list(self._sequencers.values())
      self._sequencers.clear()
    for sequencer in live:
      sequencer.terminate(terminal)


@enum.unique
class SessionState(enum.Enum):
  """Lifecycle states of a BidiSession.

  State transition flow:
  ```
        ┌─────────────┐
        │   CREATED   │  Initial state after BidiSession() __init__()
        └──────┬──────┘
               │ session.start() [sends SETUP, receives SETUP_ACK handshake]
               ▼
        ┌─────────────┐
        │   ACTIVE    │  Handshake verified; ready to stream observations
        └──────┬──────┘
               │ session.close() / server disconnect / unhandled stream error
               ▼
        ┌─────────────┐
        │   CLOSED    │  Terminal state; stream cancelled; cannot be restarted
        └─────────────┘
  ```

  Lifecycle state details and entry timings:

  - CREATED:
    * Entry timing: Set during `__init__()`.
    * Meaning: The session object has been configured and queues are created,
      but the gRPC streaming RPC has not yet been initiated.
    * Allowed operations: `session.start()` to begin the handshake, or
      `session.close()` to abort. Calling `send_observation()` in this state
      raises RuntimeError.

  - ACTIVE:
    * Entry timing: Entered inside `session.start()` immediately AFTER the
      initial `SETUP` request frame is sent and the server's `SETUP_ACK`
      response is received and verified.
    * Meaning: The bidirectional gRPC connection is established and healthy.
    * Allowed operations: Callers can stream observations via
      `send_observation()` and retrieve action chunks via `get_response()`,
      correlated by the request_id returned from `send_observation()`.

  - CLOSED:
    * Entry timing: Entered when any of the following occur:
      1. Explicit close: Caller invokes `session.close()`, or exits a `with`
         context block.
      2. Handshake failure: `start()` times out waiting for `SETUP_ACK` or
         encounters a network error before handshake completes.
      3. Stream termination: The background `_read_loop` terminates due to
         server half-close (EOF), server ERROR response, or network
         disconnection.
    * Meaning: Terminal state. The underlying gRPC stream is cancelled, queues
      are drained/terminated, and background threads are joined.
    * Allowed operations: Terminal state. Calling `start()` on a CLOSED session
      raises RuntimeError("Cannot restart a closed BidiSession."). Repeat calls
      to `close()` are safe and idempotent.
  """

  CREATED = "CREATED"
  """Session is initialized, awaiting start() and SETUP_ACK handshake."""

  ACTIVE = "ACTIVE"
  """Handshake succeeded (SETUP_ACK received); session is actively streaming."""

  CLOSED = "CLOSED"
  """Session is terminated and cannot be restarted or reused."""


class BidiSession:
  """Persistent bidirectional streaming session to Robotics Developer API."""

  def __init__(
      self,
      channel: grpc.Channel,
      model_id: str,
      method_name: str = constants.SAMPLE_ACTIONS_STREAM_METHOD_NAME,
      api_key: str | None = None,
      timeout: float | None = None,
      use_msgpack: bool = False,
      max_live_requests: int = _DEFAULT_MAX_LIVE_REQUESTS,
  ):
    self._stub = _ModelServingStub(channel)
    self._model_id = model_id
    self._method_name = method_name
    self._api_key = api_key
    self._timeout = timeout
    self._use_msgpack = use_msgpack

    self._send_queue: queue.Queue[
        modelserving_pb2.CmCustomBidiRequest | None
    ] = queue.Queue()
    self._state = SessionState.CREATED
    self._recv_thread: threading.Thread | None = None
    self._stream: Any = None
    self._setup_ack_event = threading.Event()
    self._setup_ack_received = False
    self._error: Exception | None = None

    # Owns request_id allocation as well, so that an id and the queue it
    # names always come into existence together.
    self._registry = _RequestRegistry(max_live_requests=max_live_requests)

    self._last_backend_request_time: datetime.datetime | None = None
    self._last_backend_response_time: datetime.datetime | None = None

  @property
  def state(self) -> SessionState:
    return self._state

  @property
  def is_active(self) -> bool:
    return self._state == SessionState.ACTIVE

  @property
  def is_closed(self) -> bool:
    return self._state == SessionState.CLOSED

  @property
  def last_backend_request_time(self) -> datetime.datetime | None:
    return self._last_backend_request_time

  @property
  def last_backend_response_time(self) -> datetime.datetime | None:
    return self._last_backend_response_time

  @property
  def use_msgpack(self) -> bool:
    """Whether this session uses msgpack serialization."""
    return self._use_msgpack

  @property
  def response_stats(self) -> dict[str, int]:
    """Counters describing response-routing health.

    A healthy session reports zeros. Non-zero values indicate the server named
    requests we never sent, resent chunks, or that a caller abandoned a
    request without closing it.
    """
    return self._registry.stats

  @classmethod
  def create(
      cls,
      connection_type: constants.RoboticsApiConnectionType,
      model_id: str,
      method_name: str = constants.SAMPLE_ACTIONS_STREAM_METHOD_NAME,
      api_key: str | None = None,
      timeout: float | None = None,
      use_msgpack: bool = False,
  ) -> "BidiSession":
    """Factory method to create a BidiSession with the appropriate gRPC channel.

    Args:
      connection_type: The connection type (CLOUD or LOCAL).
      model_id: The model serving ID.
      method_name: The method name on the server.
      api_key: API key for CLOUD connection. If None, fetched via
        auth.get_api_key().
      timeout: Default seconds to wait for a single response chunk, used by
        `get_response()` when its caller does not pass one. Not a bound on the
        session lifetime or on a whole request.
      use_msgpack: Whether to use msgpack serialization.

    Returns:
      A configured BidiSession instance.

    Raises:
      ValueError: If connection_type is not supported for bidi streaming.
    """
    match connection_type:
      case constants.RoboticsApiConnectionType.CLOUD:
        service_address = os.environ.get(
            "ROBOTICS_DEVELOPER_GRPC_ADDRESS",
            "dns:///roboticsdeveloper.googleapis.com:443",
        )
        if (
            "localhost" in service_address or "127.0.0.1" in service_address
        ) and loas2 is not None:
          channel_creds = loas2.loas2_channel_credentials()
          channel = grpc.secure_channel(service_address, channel_creds)
        else:
          channel_creds = grpc.ssl_channel_credentials()
          channel = grpc.secure_channel(
              service_address,
              channel_creds,
              options=[("grpc.service_config_disable_resolution", 1)],
          )
        resolved_api_key = api_key or auth.get_api_key()
      case _:
        raise ValueError(
            "Bidi streaming is not supported for connection type:"
            f" {connection_type}"
        )

    return cls(
        channel=channel,
        model_id=model_id,
        method_name=method_name,
        api_key=resolved_api_key,
        timeout=timeout,
        use_msgpack=use_msgpack,
    )

  def _request_iterator(
      self,
  ) -> Iterator[modelserving_pb2.CmCustomBidiRequest]:
    while True:
      item = self._send_queue.get()
      if item is None or self._state == SessionState.CLOSED:
        break
      yield item

  def _read_loop(self) -> None:
    """Background loop consuming responses from the server."""
    try:
      for resp in self._stream:
        if self._state == SessionState.CLOSED:
          break
        if resp.response_type == _ResponseType.ACTION_CHUNK:
          self._registry.put(resp)
        elif resp.response_type == _ResponseType.SETUP_ACK:
          self._setup_ack_received = True
          self._setup_ack_event.set()
        elif resp.response_type == _ResponseType.ERROR:
          if resp.request_id:
            # Request-scoped failure: deliver it to the caller waiting on this
            # request_id instead of tearing down the whole session.
            self._registry.put(resp)
          else:
            # Stream-scoped failure: no caller owns it, so fail the session.
            err_msg = (
                resp.status.message
                if resp.HasField("status")
                else "Server returned ERROR response"
            )
            self._error = RuntimeError(f"Server error: {err_msg}")
            break
        else:
          logging.warning(
              "Received unhandled response type: %s", resp.response_type
          )
    except Exception as e:  # pylint: disable=broad-exception-caught
      if self._state != SessionState.CLOSED:
        logging.exception("BidiSession background stream encountered an error:")
        self._error = e
    finally:
      was_closed = self._state == SessionState.CLOSED
      self._state = SessionState.CLOSED
      if (
          not self._setup_ack_received
          and self._error is None
          and not was_closed
      ):
        self._error = ConnectionError(
            "Server closed connection before SETUP_ACK."
        )
      self._setup_ack_event.set()
      # Release every blocked get_response() caller so they observe the
      # closure instead of waiting out their timeout.
      self._registry.terminate_all(
          self._error
          if self._error is not None
          else RuntimeError("BidiSession closed while waiting for response.")
      )
      self._send_queue.put(None)

  def start(
      self, handshake_timeout: float = DEFAULT_HANDSHAKE_TIMEOUT_SEC
  ) -> "BidiSession":
    """Sends setup frame and starts background response consumer thread."""
    if self._state == SessionState.ACTIVE:
      return self
    if self._state == SessionState.CLOSED:
      raise RuntimeError("Cannot restart a closed BidiSession.")

    metadata = [("x-goog-api-key", self._api_key)] if self._api_key else None

    # Queue initial Setup frame
    setup_req = modelserving_pb2.CmCustomBidiRequest(
        setup=modelserving_pb2.CmCustomBidiRequest.CmCustomSetup(
            sax_config=modelserving_pb2.CmCustomBidiRequest.SaxModelConfig(
                model_id=self._model_id,
                method_name=self._method_name,
            )
        )
    )
    self._send_queue.put(setup_req)

    try:
      self._stream = self._stub.CmCustomBidi(
          self._request_iterator(),
          metadata=metadata,
      )
    except Exception as e:
      self._error = e
      self.close()
      raise

    self._recv_thread = threading.Thread(target=self._read_loop, daemon=True)
    self._recv_thread.start()

    # Wait for handshake
    if not self._setup_ack_event.wait(timeout=handshake_timeout):
      self.close()
      raise TimeoutError(
          "Timed out waiting for SETUP_ACK handshake from server."
      )

    if self._error is not None:
      self.close()
      raise self._error

    self._state = SessionState.ACTIVE
    return self

  def send_observation(self, observation_payload: bytes) -> int:
    """Sends an observation payload frame to the server.

    Args:
      observation_payload: The serialized observation bytes.

    Returns:
      The request_id stamped on the frame. Pass it to `get_response()` to
      retrieve the matching action chunks.

    Raises:
      RuntimeError: The session is not ACTIVE.
    """
    if self._state != SessionState.ACTIVE:
      if self._error:
        raise self._error
      raise RuntimeError("BidiSession is not active.")

    # Register the sequencer before the frame goes on the wire, so a response
    # cannot arrive before the sequencer that has to receive it exists.
    sequencer = self._registry.open()
    req = modelserving_pb2.CmCustomBidiRequest(
        observation=modelserving_pb2.CmCustomBidiRequest.CmCustomObservation(
            input_bytes=observation_payload,
            request_id=sequencer.request_id,
        )
    )
    self._send_queue.put(req)
    return sequencer.request_id

  def get_response(
      self, request_id: int, timeout: float | None = None
  ) -> modelserving_pb2.CmCustomBidiResponse:
    """Returns the next response chunk for `request_id`, in `chunk_index` order.

    Blocks until the chunk with the expected `chunk_index` is deliverable. A
    chunk that arrives out of order is held back until the gap before it is
    filled.

    Args:
      request_id: The correlation ID returned by `send_observation()`.
      timeout: Seconds to wait for the next chunk. Note this bounds the wait for
        a single chunk, not for the whole request. Defaults to the session-level
        timeout.

    Returns:
      The next chunk for `request_id`, in `chunk_index` order.

    Raises:
      TimeoutError: No chunk became deliverable within `timeout`.
      RuntimeError: The session closed, the server failed this request, or
        `request_id` is no longer live.
      Exception: The stream-level error recorded by the background read loop.
    """
    effective_timeout = self._timeout if timeout is None else timeout
    sequencer = self._registry.lookup(request_id)
    response = sequencer.next_chunk(timeout=effective_timeout)
    if response.final_chunk:
      self._registry.close(request_id)

    self._last_backend_request_time = (
        response.backend_request_time.ToDatetime(tzinfo=datetime.timezone.utc)
        if response.HasField("backend_request_time")
        else None
    )
    self._last_backend_response_time = (
        response.backend_response_time.ToDatetime(tzinfo=datetime.timezone.utc)
        if response.HasField("backend_response_time")
        else None
    )
    return response

  def close(self) -> None:
    """Closes the streaming session gracefully."""
    self._state = SessionState.CLOSED
    self._send_queue.put(None)
    # Wake any thread blocked in get_response() so it observes the closure
    # immediately rather than waiting out its timeout.
    self._registry.terminate_all(
        self._error
        if self._error is not None
        else RuntimeError("BidiSession closed while waiting for response.")
    )
    if self._stream is not None and hasattr(self._stream, "cancel"):
      try:
        self._stream.cancel()
      except (grpc.RpcError, ValueError, RuntimeError):
        # Stream might already be cancelled or completed.
        pass
    if (
        self._recv_thread is not None
        and self._recv_thread.is_alive()
        and threading.current_thread() != self._recv_thread
    ):
      self._recv_thread.join(timeout=1.0)

  def __enter__(self) -> "BidiSession":
    return self.start()

  def __exit__(
      self,
      exc_type: type[BaseException] | None,
      exc_val: BaseException | None,
      exc_tb: types.TracebackType | None,
  ) -> None:
    self.close()
