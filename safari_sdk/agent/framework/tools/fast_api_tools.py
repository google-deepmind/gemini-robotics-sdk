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

"""Reusable utility classes for connecting to FastAPI endpoints."""

import abc
import asyncio
import datetime
import inspect
import re
from typing import Callable, Sequence, cast

from absl import logging
from google.genai import types as genai_types
import httpx

from safari_sdk.agent.framework import constants
from safari_sdk.agent.framework import flags
from safari_sdk.agent.framework import types
from safari_sdk.agent.framework.embodiments import fast_api_endpoint


_BLACK_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb"
    b"\x00C\x00\x03\x02\x02\x03\x02\x02\x03\x03\x03\x03\x04\x03\x03\x04\x05"
    b"\x08\x05\x05\x04\x04\x05\n\x07\x07\x06\x08\x0c\n\x0c\x0c\x0b\n\x0b\x0b"
    b"\r\x0e\x12\x10\r\x0e\x11\x0e\x0b\x0b\x10\x16\x10\x11\x13\x14\x15\x15"
    b"\x15\x0c\x0f\x17\x18\x16\x14\x18\x12\x14\x15\x14\xff\xc0\x00\x0b\x08"
    b"\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01"
    b"\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04"
    b"\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02"
    b"\x04\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12"
    b'!1A\x06\x13"2q\x14#3B\x15r\x81\x91\xa1\x07\x164V\xb1\xc1\x17\x25\x35'
    b"E\xd1\xf0\x18\xa2\xb2\xc2\xd2\x08\x26F\x82\xe1\x19\x27\x37'H\x92\xc3"
    b"\xf1\x1a\x38t\xd3\x83\xa3\xb3\x28\x34\x36\x39\x3a\x43\x44\x45\x46\x47"
    b"\x48\x49\x4a\x53\x54\x55\x56\x57\x58\x59\x5a\x63\x64\x65\x66\x67\x68"
    b"\x69\x6a\x73\x74\x75\x76\x77\x78\x79\x7a\x84\x85\x86\x87\x88\x89\x8a"
    b"\x93\x94\x95\x96\x97\x98\x99\x9a\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb4\xb5"
    b"\xb6\xb7\xb8\xb9\xba\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd3\xd4\xd5\xd6"
    b"\xd7\xd8\xd9\xda\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf2\xf3\xf4\xf5"
    b"\xf6\xf7\xf8\xf9\xfa\xff\xc4\x00\x1f\x01\x00\x03\x01\x01\x01\x01\x01"
    b"\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07"
    b"\x08\t\n\x0b\xff\xc4\x00\xb5\x11\x00\x02\x01\x02\x04\x04\x03\x04\x07"
    b"\x05\x04\x04\x00\x01\x02w\x00\x01\x02\x03\x11\x04\x05!1\x06\x12\x13"
    b'"2A\x07\x14#3B\x15q\x81\x91\xa1\x08\x16\xb1\xc1\x17\x25\x35E\xf0\xd1'
    b"\x18\xa2\xb2\xc2\xd2\x0b\x26F\x82\xe1\x19\x27\x37'H\x92\xc3\xf1\x1a"
    b"\x38t\xd3\x83\xa3\xb3\x28\x34\x36\x39\x3a\x43\x44\x45\x46\x47\x48\x49"
    b"\x4a\x53\x54\x55\x56\x57\x58\x59\x5a\x63\x64\x65\x66\x67\x68\x69\x6a"
    b"\x73\x74\x75\x76\x77\x78\x79\x7a\x84\x85\x86\x87\x88\x89\x8a\x93\x94"
    b"\x95\x96\x97\x98\x99\x9a\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb4\xb5\xb6\xb7"
    b"\xb8\xb9\xba\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd3\xd4\xd5\xd6\xd7\xd8"
    b"\xd9\xda\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf2\xf3\xf4\xf5\xf6\xf7"
    b"\xf8\xf9\xfa\xff\xcc\x00\x03\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00"
    b"\x02\x11\x03\x11\x00?\x00\xbf\x80\x00\xff\xd9"
)


class FastApiGet:
  """Function that calls a FastAPI endpoint using a dynamic signature.

  Since different endpoints have different signatures, this class creates a
  dynamic signature for the endpoint call method, based on the specified names.
  """

  def __init__(
      self,
      server: str,
      endpoint: fast_api_endpoint.FastApiEndpoint | str,
      param_names: Sequence[str],
      scheduling: genai_types.FunctionResponseScheduling = genai_types.FunctionResponseScheduling.INTERRUPT,
  ):
    """Initializes the FastAPI endpoint call function.

    Args:
      server: The server address (e.g., "http://localhost:8888").
      endpoint: The FastAPI endpoint to call. Can be either a FastApiEndpoint
        object or a string containing the endpoint path. If a string, the path
        must start and end with a forward slash, e.g., "/stop/".
      param_names: The names of the parameters to pass to the endpoint. The
        call_id parameter is implicitly added.
      scheduling: The scheduling of the function response. Defaults to
        INTERRUPT.
    """
    if isinstance(endpoint, fast_api_endpoint.FastApiEndpoint):
      endpoint_path = endpoint.path
    else:
      endpoint_path = endpoint
    self._url = f"{server}{endpoint_path}"

    # Create a signature for the call method.
    param_names = list(param_names)
    param_names.append("call_id")  # Secret name required for all functions.
    params = [
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in param_names
    ]
    self._signature = inspect.Signature(params)
    self._scheduling = scheduling

  async def __call__(self, *args, **kwargs) -> genai_types.FunctionResponse:
    # Bind the incoming arguments to our dynamic signature.
    # This step validates arguments and maps positional to keyword args.
    try:
      bound_args = self._signature.bind(*args, **kwargs)
      # Optional: if you add defaults to your signature.
      bound_args.apply_defaults()
    except TypeError as e:
      # `bind` raises a TypeError automatically if the arguments are wrong.
      raise TypeError(f"Invalid arguments for endpoint {self._url}: {e}") from e

    # The bound arguments are in a convenient dictionary
    api_params = bound_args.arguments
    api_params.pop("call_id", None)  # Call ID is currently unused.

    # Perform the actual HTTP call.

    if flags.AGENTIC_SIMULATE_COMMUNICATION.value:
      return genai_types.FunctionResponse(
          will_continue=False,
          response={"output": "Simulated success"},
          scheduling=self._scheduling,
      )

    t1 = datetime.datetime.now()
    try:
      async with httpx.AsyncClient(timeout=None) as session:
        response = await session.get(self._url, params=api_params)
        response.raise_for_status()
        response_data = {"output": response.json()}
    except httpx.HTTPError as e:
      logging.exception("HTTP error for GET at %s: %s", self._url, e)
      response_data = {
          "output": "FastAPI call execution failed",
          "error": str(e),
      }
    t2 = datetime.datetime.now()
    logging.info(
        "FastApiGet %s took %.3fs", self._url, (t2 - t1).total_seconds()
    )
    return genai_types.FunctionResponse(
        will_continue=False,
        response=response_data,
        scheduling=self._scheduling,
    )

  @property
  def __signature__(self):
    # Override the __signature__ for introspection (e.g., help()).
    return self._signature


class FastApiStream(metaclass=abc.ABCMeta):
  """Returns functions to stream data from a FastAPI endpoint."""

  def __init__(
      self,
      server: str,
      endpoint: fast_api_endpoint.FastApiEndpoint | str,
      stream_name: str = "",
      reconnect_delay: float = 3.0,
  ):
    """Initializes the FastAPI stream.

    Args:
      server: The server address (e.g., "http://localhost:8888").
      endpoint: The FastAPI endpoint to stream from. Can be either a
        FastApiEndpoint object or a string containing the endpoint path. If a
        string, the path must start and end with a forward slash, e.g.,
        "/camera_stream/".
      stream_name: The name of the stream. This is used as the metadata key for
        the events.
      reconnect_delay: Seconds to wait before attempting to reconnect after a
        stream failure.
    """
    if isinstance(endpoint, fast_api_endpoint.FastApiEndpoint):
      endpoint_path = endpoint.path
    else:
      endpoint_path = endpoint
    self._url = f"{server}{endpoint_path}"  # Note: path contains all slashes.
    self._stream_name = stream_name
    self._reconnect_delay = reconnect_delay

  @abc.abstractmethod
  async def _read_stream(
      self, response: httpx.Response
  ) -> types.EventStream[types.Event]:
    """Reads an HTTP response stream and constructs events from it.

    Args:
      response: The HTTP response to stream data from.

    Yields:
        An event read from the stream.
    """
    yield

  async def stream(self) -> types.EventStream[types.Event]:
    """Event generator for a stream of data."""

    while True:
      # Use the retry library for fuzzed retries to avoid multiple
      # connections synching and hammering the server at once, and
      # potentially exponentiating to avoid GO_AWAY.

      try:
        logging.info("Connecting to stream at: %s", self._url)
        async with httpx.AsyncClient(timeout=None) as client:
          async with client.stream("GET", self._url) as response:
            response.raise_for_status()
            logging.info("Successfully connected to stream at: %s.", self._url)
            response = cast(httpx.Response, response)
            async for event in self._read_stream(response):
              yield event
            logging.info(
                "Stream at %s closed by the server. Will reconnect.",
                self._url,
            )
      except httpx.HTTPStatusError as e:
        # For 4xx/5xx errors, log the details before retrying.
        logging.exception(
            "HTTP error during stream at %s connection: %r. Will reconnect.",
            self._url,
            e,
        )
      except asyncio.CancelledError:
        logging.info("Stream at %s cancelled by client.", self._url)
        break
      except httpx.ConnectError as e:
        logging.warning(
            "Connection failed while connecting to %s: %r. Is the backend"
            " running? Will reconnect.",
            self._url,
            e,
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception(
            "An unexpected error occurred in the stream at %s: %r. Will"
            " reconnect.",
            self._url,
            e,
        )

      logging.info(
          "Waiting %.1f seconds before attempting to reconnect to %s...",
          self._reconnect_delay,
          self._url,
      )
      await asyncio.sleep(self._reconnect_delay)


class FastApiVideoStream(FastApiStream):
  """Returns a function that streams video via a FastAPI endpoint."""

  async def stream(self) -> types.EventStream[types.Event]:
    """Event generator for a stream of data."""
    if flags.AGENTIC_SIMULATE_COMMUNICATION.value:
      logging.info("Simulating video stream for %s", self._stream_name)
      while True:
        yield types.Event(
            type=types.EventType.MODEL_IMAGE_INPUT,
            source=types.EventSource.ROBOT,
            data=_BLACK_JPEG,
            metadata={constants.STREAM_NAME_METADATA_KEY: self._stream_name},
        )
        await asyncio.sleep(0.1)
    else:
      async for event in super().stream():
        yield event

  async def _read_stream(self, response):
    # Cast to httpx.Response to access the aiter_bytes() method.
    response = cast(httpx.Response, response)
    content_type = response.headers.get("Content-Type", "")
    if "multipart/x-mixed-replace" not in content_type:
      logging.info(
          "Error: Expected 'multipart/x-mixed-replace' but got %s",
          content_type,
      )
      return

    # Extract the boundary from the Content-Type header
    match = re.search(r"boundary=(.+)", content_type)
    if not match:
      logging.info("Error: Could not find boundary in Content-Type header.")
      return
    boundary = b"--" + match.group(1).encode("utf-8")

    buffer = b""
    chunk_size = 8192  # Read in chunks
    async for chunk in response.aiter_bytes(chunk_size=chunk_size):
      if not chunk:
        logging.info("Stream ended.")
        break
      buffer += chunk

      # Find all occurrences of the boundary within the buffer
      parts = buffer.split(boundary)

      # The last part might be incomplete, keep it for the next chunk
      buffer = parts.pop()

      for part in parts:
        if b"Content-Type: image/jpeg\r\n\r\n" in part:
          # Extract the image data
          # The data starts after the header and ends before the next
          # boundary or end of stream
          header_end = part.find(b"\r\n\r\n")
          if header_end != -1:
            image_bytes = part[header_end + 4 :].strip(b"\r\n")
            if image_bytes:
              yield types.Event(
                  type=types.EventType.MODEL_IMAGE_INPUT,
                  source=types.EventSource.ROBOT,
                  data=image_bytes,
                  metadata={
                      constants.STREAM_NAME_METADATA_KEY: self._stream_name
                  },
              )


class FastApiAudioStream(FastApiStream):
  """Returns a function that streams audio via a FastAPI endpoint."""

  async def _read_stream(self, response: httpx.Response):
    chunk_size = 1024
    async for chunk in response.aiter_bytes(chunk_size):
      yield types.Event(
          type=types.EventType.MODEL_AUDIO_INPUT,
          source=types.EventSource.USER,
          data=chunk,
      )


class FastApiServerSentEventsStream(FastApiStream):
  """Returns a function that streams server-sent events data."""

  def __init__(
      self,
      server_sent_event_data_to_event_formatter: Callable[
          [str], types.Event | None
      ],
      **kwargs,
  ):
    """Initializes the FastAPI text stream function.

    Args:
      server_sent_event_data_to_event_formatter: A callback that converts the
        server-sent events data to a types.Event that will be published to the
        event bus.
      **kwargs: Additional arguments to pass to the base class.
    """
    self._sse_data_to_event_formatter = (
        server_sent_event_data_to_event_formatter
    )
    super().__init__(**kwargs)

  async def _read_stream(self, response: httpx.Response):
    # Parse the SSE messages.
    # An SSE event streams each message in a single line independently and is
    # formatted as data: <message>\n\n where message could be a JSON object or
    # a string. Note for JSON the expectation is that the server would send
    # the entire JSON object as a single line.
    sse_data_lines = []

    # Read the lines of data from the response. Note that the aiter_lines()
    # method will already decode the response to UTF-8 or the encoding
    # specified in the Content-Type header.
    async for line in response.aiter_lines():
      # The payload of the SSE messages should be prefixed with "data: " and
      # terminated with "\n\n".

      # An empty line indicates the end of the message and so we use it to
      # emit the buffered data.
      if not line:
        if sse_data_lines:
          # We join the buffered lines into a single string. It is the
          # responsibility of event formatter callback passed to this
          # class to format the data as needed and return it as a
          # types.Event that would be published to the event bus.
          sse_data = "\n".join(sse_data_lines)
          event = self._sse_data_to_event_formatter(sse_data)
          if event is not None:
            yield event
          sse_data_lines = []
        continue

      # Otherwise, we check if the line starts with "data: " and add it to
      # the buffer. Other lines that don't start with "data: " are ignored as
      # data should be the main payload of the SSE message.
      if line.startswith("data:"):
        # Remove the "data: " prefix and strip the trailing whitespace.
        sse_data_lines.append(line[5:].strip())
