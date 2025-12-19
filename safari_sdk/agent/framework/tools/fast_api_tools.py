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
import inspect
import re
from typing import Callable, Sequence, cast

from absl import logging
from google.genai import types as genai_types
import httpx

from safari_sdk.agent.framework import constants
from safari_sdk.agent.framework import types
from safari_sdk.agent.framework.embodiments import fast_api_endpoint


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
