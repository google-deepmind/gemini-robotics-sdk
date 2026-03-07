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

"""GenAI client wrapper for Gemini-based tools."""

import asyncio
import datetime
import enum
import itertools
import os
import time
from typing import Optional, TextIO

from absl import logging
from google import genai
from google.genai import types
import pytz

from safari_sdk.agent.framework import flags as agent_flags
from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.utils import image_processing


_COMPACT_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
_COMPACT_TIMESTAMP_FORMAT_WITH_MILLIS = "%Y%m%d_%H%M%S.%f"
_LOG_DIR = "/tmp/genai_client_logs"
_LOS_ANGELES_TIMEZONE = pytz.timezone("America/Los_Angeles")
_RAW_RESPONSE_LOG_FREQUENCY_SECONDS = 5.0


class LocalFileLogger:
  """Logs timestamped messages to local files in the log directory."""

  def __init__(self):
    # Ensure the log directory exists.
    os.makedirs(_LOG_DIR, exist_ok=True)
    # Name the log file after the current time in `YYYYMMDD_HHMMSS.log` format.
    now = datetime.datetime.now(_LOS_ANGELES_TIMEZONE)
    formatted_time = now.strftime(_COMPACT_TIMESTAMP_FORMAT)
    self._log_file_path = os.path.join(_LOG_DIR, f"{formatted_time}.log")
    try:
      self._log_file: Optional[TextIO] = open(self._log_file_path, "a")
    except OSError as e:
      logging.exception(
          "Failed to open log file %s: %s", self._log_file_path, e
      )
      self._log_file = None

  def log(self, message: str) -> None:
    """Writes a message to the log file."""
    if not self._log_file:
      logging.error("Log file not initialized.")
      return

    try:
      now = datetime.datetime.now(_LOS_ANGELES_TIMEZONE)
      formatted_time = now.strftime(_COMPACT_TIMESTAMP_FORMAT_WITH_MILLIS)
      log_message = f"[{formatted_time}]\n{message}"
      self._log_file.write(log_message)
      self._log_file.flush()
    except OSError as e:
      logging.exception("Failed to write to %s: %s", self._log_file_path, e)

  def close(self) -> None:
    """Closes the log file."""
    if self._log_file:
      try:
        self._log_file.close()
        logging.info("Closed %s", self._log_file_path)
        self._log_file = None
      except OSError as e:
        logging.exception("Failed to close %s: %s", self._log_file_path, e)

  def __del__(self) -> None:
    """Ensures the file is closed when the object is deleted."""
    self.close()


@enum.unique
class GeminiClientHealth(enum.Enum):
  """Class for tracking the health of the Gemini client."""

  NORMAL = "NORMAL"
  # the SD tool returned 429 quota exceeded error.
  ERROR_QUOTA_EXCEEDED = "ERROR_QUOTA_EXCEEDED"
  # the SD tool returned 503 overloaded error.
  ERROR_OVERLOADED = "ERROR_OVERLOADED"
  # SD tool returned other error.
  ERROR_OTHER = "ERROR_OTHER"


class GeminiClientWrapper:
  """A wrapper class for the genai client."""

  def __init__(
      self,
      bus: event_bus.EventBus,
      client: genai.Client,
      model_name: str,
      config: Optional[types.GenerateContentConfigOrDict],
      print_raw_response: bool = False,
      tool_name: Optional[str] = None,
  ):
    self._bus = bus
    self._client = client
    self._model_name = model_name
    self._config = config
    self._print_raw_response = print_raw_response
    self._logger = LocalFileLogger()
    self._gemini_client_health = GeminiClientHealth.NORMAL
    self._tool_name = tool_name
    self._health_lock = asyncio.Lock()

    # Sequence counter for thread-safe health status updates. Each generate()
    # call gets a unique sequence number. Health status updates are only applied
    # if the call's sequence is >= the last update's sequence, ensuring that
    # out-of-order completions (e.g., an older call completing after a newer
    # one) don't incorrectly overwrite the health status.
    self._call_sequence = itertools.count()
    self._last_health_update_sequence = -1
    logging.info("model: %s, config: %s", self._model_name, self._config)

  def set_config(self, config: types.GenerateContentConfigOrDict):
    self._config = config

  def get_config(self) -> types.GenerateContentConfigOrDict:
    return self._config

  async def _maybe_publish(
      self,
      status: GeminiClientHealth,
      exception_msg: str | None,
      call_seq: int,
  ):
    """Publishes a health event if the status has changed.

    Thread-safe: uses a lock and sequence numbers to ensure out-of-order
    generate call completions don't incorrectly overwrite the health status.

    Args:
      status: The new health status to publish.
      exception_msg: The exception message, if any.
      call_seq: The sequence number of the generate call reporting this status.
    """
    async with self._health_lock:
      if call_seq < self._last_health_update_sequence:
        return
      if self._gemini_client_health == status:
        return
      self._last_health_update_sequence = call_seq
      self._gemini_client_health = status
    await self._bus.publish(
        event=event_bus.Event(
            type=event_bus.EventType.TOOL_CLIENT_HEALTH,
            source=event_bus.EventSource.AGENTIC_TOOL,
            data={
                "health_status": status.value,
                "tool_name": self._tool_name,
                "exception_message": exception_msg,
            },
        )
    )

  async def _maybe_publish_gemini_client_health_event(
      self, e: Exception | None, call_seq: int
  ):
    """Publishes a GEMINI_CLIENT_HEALTH event to the event bus if needed."""
    if e is None:
      await self._maybe_publish(GeminiClientHealth.NORMAL, None, call_seq)
    elif "429" in str(e):
      await self._maybe_publish(
          GeminiClientHealth.ERROR_QUOTA_EXCEEDED, str(e), call_seq
      )
    elif "503" in str(e):
      await self._maybe_publish(
          GeminiClientHealth.ERROR_OVERLOADED, str(e), call_seq
      )
    else:
      await self._maybe_publish(
          GeminiClientHealth.ERROR_OTHER, str(e), call_seq
      )

  async def generate_content(self, contents):
    """Generates content using the genai client."""
    call_seq = next(self._call_sequence)
    # Preprocess the contents to convert bytes to PIL image.
    converted_contents = image_processing.convert_bytes_to_image(contents)
    # Async Gemini query.
    time_start = time.time()
    try:
      response = await self._client.aio.models.generate_content(
          model=self._model_name,
          contents=converted_contents,
          config=self._config,
      )
    except Exception as e:
      await self._maybe_publish_gemini_client_health_event(e, call_seq)
      raise e
    time_end = time.time()
    query_time = time_end - time_start
    if self._print_raw_response:
      logging.log_every_n_seconds(
          logging.INFO,
          "[Query] response: %s",
          _RAW_RESPONSE_LOG_FREQUENCY_SECONDS,
          response.text,
      )
      logging.log_every_n_seconds(
          logging.INFO,
          "[Query] latency: %.2f seconds",
          _RAW_RESPONSE_LOG_FREQUENCY_SECONDS,
          query_time,
      )
    # Log the Gemini query if enabled.
    if agent_flags.AGENTIC_LOG_GEMINI_QUERY.value:
      self._logger.log(
          f"model: {self._model_name}\nconfig: {self._config}\ncontents:"
          f" {contents}\nresponse: {response}\nquery_time: {query_time}"
      )
    await self._maybe_publish_gemini_client_health_event(None, call_seq)
    return response
