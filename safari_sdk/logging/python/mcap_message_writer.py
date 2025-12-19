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

"""Safari Robot MCAP Message Writer class."""

import queue
import threading
import time
from google.protobuf import message as message_lib
from safari_sdk.logging.python import file_handler as file_handler_lib
from safari_sdk.logging.python import message as message_holder

_SENTINEL = object()
_THREAD_TIMEOUT_SECONDS = 2 * 60


class McapMessageWriter:
  """Class that orchestrates writing proto messages to mcap files.

  Writes log protos to mcap files, asychronously via a single background thread.
  The main responsibility of this class is to manage the message queue and
  threading aspects of the logging process. The actual writing of messages to
  mcap files is delegated to the provided FileHandler object.

  Typical usage:

    message_writer = McapMessageWriter(file_handler)
    message_writer.reset_file_handler(output_file_prefix, start_nsec)
    message_writer.start()
    message_writer.write_proto_message(topic, message, publish_time_nsec)
    ...
    message_writer.stop(stop_nsec)
  """

  def __init__(
      self,
      file_handler: file_handler_lib.FileHandler,
      message_queue_size_limit: int = 0,
  ):
    """Initializes the McapMessageWriter instance.

    Args:
      file_handler: The file handler for writing messages to mcap files.
      message_queue_size_limit: The size limit of the message queue.
    """
    # If maxsize is less than or equal to zero, the queue size is infinite.
    self._message_queue: queue.Queue[message_holder.Message] = queue.Queue(
        maxsize=message_queue_size_limit
    )
    self._log_writer_thread: threading.Thread | None = None
    self._worker_thread_exception: Exception | None = None

    self._file_handler: file_handler_lib.FileHandler = file_handler

  def reset_file_handler(
      self, output_file_prefix: str, start_timestamp_nsec: int
  ) -> None:
    """Resets the file handler for a new files.

    This should be called before calling start().

    Args:
      output_file_prefix: The prefix of the new file name.
      start_timestamp_nsec: The start timestamp of the new file.
    """
    self._file_handler.reset_for_new_file(
        output_file_prefix=output_file_prefix, start_nsec=start_timestamp_nsec
    )

  def start(self) -> None:
    """Starts the worker thread if not already running."""
    if (
        self._log_writer_thread is None
        or not self._log_writer_thread.is_alive()
    ):
      self._log_writer_thread = threading.Thread(
          target=self._process_message_queue
      )
      self._log_writer_thread.start()

  def stop(
      self,
      stop_timestamp_nsec: int,
      timeout_seconds: int = _THREAD_TIMEOUT_SECONDS,
  ) -> None:
    """Flushes the message queue and stops the worker thread.

    Args:
      stop_timestamp_nsec: The stop timestamp for finalizing the mcap file.
      timeout_seconds: The maximum time to wait for the worker thread to stop.
    """
    self._flush_queue_and_stop_worker(timeout_seconds=timeout_seconds)
    self._file_handler.finalize_and_close_file(stop_nsec=stop_timestamp_nsec)

  def write_proto_message(
      self,
      topic: str,
      message: message_lib.Message,
      publish_time_nsec: int,
      log_time_nsec: int = 0,
  ) -> None:
    """Writes a proto message to the log file if logging is enabled.

    Args:
      topic: The safari_logging_topic of the message.
      message: The proto message to be written.
      publish_time_nsec: The timestamp of the message (this may be the time the
        message was published, or the time the data in the  message was
        sampled).
      log_time_nsec: The timestamp when the logger received the message. If 0,
        the current time will be used.

    Raises:
      RuntimeError: If the log writer thread has failed.
    """
    if self._worker_thread_exception:
      raise RuntimeError(
          'Log writer thread has failed. Cannot write new messages.'
      ) from self._worker_thread_exception

    # By default, this will block if the queue is full. It will not raise an
    # exception.
    self._message_queue.put(
        message_holder.Message(
            topic=topic,
            message=message,
            publish_time_nsec=publish_time_nsec,
            log_time_nsec=log_time_nsec if log_time_nsec else time.time_ns(),
        )
    )

  def _process_message_queue(self) -> None:
    """Processes messages from the queue and writes them to the file handler."""
    try:
      while True:
        message = self._message_queue.get()
        if message is _SENTINEL:  # Sentinel to stop the thread
          self._message_queue.task_done()
          break
        try:
          self._file_handler.write_message(message=message)
        finally:
          self._message_queue.task_done()
    except Exception as e:  # pylint: disable=broad-exception-caught
      self._worker_thread_exception = e

  def _drain_message_queue(self) -> None:
    """Drains all items from the message queue.

    Called when the worker stops, either cleanly or due to an error,
    to ensure the queue is empty.
    """
    while not self._message_queue.empty():
      try:
        self._message_queue.get_nowait()
        self._message_queue.task_done()
      except queue.Empty:
        # This can happen in a race condition if queue becomes empty
        # between check and get_nowait(). It's safe to ignore.
        break

  def _flush_queue_and_stop_worker(self, timeout_seconds: int) -> None:
    """Signals the worker thread to stop and waits for queue processing.

    Args:
      timeout_seconds: The maximum time to wait for the worker thread to stop.

    Raises:
      Exception: Re-raises an exception if the worker thread recorded one.
      RuntimeError: If the worker thread fails to stop within the timeout
                    and no specific exception was recorded by the worker.
    """
    current_thread = self._log_writer_thread
    if current_thread and current_thread.is_alive():
      self._message_queue.put(_SENTINEL)
      current_thread.join(timeout=timeout_seconds)  # Wait for thread to finish
    recorded_exception = self._worker_thread_exception
    self._log_writer_thread = None
    self._worker_thread_exception = None
    # Always drain the queue before raising exceptions.
    self._drain_message_queue()

    if recorded_exception:
      raise RuntimeError(
          'Log writer thread failed with an exception.'
      ) from recorded_exception
    # If the thread is still alive after the timeout, raise an error.
    if current_thread and current_thread.is_alive():
      raise RuntimeError(
          f'Failed to stop log writer thread within {timeout_seconds} seconds.'
      )
