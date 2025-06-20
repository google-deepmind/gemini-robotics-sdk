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

"""Safari robot Base Logger class."""

from collections.abc import Collection
import queue
import threading
import time

from google.protobuf import message as message_lib
from safari_sdk.logging.python import constants
from safari_sdk.logging.python import file_handler
from safari_sdk.logging.python import message as message_holder
from safari_sdk.protos import label_pb2
from safari_sdk.protos.logging import metadata_pb2


_SENTINEL = object()


def _is_reserved_topic_name(topic: str) -> bool:
  return topic in file_handler.RESERVED_TOPICS


class BaseLogger:
  """Safari robot Base Logger class.

  Logs messages to mcap files which can be uploaded into SSOT.

  Logging can be done with or without creating a session.

  Use start_session() and stop_session() to log a session.

  Use start_outside_session_logging() and
  stop_outside_session_logging_and_finalize_file() to log without creating a
  session.

  In both cases write_proto_message() is used to write a message to the log.
  Calls to write_proto_message() are ignored if logging is not started (if
  is_recording() returns False).
  """

  def __init__(
      self,
      agent_id: str,
      output_directory: str,
      required_topics: Collection[str],
      optional_topics: Collection[str] | None = None,
      internal_topics: Collection[str] | None = None,
      file_shard_size_limit_bytes: int = constants.DEFAULT_FILE_SHARD_SIZE_LIMIT_BYTES,
      message_queue_size_limit: int = 0,
  ):

    self._agent_id: str = agent_id
    self._session_started: bool = False
    self._session: metadata_pb2.Session = None
    # If True, data outside of sessions will be logged to the same file too.
    self._log_outside_session: bool = False
    self._is_recording: bool = False
    self._stop_nsec = None

    # For non-blocking writes.
    # It's not pratical to prescribe one meaningful limit to handle different
    # traffic patterns. The expectation is the robot system should be able to
    # handle writes fast enough than produced log data so this queue never grows
    # big. This issue typically manifest as:
    #   1) memory use consistently grows during data collection and/or
    #   2) self._log_writer_thread.join(timeout=...) is frequently hit.
    # If maxsize is less than or equal to zero, the queue size is infinite.
    self._message_queue: queue.Queue[message_holder.Message] = queue.Queue(
        maxsize=message_queue_size_limit
    )
    self._log_writer_thread: threading.Thread | None = None
    self._worker_exception: Exception | None = None

    optional_topics = optional_topics or []
    internal_topics = internal_topics or []

    self._required_topics: set[str] = set(required_topics)
    self._all_topics: set[str] = self._required_topics.union(
        optional_topics
    ).union(internal_topics)
    duplicate_topics: set[str] = self._required_topics.intersection(
        set(optional_topics)
    )
    if duplicate_topics:
      raise ValueError(
          'required_topics and optional_topics must not have common elements; '
          f'elements present in both: {duplicate_topics}.'
      )
    for topic in self._required_topics.union(optional_topics):
      if _is_reserved_topic_name(topic):
        raise ValueError(f'Topic name "{topic}" is reserved.')

    self._file_handler: file_handler.FileHandler = file_handler.FileHandler(
        agent_id=self._agent_id,
        topics=self._all_topics,
        output_directory=output_directory,
        file_shard_size_limit_bytes=file_shard_size_limit_bytes,
    )

  def is_session_started(self) -> bool:
    return self._session_started

  def is_logging_outside_session(self) -> bool:
    return self._log_outside_session

  def is_recording(self) -> bool:
    return self._is_recording

  def start_session(
      self,
      *,
      start_nsec: int,
      task_id: str,
      output_file_prefix: str = '',
  ) -> bool:
    """Starts a new session for logging.

    Args:
      start_nsec: The start time of this new session.
      task_id: The task id of this new session.
      output_file_prefix: file name before the shard number and .mcap extension.
        If is_logging_outside_session() is true (because _log_outside_session()
        was called), then this value is ignored because the log file is already
        created and opened.

    Returns:
      True if the session is started successfully.

    Raises:
      ValueError: If the session has already been started.
      ValueError: If outside session logging is started.
    """
    if self._session_started:
      raise ValueError('Session has already been started.')
    if self._log_outside_session:
      raise ValueError(
          'Cannot start a session when outside session logging is started.'
      )
    assert not self._is_recording
    self._worker_exception = None
    self._file_handler.reset_for_new_file(
        output_file_prefix=output_file_prefix, start_nsec=start_nsec
    )
    self._session = metadata_pb2.Session(
        interval=label_pb2.IntervalValue(
            start_nsec=start_nsec,
        ),
        task_id=task_id,
    )
    for topic in self._all_topics:
      self._session.streams.append(
          metadata_pb2.Session.StreamMetadata(
              key_range=metadata_pb2.KeyRange(
                  topic=topic,
                  interval=label_pb2.IntervalValue(
                      start_nsec=start_nsec,
                  ),
              ),
              is_required=topic in self._required_topics,
          )
      )
    self._session_started = True
    self._is_recording = True
    self._stop_nsec = None
    self._ensure_worker_started()
    return True

  def stop_recording_without_saving_session(self, stop_nsec: int) -> None:
    """Stops recording the current session, but does not save the metadata.

    IMPORTANT: stop_recording_without_saving_session() DOES NOT SAVE THE
    SESSION.  Keep reading for details.

    Note: This method is optional.  The straightforward way to record a session
    is to call start_session() and then call stop_session().

    Calling stop_recording_without_saving_session() is useful when some of the
    labels are not available at the time the recoding is stopped.  Once the
    labels have all been added (with add_session_label()) the session must be
    saved with stop_session().  If you call
    stop_recording_without_saving_session() but do not call stop_session() then
    the session WILL NOT BE SAVED.  The start_session() method will raise an
    exception if stop_recording_without_saving_session() was called but
    stop_session() was not called afterwards.

    Args:
      stop_nsec: The stop time of this session.  This is typically the current
        time when stop_recording_without_saving_session() is called.

    Raises:
      ValueError: If the session has not been started.
    """
    if not self._session_started:
      raise ValueError('Session is not started.')
    self._is_recording = False
    self._stop_nsec = stop_nsec

  def stop_session(self, stop_nsec: int) -> None:
    """Stops the current session, updates the metadata and writes to file.

    Args:
      stop_nsec: The stop time of this session.  If
        stop_recording_without_saving_session() was called then the stop_nsec
        passed to stop_recording_without_saving_session is used and this
        argument is ignored.

    Raises:
      ValueError: If the session has not been started.
    """
    if not self._session_started:
      raise ValueError('Session is not started.')
    assert not self._log_outside_session
    self._is_recording = False
    if self._stop_nsec is not None:
      stop_nsec = self._stop_nsec
      self._stop_nsec = None
    self._session.interval.stop_nsec = stop_nsec
    for stream in self._session.streams:
      stream.key_range.interval.stop_nsec = stop_nsec
    self._message_queue.put(
        message_holder.Message(
            topic=constants.SESSION_TOPIC_NAME,
            message=self._session,
            log_time_nsec=stop_nsec,
            publish_time_nsec=stop_nsec,
        )
    )
    self._flush_queue_and_stop_worker()
    self._session_started = False
    self._file_handler.finalize_and_close_file(stop_nsec=stop_nsec)

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
      log_time_nsec: The time when the logger received the message. If 0, the
        current time will be used.

    Raises:
      RuntimeError: If the log writer thread has failed.
    """
    if self._worker_exception:
      raise RuntimeError(
          'Log writer thread has failed. Cannot write new messages.'
      ) from self._worker_exception
    if self.is_recording():
      self._message_queue.put(
          message_holder.Message(
              topic=topic,
              message=message,
              publish_time_nsec=publish_time_nsec,
              log_time_nsec=log_time_nsec if log_time_nsec else time.time_ns(),
          )
      )

  def add_session_label(self, label: label_pb2.LabelMessage) -> None:
    if not self._session_started:
      raise ValueError(
          'add_session_label is called before session has been started.'
      )
    self._session.labels.append(label)

  def start_outside_session_logging(
      self,
      start_nsec: int,
      output_file_prefix: str,
  ) -> None:
    """Enables logging outside of sessions.

    This starts logging to a file, just like start_session does.  But this does
    not create a session object.

    Args:
      start_nsec: The time when this file starts (should be the current time).
      output_file_prefix: file name before the shard number and .mcap extension.

    Raises:
      ValueError: If a session has been started.
    """
    if self._session_started:
      raise ValueError(
          'Cannot start outside session logging when a session is started.'
      )
    if not self._log_outside_session:
      self._worker_exception = None
      self._log_outside_session = True
      self._is_recording = True
      self._file_handler.reset_for_new_file(
          output_file_prefix=output_file_prefix, start_nsec=start_nsec
      )
      self._ensure_worker_started()

  def stop_outside_session_logging_and_finalize_file(
      self, stop_nsec: int
  ) -> None:
    """Stops logging outside of sessions and finalizes the log file.

    Args:
      stop_nsec: The time when logging is stopped.

    Raises:
      ValueError: If a session has been started.
    """
    if self._session_started:
      raise ValueError(
          'stop_outside_session_logging_and_finalize_file() should not be'
          ' called while recording a sesison.'
      )
    if self._log_outside_session:
      self._log_outside_session = False
      self._is_recording = False
      self._flush_queue_and_stop_worker()
      self._file_handler.finalize_and_close_file(stop_nsec=stop_nsec)

  def _ensure_worker_started(self) -> None:
    """Starts the log writer thread if not already running and recording."""
    if self._is_recording and (
        self._log_writer_thread is None
        or not self._log_writer_thread.is_alive()
    ):
      self._log_writer_thread = threading.Thread(
          target=self._process_message_queue
      )
      self._log_writer_thread.start()

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
      self._worker_exception = e

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

  def _flush_queue_and_stop_worker(self) -> None:
    """Signals the worker thread to stop and waits for queue processing.

    Raises:
      Exception: Re-raises an exception if the worker thread recorded one.
      RuntimeError: If the worker thread fails to stop within the timeout
                    and no specific exception was recorded by the worker.
    """
    current_thread = self._log_writer_thread
    if current_thread and current_thread.is_alive():
      self._message_queue.put(_SENTINEL)
      current_thread.join(timeout=2 * 60)  # Wait for thread to finish
    recorded_exception = self._worker_exception
    self._log_writer_thread = None
    self._worker_exception = None
    # Always drain the queue before raising exceptions.
    self._drain_message_queue()

    if recorded_exception:
      raise RuntimeError(
          'Log writer thread failed with an exception.'
      ) from recorded_exception
    if current_thread and current_thread.is_alive():
      raise RuntimeError('Failed to stop log writer thread within 2 minutes.')
