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

"""Safari robot Stream Logger class."""

from collections.abc import Collection
import copy
import threading

from safari_sdk.logging.python import base_logger
from safari_sdk.logging.python import constants
from safari_sdk.logging.python import stream_logger_interface
from safari_sdk.protos.logging import metadata_pb2


class StreamLogger(
    base_logger.BaseLogger, stream_logger_interface.StreamLoggerInterface
):
  """Safari robot Stream Logger class."""

  def __init__(
      self,
      agent_id: str,
      output_directory: str,
      required_topics: Collection[str],
      optional_topics: Collection[str] | None = None,
      file_shard_size_limit_bytes: int = constants.DEFAULT_FILE_SHARD_SIZE_LIMIT_BYTES,
      message_queue_size_limit: int = 0,
  ):
    super().__init__(
        agent_id=agent_id,
        output_directory=output_directory,
        required_topics=required_topics,
        optional_topics=optional_topics,
        internal_topics=set([constants.SYNC_TOPIC_NAME]),
        file_shard_size_limit_bytes=file_shard_size_limit_bytes,
        message_queue_size_limit=message_queue_size_limit,
    )

    # Tracks the time of the most recent message on each topic.
    # Protected by self._sync_message_lock.
    self._sync_message: metadata_pb2.TimeSynchronization = (
        metadata_pb2.TimeSynchronization()
    )
    self._sync_message_lock: threading.Lock = threading.Lock()
    self._have_all_required_topics: bool = False

  def has_received_all_required_topics(self) -> bool:
    if not self._have_all_required_topics:
      with self._sync_message_lock:
        for topic in self._required_topics:
          if topic not in self._sync_message.last_timestamp_by_topic:
            # Have not received all required topics. Cannot start session
            # logging.
            return False
      # Once we have seen all required topics, we will always see all topics,
      # because the sync_message is never cleared.
      self._have_all_required_topics = True
    return True

  def start_session(
      self,
      *,
      start_nsec: int,
      task_id: str,
      output_file_prefix: str = '',
  ) -> bool:

    if not self.has_received_all_required_topics():
      return False

    if not super().start_session(
        task_id=task_id,
        start_nsec=start_nsec,
        output_file_prefix=output_file_prefix,
    ):
      return False
    return True

  def stop_session(self, stop_nsec: int) -> None:
    super().stop_session(stop_nsec=stop_nsec)
    self._session_started = False

  def write_sync_message(self, publish_time_nsec: int) -> None:
    if not self.has_received_all_required_topics():
      raise ValueError(
          'write_sync_message is called before all required topics have been'
          ' received.'
      )
    if not self.is_recording():
      raise ValueError(
          'write_sync_message was called, but no session is active and'
          ' start_outside_session_logging was not called..'
      )
    with self._sync_message_lock:
      sync_message: metadata_pb2.TimeSynchronization = copy.deepcopy(
          self._sync_message
      )
    super().write_proto_message(
        topic=constants.SYNC_TOPIC_NAME,
        message=sync_message,
        log_time_nsec=publish_time_nsec,
        publish_time_nsec=publish_time_nsec,
    )

  # Called within callback functions, maybe multi-threaded.
  def update_synchronization_and_maybe_write_message(
      self,
      topic: str,
      message: stream_logger_interface.LOG_MESSAGE_TYPE,
      publish_time_nsec: int,
      log_time_nsec: int = 0,
  ) -> None:
    if topic not in self._all_topics:
      raise ValueError(
          'Unknown topic not present in during initialization: %s' % topic
      )
    with self._sync_message_lock:
      self._sync_message.last_timestamp_by_topic[topic] = publish_time_nsec
    if self.is_recording():
      super().write_proto_message(
          topic=topic,
          message=message,
          log_time_nsec=log_time_nsec,
          publish_time_nsec=publish_time_nsec,
      )

  def get_latest_sync_message(self) -> metadata_pb2.TimeSynchronization:
    with self._sync_message_lock:
      return copy.deepcopy(self._sync_message)
