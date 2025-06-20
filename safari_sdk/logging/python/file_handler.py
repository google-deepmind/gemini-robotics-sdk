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

"""A helper class for writing log entries to a mcap file, with file rotation."""

from collections.abc import Collection
import datetime
import os
import pathlib
import stat
import sys
import threading

from mcap_protobuf import writer as mcap_protobuf_writer

from safari_sdk.logging.python import constants
from safari_sdk.logging.python import message as message_lib
from safari_sdk.protos import label_pb2
from safari_sdk.protos.logging import metadata_pb2


RESERVED_TOPICS = frozenset([
    constants.FILE_METADATA_TOPIC_NAME,
    constants.SESSION_TOPIC_NAME,
    constants.SYNC_TOPIC_NAME,
    constants.TIMESTEP_TOPIC_NAME,
    constants.ACTION_TOPIC_NAME,
])


class FileHandler:
  """A helper class for writing log entries to a mcap file, with file rotation."""

  def __init__(
      self,
      agent_id: str,
      topics: Collection[str],
      output_directory: str,
      file_shard_size_limit_bytes: int = constants.DEFAULT_FILE_SHARD_SIZE_LIMIT_BYTES,
  ):
    # Invariant throughout the lifetime of the object.
    self._agent_id: str = agent_id
    self._topics: set[str] = set(topics)
    self._recognized_topics: set[str] = self._topics.union(RESERVED_TOPICS)
    self._output_directory: str = output_directory
    self._file_shard_size_limit_bytes: int = file_shard_size_limit_bytes

    # Invariant until reset_for_new_file call.
    self._output_file_prefix: str = ''

    # Invariant during operation on a particular file shard.
    self._shard: int = 0
    self._file_handle = None
    self._mcap_writer: mcap_protobuf_writer.Writer = None

    # Variable with each write_message call.
    self._file_shard_bytes: int = 0

    # start timestamp of current file shard.
    self._start_nsec: int = sys.maxsize
    # stop timestamp of current file shard.
    self._stop_nsec: int = 0

    self._lock: threading.Lock = threading.Lock()

  def reset_for_new_file(
      self, output_file_prefix: str, start_nsec: int
  ) -> None:
    with self._lock:
      self._output_file_prefix = output_file_prefix
      self._reset_for_new_shard(is_first_shard=True)

      self._start_nsec = start_nsec
      self._stop_nsec = start_nsec

  def _reset_for_new_shard(self, is_first_shard: bool) -> None:
    """Reset the file handler states for a new shard.

    Args:
      is_first_shard: If True, the shard number is set to 0.
    """
    if is_first_shard:
      self._shard = 0
    else:
      self._shard += 1
    tmp_dir = f'{self._output_directory}/tmp'
    if not os.path.exists(tmp_dir):
      os.makedirs(tmp_dir)
    self._file_handle = open(
        pathlib.Path(tmp_dir)
        / f'{self._output_file_prefix}-shard{self._shard}.mcap',
        'wb',
    )
    self._mcap_writer = mcap_protobuf_writer.Writer(self._file_handle)
    self._file_shard_bytes = 0

  def write_message(
      self,
      message: message_lib.Message,
  ) -> None:
    """Write message with file rotation.

    Args:
      message: The Safari message object.
    """
    with self._lock:
      if message.topic not in self._recognized_topics:
        raise ValueError(
            'Unknown topic not present in during initialization: %s'
            % message.topic
        )
      self._topics.add(message.topic)
      msg_size = message.message.ByteSize()
      if self._file_shard_bytes > 0 and (
          self._file_shard_bytes + msg_size > self._file_shard_size_limit_bytes
      ):
        self._finalize_and_close_file()
        self._reset_for_new_shard(is_first_shard=False)
      self._start_nsec = min(self._start_nsec, message.publish_time_nsec)
      self._stop_nsec = max(self._stop_nsec, message.publish_time_nsec + 1)
      self._mcap_writer.write_message(
          topic=message.topic,
          message=message.message,
          log_time=message.log_time_nsec,
          publish_time=message.publish_time_nsec,
      )
      self._file_shard_bytes += msg_size

  def finalize_and_close_file(self, stop_nsec: int) -> None:
    """Finalize the file metadata and the mcap writer and close the file handle.

    Args:
      stop_nsec: The stop time of data coverage in the this new file shard.
    """
    with self._lock:
      self._start_nsec = min(self._start_nsec, stop_nsec)
      self._stop_nsec = max(self._stop_nsec, stop_nsec)
      self._finalize_and_close_file()

  def _finalize_and_close_file(self) -> None:
    """Private method to finalize and close the mcap writer and file handle."""
    file_metadata = metadata_pb2.FileMetadata(
        agent_id=self._agent_id,
    )
    if self._stop_nsec > self._start_nsec:
      # there's actually data in the file, so we need to add the stream
      # coverages.
      for topic in self._topics:
        file_metadata.stream_coverages.append(
            metadata_pb2.KeyRange(
                topic=topic,
                interval=label_pb2.IntervalValue(
                    start_nsec=self._start_nsec,
                    stop_nsec=self._stop_nsec,
                ),
            )
        )
      # the start_nsec of the next file shard should <= the stop_nsec of
      # the current file shard so the backend can observe continous data
      # coverage after received both shards.
      self._start_nsec = self._stop_nsec
    self._mcap_writer.write_message(
        topic=constants.FILE_METADATA_TOPIC_NAME,
        message=file_metadata,
        log_time=self._stop_nsec,
        publish_time=self._stop_nsec,
    )
    self._mcap_writer.finish()
    if self._file_handle:
      tmp_file_path = pathlib.Path(self._file_handle.name)
      self._file_handle.close()
      file_name = tmp_file_path.name
      date_now = datetime.datetime.now()
      final_dir = (
          pathlib.Path(self._output_directory)
          / date_now.strftime('%Y')
          / date_now.strftime('%m')
          / date_now.strftime('%d')
      )
      final_file_path = final_dir / file_name
      if not os.path.exists(final_dir):
        os.makedirs(final_dir)
      os.rename(tmp_file_path, final_file_path)
      current_permissions = os.stat(final_file_path).st_mode
      # Remove write permissions for all users (owner, group, others)
      os.chmod(
          final_file_path,
          current_permissions & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH,
      )
