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
from safari_sdk.protos import image_pb2
from safari_sdk.protos import joints_pb2
from safari_sdk.protos import pose_pb2
from safari_sdk.protos import sensor_calibration_pb2
from safari_sdk.protos import transform_pb2
from safari_sdk.protos import vector_pb2
from safari_sdk.protos.logging import contact_surface_pb2
from safari_sdk.protos.logging import metadata_pb2
from safari_sdk.protos.logging import robot_base_pb2
from safari_sdk.protos.logging import tracker_pb2
from tensorflow.core.example import example_pb2


_LOG_MESSAGE_TYPE = (
    contact_surface_pb2.ContactSurface
    | image_pb2.Image
    | joints_pb2.Joints
    | joints_pb2.JointsTrajectory
    | metadata_pb2.Session
    | metadata_pb2.FileMetadata
    | metadata_pb2.TimeSynchronization
    | pose_pb2.Poses
    | robot_base_pb2.RobotBase
    | sensor_calibration_pb2.SensorCalibration
    | tracker_pb2.Trackers
    | transform_pb2.Transforms
    | vector_pb2.NamedVectorDouble
    | vector_pb2.NamedVectorInt64
    | example_pb2.Example
)


class StreamLogger(base_logger.BaseLogger):
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
    """True if we have seen at least one message on each rwquired topic."""
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
    """Writes the sync message.

    This must not be called unless we are recording (start_session or
    start_outside_session_logging has been called).

    This must not be called until we have seen at least one message on each
    topic.

    Args:
      publish_time_nsec: The publish time of the sync message.
    """
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
      message: _LOG_MESSAGE_TYPE,
      publish_time_nsec: int,
      log_time_nsec: int = 0,
  ) -> None:
    """Updates the synchronization message and maybe writes the message.

    Args:
      topic: The safari_logging_topic of the message.
      message: The proto message to be written.
      publish_time_nsec: The timestamp of the message (this may be the time the
        message was published, or the time the data in the  message was
        sampled).
      log_time_nsec: The time when the logger received the message. If 0, will
        be set to the system's current time.
    """
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
