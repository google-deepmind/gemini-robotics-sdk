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

"""Interface for stream loggers."""

import abc
from typing import Collection

from google.protobuf import struct_pb2
from google.protobuf import message as message_lib
from safari_sdk.protos import image_pb2
from safari_sdk.protos import joints_pb2
from safari_sdk.protos import label_pb2
from safari_sdk.protos import pose_pb2
from safari_sdk.protos import sensor_calibration_pb2
from safari_sdk.protos import transform_pb2
from safari_sdk.protos import vector_pb2
from safari_sdk.protos.logging import audio_pb2
from safari_sdk.protos.logging import contact_surface_pb2
from safari_sdk.protos.logging import imu_pb2
from safari_sdk.protos.logging import metadata_pb2
from safari_sdk.protos.logging import robot_base_pb2
from safari_sdk.protos.logging import tracker_pb2
from tensorflow.core.example import example_pb2

LOG_MESSAGE_TYPE = (
    # go/keep-sorted start
    audio_pb2.Audio
    | contact_surface_pb2.ContactSurface
    | example_pb2.Example
    | image_pb2.Image
    | imu_pb2.Imu
    | joints_pb2.Joints
    | joints_pb2.JointsTrajectory
    | metadata_pb2.FileMetadata
    | metadata_pb2.Session
    | metadata_pb2.TimeSynchronization
    | pose_pb2.Poses
    | robot_base_pb2.RobotBase
    | sensor_calibration_pb2.SensorCalibration
    | struct_pb2.Struct
    | struct_pb2.Value
    | tracker_pb2.Trackers
    | transform_pb2.Transforms
    | vector_pb2.NamedVectorDouble
    | vector_pb2.NamedVectorInt64
    # go/keep-sorted end
)


class StreamLoggerInterface(abc.ABC):
  """Interface for StreamLogger."""

  def __init__(
      self,
      agent_id: str,
      output_directory: str,
      required_topics: Collection[str],
      optional_topics: Collection[str] | None = None,
      file_shard_size_limit_bytes: int = 2 * 1024 * 1024 * 1024,
  ):
    """Initializes the stream logger."""

  @abc.abstractmethod
  def is_session_started(self) -> bool:
    """Returns whether a session has been started."""

  @abc.abstractmethod
  def is_logging_outside_session(self) -> bool:
    """Returns whether logging outside of sessions is started."""

  @abc.abstractmethod
  def is_recording(self) -> bool:
    """Returns whether the stream logger is recording."""

  @abc.abstractmethod
  def write_sync_message(self, publish_time_nsec: int) -> None:
    """Writes the sync message.

    This must not be called unless we are recording (start_session or
    start_outside_session_logging has been called).

    This must not be called until we have seen at least one message on each
    topic.

    Args:
      publish_time_nsec: The publish time of the sync message.
    """

  @abc.abstractmethod
  def get_latest_sync_message(self) -> metadata_pb2.TimeSynchronization:
    """Returns the latest sync message."""

  @abc.abstractmethod
  def has_received_all_required_topics(self) -> bool:
    """True if we have seen at least one message on each rwquired topic."""

  @abc.abstractmethod
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

  @abc.abstractmethod
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

  @abc.abstractmethod
  def add_session_label(self, label: label_pb2.LabelMessage) -> None:
    """Adds a session label."""

  @abc.abstractmethod
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

  @abc.abstractmethod
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

  @abc.abstractmethod
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

  @abc.abstractmethod
  def stop_outside_session_logging_and_finalize_file(
      self, stop_nsec: int
  ) -> None:
    """Stops logging outside of sessions and finalizes the log file.

    Args:
      stop_nsec: The time when logging is stopped.

    Raises:
      ValueError: If a session has been started.
    """

  @abc.abstractmethod
  def update_synchronization_and_maybe_write_message(
      self,
      topic: str,
      message: LOG_MESSAGE_TYPE,
      publish_time_nsec: int,
      log_time_nsec: int = 0,
  ) -> None:
    """Updates the synchronization message and maybe writes the message.

    This method is called within callback functions, maybe multi-threaded.

    Args:
      topic: The safari_logging_topic of the message.
      message: The proto message to be written.
      publish_time_nsec: The timestamp of the message (this may be the time the
        message was published, or the time the data in the  message was
        sampled).
      log_time_nsec: The time when the logger received the message. If 0, will
        be set to the system's current time.
    """
