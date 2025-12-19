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

"""SessionManager class for managing the lifecycle of a session."""

from typing import Set
from safari_sdk.logging.python import metadata_utils
from safari_sdk.protos import label_pb2
from safari_sdk.protos.logging import metadata_pb2


class SessionManager:
  """Class for managing the lifecycle of a session."""

  def __init__(
      self,
      topics: Set[str],
      required_topics: Set[str],
      policy_environment_metadata_params: metadata_utils.PolicyEnvironmentMetadataParams,
  ):
    """Initializes the SessionManager.

    Args:
      topics: The topics to be logged.
      required_topics: The topics that are required to be logged.
      policy_environment_metadata_params: parameters for setting the policy
        environment metadata. Note that the dictionaries passed must be
        flattened.
    """

    self._session_started: bool = False
    self._session: metadata_pb2.Session | None = None

    self._topics: Set[str] = topics
    self._required_topics: Set[str] = required_topics

    self._policy_environment_metadata_params = (
        policy_environment_metadata_params
    )

    self._validate_topics()

  def _validate_topics(self) -> None:
    """Validates that required topics are a subset of all topics.

    Raises:
        ValueError: If required_topics is not a subset of topics or if a
        reserved topic is present.
    """
    # Checks that required topics are a subset of all topics.
    if not self._required_topics.issubset(self._topics):
      missing_topics = self._required_topics - self._topics
      raise ValueError(
          'required_topics must be a subset of topics. '
          f'Missing topics: {missing_topics}'
      )

  @property
  def session_started(self) -> bool:
    return self._session_started

  def _set_policy_environment_metadata(self) -> None:
    """Sets the policy environment metadata in the session.

    Raises:
      ValueError: If the session has not been started.
    """
    if self._session is None:
      raise ValueError(
          'Session is None. Cannot set policy environment metadata.'
      )
    feature_specs = metadata_utils.create_feature_specs_proto(
        self._policy_environment_metadata_params
    )
    self._session.policy_environment_metadata.feature_specs.CopyFrom(
        feature_specs
    )

  def start_session(self, *, start_timestamp_nsec: int, task_id: str) -> None:
    """Starts a new session for logging.

    Args:
      start_timestamp_nsec: The start timestamp of the session.
      task_id: The task ID of the session.
    """
    if self._session_started:
      raise ValueError('Session has already been started.')

    self._session = metadata_pb2.Session(
        interval=label_pb2.IntervalValue(
            start_nsec=start_timestamp_nsec,
        ),
        task_id=task_id,
    )
    # Set the policy environment metadata once the session is created.
    self._set_policy_environment_metadata()

    for topic in self._topics:
      self._session.streams.append(
          metadata_pb2.Session.StreamMetadata(
              key_range=metadata_pb2.KeyRange(
                  topic=topic,
                  interval=label_pb2.IntervalValue(
                      start_nsec=start_timestamp_nsec,
                  ),
              ),
              is_required=topic in self._required_topics,
          )
      )
    self._session_started = True

  def add_session_label(self, label: label_pb2.LabelMessage) -> None:
    """Adds a label to the session.

    Args:
      label: The label to be added to the session. This will be added to the
        session aspects.

    Raises:
      ValueError: If the session has not been started.
    """
    if not self._session_started or self._session is None:
      raise ValueError(
          'add_session_label is called before session has been started.'
      )
    self._session.labels.append(label)

  def stop_session(self, stop_timestamp_nsec: int) -> metadata_pb2.Session:
    """Stops the current session, updates the metadata and writes to file.

    Args:
      stop_timestamp_nsec: The stop timestamp of the session.

    Returns:
      The Session object.

    Raises:
      ValueError: If the session has not been started.
    """
    if not self._session_started or self._session is None:
      raise ValueError('Session is not started.')

    self._session.interval.stop_nsec = stop_timestamp_nsec

    for stream in self._session.streams:
      stream.key_range.interval.stop_nsec = stop_timestamp_nsec

    self._session_started = False

    return self._session
