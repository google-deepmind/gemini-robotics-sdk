# Copyright 2026 Google LLC
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

"""Tests for stream_logger session metadata behavior."""

import time

from absl.testing import absltest
from absl.testing import parameterized
from safari_sdk.logging.python import constants
from safari_sdk.logging.python import mcap_parser_utils
from safari_sdk.logging.python import session_metadata as session_metadata_lib
from safari_sdk.logging.python import stream_logger as stream_logger_lib
from safari_sdk.protos.logging import metadata_pb2
from safari_sdk.protos.logging import orchestrator_info_pb2
from safari_sdk.protos.logging import policy_type_pb2

_TEST_AGENT_ID = "test_agent"
_TEST_TASK_ID = "test_task"
_TEST_TOPIC = "test_topic"


def _make_logger(
    output_directory: str,
    session_metadata_config: (
        session_metadata_lib.SessionMetadataConfig | None
    ) = None,
) -> stream_logger_lib.StreamLogger:
  return stream_logger_lib.StreamLogger(
      agent_id=_TEST_AGENT_ID,
      output_directory=output_directory,
      required_topics=[_TEST_TOPIC],
      session_metadata_config=session_metadata_config,
  )


def _start_session(logger: stream_logger_lib.StreamLogger) -> None:
  sync_msg = metadata_pb2.TimeSynchronization()
  logger.update_synchronization_and_maybe_write_message(
      topic=_TEST_TOPIC,
      message=sync_msg,
      publish_time_nsec=int(time.time() * 1e9),
  )
  logger.start_session(
      start_nsec=int(time.time() * 1e9),
      task_id=_TEST_TASK_ID,
  )


def _stop_and_read_session(logger, output_directory):
  logger.stop_session(stop_nsec=int(time.time() * 1e9))
  sessions = mcap_parser_utils.read_session_proto_data(
      output_directory, constants.SESSION_TOPIC_NAME
  )
  return sessions[0]


class StreamLoggerMetadataTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._output_dir = self.create_tempdir().full_path

  def test_no_metadata_config_starts_session_successfully(self):
    logger = _make_logger(self._output_dir)
    _start_session(logger)
    session = _stop_and_read_session(logger, self._output_dir)
    self.assertEqual(
        session.policy_environment_metadata.policy_type,
        policy_type_pb2.PolicyType.POLICY_TYPE_UNSPECIFIED,
    )

  def test_policy_type_is_written_to_session(self):
    config = session_metadata_lib.SessionMetadataConfig(
        policy_type=policy_type_pb2.PolicyType.POLICY_TYPE_ROBOT_EVALUATION,
    )
    logger = _make_logger(self._output_dir, session_metadata_config=config)
    _start_session(logger)
    session = _stop_and_read_session(logger, self._output_dir)
    self.assertEqual(
        session.policy_environment_metadata.policy_type,
        policy_type_pb2.PolicyType.POLICY_TYPE_ROBOT_EVALUATION,
    )

  def test_embodiment_version_is_written_to_session(self):
    config = session_metadata_lib.SessionMetadataConfig(
        embodiment_version="apollo_v2",
    )
    logger = _make_logger(self._output_dir, session_metadata_config=config)
    _start_session(logger)
    session = _stop_and_read_session(logger, self._output_dir)
    self.assertEqual(
        session.policy_environment_metadata.embodiment_version, "apollo_v2"
    )

  def test_control_timestep_is_written_to_session(self):
    config = session_metadata_lib.SessionMetadataConfig(
        control_timestep_seconds=0.05,
    )
    logger = _make_logger(self._output_dir, session_metadata_config=config)
    _start_session(logger)
    session = _stop_and_read_session(logger, self._output_dir)
    self.assertAlmostEqual(
        session.policy_environment_metadata.control_timestep, 0.05
    )

  def test_fixed_tags_are_written_to_session(self):
    config = session_metadata_lib.SessionMetadataConfig(
        fixed_tags=["tag_a", "tag_b"],
    )
    logger = _make_logger(self._output_dir, session_metadata_config=config)
    _start_session(logger)
    session = _stop_and_read_session(logger, self._output_dir)
    self.assertSequenceEqual(session.tags, ["tag_a", "tag_b"])

  def test_dynamic_tags_are_written_to_session(self):
    config = session_metadata_lib.SessionMetadataConfig(
        dynamic_episode_taggers=[
            lambda: ["dynamic_tag_1", "dynamic_tag_2"],
        ],
    )
    logger = _make_logger(self._output_dir, session_metadata_config=config)
    _start_session(logger)
    session = _stop_and_read_session(logger, self._output_dir)
    self.assertSequenceEqual(session.tags, ["dynamic_tag_1", "dynamic_tag_2"])

  @parameterized.named_parameters(
      dict(testcase_name="success", is_success=True),
      dict(testcase_name="failure", is_success=False),
  )
  def test_is_success_provider_writes_label(self, is_success):
    config = session_metadata_lib.SessionMetadataConfig(
        is_success_provider=lambda: is_success,
    )
    logger = _make_logger(self._output_dir, session_metadata_config=config)
    _start_session(logger)
    session = _stop_and_read_session(logger, self._output_dir)
    success_labels = [l for l in session.labels if l.key == "success"]
    self.assertLen(success_labels, 1)
    self.assertEqual(success_labels[0].label_value.bool_value, is_success)

  def test_dynamic_metadata_provider_writes_labels(self):
    config = session_metadata_lib.SessionMetadataConfig(
        dynamic_metadata_provider=lambda: {"key1": "val1", "key2": "val2"},
    )
    logger = _make_logger(self._output_dir, session_metadata_config=config)
    _start_session(logger)
    session = _stop_and_read_session(logger, self._output_dir)
    labels_by_key = {l.key: l.label_value.string_value for l in session.labels}
    self.assertEqual(labels_by_key["key1"], "val1")
    self.assertEqual(labels_by_key["key2"], "val2")

  def test_orchestrator_info_is_written_to_session(self):
    expected_info = orchestrator_info_pb2.OrchestratorInfo(
        task_id="orca_task_123",
        robot_job_id="job_456",
    )
    config = session_metadata_lib.SessionMetadataConfig(
        orchestrator_info_provider=lambda: expected_info,
    )
    logger = _make_logger(self._output_dir, session_metadata_config=config)
    _start_session(logger)
    session = _stop_and_read_session(logger, self._output_dir)
    self.assertEqual(session.orchestrator_info.task_id, "orca_task_123")
    self.assertEqual(session.orchestrator_info.robot_job_id, "job_456")


if __name__ == "__main__":
  absltest.main()
