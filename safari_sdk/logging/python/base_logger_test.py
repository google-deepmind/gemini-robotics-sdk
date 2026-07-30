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

import time
from unittest import mock

from absl.testing import absltest
from safari_sdk.logging.python import base_logger
from safari_sdk.logging.python import constants
from safari_sdk.protos import label_pb2


class BaseLoggerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.output_dir = self.create_tempdir().full_path
    self.agent_id = 'test_agent'
    self.required_topics = {'/sensor/cam0'}
    self.optional_topics = {'/sensor/imu'}

  def test_constructor_duplicate_topics_raises_value_error(self):
    with self.assertRaisesRegex(ValueError, 'must not have common elements'):
      base_logger.BaseLogger(
          agent_id=self.agent_id,
          output_directory=self.output_dir,
          required_topics={'/sensor/cam0'},
          optional_topics={'/sensor/cam0'},
      )

  def test_constructor_reserved_topic_raises_value_error(self):
    with self.assertRaisesRegex(ValueError, 'reserved'):
      base_logger.BaseLogger(
          agent_id=self.agent_id,
          output_directory=self.output_dir,
          required_topics={constants.SESSION_TOPIC_NAME},
      )

  def test_state_machine_session_lifecycle(self):
    logger = base_logger.BaseLogger(
        agent_id=self.agent_id,
        output_directory=self.output_dir,
        required_topics=self.required_topics,
        optional_topics=self.optional_topics,
    )
    self.assertFalse(logger.is_session_started())
    self.assertFalse(logger.is_recording())

    # Start session
    logger.start_session(
        start_nsec=1000, task_id='task_123', output_file_prefix='test_pref'
    )
    self.assertTrue(logger.is_session_started())
    self.assertTrue(logger.is_recording())

    # Double start raises
    with self.assertRaisesRegex(ValueError, 'already been started'):
      logger.start_session(start_nsec=1100, task_id='task_123')

    # Stop recording without saving session
    logger.stop_recording_without_saving_session(stop_nsec=1500)
    self.assertTrue(logger.is_session_started())
    self.assertFalse(logger.is_recording())

    # Stop session
    logger.stop_session(stop_nsec=2000)
    self.assertFalse(logger.is_session_started())
    self.assertFalse(logger.is_recording())

    # Stop session again raises
    with self.assertRaisesRegex(ValueError, 'Session is not started'):
      logger.stop_session(stop_nsec=2100)

  def test_outside_session_logging_lifecycle(self):
    logger = base_logger.BaseLogger(
        agent_id=self.agent_id,
        output_directory=self.output_dir,
        required_topics=self.required_topics,
    )
    self.assertFalse(logger.is_logging_outside_session())

    logger.start_outside_session_logging(
        start_nsec=1000, output_file_prefix='outside_pref'
    )
    self.assertTrue(logger.is_logging_outside_session())
    self.assertTrue(logger.is_recording())

    # Cannot start session while outside logging is active
    with self.assertRaisesRegex(
        ValueError, 'Cannot start a session when outside session logging'
    ):
      logger.start_session(start_nsec=1100, task_id='task_123')

    logger.stop_outside_session_logging_and_finalize_file(stop_nsec=2000)
    self.assertFalse(logger.is_logging_outside_session())
    self.assertFalse(logger.is_recording())

  def test_write_proto_message_queue_and_worker(self):
    logger = base_logger.BaseLogger(
        agent_id=self.agent_id,
        output_directory=self.output_dir,
        required_topics=self.required_topics,
    )
    fake_msg = label_pb2.LabelMessage(key='test')

    # Outside recording, message is ignored
    logger.write_proto_message(
        topic='/sensor/cam0', message=fake_msg, publish_time_nsec=1000
    )
    self.assertTrue(logger._message_queue.empty())

    logger.start_session(start_nsec=1000, task_id='task_1')
    logger.write_proto_message(
        topic='/sensor/cam0', message=fake_msg, publish_time_nsec=1050
    )
    logger.stop_session(stop_nsec=2000)
    self.assertTrue(logger._message_queue.empty())

  def test_add_session_label(self):
    logger = base_logger.BaseLogger(
        agent_id=self.agent_id,
        output_directory=self.output_dir,
        required_topics=self.required_topics,
    )
    with self.assertRaisesRegex(ValueError, 'add_session_label is called'):
      logger.add_session_label(label_pb2.LabelMessage())

    logger.start_session(start_nsec=1000, task_id='task_1')
    logger.add_session_label(label_pb2.LabelMessage())
    self.assertLen(logger._session.labels, 1)
    logger.stop_session(stop_nsec=2000)

  def test_worker_exception_propagation(self):
    logger = base_logger.BaseLogger(
        agent_id=self.agent_id,
        output_directory=self.output_dir,
        required_topics=self.required_topics,
    )
    logger.start_session(start_nsec=1000, task_id='task_1')

    # Force file_handler to raise Exception
    logger._file_handler.write_message = mock.Mock(
        side_effect=ValueError('Write failure')
    )

    fake_msg = label_pb2.LabelMessage(key='test')
    logger.write_proto_message(
        topic='/sensor/cam0', message=fake_msg, publish_time_nsec=1050
    )

    # Give worker thread time to process message and catch exception
    time.sleep(0.1)

    # Subsequent write_proto_message should raise RuntimeError
    with self.assertRaisesRegex(
        RuntimeError, 'Log writer thread has failed'
    ):
      logger.write_proto_message(
          topic='/sensor/cam0', message=fake_msg, publish_time_nsec=1100
      )

    # stop_session should also propagate exception
    with self.assertRaisesRegex(RuntimeError, 'Log writer thread failed'):
      logger.stop_session(stop_nsec=2000)

  def test_worker_thread_timeout_guard(self):
    logger = base_logger.BaseLogger(
        agent_id=self.agent_id,
        output_directory=self.output_dir,
        required_topics=self.required_topics,
    )
    logger.start_session(start_nsec=1000, task_id='task_1')

    # Stop and join the real background thread first so it isn't orphaned
    real_thread = logger._log_writer_thread
    logger._message_queue.put(base_logger._SENTINEL)
    real_thread.join()

    # Mock the thread join to simulate a hung thread that stays alive
    mock_thread = mock.Mock()
    mock_thread.is_alive.return_value = True
    logger._log_writer_thread = mock_thread

    with self.assertRaisesRegex(
        RuntimeError, 'Failed to stop log writer thread within 2 minutes'
    ):
      logger.stop_session(stop_nsec=2000)


if __name__ == '__main__':
  absltest.main()
