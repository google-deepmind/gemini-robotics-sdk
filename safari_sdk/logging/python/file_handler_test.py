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

import datetime
import os
import pathlib
import stat
from absl.testing import absltest
from safari_sdk.logging.python import file_handler
from safari_sdk.logging.python import message as message_lib
from safari_sdk.protos import label_pb2


class FileHandlerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.output_dir = self.create_tempdir().full_path
    self.topics = {'/sensor/cam0', '/sensor/imu'}
    self.handler = file_handler.FileHandler(
        agent_id='test_agent',
        topics=self.topics,
        output_directory=self.output_dir,
        file_shard_size_limit_bytes=1000,
    )

  def test_reset_for_new_file_creates_tmp_file(self):
    self.handler.reset_for_new_file('test_session', start_nsec=1000)
    expected_tmp_file = (
        pathlib.Path(self.output_dir) / 'tmp' / 'test_session-shard0.mcap'
    )
    self.assertTrue(os.path.exists(expected_tmp_file))
    self.assertEqual(self.handler._start_nsec, 1000)
    self.assertEqual(self.handler._stop_nsec, 1000)

  def test_write_message_success(self):
    self.handler.reset_for_new_file('test_session', start_nsec=1000)
    fake_proto = label_pb2.LabelMessage(key='test_label')

    msg = message_lib.Message(
        topic='/sensor/cam0',
        message=fake_proto,
        log_time_nsec=1050,
        publish_time_nsec=1050,
    )
    self.handler.write_message(msg)

    self.assertEqual(self.handler._start_nsec, 1000)
    self.assertEqual(self.handler._stop_nsec, 1051)
    self.assertGreater(self.handler._file_shard_bytes, 0)

  def test_write_message_unknown_topic_raises_value_error(self):
    self.handler.reset_for_new_file('test_session', start_nsec=1000)
    fake_proto = label_pb2.LabelMessage()
    msg = message_lib.Message(
        topic='/unknown/topic',
        message=fake_proto,
        log_time_nsec=1000,
        publish_time_nsec=1000,
    )
    with self.assertRaisesRegex(ValueError, 'Unknown topic not present'):
      self.handler.write_message(msg)

  def test_automatic_shard_rotation(self):
    small_handler = file_handler.FileHandler(
        agent_id='test_agent',
        topics=self.topics,
        output_directory=self.output_dir,
        file_shard_size_limit_bytes=30,
    )
    small_handler.reset_for_new_file('rotate_session', start_nsec=1000)

    fake_proto1 = label_pb2.LabelMessage(key='x' * 25)
    msg1 = message_lib.Message(
        topic='/sensor/cam0',
        message=fake_proto1,
        log_time_nsec=1050,
        publish_time_nsec=1050,
    )
    small_handler.write_message(msg1)

    # Second message exceeds limit
    fake_proto2 = label_pb2.LabelMessage(key='y' * 25)
    msg2 = message_lib.Message(
        topic='/sensor/cam0',
        message=fake_proto2,
        log_time_nsec=1100,
        publish_time_nsec=1100,
    )
    small_handler.write_message(msg2)

    self.assertEqual(small_handler._shard, 1)
    expected_tmp_shard1 = (
        pathlib.Path(self.output_dir) / 'tmp' / 'rotate_session-shard1.mcap'
    )
    self.assertTrue(os.path.exists(expected_tmp_shard1))

  def test_finalize_and_close_file_moves_and_sets_permissions(self):
    self.handler.reset_for_new_file('test_session', start_nsec=1000)
    fake_proto = label_pb2.LabelMessage(key='test_label')
    msg = message_lib.Message(
        topic='/sensor/cam0',
        message=fake_proto,
        log_time_nsec=1050,
        publish_time_nsec=1050,
    )
    self.handler.write_message(msg)

    self.handler.finalize_and_close_file(stop_nsec=2000)

    date_now = datetime.datetime.now()
    expected_dir = (
        pathlib.Path(self.output_dir)
        / date_now.strftime('%Y')
        / date_now.strftime('%m')
        / date_now.strftime('%d')
    )
    expected_final_file = expected_dir / 'test_session-shard0.mcap'

    self.assertTrue(os.path.exists(expected_final_file))
    # Verify write bits are removed
    mode = os.stat(expected_final_file).st_mode
    self.assertFalse(mode & stat.S_IWUSR)
    self.assertFalse(mode & stat.S_IWGRP)
    self.assertFalse(mode & stat.S_IWOTH)

  def test_continuous_time_tracking(self):
    self.handler.reset_for_new_file('time_session', start_nsec=1000)
    fake_proto = label_pb2.LabelMessage(key='test_label')
    msg = message_lib.Message(
        topic='/sensor/cam0',
        message=fake_proto,
        log_time_nsec=1050,
        publish_time_nsec=1050,
    )
    self.handler.write_message(msg)

    self.handler.finalize_and_close_file(stop_nsec=2000)
    # Check that _start_nsec was updated to _stop_nsec (2000) for the next shard
    self.assertEqual(self.handler._start_nsec, 2000)


if __name__ == '__main__':
  absltest.main()
