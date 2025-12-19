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

from unittest import mock
from google.protobuf import struct_pb2
from absl.testing import absltest
from safari_sdk.logging.python import file_handler as file_handler_lib
from safari_sdk.logging.python import mcap_message_writer
from safari_sdk.logging.python import message as message_lib

_TEST_MESSAGE = "test_message"
_TEST_FILE_PREFIX = "test_file"
_TEST_START_NSEC = 1234567890
_TEST_STOP_NSEC = 1234567891
_TEST_TOPIC = "test_topic"


class McapMessageWriterTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._file_handler = mock.MagicMock(spec=file_handler_lib.FileHandler)
    self._message_writer = mcap_message_writer.McapMessageWriter(
        file_handler=self._file_handler,
    )

  def test_reset_file_handler_calls_underlying_file_handler(self):
    self._message_writer.reset_file_handler(
        output_file_prefix=_TEST_FILE_PREFIX,
        start_timestamp_nsec=_TEST_START_NSEC,
    )
    self._file_handler.reset_for_new_file.assert_called_once_with(
        output_file_prefix=_TEST_FILE_PREFIX, start_nsec=_TEST_START_NSEC
    )

  def test_start_worker_thread_if_not_already_running(self):
    message_writer = mcap_message_writer.McapMessageWriter(
        file_handler=self._file_handler,
    )
    message_writer.start()
    self.assertIsNotNone(message_writer._log_writer_thread)
    self.assertTrue(message_writer._log_writer_thread.is_alive())
    message_writer.stop(stop_timestamp_nsec=_TEST_STOP_NSEC)

  def test_stop_waits_for_thread_to_finish(self):
    self._message_writer.reset_file_handler(
        output_file_prefix=_TEST_FILE_PREFIX,
        start_timestamp_nsec=_TEST_START_NSEC,
    )
    self._message_writer.start()
    self._message_writer.stop(stop_timestamp_nsec=_TEST_STOP_NSEC)

    self._file_handler.finalize_and_close_file.assert_called_once_with(
        stop_nsec=_TEST_STOP_NSEC
    )
    self.assertIsNone(self._message_writer._log_writer_thread)

  def test_stop_raises_error_if_thread_fails_with_exception(self):
    message_writer = mcap_message_writer.McapMessageWriter(
        file_handler=self._file_handler,
    )
    message_writer.reset_file_handler(
        output_file_prefix=_TEST_FILE_PREFIX,
        start_timestamp_nsec=_TEST_START_NSEC,
    )
    message_writer.start()
    message_writer._worker_thread_exception = ValueError("test_error")
    with self.assertRaisesRegex(
        RuntimeError, "Log writer thread failed with an exception"
    ):
      message_writer.stop(stop_timestamp_nsec=_TEST_STOP_NSEC)

  def test_write_proto_message_raises_error_if_worker_thread_fails(self):
    message_writer = mcap_message_writer.McapMessageWriter(
        file_handler=self._file_handler,
    )
    message_writer.reset_file_handler(
        output_file_prefix=_TEST_FILE_PREFIX,
        start_timestamp_nsec=_TEST_START_NSEC,
    )

    # Simulate a failure in the worker thread.
    message_writer._worker_thread_exception = ValueError("test_error")

    with self.assertRaisesRegex(RuntimeError, "Log writer thread has failed"):
      message_writer.write_proto_message(
          topic=_TEST_TOPIC,
          message=struct_pb2.Value(string_value=_TEST_MESSAGE),
          publish_time_nsec=_TEST_START_NSEC,
          log_time_nsec=_TEST_START_NSEC,
      )

  def test_write_proto_message_writes_message_to_queue(self):
    message = struct_pb2.Value(string_value=_TEST_MESSAGE)
    self._message_writer.reset_file_handler(
        output_file_prefix=_TEST_FILE_PREFIX,
        start_timestamp_nsec=_TEST_START_NSEC,
    )
    self._message_writer.start()
    self._message_writer.write_proto_message(
        topic=_TEST_TOPIC,
        message=message,
        publish_time_nsec=_TEST_START_NSEC,
        log_time_nsec=_TEST_START_NSEC,
    )
    self._message_writer.stop(stop_timestamp_nsec=_TEST_STOP_NSEC)

    self._file_handler.write_message.assert_called_once_with(
        message=message_lib.Message(
            topic=_TEST_TOPIC,
            message=message,
            publish_time_nsec=_TEST_START_NSEC,
            log_time_nsec=_TEST_START_NSEC,
        )
    )

  def test_write_multiple_messages_to_queue(self):
    num_messages = 10
    messages = [
        struct_pb2.Value(string_value=f"{_TEST_MESSAGE}_{i}")
        for i in range(num_messages)
    ]

    self._message_writer.reset_file_handler(
        output_file_prefix=_TEST_FILE_PREFIX,
        start_timestamp_nsec=_TEST_START_NSEC,
    )
    self._message_writer.start()

    for message in messages:
      self._message_writer.write_proto_message(
          topic=_TEST_TOPIC,
          message=message,
          publish_time_nsec=_TEST_START_NSEC,
          log_time_nsec=_TEST_START_NSEC,
      )

    self._message_writer.stop(stop_timestamp_nsec=_TEST_STOP_NSEC)

    self._file_handler.write_message.assert_has_calls([
        mock.call(
            message=message_lib.Message(
                topic=_TEST_TOPIC,
                message=message,
                publish_time_nsec=_TEST_START_NSEC,
                log_time_nsec=_TEST_START_NSEC,
            )
        )
        for message in messages
    ])

    self._file_handler.finalize_and_close_file.assert_called_once_with(
        stop_nsec=_TEST_STOP_NSEC
    )
    # Assert that the queue is empty after the messages are written.
    self.assertTrue(self._message_writer._message_queue.empty())


if __name__ == "__main__":
  absltest.main()
