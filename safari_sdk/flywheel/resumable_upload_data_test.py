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
from unittest import mock

import mcap.exceptions
import pytz
import requests

from absl.testing import absltest
from absl.testing import parameterized
from safari_sdk.flywheel import resumable_upload_data


class CheckSessionSizeTest(absltest.TestCase):

  def _make_fake_message(self, size_bytes):
    fake_message = mock.Mock()
    fake_message.data = b'x' * size_bytes
    return fake_message

  @mock.patch.object(resumable_upload_data.mcap_reader, 'make_reader')
  def test_raises_if_session_too_large(self, mock_make_reader):
    oversized = self._make_fake_message(
        resumable_upload_data._SESSION_SIZE_LIMIT_BYTES + 1
    )
    mock_make_reader.return_value.iter_messages.return_value = [
        (None, None, oversized)
    ]
    with self.assertRaisesRegex(ValueError, '/session message is'):
      resumable_upload_data._check_session_size(b'fake-mcap-bytes')

  @mock.patch.object(resumable_upload_data.mcap_reader, 'make_reader')
  def test_passes_if_session_small(self, mock_make_reader):
    small = self._make_fake_message(700)
    mock_make_reader.return_value.iter_messages.return_value = [
        (None, None, small)
    ]
    resumable_upload_data._check_session_size(b'fake-mcap-bytes')

  @mock.patch.object(resumable_upload_data.mcap_reader, 'make_reader')
  def test_check_session_size_invalid_mcap(self, mock_make_reader):
    mock_make_reader.side_effect = mcap.exceptions.McapError('Corrupt MCAP')

    with self.assertRaisesRegex(ValueError, 'File is not a valid MCAP'):
      resumable_upload_data._check_session_size(b'corrupt-mcap-bytes')


class UploadFileTest(absltest.TestCase):

  @mock.patch.object(resumable_upload_data.requests, 'post')
  def test_upload_file_resumable_success(self, mock_post):
    r_start = mock.Mock()
    r_start.status_code = 200
    r_start.reason = 'OK'
    r_start.headers = {
        'X-Goog-Upload-Status': 'active',
        'X-Goog-Upload-URL': 'https://upload.example.com/session/123',
    }

    r_upload = mock.Mock()
    r_upload.status_code = 200
    r_upload.reason = 'OK'
    r_upload.headers = {'X-Goog-Upload-Status': 'final'}

    mock_post.side_effect = [r_start, r_upload]

    api_endpoint = 'https://example.com/upload'
    agent_id = 'test_agent_001'
    filename = 'data.mcap'
    file_content_bytes = b'dummy file content'
    api_key = 'test_api_key_123'
    now = datetime.datetime(2023, 10, 26, 10, 0, 0, tzinfo=pytz.utc)  # pylint: disable=g-tzinfo-datetime

    status_code, reason = resumable_upload_data._upload_file(
        api_endpoint=api_endpoint,
        agent_id=agent_id,
        filename=filename,
        file_content_bytes=file_content_bytes,
        api_key=api_key,
        now=now,
    )

    self.assertEqual(status_code, 200)
    self.assertEqual(reason, 'OK')
    self.assertEqual(mock_post.call_count, 2)

    # Check start call
    start_call = mock_post.call_args_list[0]
    self.assertEqual(start_call[0][0], api_endpoint)
    self.assertEqual(
        start_call[1]['headers']['X-Goog-Upload-Protocol'], 'resumable'
    )
    self.assertEqual(start_call[1]['headers']['X-Goog-Upload-Command'], 'start')

    # Check upload call
    upload_call = mock_post.call_args_list[1]
    self.assertEqual(
        upload_call[0][0], 'https://upload.example.com/session/123'
    )
    self.assertEqual(
        upload_call[1]['headers']['X-Goog-Upload-Command'], 'upload, finalize'
    )
    self.assertEqual(upload_call[1]['headers']['X-Goog-Upload-Offset'], '0')

  @mock.patch.object(resumable_upload_data.requests, 'post')
  def test_upload_file_chunked_success(self, mock_post):
    r_start = mock.Mock()
    r_start.status_code = 200
    r_start.reason = 'OK'
    r_start.headers = {
        'X-Goog-Upload-Status': 'active',
        'X-Goog-Upload-URL': 'https://upload.example.com/session/123',
    }

    r_chunk1 = mock.Mock()
    r_chunk1.status_code = 200
    r_chunk1.reason = 'OK'
    r_chunk1.headers = {'X-Goog-Upload-Status': 'active'}

    r_chunk2 = mock.Mock()
    r_chunk2.status_code = 200
    r_chunk2.reason = 'OK'
    r_chunk2.headers = {'X-Goog-Upload-Status': 'final'}

    mock_post.side_effect = [r_start, r_chunk1, r_chunk2]

    now = datetime.datetime(2023, 10, 26, 10, 0, 0, tzinfo=pytz.utc)  # pylint: disable=g-tzinfo-datetime
    status_code, reason = resumable_upload_data._upload_file(
        api_endpoint='https://example.com/upload',
        agent_id='test_agent',
        filename='data.mcap',
        file_content_bytes=b'0123456789',
        api_key='key',
        now=now,
        chunk_size=5,
    )

    self.assertEqual(status_code, 200)
    self.assertEqual(reason, 'OK')
    self.assertEqual(mock_post.call_count, 3)

    # Check 1st chunk command is 'upload'
    chunk1_call = mock_post.call_args_list[1]
    self.assertEqual(
        chunk1_call[1]['headers']['X-Goog-Upload-Command'], 'upload'
    )
    self.assertEqual(chunk1_call[1]['headers']['X-Goog-Upload-Offset'], '0')

    # Check 2nd chunk command is 'upload, finalize'
    chunk2_call = mock_post.call_args_list[2]
    self.assertEqual(
        chunk2_call[1]['headers']['X-Goog-Upload-Command'], 'upload, finalize'
    )
    self.assertEqual(chunk2_call[1]['headers']['X-Goog-Upload-Offset'], '5')

  @mock.patch.object(resumable_upload_data.requests, 'post')
  def test_upload_file_fatal_4xx_error(self, mock_post):
    r_start = mock.Mock()
    r_start.status_code = 200
    r_start.reason = 'OK'
    r_start.headers = {
        'X-Goog-Upload-Status': 'active',
        'X-Goog-Upload-URL': 'https://upload.example.com/session/123',
    }

    r_upload = mock.Mock()
    r_upload.status_code = 400
    r_upload.reason = 'Bad Request'
    r_upload.headers = {'X-Goog-Upload-Status': 'active'}

    mock_post.side_effect = [r_start, r_upload]

    now = datetime.datetime(2023, 10, 26, 10, 0, 0, tzinfo=pytz.utc)  # pylint: disable=g-tzinfo-datetime
    status_code, reason = resumable_upload_data._upload_file(
        api_endpoint='https://example.com/upload',
        agent_id='test_agent',
        filename='data.mcap',
        file_content_bytes=b'data',
        api_key='key',
        now=now,
    )

    self.assertEqual(status_code, 400)
    self.assertEqual(reason, 'Bad Request')
    # Ensures query status was NOT called after 400 Bad Request
    self.assertEqual(mock_post.call_count, 2)

  @mock.patch.object(resumable_upload_data.time, 'sleep')
  @mock.patch.object(resumable_upload_data.requests, 'post')
  def test_upload_file_no_progress_max_retries(self, mock_post, mock_sleep):
    del mock_sleep
    r_start = mock.Mock()
    r_start.status_code = 200
    r_start.reason = 'OK'
    r_start.headers = {
        'X-Goog-Upload-Status': 'active',
        'X-Goog-Upload-URL': 'https://upload.example.com/session/123',
    }

    r_query = mock.Mock()
    r_query.status_code = 200
    r_query.reason = 'OK'
    r_query.headers = {
        'X-Goog-Upload-Status': 'active',
        'X-Goog-Upload-Size-Received': '0',
    }

    # Upload throws ConnectionError repeatedly, query returns 0 bytes received
    side_effects = [r_start]
    for _ in range(3):
      side_effects.append(requests.exceptions.ConnectionError('Error'))
      side_effects.append(r_query)

    mock_post.side_effect = side_effects

    now = datetime.datetime(2023, 10, 26, 10, 0, 0, tzinfo=pytz.utc)  # pylint: disable=g-tzinfo-datetime
    status_code, reason = resumable_upload_data._upload_file(
        api_endpoint='https://example.com/upload',
        agent_id='test_agent',
        filename='data.mcap',
        file_content_bytes=b'data',
        api_key='key',
        now=now,
        max_retries=3,
    )

    self.assertEqual(status_code, -1)
    self.assertIn('No upload progress after multiple retry attempts', reason)

  @mock.patch.object(resumable_upload_data.requests, 'post')
  def test_upload_file_missing_upload_url(self, mock_post):
    r_start = mock.Mock()
    r_start.status_code = 200
    r_start.reason = 'OK'
    r_start.headers = {'X-Goog-Upload-Status': 'active'}
    mock_post.return_value = r_start

    now = datetime.datetime(2023, 10, 26, 10, 0, 0, tzinfo=pytz.utc)  # pylint: disable=g-tzinfo-datetime
    status_code, reason = resumable_upload_data._upload_file(
        api_endpoint='https://example.com/upload',
        agent_id='test_agent',
        filename='data.mcap',
        file_content_bytes=b'data',
        api_key='key',
        now=now,
    )
    self.assertEqual(status_code, 200)
    self.assertIn('Missing X-Goog-Upload-URL header', reason)


class UploadDataDirectoryTest(parameterized.TestCase):

  @mock.patch.object(resumable_upload_data, '_check_session_size')
  @mock.patch.object(resumable_upload_data, '_upload_file')
  @mock.patch.object(resumable_upload_data.auth, 'get_api_key')
  def test_upload_data_directory_success_and_rename(
      self,
      mock_get_api_key,
      mock_upload_file,
      mock_check_session_size,
  ):
    del mock_check_session_size
    upload_data_dir = self.create_tempdir()
    upload_data_dir.create_file('data1.mcap', content='dummy file content 1')
    upload_data_dir.create_file('data2.mcap', content='dummy file content 2')

    upload_sub_dir = upload_data_dir.mkdir()
    upload_sub_dir.create_file('data3.mcap', content='dummy file content 3')

    mock_upload_file.return_value = (200, 'OK')
    mock_get_api_key.return_value = 'test_api_key_123'

    resumable_upload_data.upload_data_directory(
        api_endpoint='https://example.com/upload',
        data_directory=upload_data_dir.full_path,
        robot_id='test_agent_001',
    )

    self.assertEqual(mock_upload_file.call_count, 3)
    self.assertTrue(
        os.path.exists(
            os.path.join(upload_data_dir.full_path, 'data1.mcap.uploaded')
        )
    )
    self.assertFalse(
        os.path.exists(os.path.join(upload_data_dir.full_path, 'data1.mcap'))
    )

  @mock.patch.object(resumable_upload_data.auth, 'get_api_key')
  def test_upload_data_directory_no_api_key_raises_error(
      self, mock_get_api_key
  ):
    mock_get_api_key.return_value = None
    with self.assertRaises(ValueError):
      resumable_upload_data.upload_data_directory(
          api_endpoint='https://example.com/upload',
          data_directory='test_data_dir',
          robot_id='test_agent_001',
      )


class UploadSingleFilePublicTest(parameterized.TestCase):

  @mock.patch.object(resumable_upload_data, '_check_session_size')
  @mock.patch.object(resumable_upload_data, '_upload_file')
  @mock.patch.object(resumable_upload_data.auth, 'get_api_key')
  def test_upload_single_file_success(
      self,
      mock_get_api_key,
      mock_upload_file,
      mock_check_session_size,
  ):
    del mock_check_session_size
    mock_get_api_key.return_value = 'test_api_key_123'
    mock_upload_file.return_value = (200, 'OK')

    temp_dir = self.create_tempdir()
    file_path = temp_dir.create_file(
        'data.mcap', content='dummy content'
    ).full_path

    success, msg = resumable_upload_data.upload_single_file(
        api_endpoint='https://example.com/upload',
        file_path=file_path,
        robot_id='test_agent_001',
    )

    self.assertTrue(success)
    self.assertIn('Uploaded successfully', msg)
    self.assertTrue(os.path.exists(file_path + '.uploaded'))

  @mock.patch.object(resumable_upload_data.auth, 'get_api_key')
  def test_upload_single_file_not_mcap(self, mock_get_api_key):
    mock_get_api_key.return_value = 'test_api_key_123'
    with self.assertRaisesRegex(ValueError, 'File must be an MCAP file.'):
      resumable_upload_data.upload_single_file(
          api_endpoint='https://example.com/upload',
          file_path='some_file.txt',
          robot_id='test_agent_001',
      )


if __name__ == '__main__':
  absltest.main()
