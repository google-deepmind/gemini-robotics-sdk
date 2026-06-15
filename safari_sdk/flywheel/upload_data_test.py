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

from absl.testing import absltest
from absl.testing import parameterized
from safari_sdk.flywheel import upload_data


class CheckSessionSizeTest(absltest.TestCase):

  def _make_fake_message(self, size_bytes):
    fake_message = mock.Mock()
    fake_message.data = b'x' * size_bytes
    return fake_message

  @mock.patch.object(upload_data.mcap_reader, 'make_reader')
  def test_raises_if_session_too_large(self, mock_make_reader):
    oversized = self._make_fake_message(
        upload_data._SESSION_SIZE_LIMIT_BYTES + 1
    )
    mock_make_reader.return_value.iter_messages.return_value = [
        (None, None, oversized)
    ]
    with self.assertRaisesRegex(ValueError, '/session message is'):
      upload_data._check_session_size(b'fake-mcap-bytes')

  @mock.patch.object(upload_data.mcap_reader, 'make_reader')
  def test_passes_if_session_small(self, mock_make_reader):
    small = self._make_fake_message(700)
    mock_make_reader.return_value.iter_messages.return_value = [
        (None, None, small)
    ]
    upload_data._check_session_size(b'fake-mcap-bytes')

  @mock.patch.object(upload_data.mcap_reader, 'make_reader')
  def test_check_session_size_invalid_mcap(self, mock_make_reader):
    mock_make_reader.side_effect = mcap.exceptions.McapError('Corrupt MCAP')

    with self.assertRaisesRegex(ValueError, 'File is not a valid MCAP'):
      upload_data._check_session_size(b'corrupt-mcap-bytes')


class UploadFileTest(absltest.TestCase):

  @mock.patch.object(upload_data.requests, 'post')
  def test_upload_file_calls_requests_post_correctly(self, mock_post):
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.reason = 'OK'
    mock_post.return_value = mock_response

    api_endpoint = 'https://example.com/upload'
    agent_id = 'test_agent_001'
    filename = 'data.mcap'
    file_content_bytes = b'dummy file content'
    api_key = 'test_api_key_123'
    # Provide a timezone-aware datetime object for 'now'
    now = datetime.datetime(2023, 10, 26, 10, 0, 0, tzinfo=pytz.utc)  # pylint: disable=g-tzinfo-datetime

    status_code, reason = upload_data._upload_file(
        api_endpoint=api_endpoint,
        agent_id=agent_id,
        filename=filename,
        file_content_bytes=file_content_bytes,
        api_key=api_key,
        now=now,
    )

    self.assertEqual(status_code, 200)
    self.assertEqual(reason, 'OK')

    mock_post.assert_called_once()


class UploadDataDirectoryTest(parameterized.TestCase):

  @mock.patch.object(upload_data, '_check_session_size')
  @mock.patch.object(upload_data, '_upload_file')
  @mock.patch.object(upload_data.auth, 'get_api_key')
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

    upload_data.upload_data_directory(
        api_endpoint='https://example.com/upload',
        data_directory=upload_data_dir.full_path,
        robot_id='test_agent_001',
    )
    # check calls of upload_file,
    self.assertEqual(mock_upload_file.call_count, 3)
    # check calls of upload_file one by one
    mock_upload_file.assert_has_calls(
        any_order=True,
        calls=[
            mock.call(
                api_endpoint='https://example.com/upload',
                agent_id='test_agent_001',
                filename='data1.mcap',
                file_content_bytes=b'dummy file content 1',
                api_key='test_api_key_123',
                now=mock.ANY,
            ),
            mock.call(
                api_endpoint='https://example.com/upload',
                agent_id='test_agent_001',
                filename='data2.mcap',
                file_content_bytes=b'dummy file content 2',
                api_key='test_api_key_123',
                now=mock.ANY,
            ),
            mock.call(
                api_endpoint='https://example.com/upload',
                agent_id='test_agent_001',
                filename='data3.mcap',
                file_content_bytes=b'dummy file content 3',
                api_key='test_api_key_123',
                now=mock.ANY,
            ),
        ],
    )

    # check file name changed
    self.assertTrue(
        os.path.exists(
            os.path.join(upload_data_dir.full_path, 'data1.mcap.uploaded')
        )
    )
    self.assertFalse(
        os.path.exists(os.path.join(upload_data_dir.full_path, 'data1.mcap'))
    )
    self.assertTrue(
        os.path.exists(
            os.path.join(upload_data_dir.full_path, 'data2.mcap.uploaded')
        )
    )
    self.assertFalse(
        os.path.exists(os.path.join(upload_data_dir.full_path, 'data2.mcap'))
    )
    self.assertTrue(
        os.path.exists(
            os.path.join(upload_sub_dir.full_path, 'data3.mcap.uploaded')
        )
    )
    self.assertFalse(
        os.path.exists(os.path.join(upload_sub_dir.full_path, 'data3.mcap'))
    )

  @mock.patch.object(upload_data.auth, 'get_api_key')
  def test_upload_data_directory_no_api_key_raises_error(
      self, mock_get_api_key
  ):
    mock_get_api_key.return_value = None
    with self.assertRaises(ValueError):
      upload_data.upload_data_directory(
          api_endpoint='https://example.com/upload',
          data_directory='test_data_dir',
          robot_id='test_agent_001',
      )

  @mock.patch.object(upload_data.auth, 'get_api_key')
  def test_upload_data_directory_already_uploaded_prints_message(
      self, mock_get_api_key
  ):
    mock_get_api_key.return_value = 'test_api_key_123'
    upload_data_dir = self.create_tempdir()
    upload_data_dir.create_file(
        'data1.mcap.uploaded', content='dummy file content 1'
    )
    upload_data_dir.create_file(
        'data2.mcap.uploaded', content='dummy file content 2'
    )

    with mock.patch('builtins.print') as mock_print:
      upload_data.upload_data_directory(
          api_endpoint='https://example.com/upload',
          data_directory=upload_data_dir.full_path,
          robot_id='test_agent_001',
      )
      mock_print.assert_called_once()
      printed_msg = mock_print.call_args[0][0]
      self.assertIn('No new .mcap files found', printed_msg)
      self.assertIn('2 file(s) were already uploaded', printed_msg)

  @mock.patch.object(upload_data.auth, 'get_api_key')
  def test_upload_data_directory_empty_prints_message(self, mock_get_api_key):
    mock_get_api_key.return_value = 'test_api_key_123'
    upload_data_dir = self.create_tempdir()

    with mock.patch('builtins.print') as mock_print:
      upload_data.upload_data_directory(
          api_endpoint='https://example.com/upload',
          data_directory=upload_data_dir.full_path,
          robot_id='test_agent_001',
      )
      mock_print.assert_called_once_with(
          f'No .mcap files found in {upload_data_dir.full_path}.'
      )

  @mock.patch.object(upload_data, '_upload_file')
  @mock.patch.object(upload_data.auth, 'get_api_key')
  def test_upload_data_directory_mixed_success_failure(
      self, mock_get_api_key, mock_upload_file
  ):
    mock_get_api_key.return_value = 'test_api_key_123'
    upload_data_dir = self.create_tempdir()
    file1 = upload_data_dir.create_file(
        'data1.mcap', content='dummy file content 1'
    )
    file2 = upload_data_dir.create_file(
        'data2.mcap', content='dummy file content 2'
    )

    # Mock _upload_file to succeed for file1 and fail for file2
    def side_effect(**kwargs):
      filename = kwargs.get('filename')
      if 'data1.mcap' in filename:
        return 200, 'OK'
      return 500, 'Internal Server Error'

    mock_upload_file.side_effect = side_effect

    with mock.patch('builtins.print') as mock_print:
      with mock.patch.object(upload_data, '_check_session_size'):
        upload_data.upload_data_directory(
            api_endpoint='https://example.com/upload',
            data_directory=upload_data_dir.full_path,
            robot_id='test_agent_001',
        )

    # Check that file1 was renamed and file2 was not
    self.assertTrue(os.path.exists(file1.full_path + '.uploaded'))
    self.assertFalse(os.path.exists(file2.full_path + '.uploaded'))
    self.assertTrue(os.path.exists(file2.full_path))

    # Verify printed calls
    printed_calls = [call[0][0] for call in mock_print.call_args_list]
    self.assertTrue(any('Uploaded data1.mcap' in msg for msg in printed_calls))
    self.assertTrue(
        any('Failed to upload data2.mcap' in msg for msg in printed_calls)
    )


class UploadSingleFilePublicTest(parameterized.TestCase):

  @mock.patch.object(upload_data, '_check_session_size')
  @mock.patch.object(upload_data, '_upload_file')
  @mock.patch.object(upload_data.auth, 'get_api_key')
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

    success, msg = upload_data.upload_single_file(
        api_endpoint='https://example.com/upload',
        file_path=file_path,
        robot_id='test_agent_001',
    )

    self.assertTrue(success)
    self.assertIn('Uploaded successfully', msg)
    self.assertTrue(os.path.exists(file_path + '.uploaded'))
    self.assertFalse(os.path.exists(file_path))

    mock_upload_file.assert_called_once_with(
        api_endpoint='https://example.com/upload',
        agent_id='test_agent_001',
        filename='data.mcap',
        file_content_bytes=b'dummy content',
        api_key='test_api_key_123',
        now=mock.ANY,
    )

  @mock.patch.object(upload_data.auth, 'get_api_key')
  def test_upload_single_file_no_api_key(self, mock_get_api_key):
    mock_get_api_key.return_value = None
    with self.assertRaisesRegex(ValueError, 'No API key found.'):
      upload_data.upload_single_file(
          api_endpoint='https://example.com/upload',
          file_path='some_file.mcap',
          robot_id='test_agent_001',
      )

  @mock.patch.object(upload_data.auth, 'get_api_key')
  def test_upload_single_file_not_mcap(self, mock_get_api_key):
    mock_get_api_key.return_value = 'test_api_key_123'
    with self.assertRaisesRegex(ValueError, 'File must be an MCAP file.'):
      upload_data.upload_single_file(
          api_endpoint='https://example.com/upload',
          file_path='some_file.txt',
          robot_id='test_agent_001',
      )

  @mock.patch.object(upload_data.auth, 'get_api_key')
  def test_upload_single_file_not_found(self, mock_get_api_key):
    mock_get_api_key.return_value = 'test_api_key_123'
    with self.assertRaises(FileNotFoundError):
      upload_data.upload_single_file(
          api_endpoint='https://example.com/upload',
          file_path='non_existent_file.mcap',
          robot_id='test_agent_001',
      )

  @mock.patch.object(upload_data, '_check_session_size')
  @mock.patch.object(upload_data, '_upload_file')
  @mock.patch.object(upload_data.auth, 'get_api_key')
  def test_upload_single_file_api_failure(
      self,
      mock_get_api_key,
      mock_upload_file,
      mock_check_session_size,
  ):
    del mock_check_session_size
    mock_get_api_key.return_value = 'test_api_key_123'
    mock_upload_file.return_value = (400, 'Bad Request')

    temp_dir = self.create_tempdir()
    file_path = temp_dir.create_file(
        'data.mcap', content='dummy content'
    ).full_path

    success, msg = upload_data.upload_single_file(
        api_endpoint='https://example.com/upload',
        file_path=file_path,
        robot_id='test_agent_001',
    )

    self.assertFalse(success)
    self.assertEqual(msg, 'Bad Request')
    self.assertFalse(os.path.exists(file_path + '.uploaded'))
    self.assertTrue(os.path.exists(file_path))


if __name__ == '__main__':
  absltest.main()
