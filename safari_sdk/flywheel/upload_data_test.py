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

import pytz

# Assuming the module is accessible like this for testing
from absl.testing import absltest
from absl.testing import parameterized
from safari_sdk.flywheel import upload_data


class UploadFileTest(absltest.TestCase):

  @mock.patch(
      'safari_sdk.flywheel.upload_data.requests.post'
  )
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

  @mock.patch(
      'safari_sdk.flywheel.upload_data._upload_file'
  )
  @mock.patch(
      'safari_sdk.flywheel.upload_data.auth.get_api_key'
  )
  def test_upload_data_directory_success_and_rename(
      self,
      mock_get_api_key,
      mock_upload_file,
  ):
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

  @mock.patch(
      'safari_sdk.flywheel.upload_data.auth.get_api_key'
  )
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


if __name__ == '__main__':
  absltest.main()
