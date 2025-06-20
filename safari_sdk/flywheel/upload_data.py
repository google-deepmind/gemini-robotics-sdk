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

"""Upload data library."""

import datetime
import json
import os
import time

import pytz
import requests

from safari_sdk import auth


def _upload_file(
    *,
    api_endpoint,
    agent_id,
    filename,
    file_content_bytes,
    api_key,
    now,
):
  """Calls the data ingestion service to upload the file."""

  def to_multi_part(metadata, body, ct):
    """Returns a multi-part request for the metadata and body."""
    boundary_ = b'BOUNDARY'
    data_ct = b'Content-Type: application/octet-stream'
    payload = b''.join([
        b'--',
        boundary_,
        b'\r\n',
        data_ct,
        b'\r\n\r\n',
        metadata,
        b'\r\n--',
        boundary_,
        b'\r\n',
        data_ct,
        b'\r\n\r\n',
        body,
        b'\r\n--',
        boundary_,
        b'--\r\n',
    ])
    headers = {
        'X-Goog-Upload-Protocol': 'multipart',
        'X-Goog-Upload-Header-Content-Type': ct.decode('utf-8'),
        'Content-Type': (
            'multipart/related; boundary=%s' % boundary_.decode('utf-8')
        ),
    }
    return headers, payload

  request_dict = {
      'date': {'year': now.year, 'month': now.month, 'day': now.day},
      'agentId': agent_id,
      'filename': filename,
  }
  headers, body = to_multi_part(
      json.dumps(request_dict).encode(), file_content_bytes, b'text/plain'
  )
  r = requests.post(
      api_endpoint,
      params={'key': api_key},
      headers=headers,
      data=body,
  )
  return (r.status_code, r.reason)


def upload_data_directory(
    api_endpoint,
    data_directory,
    robot_id,
):
  """Upload data directory."""

  api_key = auth.get_api_key()
  if not api_key:
    raise ValueError('No API key found.')

  for root, dirs, files in os.walk(data_directory):
    del dirs
    for file in files:
      if file.endswith('.mcap'):
        file_path = os.path.join(root, file)

        with open(file_path, 'rb') as f:
          file_content_bytes = f.read()
        file_size_mb = len(file_content_bytes) / (1024 * 1024)

        t_start = time.time()
        status_code, reason = _upload_file(
            api_endpoint=api_endpoint,
            agent_id=robot_id,
            filename=file,
            file_content_bytes=file_content_bytes,
            api_key=api_key,
            now=datetime.datetime.now(pytz.timezone('America/Los_Angeles')),
        )
        t_end = time.time()

        if status_code == 200:
          uploaded_file_path = file_path + '.uploaded'
          os.rename(file_path, uploaded_file_path)

          upload_speed_mb_s = file_size_mb / (t_end - t_start)
          print(
              f'Uploaded {file} ({file_size_mb:.2f} MB) and renamed to'
              f' {uploaded_file_path} in {t_end - t_start:.2f}s'
              f' ({upload_speed_mb_s:.2f} MB/s)'
          )
        else:
          print(f'Failed to upload {file} ({file_size_mb:.2f} MB): {reason}')
