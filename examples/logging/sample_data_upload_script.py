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

"""Sample data upload script.

Google Deepmind Robotics team has set up an endpoint to upload data. Each log
file is uploaded as a single HTTP POST request.
"""

import datetime
import json
import os
import time

from absl import app
from absl import flags
import pytz
import requests


_API_ENDPOINT = flags.DEFINE_string(
    'api_endpoint',
    'https://roboticsdeveloper.googleapis.com/upload/v1/dataIngestion:uploadData',
    'Data ingestion service endpoint.',
)
_AGENT_ID = flags.DEFINE_string(
    'agent_id',
    None,
    'Typically the identifier of the robot or human collector. Alphanumeric '
    'and fewer than 60 characters.',
    required=True,
)
_API_KEY = flags.DEFINE_string(
    'api_key',
    None,
    'Api key to call the data ingestion service, please contact Google '
    'Deepmind Robotics team for this',
    required=True,
)
_DATA_DIRECTORY = flags.DEFINE_string(
    'data_dir',
    None,
    'Directory where the data files are stored.',
    required=True,
)


def upload(
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
        'Content-Type': 'multipart/related; boundary=%s' % boundary_.decode(
            'utf-8'
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


def main(_):

  def walk_and_upload(directory):
    for root, dirs, files in os.walk(directory):
      del dirs
      for file in files:
        if file.endswith('.mcap'):
          file_path = os.path.join(root, file)

          with open(file_path, 'rb') as f:
            file_content_bytes = f.read()
          file_size_mb = len(file_content_bytes) / (1024 * 1024)

          t_start = time.time()
          status_code, reason = upload(
              api_endpoint=_API_ENDPOINT.value,
              agent_id=_AGENT_ID.value,
              filename=file,
              file_content_bytes=file_content_bytes,
              api_key=_API_KEY.value,
              now=datetime.datetime.now(pytz.timezone('America/Los_Angeles')),
          )
          t_end = time.time()

          if status_code == 200:
            uploaded_file_path = os.path.splitext(file_path)[0] + '.uploaded'
            os.rename(file_path, uploaded_file_path)

            upload_speed_mb_s = file_size_mb / (t_end - t_start)
            print(
                f'Uploaded {file} ({file_size_mb:.2f} MB) and renamed to'
                f' {uploaded_file_path} in {t_end - t_start:.2f}s'
                f' ({upload_speed_mb_s:.2f} MB/s)'
            )
          else:
            print(f'Failed to upload {file} ({file_size_mb:.2f} MB): {reason}')

  walk_and_upload(_DATA_DIRECTORY.value)


if __name__ == '__main__':
  app.run(main)
