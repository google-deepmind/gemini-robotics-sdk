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

r"""Resumable upload data library.

                   Resumable Upload Protocol State Machine
================================================================================

  +-----------------------------------------------------------------------+
  |                           START SESSION                               |
  | POST api_endpoint (X-Goog-Upload-Command: start, protocol: resumable) |
  +-----------------------------------------------------------------------+
                                    |
                                    v
                       +-------------------------+
                       |  Active Upload Session  |
                       |  (Extract Upload URL)   |
                       +-------------------------+
                                    |
                                    v  <----------------------------------+
                     +-----------------------------+                      |
                     |     UPLOAD CHUNK LOOP       |                      |
                     | POST upload_url             |                      |
                     | cmd: 'upload' / 'finalize'  |                      |
                     | offset: N                   |                      |
                     +-----------------------------+                      |
                       /           |            \                         |
      Status: 'final' /       4xx  |   Error /   \                        |
                     /       Fatal |   Non-active \                       |
                    v              v               v                      |
            +---------------+  +-------+   +---------------+              |
            |    SUCCESS    |  | FAIL  |   | QUERY STATUS  |              |
            +---------------+  +-------+   +---------------+              |
                                              /   |       \               |
                             Status: 'final' /    | 4xx    \  Status:     |
                             (Completed)    /     | Fatal   \ 'active'    |
                                           v      |          v            |
                              +---------------+   |   +----------------+  |
                              |   SUCCESS     |   |   | Check Progress |  |
                              +---------------+   |   +----------------+  |
                                                  |      /       \        |
                                No Progress /     |     /         \       |
                                Progress          |    /           \      |
                                Retries >= Max    |   /             \     |
                                Updated Offset    |  /               \    |
                                                  v v                 v   |
                                             +-------------+   +-------------+
                                             |     FAIL    |   |  NEXT CHUNK |
                                             +-------------+   +-------------+
"""

import datetime
import io
import json
import os
import time
from typing import BinaryIO, Union

import mcap.exceptions as mcap_exceptions
import mcap.reader as mcap_reader
import pytz
import requests

from safari_sdk import auth

_SESSION_SIZE_LIMIT_BYTES = 1 * 1024 * 1024
_CHUNK_SIZE_BYTES = (
    32 * 1024 * 1024
)  # 32 MiB default chunk size for streaming upload


def _check_session_size(
    file_input: Union[str, os.PathLike[str], bytes, BinaryIO],
) -> None:
  """Raises ValueError if any /session message is too big."""
  try:
    if isinstance(file_input, (str, os.PathLike)):
      with open(file_input, 'rb') as f:
        reader = mcap_reader.make_reader(f)
        _verify_session_messages(reader)
    elif isinstance(file_input, bytes):
      reader = mcap_reader.make_reader(io.BytesIO(file_input))
      _verify_session_messages(reader)
    else:
      reader = mcap_reader.make_reader(file_input)
      _verify_session_messages(reader)
  except mcap_exceptions.McapError as e:
    raise ValueError(f'File is not a valid MCAP: {e}') from e


def _verify_session_messages(reader: mcap_reader.McapReader) -> None:
  for _, _, message in reader.iter_messages(topics=['/session']):
    size = len(message.data)
    if size > _SESSION_SIZE_LIMIT_BYTES:
      raise ValueError(
          f'/session message is {size:,} bytes'
          f' ({size / 1024 / 1024:.1f} MiB), which exceeds the'
          f' {_SESSION_SIZE_LIMIT_BYTES // 1024 // 1024} MiB limit.'
          ' This is likely caused by inefficient per-pixel image bounds in'
          ' a gym.Box observation space. Use low=-np.inf, high=np.inf for'
          ' image observations, then re-record, or reach out to your TTP'
          ' contact for assistance.'
      )


def _upload_file(
    *,
    api_endpoint: str,
    agent_id: str,
    filename: str,
    api_key: str,
    now: datetime.datetime,
    file_path: str | None = None,
    file_content_bytes: bytes | None = None,
    chunk_size: int = _CHUNK_SIZE_BYTES,
    max_retries: int = 5,
) -> tuple[int, str]:
  """Calls the data ingestion service to upload the file using resumable UUP protocol."""
  if file_path is not None:
    file_size = os.path.getsize(file_path)
  elif file_content_bytes is not None:
    file_size = len(file_content_bytes)
  else:
    raise ValueError('Either file_path or file_content_bytes must be provided.')

  request_dict = {
      'date': {'year': now.year, 'month': now.month, 'day': now.day},
      'agentId': agent_id,
      'filename': filename,
  }

  metadata_json = json.dumps(request_dict)
  start_headers = {
      'X-Goog-Upload-Protocol': 'resumable',
      'X-Goog-Upload-Command': 'start',
      'X-Goog-Upload-Header-Content-Length': str(file_size),
      'X-Goog-Upload-Header-Content-Type': 'application/octet-stream',
      'Content-Type': 'application/json',
  }

  r_start = None
  start_error = None
  start_backoff = 1
  for _ in range(max_retries):
    try:
      r_start = requests.post(
          api_endpoint,
          params={'key': api_key},
          headers=start_headers,
          data=metadata_json.encode('utf-8'),
      )
      if r_start.status_code == 200 or 400 <= r_start.status_code < 500:
        break
    except requests.exceptions.RequestException as e:
      start_error = f'Connection Error: {e}'

    time.sleep(start_backoff)
    start_backoff *= 2

  if r_start is None:
    return (-1, start_error or 'Connection error starting upload session')

  if r_start.headers.get('X-Goog-Upload-Status') == 'final':
    return (r_start.status_code, r_start.reason)

  if (
      r_start.status_code != 200
      or r_start.headers.get('X-Goog-Upload-Status') != 'active'
  ):
    return (r_start.status_code, r_start.reason)

  upload_url = r_start.headers.get('X-Goog-Upload-URL')
  if not upload_url:
    return (r_start.status_code, 'Missing X-Goog-Upload-URL header')

  offset = 0
  consecutive_no_progress_retries = 0
  backoff_delay = 1

  # If reading from disk, open file handle once for all chunks
  f_stream: BinaryIO | None = None
  if file_path is not None:
    f_stream = open(file_path, 'rb')

  try:
    while offset < file_size or (file_size == 0 and offset == 0):
      if f_stream is not None:
        f_stream.seek(offset)
        chunk_data = f_stream.read(chunk_size)
      else:
        assert file_content_bytes is not None
        chunk_data = file_content_bytes[offset : offset + chunk_size]

      chunk_len = len(chunk_data)
      is_last_chunk = offset + chunk_len >= file_size

      command = 'upload, finalize' if is_last_chunk else 'upload'
      upload_headers = {
          'X-Goog-Upload-Command': command,
          'X-Goog-Upload-Offset': str(offset),
      }

      try:
        r_upload = requests.post(
            upload_url,
            headers=upload_headers,
            data=chunk_data,
        )

        if r_upload.headers.get('X-Goog-Upload-Status') == 'final':
          return (r_upload.status_code, r_upload.reason)

        if 400 <= r_upload.status_code < 500:
          return (r_upload.status_code, r_upload.reason)

        if (
            r_upload.status_code == 200
            and r_upload.headers.get('X-Goog-Upload-Status') == 'active'
        ):
          offset += chunk_len
          consecutive_no_progress_retries = 0
          backoff_delay = 1
          if file_size == 0:
            break
          continue

      except requests.exceptions.RequestException:
        pass

      # Query status to resume
      query_headers = {'X-Goog-Upload-Command': 'query'}
      query_success = False
      r_query = None

      for _ in range(max_retries):
        try:
          r_query = requests.post(upload_url, headers=query_headers)

          if r_query.headers.get('X-Goog-Upload-Status') == 'final':
            return (r_query.status_code, r_query.reason)

          if 400 <= r_query.status_code < 500:
            return (r_query.status_code, r_query.reason)

          if (
              r_query.status_code == 200
              and r_query.headers.get('X-Goog-Upload-Status') == 'active'
          ):
            query_success = True
            break
        except requests.exceptions.RequestException:
          pass

        time.sleep(backoff_delay)
        backoff_delay *= 2

      if not query_success or r_query is None:
        return (-1, 'Failed to retrieve upload status after multiple attempts')

      new_offset = int(r_query.headers.get('X-Goog-Upload-Size-Received', 0))

      if new_offset == offset:
        consecutive_no_progress_retries += 1
        if consecutive_no_progress_retries >= max_retries:
          return (-1, 'No upload progress after multiple retry attempts')
        time.sleep(backoff_delay)
        backoff_delay *= 2
      else:
        consecutive_no_progress_retries = 0
        backoff_delay = 1
        offset = new_offset

      if file_size == 0:
        break
  finally:
    if f_stream is not None:
      f_stream.close()

  return (200, 'OK')


def upload_data_directory(
    api_endpoint: str,
    data_directory: str,
    robot_id: str,
) -> tuple[int, int, int]:
  """Upload data directory using resumable uploads."""
  api_key = auth.get_api_key()
  if not api_key:
    raise ValueError('No API key found.')

  uploaded_count = 0
  failed_count = 0
  already_uploaded_count = 0

  for root, dirs, files in os.walk(data_directory):
    del dirs
    for file in files:
      if file.endswith('.mcap'):
        file_path = os.path.join(root, file)
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

        try:
          _check_session_size(file_path)
        except ValueError as e:
          failed_count += 1
          print(f'Failed to validate session in {file}: {e}')
          continue

        t_start = time.time()
        status_code, reason = _upload_file(
            api_endpoint=api_endpoint,
            agent_id=robot_id,
            filename=file,
            file_path=file_path,
            api_key=api_key,
            now=datetime.datetime.now(pytz.timezone('America/Los_Angeles')),
        )
        t_end = time.time()

        if status_code == 200:
          uploaded_count += 1
          uploaded_file_path = file_path + '.uploaded'
          os.rename(file_path, uploaded_file_path)

          upload_speed_mb_s = file_size_mb / (t_end - t_start)
          print(
              f'Uploaded {file} ({file_size_mb:.2f} MB) and renamed to'
              f' {uploaded_file_path} in {t_end - t_start:.2f}s'
              f' ({upload_speed_mb_s:.2f} MB/s)'
          )
        else:
          failed_count += 1
          print(f'Failed to upload {file} ({file_size_mb:.2f} MB): {reason}')
      elif file.endswith('.mcap.uploaded'):
        already_uploaded_count += 1

  if not uploaded_count and not failed_count:
    if already_uploaded_count:
      print(
          f'No new .mcap files found in {data_directory}. '
          f'{already_uploaded_count} file(s) were already uploaded.'
      )
    else:
      print(f'No .mcap files found in {data_directory}.')

  return uploaded_count, failed_count, already_uploaded_count


def upload_single_file(
    api_endpoint: str,
    file_path: str,
    robot_id: str,
) -> tuple[bool, str]:
  """Upload a single file using resumable upload."""
  api_key = auth.get_api_key()
  if not api_key:
    raise ValueError('No API key found.')

  if not file_path.endswith('.mcap'):
    raise ValueError('File must be an MCAP file.')

  if not os.path.exists(file_path):
    raise FileNotFoundError(f'File not found: {file_path}')

  file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

  _check_session_size(file_path)

  t_start = time.time()
  status_code, reason = _upload_file(
      api_endpoint=api_endpoint,
      agent_id=robot_id,
      filename=os.path.basename(file_path),
      file_path=file_path,
      api_key=api_key,
      now=datetime.datetime.now(pytz.timezone('America/Los_Angeles')),
  )
  t_end = time.time()

  if status_code == 200:
    uploaded_file_path = file_path + '.uploaded'
    os.rename(file_path, uploaded_file_path)
    upload_speed_mb_s = file_size_mb / (t_end - t_start)
    print(
        f'Uploaded {os.path.basename(file_path)} ({file_size_mb:.2f} MB) and'
        f' renamed to {uploaded_file_path} in {t_end - t_start:.2f}s'
        f' ({upload_speed_mb_s:.2f} MB/s)'
    )
    return True, f'Uploaded successfully in {t_end - t_start:.2f}s'
  else:
    print(
        f'Failed to upload {os.path.basename(file_path)}'
        f' ({file_size_mb:.2f} MB): {reason}'
    )
    return False, reason
