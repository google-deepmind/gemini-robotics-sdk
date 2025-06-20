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

"""Forward compatibility layer for Gemini API. Will use genai in the future."""

import base64
import enum
import json
import logging
import time
from typing import Any, Callable, Optional, Union

from google import genai
from google.genai import types
import grpc
import numpy as np
import tensorflow as tf

from safari_sdk import auth


class RoboticsApiConnectionType(enum.Enum):
  """Connection types for the Robotics API."""
  CLOUD = 'cloud'  # Connects to a Google Cloud-based server.
  LOCAL = 'local'  # Connects to a local server.


class Client:
  """Forward compatibility layer for Gemini API.

  For general gemini use cases, see https://ai.google.dev/gemini-api/docs.

  For robotics use cases, contents are images followed by a JSON string
  representing the observations. Image observations in the JSON is index of the
  image in the contents list.

    client = genai_robotics.Client(use_robotics_api=True)
    response = client.models.generate_content(
        model="serve_id",
        contents=[
            types.Part.from_bytes(data=image_1,
                                  mime_type="image/jpeg"),
            types.Part.from_bytes(data=image_2,
                                  mime_type="image/jpeg"),
            '''{
             "images/overhead_cam": 0,
             "images/worms_eye_cam": 1,
             "task_instruction": "pick up coke can",
             "joints_pos": [1,2,3,4,5,6]
            }''',
        ]
    )

  Also works for images in numpy arrays and tensors.
      response = client.models.generate_content(
        model="serve_id",
        contents=[
            np.zeros((100, 100, 3), dtype=np.uint8),
            tf.zeros((100, 100, 3), dtype=tf.uint8),
            '''{
             "images/overhead_cam": 0,
             "images/worms_eye_cam": 1,
             "task_instruction": "pick up coke can",
             "joints_pos": [1,2,3,4,5,6]
            }''',
        ]
    )

  The client can connect to either a Google Cloud-based server or a local
  server. You can specify the connection type using the
  `robotics_api_connection` argument. In case of a local connection, the client
  connects to a gRPC server running on the local machine. In this case, the
  `model` argument to `generate_content` is ignored.
  """

  def __init__(
      self,
      *,
      use_robotics_api: bool = False,
      robotics_api_connection: RoboticsApiConnectionType = RoboticsApiConnectionType.CLOUD,
      api_key: str | None = None,
      **kwargs,
  ):
    self._use_robotics_api = use_robotics_api

    if not self._use_robotics_api:
      if not api_key:
        api_key = auth.get_api_key()
      self._client = genai.Client(api_key=api_key, **kwargs)
      return

    self._robotics_api_connection = robotics_api_connection
    match self._robotics_api_connection:
      case RoboticsApiConnectionType.CLOUD:
        service = auth.get_service()
        self._client = service.modelServing()
      case RoboticsApiConnectionType.LOCAL:
        self._client = _connect_to_grpc(_LOCAL_GRPC_URL)
      case _:
        raise ValueError(
            f'Unsupported robotics_api_connection: {robotics_api_connection}.'
            ' Only Cloud and local are supported.'
        )
    self.models: Any = lambda: None
    self.models.generate_content = self._robotics_generate_content

  def _robotics_generate_content(
      self,
      *,
      model: str,
      contents: Union[types.ContentListUnion, types.ContentListUnionDict],
      config: Optional[types.GenerateContentConfigOrDict] = None,
  ) -> types.GenerateContentResponse:
    """Generate content using the robotics API."""
    del config

    if not isinstance(contents, list):
      raise ValueError('contents must be a list of items.')
    if not isinstance(contents[-1], str):
      raise ValueError(
          'contents[-1] must be a JSON string representing the observations.'
      )

    query = {}
    try:
      input_query = json.loads(contents[-1])
    except json.JSONDecodeError as e:
      raise ValueError(
          f'Failed to parse contents[-1] as JSON: {contents[-1]}'
      ) from e

    for key, value in input_query.items():
      if key.startswith('images/'):
        query[key] = base64.b64encode(
            _coerced_to_image_bytes(contents[value])
        ).decode('utf-8')
      elif isinstance(value, (str, int, float)):
        query[key] = value
      elif isinstance(value, list):
        if not _is_list_of_numbers(value):
          raise ValueError(
              f'If value is a list, it must be a list of numbers, key: {key}.'
          )
        query[key] = value
      elif isinstance(value, np.ndarray):
        query[key] = value.tolist()
      elif isinstance(value, tf.Tensor):
        query[key] = value.numpy().tolist()
      else:
        raise ValueError(
            f'Unsupported value type: {type(value)} for key {key}.'
        )
    match self._robotics_api_connection:
      case RoboticsApiConnectionType.CLOUD:
        req_body = {
            'modelId': model,
            'methodName': 'sample_actions_json_flat',
            'inputBytes': (
                base64.b64encode(json.dumps(query).encode('utf-8')).decode(
                    'utf-8'
                )
            ),
            'requestId': time.time_ns(),
        }
        logging.debug('Request: %s', req_body)
        req = self._client.cmCustom(body=req_body)  # pytype: disable=attribute-error
        res = req.execute()
        logging.debug('Response: %s', res)
        response = lambda: None
        response.text = base64.b64decode(res['outputBytes']).decode('utf-8')
      case RoboticsApiConnectionType.LOCAL:
        response = lambda: None
        response.text = self._client(query)
      case _:
        raise ValueError(
            'Unsupported robotics_api_connection:'
            f' {self._robotics_api_connection}. Only Cloud and local are'
            ' supported.'
        )
    return response

  def __getattr__(self, name):
    if not self._use_robotics_api:
      return getattr(self._client, name)

    raise NameError(f'Attribute {name} not found.')


def _coerced_to_image_bytes(content) -> bytes:
  """Coerce content to image bytes."""
  if isinstance(content, types.Part):
    if content.inline_data.mime_type in ('image/jpeg', 'image/png'):
      return content.inline_data.data
    raise ValueError(f'Unsupported image mime type: {content.mime_type}')
  elif isinstance(content, bytes):
    if content[:4] == b'\x89PNG':
      return content
    elif content[:3] == b'\xff\xd8\xff':
      return content
    else:
      raise ValueError('Invalid PNG or JPEG image bytes.')
  elif isinstance(content, (np.ndarray, tf.Tensor)):
    return tf.io.encode_jpeg(content).numpy()
  else:
    raise ValueError(f'Unsupported image type: {type(content)}')


def _is_list_of_numbers(value):
  """Check if value is a list of numbers or list of lists of numbers...."""
  for v in value:
    if isinstance(v, (int, float)):
      continue
    if isinstance(v, list):
      if not _is_list_of_numbers(v):
        return False
    else:
      return False
  return True


def _connect_to_grpc(base_url: str) -> Callable[[dict[str, Any]], str]:
  """Connects to gRPC server."""
  if not base_url.startswith('grpc://'):
    raise ValueError(
        f'Unsupported base_url: {base_url}. Only gRPC is supported'
        ' (grpc://).'
    )
  grpc_channel = grpc.insecure_channel(base_url[7:])
  grpc_stub = grpc_channel.unary_unary(
      _LOCAL_GRPC_METHOD_NAME,
      request_serializer=lambda v: v,
      response_deserializer=lambda v: v,
  )

  def query(query: dict[str, Any]) -> str:
    encoded_query = json.dumps(query).encode('utf-8')
    return grpc_stub(encoded_query).decode('utf-8')

  return query


_LOCAL_GRPC_URL = 'grpc://localhost:60061'
_LOCAL_GRPC_METHOD_NAME = '/gemini_robotics/sample_actions_json_flat'
