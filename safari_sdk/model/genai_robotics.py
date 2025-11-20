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
import functools
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
from safari_sdk.model import constants

_CONNECTION = constants.RoboticsApiConnectionType
_LOCAL_GRPC_URL = 'grpc://localhost:60061'
_LOCAL_GRPC_METHOD_NAME = '/gemini_robotics/sample_actions_json_flat'


def update_robotics_content_to_genai_format(
    contents: Union[types.ContentListUnion, types.ContentListUnionDict],
    image_compression_jpeg_quality: int = 95,
) -> Union[types.ContentListUnion, types.ContentListUnionDict]:
  """Update robotics contents to required GenAI API format."""

  if not isinstance(contents, list):
    return contents

  new_contents = []
  for content in contents:
    if isinstance(content, types.Part):
      new_contents.append(content)
    elif isinstance(content, str):
      new_contents.append(content.replace('Infinity', '0.0'))
    elif isinstance(content, (np.ndarray, tf.Tensor)):
      # automatically convert images to jpeg bytes.
      if (
          content.dtype in (np.uint8, tf.uint8)
          and content.ndim == 3
          and content.shape[-1] == 3
      ):
        new_contents.append(
            types.Part.from_bytes(
                data=_coerced_to_image_bytes(
                    content,
                    image_compression_jpeg_quality=image_compression_jpeg_quality,
                ),
                mime_type='image/jpeg',
            )
        )
      else:
        raise ValueError(
            f'Unsupported numpy array/tensor dtype: {content.dtype} with'
            f' shape {content.shape}'
        )
    elif isinstance(content, tf.Tensor):
      new_contents.append(types.Part(text=content.numpy().tolist()))
    else:
      raise ValueError(f'Unsupported content type: {type(content)}')

  return new_contents


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
      robotics_api_connection: _CONNECTION = _CONNECTION.CLOUD,
      api_key: str | None = None,
      method_name: str = 'sample_actions_json_flat',
      image_compression_jpeg_quality: int = 95,
      **kwargs,
  ):
    self._method_name = method_name
    self._robotics_api_connection = robotics_api_connection
    match self._robotics_api_connection:
      case _CONNECTION.CLOUD:
        service = auth.get_service()
        self._client = service.modelServing()
        self.models: Any = lambda: None
        self.models.generate_content = functools.partial(
            self._robotics_generate_content,
            image_compression_jpeg_quality=image_compression_jpeg_quality,
        )
      case _CONNECTION.LOCAL:
        self._client = _connect_to_grpc(_LOCAL_GRPC_URL)
        self.models: Any = lambda: None
        self.models.generate_content = functools.partial(
            self._robotics_generate_content,
            image_compression_jpeg_quality=image_compression_jpeg_quality,
        )
      case _CONNECTION.CLOUD_GENAI:
        if not api_key:
          api_key = auth.get_api_key()
        self._client = genai.Client(api_key=api_key, **kwargs)
      case _:
        raise ValueError(
            f'Unsupported robotics_api_connection: {robotics_api_connection}.'
            ' Only cloud, cloud_genai, and local are supported.'
        )

  def _robotics_generate_content(
      self,
      *,
      model: str,
      contents: Union[types.ContentListUnion, types.ContentListUnionDict],
      config: Optional[types.GenerateContentConfigOrDict] = None,
      image_compression_jpeg_quality: int = 95,
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
            _coerced_to_image_bytes(
                contents[value],
                image_compression_jpeg_quality=image_compression_jpeg_quality,
            )
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
      case _CONNECTION.CLOUD:
        req_body = {
            'modelId': model,
            'methodName': self._method_name,
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
      case _CONNECTION.LOCAL:
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
    if self._robotics_api_connection == _CONNECTION.CLOUD_GENAI:
      return getattr(self._client, name)

    raise NameError(f'Attribute {name} not found.')


def _coerced_to_image_bytes(content, image_compression_jpeg_quality) -> bytes:
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
    return tf.io.encode_jpeg(content,
                             quality=image_compression_jpeg_quality).numpy()
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
        f'Unsupported base_url: {base_url}. Only gRPC is supported (grpc://).'
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
