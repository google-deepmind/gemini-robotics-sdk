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
import dataclasses
import datetime
import functools
import json
import logging
import os
import time
from typing import Any, Callable, Optional, Union
import warnings

from google import genai
from google.genai import types
import grpc
import msgpack
import numpy as np
from packaging import version as packaging_version
import tensorflow as tf

from safari_sdk import _version as _sdk_version
from safari_sdk import auth
from safari_sdk.model import constants

_CONNECTION = constants.RoboticsApiConnectionType
_LOCAL_GRPC_URL = 'grpc://localhost:60061'
_LOCAL_SERVICE_NAME = 'gemini_robotics'
# Minimum server version this client requires. Bump when the client starts
# relying on server-side features that older servers don't have.
_MIN_SERVER_VERSION = '0.0.0'
# Server versions >= this threshold use msgpack serialization and raw image
# bytes instead of JSON + base64.  Older servers keep the legacy JSON protocol.
_MSGPACK_MIN_SERVER_VERSION = '2.0.0'


@dataclasses.dataclass
class RoboticsGenerateContentResponse:
  """Response object returned by robotics generate_content calls."""

  text: str | None = None
  candidates: list[Any] | None = None
  backend_request_time: str | None = None
  backend_response_time: str | None = None
  client_image_encode_ms: float | None = None
  client_rpc_ms: float | None = None


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


class Models:
  """Models namespace on Client exposing generate_content."""

  def __init__(
      self,
      generate_content_fn: Callable[..., Any],
      genai_client: genai.Client | None = None,
  ):
    self.generate_content = generate_content_fn
    self._genai_client = genai_client

  def __getattr__(self, name: str) -> Any:
    if self._genai_client is not None:
      return getattr(self._genai_client.models, name)
    raise AttributeError(f"'Models' object has no attribute '{name}'")


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
      num_retries: int = 1,
      grpc_url: str | None = None,
      skip_version_check: bool = False,
      timeout: int | None = None,
      genai_base_url: str | None = None,
      **kwargs,
  ):
    """Initializes the GenAI Robotics Client.

    Args:
      robotics_api_connection: The type of connection to use for the robotics
        API. Defaults to `_CONNECTION.CLOUD`.
      api_key: The API key to use for `_CONNECTION.CLOUD_GENAI`. If None, it
        will be fetched using `auth.get_api_key()`.
      method_name: The method name to call on the robotics API. Only used for
        `_CONNECTION.CLOUD`.
      image_compression_jpeg_quality: The quality level (0-100) to use when
        compressing images to JPEG. Only used for `_CONNECTION.CLOUD` and
        `_CONNECTION.LOCAL`.
      num_retries: The number of times to retry HTTP calls to the cloud if it
        fails. Only used for `_CONNECTION.CLOUD`.
      grpc_url: The gRPC URL to connect to when using `_CONNECTION.LOCAL`.
        If None, the default `_LOCAL_GRPC_URL` is used. Example:
        'grpc://10.0.0.5:10100'.
      skip_version_check: If True, skip the server version check when using
        `_CONNECTION.LOCAL`. Useful for benchmarking or when the server is
        not yet running at client construction time.
      timeout: Timeout in seconds for remote model inference requests.
      genai_base_url: The base URL to use for `CLOUD_GENAI`. If
        None, the default base URL is used.
      **kwargs: Additional keyword arguments to pass to the `genai.Client` when
        using `_CONNECTION.CLOUD_GENAI`.
    """
    self._method_name = method_name
    self._robotics_api_connection = robotics_api_connection
    self._num_retries = num_retries
    self._server_version: str | None = None
    self._image_compression_jpeg_quality = image_compression_jpeg_quality
    self._call_counter = 0
    match self._robotics_api_connection:
      case _CONNECTION.CLOUD:
        if timeout is not None:
          service = auth.get_service(
              timeout=timeout, dscp=constants.DEFAULT_INFERENCE_DSCP
          )
        else:
          service = auth.get_service(dscp=constants.DEFAULT_INFERENCE_DSCP)
        self._client = service.modelServing()

        partial_kwargs = {
            'image_compression_jpeg_quality': image_compression_jpeg_quality,
        }
        if timeout is not None:
          partial_kwargs['config'] = types.GenerateContentConfig(  # pyrefly: ignore[bad-assignment]
              http_options=types.HttpOptions(timeout=timeout * 1000)
          )

        generate_content_fn = functools.partial(
            self._robotics_generate_content, **partial_kwargs
        )
        self.models = Models(generate_content_fn=generate_content_fn)
      case _CONNECTION.LOCAL:
        url = grpc_url or os.environ.get('GOOGLE_GEMINI_BASE_URL')
        if url is None:
          url = _LOCAL_GRPC_URL
        if not url.startswith('grpc://'):
          url = 'grpc://' + url
        channel = grpc.insecure_channel(url[7:])
        if not skip_version_check:
          server_info = _check_server_compatibility(
              channel, _sdk_version.__version__
          )
          self._server_version = server_info.get('server_version')
        self._client = _connect_to_grpc_json(
            channel, method_name, use_msgpack=self._use_msgpack
        )
        generate_content_fn = functools.partial(
            self._robotics_generate_content,
            image_compression_jpeg_quality=image_compression_jpeg_quality,
        )
        self.models = Models(generate_content_fn=generate_content_fn)
      case _CONNECTION.CLOUD_GENAI:
        if not api_key:
          api_key = auth.get_api_key(key_name='api_key_gemini')
        http_options = kwargs.get('http_options') or types.HttpOptions()
        if isinstance(http_options, dict):
          http_options = types.HttpOptions(**http_options)
        updates: dict[str, Any] = {}
        if genai_base_url:
          updates['base_url'] = genai_base_url
        if updates:
          kwargs['http_options'] = http_options.model_copy(update=updates)
        self._client = genai.Client(
            api_key=api_key,
            **kwargs,
        )
      case _:
        raise ValueError(
            f'Unsupported robotics_api_connection: {robotics_api_connection}.'
            ' Only cloud, cloud_genai, and local are supported.'
        )

  @property
  def _use_msgpack(self) -> bool:
    """Whether to use msgpack serialization instead of JSON.

    Returns True when the server version is >= _MSGPACK_MIN_SERVER_VERSION
    (i.e. grodv2+). Falls back to False for None or older servers.
    """
    if self._server_version is None:
      return False
    return not _version_lt(self._server_version, _MSGPACK_MIN_SERVER_VERSION)

  def ping(self) -> float | None:
    """Measures round-trip ping latency to the server in milliseconds."""
    if self._robotics_api_connection == _CONNECTION.CLOUD:
      try:
        assert hasattr(self._client, 'doPing')
        t0 = time.perf_counter()
        self._client.doPing().execute(num_retries=0)
        return (time.perf_counter() - t0) * 1000.0
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning('Failed to ping server: %s', e)
        return None
    return None

  def _robotics_generate_content(
      self,
      *,
      model: str,
      contents: Union[types.ContentListUnion, types.ContentListUnionDict],
      config: Optional[types.GenerateContentConfigOrDict] = None,
      image_compression_jpeg_quality: int = 95,
  ) -> RoboticsGenerateContentResponse:
    """Generate content using the robotics API."""
    if not isinstance(contents, list):
      raise ValueError('contents must be a list of items.')
    if not isinstance(contents[-1], str):
      raise ValueError(
          'contents[-1] must be a JSON string representing the observations.'
      )

    # Only GenerateContentConfig type is supported for now.
    # Only the timeout option in the config is supported for now.
    timeout_ms = None
    if (
        config
        and isinstance(config, types.GenerateContentConfig)
        and config.http_options
    ):
      timeout_ms = config.http_options.timeout

    query = {}
    try:
      input_query = json.loads(contents[-1])
    except json.JSONDecodeError as e:
      raise ValueError(
          f'Failed to parse contents[-1] as JSON: {contents[-1]}'
      ) from e

    t_img_start = time.perf_counter()
    for key, value in input_query.items():
      if key.startswith('images/'):
        image_bytes = _coerced_to_image_bytes(
            contents[value],
            image_compression_jpeg_quality=image_compression_jpeg_quality,
        )
        if self._use_msgpack:
          query[key] = image_bytes
        else:
          query[key] = base64.b64encode(image_bytes).decode('utf-8')
      elif isinstance(value, (str, int, float)):
        query[key] = value
      elif isinstance(value, list):
        if not _is_list_of_numbers(value):
          raise ValueError(
              f'If value is a list, it must be a list of numbers, key: {key}.'
          )
        query[key] = value
      elif isinstance(value, np.ndarray):
        res = value.tolist()
        if isinstance(res, bytes):
          res = res.decode('utf-8', errors='ignore')
        query[key] = res
      elif isinstance(value, tf.Tensor):
        query[key] = value.numpy().tolist()
      else:
        raise ValueError(
            f'Unsupported value type: {type(value)} for key {key}.'
        )
    client_image_encode_ms = (time.perf_counter() - t_img_start) * 1000.0

    match self._robotics_api_connection:
      case _CONNECTION.CLOUD:
        if self._use_msgpack:
          input_bytes = base64.b64encode(
              msgpack.packb(query, use_bin_type=True)
          ).decode('utf-8')
        else:
          input_bytes = base64.b64encode(
              json.dumps(query).encode('utf-8')
          ).decode('utf-8')
        req_body = {
            'modelId': model,
            'methodName': self._method_name,
            'inputBytes': input_bytes,
            'requestId': time.time_ns(),
        }
        if timeout_ms:
          timeout_seconds = timeout_ms // 1000
          timeout_nanos = (timeout_ms % 1000) * 1000000
          req_body['modelOptions'] = {  # pyrefly: ignore[bad-assignment]
              'timeout': {'seconds': timeout_seconds, 'nanos': timeout_nanos}
          }

        t_rpc_start = time.perf_counter()
        logging.debug('Request: %s', req_body)
        req = self._client.cmCustom(body=req_body)  # pytype: disable=attribute-error
        res = req.execute(num_retries=self._num_retries)
        logging.debug('Response: %s', res)
        client_rpc_ms = (time.perf_counter() - t_rpc_start) * 1000.0

        if self._use_msgpack:
          response_data = msgpack.unpackb(
              base64.b64decode(res['outputBytes']), raw=False
          )
          res_text = json.dumps(response_data)
        else:
          res_text = base64.b64decode(res['outputBytes']).decode('utf-8')
        response = RoboticsGenerateContentResponse(
            text=res_text,
            backend_request_time=res.get('backendRequestTime'),
            backend_response_time=res.get('backendResponseTime'),
            client_image_encode_ms=client_image_encode_ms,
            client_rpc_ms=client_rpc_ms,
        )
      case _CONNECTION.LOCAL:
        t_rpc_start = time.perf_counter()
        req_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
        res_text = self._client(query)  # pyrefly: ignore[not-callable]
        res_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
        client_rpc_ms = (time.perf_counter() - t_rpc_start) * 1000.0

        response = RoboticsGenerateContentResponse(
            text=res_text,
            backend_request_time=req_time,
            backend_response_time=res_time,
            client_image_encode_ms=client_image_encode_ms,
            client_rpc_ms=client_rpc_ms,
        )
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
    if content.inline_data.mime_type in ('image/jpeg', 'image/png'):  # pyrefly: ignore[missing-attribute]
      return content.inline_data.data  # pyrefly: ignore[missing-attribute]
    raise ValueError(f'Unsupported image mime type: {content.mime_type}')  # pyrefly: ignore[missing-attribute]
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


def _connect_to_grpc_json(
    channel: grpc.Channel,
    method_name: str,
    use_msgpack: bool = False,
) -> Callable[[dict[str, Any]], str]:
  """Creates a query function over an existing gRPC channel.

  Args:
    channel: An open gRPC channel.
    method_name: The name of the method to call on the gRPC server.
    use_msgpack: If True, use msgpack serialization. Otherwise use JSON.

  Returns:
    A callable that takes a query dict and returns the server's JSON response
    as a string.
  """
  full_method_name = f'/{_LOCAL_SERVICE_NAME}/{method_name}'
  grpc_stub = channel.unary_unary(
      full_method_name,
      request_serializer=lambda v: v,
      response_deserializer=lambda v: v,
  )

  def query(query: dict[str, Any]) -> str:
    if use_msgpack:
      encoded_query = msgpack.packb(query, use_bin_type=True)
      response_bytes = grpc_stub(encoded_query)
      response = msgpack.unpackb(response_bytes, raw=False)
      return json.dumps(response)
    else:
      encoded_query = json.dumps(query).encode('utf-8')
      return grpc_stub(encoded_query).decode('utf-8')

  return query


def _version_lt(a: str, b: str) -> bool:
  """Returns True if version string a is less than version string b.

  Args:
    a: First version string (e.g. '2.103.0').
    b: Second version string (e.g. '2.122.0').

  Returns:
    True if a < b per PEP 440 semantics. Returns False if either
    string is not a valid version.
  """
  try:
    return packaging_version.Version(a) < packaging_version.Version(b)
  except packaging_version.InvalidVersion:
    return False


def _check_server_compatibility(
    channel: grpc.Channel,
    client_version: str,
    timeout: float = 5.0,
) -> dict[str, Any]:
  """Queries get_server_info for version negotiation.

  Handles three cases:
  1. Server supports get_server_info -> returns server info dict.
  2. Server doesn't implement it (UNIMPLEMENTED) -> warns, returns
     JSON fallback.
  3. Server unreachable (UNAVAILABLE/DEADLINE_EXCEEDED) -> warns,
     returns JSON.

  Args:
    channel: An open gRPC channel to the server.
    client_version: The client SDK version string.
    timeout: Seconds to wait for the server response.

  Returns:
    A dict with at least 'supported_protocols'. May also contain
      'server_version' if the server reported one.

  Raises:
    RuntimeError: If the client version is below the server's
      min_client_version.
  """
  try:
    stub = channel.unary_unary(
        '/gemini_robotics/get_server_info',
        request_serializer=lambda v: v,
        response_deserializer=lambda v: v,
    )
    response = stub(b'', timeout=timeout)
    server_info = json.loads(response.decode('utf-8'))

    min_client = server_info.get('min_client_version', '0.0.0')
    if _version_lt(client_version, min_client):
      raise RuntimeError(
          f'Safari SDK {client_version} is too old for this server '
          f'(requires >= {min_client}). '
          f'Upgrade: pip install --upgrade google-genai>={min_client}'
      )

    server_ver = server_info.get('server_version')
    if server_ver is not None and _version_lt(server_ver, _MIN_SERVER_VERSION):
      warnings.warn(
          f'Server version {server_ver} is older than the minimum '
          f'this client supports ({_MIN_SERVER_VERSION}). '
          'Some features may not work correctly.',
          stacklevel=2,
      )
    return server_info

  except json.JSONDecodeError:
    warnings.warn(
        'Server returned invalid JSON from get_server_info. '
        'Proceeding with JSON protocol.',
        stacklevel=2,
    )
    return {'supported_protocols': ['msgpack', 'json']}

  except grpc.RpcError as e:
    status = e.code()  # pytype: disable=attribute-error
    if status == grpc.StatusCode.UNIMPLEMENTED:
      warnings.warn(
          'Server does not support get_server_info. '
          'Assuming legacy JSON protocol.',
          stacklevel=2,
      )
    elif status in (
        grpc.StatusCode.UNAVAILABLE,
        grpc.StatusCode.DEADLINE_EXCEEDED,
    ):
      warnings.warn(
          f'Server version check failed ({status.name}). '
          'Proceeding with JSON protocol.',
          stacklevel=2,
      )
    else:
      raise
    return {'supported_protocols': ['msgpack', 'json']}
