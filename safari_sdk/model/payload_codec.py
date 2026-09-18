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

"""Payload codec for encoding observations and decoding actions for Sax/CustomModel.

Provides serialization and deserialization helpers for robotics observation
payloads and model action chunks across JSON and msgpack formats.
"""

import base64
from collections.abc import Sequence
import dataclasses
import json
import time
from typing import Any, TypeAlias

from google.genai import types
import msgpack
import numpy as np
import tensorflow as tf

from safari_sdk.model import constants

# Type aliases for robotics contents and images.
ImageContent: TypeAlias = types.Part | bytes | np.ndarray | tf.Tensor
RoboticsContentItem: TypeAlias = ImageContent | str

__all__ = [
    "DecodedActionChunk",
    "EncodedObservationPayload",
    "ImageContent",
    "RoboticsContentItem",
    "decode_action_chunk_payload",
    "encode_observation_payload",
]


def _coerced_to_image_bytes(
    content: ImageContent, image_compression_jpeg_quality: int = 95
) -> bytes:
  """Coerces various image types to encoded image bytes (JPEG or PNG)."""
  if isinstance(content, types.Part):
    if content.inline_data is not None and content.inline_data.mime_type in (
        "image/jpeg",
        "image/png",
    ):
      if content.inline_data.data is not None:
        return content.inline_data.data
    mime = content.inline_data.mime_type if content.inline_data else None
    raise ValueError(f"Unsupported image mime type: {mime}")
  elif isinstance(content, bytes):
    if content[:4] == b"\x89PNG":
      return content
    elif content[:3] == b"\xff\xd8\xff":
      return content
    else:
      raise ValueError("Invalid PNG or JPEG image bytes.")
  elif isinstance(content, (np.ndarray, tf.Tensor)):
    return tf.io.encode_jpeg(
        content, quality=image_compression_jpeg_quality
    ).numpy()
  else:
    raise ValueError(f"Unsupported image type: {type(content)}")


@dataclasses.dataclass(frozen=True)
class EncodedObservationPayload:
  """Encoded observation payload ready for transport over gRPC wire.

  Attributes:
    payload_bytes: Serialized bytes (JSON or msgpack) of the observation.
    encode_time_ms: Client CPU time in ms spent JPEG compressing images.
  """

  payload_bytes: bytes
  encode_time_ms: float


@dataclasses.dataclass(frozen=True)
class DecodedActionChunk:
  """Result of decoding an action chunk from raw response bytes.

  Attributes:
    action_chunk: 2D numpy array of actions (shape: [chunk_size, action_dim]).
    metadata: Parsed response metadata dictionary (e.g. rng_key).
  """

  action_chunk: np.ndarray
  metadata: dict[str, Any]


def encode_observation_payload(
    contents: Sequence[RoboticsContentItem],
    *,
    image_compression_jpeg_quality: int = 95,
    use_msgpack: bool = False,
) -> EncodedObservationPayload:
  """Encodes robotics contents into raw payload bytes (JSON or msgpack).

  Args:
    contents: Sequence of robotics content items where contents[-1] is a JSON
      string describing observation keys and image indices.
    image_compression_jpeg_quality: JPEG compression quality level (0-100).
    use_msgpack: Whether to serialize with msgpack instead of JSON.

  Returns:
    EncodedObservationPayload containing serialized bytes and encode time in ms.

  Raises:
    ValueError: If contents format is invalid, JSON parsing fails, or image
      indexing fails.
  """
  if not contents:
    raise ValueError("contents must be a non-empty sequence.")
  if not isinstance(contents[-1], str):
    raise ValueError(
        "contents[-1] must be a JSON string representing the observations."
    )

  try:
    input_query = json.loads(contents[-1])
  except json.JSONDecodeError as e:
    raise ValueError(
        f"Failed to parse contents[-1] as JSON: {contents[-1]}"
    ) from e

  t_start = time.perf_counter()
  query: dict[str, Any] = {}
  for key, value in input_query.items():
    if key.startswith("images/"):
      image_content = contents[value]
      if isinstance(image_content, str):
        raise ValueError(
            f"Expected image content at index {value} for key {key}, got str."
        )
      image_bytes = _coerced_to_image_bytes(
          image_content,
          image_compression_jpeg_quality=image_compression_jpeg_quality,
      )
      if use_msgpack:
        query[key] = image_bytes
      else:
        query[key] = base64.b64encode(image_bytes).decode("utf-8")
    elif isinstance(value, (str, int, float, list)):
      query[key] = value
    else:
      raise ValueError(f"Unsupported value type: {type(value)} for key {key}.")

  encode_time_ms = (time.perf_counter() - t_start) * 1000.0

  if use_msgpack:
    payload_bytes = msgpack.packb(query, use_bin_type=True)
  else:
    payload_bytes = json.dumps(query).encode("utf-8")

  return EncodedObservationPayload(
      payload_bytes=payload_bytes, encode_time_ms=encode_time_ms
  )


def decode_action_chunk_payload(
    output_bytes: bytes,
    *,
    use_msgpack: bool = False,
) -> DecodedActionChunk:
  """Decodes action chunk from raw bytes (JSON or msgpack).

  Args:
    output_bytes: Raw response bytes from the server.
    use_msgpack: Whether to unpack using msgpack instead of JSON.

  Returns:
    DecodedActionChunk containing the numpy action chunk and response metadata.
  """
  data = (
      msgpack.unpackb(output_bytes, raw=False)
      if use_msgpack
      else json.loads(output_bytes)
  )

  if not isinstance(data, dict):
    raise ValueError(
        "Response data does not have a single object as root object, got"
        f" {type(data)}: not a dictionary"
    )

  action_chunk = data.get(constants.ACTION_CHUNK_RESPONSE_KEY)
  if action_chunk is None:
    raise ValueError(
        "Response JSON does not contain"
        f" '{constants.ACTION_CHUNK_RESPONSE_KEY}'"
    )
  action_dtype = data.get(constants.DTYPE_RESPONSE_KEY) or np.float64
  action_chunk = np.array(action_chunk, dtype=action_dtype)
  if action_chunk.ndim != 2:
    raise ValueError(
        "Action chunk has more than 2 dimensions or is not a 2D array, got"
        f" shape {action_chunk.shape}: must be a 2D array"
    )
  return DecodedActionChunk(action_chunk=action_chunk, metadata=data)
