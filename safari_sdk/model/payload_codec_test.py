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

"""Unit tests for payload_codec."""

import base64
import json
import typing
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from google.genai import types
import msgpack
import numpy as np
import tensorflow as tf

from safari_sdk.model import payload_codec


class PayloadCodecTest(parameterized.TestCase):

  def test_coerced_to_image_bytes_numpy(self):
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img_bytes = payload_codec._coerced_to_image_bytes(img)
    self.assertTrue(img_bytes.startswith(b"\xff\xd8\xff"))

  def test_coerced_to_image_bytes_tensor(self):
    tensor = tf.zeros((10, 10, 3), dtype=tf.uint8)
    img_bytes = payload_codec._coerced_to_image_bytes(tensor)
    self.assertTrue(img_bytes.startswith(b"\xff\xd8\xff"))

  def test_coerced_to_image_bytes_raw_bytes(self):
    jpeg_bytes = b"\xff\xd8\xff\xe0mock_jpeg"
    png_bytes = b"\x89PNGmock_png"
    self.assertEqual(
        payload_codec._coerced_to_image_bytes(jpeg_bytes), jpeg_bytes
    )
    self.assertEqual(
        payload_codec._coerced_to_image_bytes(png_bytes), png_bytes
    )

  def test_coerced_to_image_bytes_part(self):
    part = types.Part(
        inline_data=types.Blob(
            mime_type="image/jpeg", data=b"\xff\xd8\xff\xe0mock"
        )
    )
    self.assertEqual(
        payload_codec._coerced_to_image_bytes(part),
        b"\xff\xd8\xff\xe0mock",
    )

  def test_coerced_to_image_bytes_errors(self):
    with self.assertRaisesRegex(ValueError, "Invalid PNG or JPEG image bytes"):
      payload_codec._coerced_to_image_bytes(b"invalid_bytes")

    part_invalid = types.Part(
        inline_data=types.Blob(mime_type="text/plain", data=b"text")
    )
    with self.assertRaisesRegex(ValueError, "Unsupported image mime type"):
      payload_codec._coerced_to_image_bytes(part_invalid)

    with self.assertRaisesRegex(ValueError, "Unsupported image type"):
      payload_codec._coerced_to_image_bytes(typing.cast(Any, 12345))

  def test_encode_observation_payload_json(self):
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    meta = json.dumps({
        "images/cam_0": 0,
        "instruction": "pick up cup",
        "joints": [0.1, 0.2],
    })
    contents = [img, meta]

    encoded = payload_codec.encode_observation_payload(
        contents, image_compression_jpeg_quality=80, use_msgpack=False
    )
    self.assertIsInstance(encoded, payload_codec.EncodedObservationPayload)
    self.assertGreaterEqual(encoded.encode_time_ms, 0.0)

    parsed = json.loads(encoded.payload_bytes.decode("utf-8"))
    self.assertIn("images/cam_0", parsed)
    self.assertIsInstance(parsed["images/cam_0"], str)
    decoded_img = base64.b64decode(parsed["images/cam_0"])
    self.assertTrue(decoded_img.startswith(b"\xff\xd8\xff"))
    self.assertEqual(parsed["instruction"], "pick up cup")
    self.assertEqual(parsed["joints"], [0.1, 0.2])

  def test_encode_observation_payload_msgpack(self):
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    meta = json.dumps({"images/cam_0": 0, "val": 1.0})
    contents = [img, meta]

    encoded = payload_codec.encode_observation_payload(
        contents, use_msgpack=True
    )
    self.assertIsInstance(encoded, payload_codec.EncodedObservationPayload)
    self.assertGreaterEqual(encoded.encode_time_ms, 0.0)

    unpacked = msgpack.unpackb(encoded.payload_bytes, raw=False)
    self.assertIn("images/cam_0", unpacked)
    self.assertIsInstance(unpacked["images/cam_0"], bytes)
    self.assertTrue(unpacked["images/cam_0"].startswith(b"\xff\xd8\xff"))
    self.assertEqual(unpacked["val"], 1.0)

  def test_encode_observation_payload_errors(self):
    with self.assertRaisesRegex(ValueError, "non-empty sequence"):
      payload_codec.encode_observation_payload([])

    with self.assertRaisesRegex(ValueError, "must be a JSON string"):
      payload_codec.encode_observation_payload(typing.cast(Any, [123]))

    with self.assertRaisesRegex(ValueError, "Failed to parse contents"):
      payload_codec.encode_observation_payload(["not_json{"])

    with self.assertRaisesRegex(ValueError, "Unsupported value type"):
      payload_codec.encode_observation_payload([json.dumps({"k": None})])

    with self.assertRaisesRegex(
        ValueError, "Expected image content at index 0"
    ):
      payload_codec.encode_observation_payload(
          ["not_an_image", json.dumps({"images/cam": 0})]
      )

  def test_exported_symbols(self):
    self.assertIn("ImageContent", payload_codec.__all__)
    self.assertIn("RoboticsContentItem", payload_codec.__all__)
    self.assertIn("EncodedObservationPayload", payload_codec.__all__)
    self.assertIn("DecodedActionChunk", payload_codec.__all__)

  def test_decode_action_chunk_payload_json(self):
    data = {"action_chunk": [[1.0, 2.0], [3.0, 4.0]], "rng_key": [10, 20]}
    payload = json.dumps(data).encode("utf-8")
    decoded = payload_codec.decode_action_chunk_payload(
        payload, use_msgpack=False
    )
    self.assertIsInstance(decoded, payload_codec.DecodedActionChunk)
    np.testing.assert_array_equal(
        decoded.action_chunk, np.array([[1.0, 2.0], [3.0, 4.0]])
    )
    self.assertEqual(decoded.metadata["rng_key"], [10, 20])

  def test_decode_action_chunk_payload_msgpack(self):
    data = {"action_chunk": [[5.0, 6.0]], "rng_key": [99]}
    payload = msgpack.packb(data, use_bin_type=True)
    decoded = payload_codec.decode_action_chunk_payload(
        payload, use_msgpack=True
    )
    self.assertIsInstance(decoded, payload_codec.DecodedActionChunk)
    np.testing.assert_array_equal(decoded.action_chunk, np.array([[5.0, 6.0]]))
    self.assertEqual(decoded.metadata["rng_key"], [99])

  def test_decode_action_chunk_errors(self):
    with self.assertRaisesRegex(ValueError, "not a dictionary"):
      payload_codec.decode_action_chunk_payload(b"123")

    with self.assertRaisesRegex(
        ValueError, "Response data does not have a single object as root object"
    ):
      payload_codec.decode_action_chunk_payload(
          json.dumps([1, 2, 3]).encode("utf-8")
      )

    invalid_missing = json.dumps({"foo": "bar"}).encode("utf-8")
    with self.assertRaisesRegex(ValueError, "does not contain 'action_chunk'"):
      payload_codec.decode_action_chunk_payload(invalid_missing)

    invalid_1d = json.dumps({"action_chunk": [1.0, 2.0]}).encode("utf-8")
    with self.assertRaisesRegex(ValueError, "must be a 2D array"):
      payload_codec.decode_action_chunk_payload(invalid_1d)

    with self.assertRaisesRegex(
        ValueError, "Action chunk has more than 2 dimensions"
    ):
      invalid_3d = json.dumps({"action_chunk": [[[1.0], [2.0]]]}).encode(
          "utf-8"
      )
      payload_codec.decode_action_chunk_payload(invalid_3d)


if __name__ == "__main__":
  absltest.main()
