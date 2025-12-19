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

import sys
from dm_env import specs
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from safari_sdk.logging.python import constants
from safari_sdk.logging.python import metadata_utils
from safari_sdk.protos.logging import codec_pb2
from safari_sdk.protos.logging import dtype_pb2


class MetadataUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="uint8",
          dtype=np.uint8,
          expected_dtype=dtype_pb2.DTYPE_UINT8,
      ),
      dict(
          testcase_name="uint16",
          dtype=np.uint16,
          expected_dtype=dtype_pb2.DTYPE_UINT16,
      ),
      dict(
          testcase_name="int32",
          dtype=np.int32,
          expected_dtype=dtype_pb2.DTYPE_INT32,
      ),
      dict(
          testcase_name="int64",
          dtype=np.int64,
          expected_dtype=dtype_pb2.DTYPE_INT64,
      ),
      dict(
          testcase_name="float32",
          dtype=np.float32,
          expected_dtype=dtype_pb2.DTYPE_FLOAT32,
      ),
      dict(
          testcase_name="float64",
          dtype=np.float64,
          expected_dtype=dtype_pb2.DTYPE_FLOAT64,
      ),
      dict(
          testcase_name="string",
          dtype=np.str_,
          expected_dtype=dtype_pb2.DTYPE_STRING,
      ),
      dict(
          testcase_name="object",
          dtype=np.object_,
          expected_dtype=dtype_pb2.DTYPE_STRING,
      ),
  )
  def test_create_dtype_proto_for_valid_dtypes(self, dtype, expected_dtype):
    self.assertEqual(metadata_utils.create_dtype_proto(dtype), expected_dtype)

  @parameterized.named_parameters(
      dict(
          testcase_name="uint32",
          dtype=np.uint32,
      ),
      dict(
          testcase_name="uint64",
          dtype=np.uint64,
      ),
      dict(
          testcase_name="int16",
          dtype=np.int16,
      ),
      dict(
          testcase_name="int8",
          dtype=np.int8,
      ),
      dict(
          testcase_name="bool",
          dtype=np.bool_,
      ),
  )
  def test_create_dtype_proto_raises_error_for_invalid_dtype(self, dtype):
    with self.assertRaises(ValueError):
      metadata_utils.create_dtype_proto(dtype)

  @parameterized.named_parameters(
      dict(
          testcase_name="float",
          bound=1.0,
          expected_values=[1.0],
      ),
      dict(
          testcase_name="int",
          bound=1,
          expected_values=[1.0],
      ),
      dict(
          testcase_name="positive_infinity",
          bound=np.inf,
          expected_values=[sys.float_info.max],
      ),
      dict(
          testcase_name="negative_infinity",
          bound=-np.inf,
          expected_values=[-sys.float_info.max],
      ),
  )
  def test_convert_spec_bound_for_scalar_value(self, bound, expected_values):
    self.assertEqual(metadata_utils.convert_spec_bound(bound), expected_values)

  @parameterized.named_parameters(
      dict(
          testcase_name="numpy_array",
          bound=np.array([1.0, 2.0, 3.0]),
          expected_values=[1.0, 2.0, 3.0],
      ),
      dict(
          testcase_name="numpy_array_with_infinity",
          bound=np.array([1.0, np.inf, 3.0, -np.inf]),
          expected_values=[1.0, sys.float_info.max, 3.0, -sys.float_info.max],
      ),
  )
  def test_convert_spec_bound_for_numpy_array(self, bound, expected_values):
    self.assertEqual(metadata_utils.convert_spec_bound(bound), expected_values)

  def test_creates_spec_proto_for_array_spec(self):
    array_spec = specs.Array(shape=(1, 2, 3), dtype=np.float32)
    spec_proto = metadata_utils.create_spec_proto(
        array_spec, codec_pb2.CODEC_NONE
    )
    self.assertSequenceEqual(spec_proto.shape, array_spec.shape)
    self.assertEqual(spec_proto.dtype, dtype_pb2.DTYPE_FLOAT32)
    self.assertEqual(spec_proto.codec, codec_pb2.CODEC_NONE)
    self.assertEmpty(spec_proto.minimum_values)
    self.assertEmpty(spec_proto.maximum_values)

  def test_creates_spec_proto_for_bounded_array_spec(self):
    bounded_array_spec = specs.BoundedArray(
        shape=(1, 2, 3),
        dtype=np.float32,
        minimum=np.array([1.0, 2.0, 3.0]),
        maximum=np.array([4.0, 5.0, 6.0]),
    )
    spec_proto = metadata_utils.create_spec_proto(
        bounded_array_spec, codec_pb2.CODEC_NONE
    )
    self.assertSequenceEqual(spec_proto.shape, bounded_array_spec.shape)
    self.assertEqual(spec_proto.dtype, dtype_pb2.DTYPE_FLOAT32)
    self.assertEqual(spec_proto.codec, codec_pb2.CODEC_NONE)
    self.assertSequenceEqual(spec_proto.minimum_values, [1.0, 2.0, 3.0])
    self.assertSequenceEqual(spec_proto.maximum_values, [4.0, 5.0, 6.0])

  def test_create_feature_specs_proto_with_array_specs(self):

    params = metadata_utils.PolicyEnvironmentMetadataParams(
        jpeg_compression_keys=["observation1"],
        observation_spec={
            "observation1": specs.Array(shape=(1, 2, 3), dtype=np.float32),
            "observation2": specs.Array(shape=(4, 5), dtype=np.int32),
        },
        reward_spec=specs.Array(shape=(), dtype=np.float32),
        discount_spec=specs.Array(shape=(), dtype=np.float32),
        action_spec=specs.BoundedArray(
            shape=(1, 2, 3),
            dtype=np.float32,
            minimum=np.array([1.0, 2.0, 3.0]),
            maximum=np.array([4.0, 5.0, 6.0]),
        ),
        policy_extra_spec={
            "extra1": specs.Array(shape=(1, 2, 3), dtype=np.float32),
            "extra2": specs.Array(shape=(4, 5), dtype=np.int32),
        },
    )

    spec_proto = metadata_utils.create_feature_specs_proto(params)

    print(spec_proto)

    self.assertSequenceEqual(
        spec_proto.observation[
            constants.OBSERVATION_KEY_TEMPLATE.format("observation1")
        ].shape,
        [1, 2, 3],
    )
    self.assertEqual(
        spec_proto.observation[
            constants.OBSERVATION_KEY_TEMPLATE.format("observation1")
        ].dtype,
        dtype_pb2.DTYPE_FLOAT32,
    )
    self.assertEqual(
        spec_proto.observation[
            constants.OBSERVATION_KEY_TEMPLATE.format("observation1")
        ].codec,
        codec_pb2.CODEC_IMAGE_JPEG,
    )
    self.assertSequenceEqual(
        spec_proto.observation[
            constants.OBSERVATION_KEY_TEMPLATE.format("observation2")
        ].shape,
        [4, 5],
    )
    self.assertEqual(
        spec_proto.observation[
            constants.OBSERVATION_KEY_TEMPLATE.format("observation2")
        ].dtype,
        dtype_pb2.DTYPE_INT32,
    )
    # observation2 is not in the jpeg_compression_keys, so it should not be
    # compressed.
    self.assertEqual(
        spec_proto.observation[
            constants.OBSERVATION_KEY_TEMPLATE.format("observation2")
        ].codec,
        codec_pb2.CODEC_NONE,
    )

    self.assertSequenceEqual(spec_proto.reward[constants.REWARD_KEY].shape, [])
    self.assertEqual(
        spec_proto.reward[constants.REWARD_KEY].dtype, dtype_pb2.DTYPE_FLOAT32
    )
    self.assertEqual(
        spec_proto.reward[constants.REWARD_KEY].codec, codec_pb2.CODEC_NONE
    )

    self.assertSequenceEqual(
        spec_proto.discount[constants.DISCOUNT_KEY].shape, []
    )
    self.assertEqual(
        spec_proto.discount[constants.DISCOUNT_KEY].dtype,
        dtype_pb2.DTYPE_FLOAT32,
    )
    self.assertEqual(
        spec_proto.discount[constants.DISCOUNT_KEY].codec, codec_pb2.CODEC_NONE
    )

    self.assertSequenceEqual(
        spec_proto.action[constants.ACTION_KEY_PREFIX].shape, [1, 2, 3]
    )
    self.assertEqual(
        spec_proto.action[constants.ACTION_KEY_PREFIX].dtype,
        dtype_pb2.DTYPE_FLOAT32,
    )
    self.assertEqual(
        spec_proto.action[constants.ACTION_KEY_PREFIX].codec,
        codec_pb2.CODEC_NONE,
    )

    self.assertSequenceEqual(
        spec_proto.policy_extra_output[
            constants.POLICY_EXTRA_KEY_TEMPLATE.format("extra1")
        ].shape,
        [1, 2, 3],
    )
    self.assertEqual(
        spec_proto.policy_extra_output[
            constants.POLICY_EXTRA_KEY_TEMPLATE.format("extra1")
        ].dtype,
        dtype_pb2.DTYPE_FLOAT32,
    )
    self.assertEqual(
        spec_proto.policy_extra_output[
            constants.POLICY_EXTRA_KEY_TEMPLATE.format("extra1")
        ].codec,
        codec_pb2.CODEC_NONE,
    )

    self.assertSequenceEqual(
        spec_proto.policy_extra_output[
            constants.POLICY_EXTRA_KEY_TEMPLATE.format("extra2")
        ].shape,
        [4, 5],
    )
    self.assertEqual(
        spec_proto.policy_extra_output[
            constants.POLICY_EXTRA_KEY_TEMPLATE.format("extra2")
        ].dtype,
        dtype_pb2.DTYPE_INT32,
    )
    self.assertEqual(
        spec_proto.policy_extra_output[
            constants.POLICY_EXTRA_KEY_TEMPLATE.format("extra2")
        ].codec,
        codec_pb2.CODEC_NONE,
    )

  def test_create_feature_specs_proto_with_mapping_of_array_specs(self):
    params = metadata_utils.PolicyEnvironmentMetadataParams(
        jpeg_compression_keys=["observation1"],
        observation_spec={
            "observation1": specs.Array(shape=(1, 2, 3), dtype=np.float32),
        },
        reward_spec={"reward1": specs.Array(shape=(), dtype=np.float32)},
        discount_spec={
            "discount1": specs.Array(shape=(), dtype=np.float32),
        },
        action_spec=specs.BoundedArray(
            shape=(1, 2, 3),
            dtype=np.float32,
            minimum=np.array([1.0, 2.0, 3.0]),
            maximum=np.array([4.0, 5.0, 6.0]),
        ),
        policy_extra_spec={
            "extra1": specs.Array(shape=(1, 2, 3), dtype=np.float32),
        },
    )
    spec_proto = metadata_utils.create_feature_specs_proto(params)

    self.assertSequenceEqual(
        spec_proto.reward[
            constants.REWARD_KEY_TEMPLATE.format("reward1")
        ].shape,
        [],
    )
    self.assertEqual(
        spec_proto.reward[
            constants.REWARD_KEY_TEMPLATE.format("reward1")
        ].dtype,
        dtype_pb2.DTYPE_FLOAT32,
    )
    self.assertEqual(
        spec_proto.reward[
            constants.REWARD_KEY_TEMPLATE.format("reward1")
        ].codec,
        codec_pb2.CODEC_NONE,
    )

    self.assertSequenceEqual(
        spec_proto.discount[
            constants.DISCOUNT_KEY_TEMPLATE.format("discount1")
        ].shape,
        [],
    )
    self.assertEqual(
        spec_proto.discount[
            constants.DISCOUNT_KEY_TEMPLATE.format("discount1")
        ].dtype,
        dtype_pb2.DTYPE_FLOAT32,
    )
    self.assertEqual(
        spec_proto.discount[
            constants.DISCOUNT_KEY_TEMPLATE.format("discount1")
        ].codec,
        codec_pb2.CODEC_NONE,
    )

    self.assertSequenceEqual(
        spec_proto.action[constants.ACTION_KEY_PREFIX].shape,
        [1, 2, 3],
    )


if __name__ == "__main__":
  absltest.main()
