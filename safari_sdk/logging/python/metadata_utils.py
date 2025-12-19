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

"""Utils for managing metadata."""

import dataclasses
import sys
from typing import Mapping, Sequence
from dm_env import specs
import numpy as np
from safari_sdk.logging.python import constants
from safari_sdk.protos.logging import codec_pb2
from safari_sdk.protos.logging import dtype_pb2
from safari_sdk.protos.logging import feature_specs_pb2
from safari_sdk.protos.logging import spec_pb2


@dataclasses.dataclass(frozen=True)
class PolicyEnvironmentMetadataParams:
  """Class for managing the policy environment metadata.

  Note that the dictionaries passed must be flattened.
  """

  jpeg_compression_keys: Sequence[str]
  observation_spec: Mapping[str, specs.Array]
  reward_spec: specs.Array | Mapping[str, specs.Array]
  discount_spec: specs.Array | Mapping[str, specs.Array]
  action_spec: specs.BoundedArray
  policy_extra_spec: Mapping[str, specs.Array]


def create_feature_specs_proto(
    params: PolicyEnvironmentMetadataParams,
) -> feature_specs_pb2.FeatureSpecs:
  """Generates the feature specs proto.

  Args:
    params: The policy environment metadata params.

  Returns:
    The feature specs proto.
  """
  observation_spec = {}
  reward_spec = {}
  discount_spec = {}
  action_spec = {}
  policy_extra_spec = {}

  # Process the observation spec
  for key, spec in params.observation_spec.items():
    codec = codec_pb2.CODEC_NONE

    if key in params.jpeg_compression_keys:
      codec = codec_pb2.CODEC_IMAGE_JPEG

    observation_spec[constants.OBSERVATION_KEY_TEMPLATE.format(key)] = (
        create_spec_proto(spec, codec)
    )

  # Process the reward spec
  if isinstance(params.reward_spec, specs.Array):
    reward_spec[constants.REWARD_KEY] = create_spec_proto(params.reward_spec)
  elif isinstance(params.reward_spec, Mapping):
    for key, spec in params.reward_spec.items():
      reward_spec[constants.REWARD_KEY_TEMPLATE.format(key)] = (
          create_spec_proto(spec)
      )

  # Process the discount spec
  if isinstance(params.discount_spec, specs.Array):
    discount_spec[constants.DISCOUNT_KEY] = create_spec_proto(
        params.discount_spec
    )
  elif isinstance(params.discount_spec, Mapping):
    for key, spec in params.discount_spec.items():
      discount_spec[constants.DISCOUNT_KEY_TEMPLATE.format(key)] = (
          create_spec_proto(spec)
      )

  # Process the action spec
  action_spec[constants.ACTION_KEY_PREFIX] = create_spec_proto(
      params.action_spec
  )

  # Process the policy extra spec
  for key, spec in params.policy_extra_spec.items():
    policy_extra_spec[constants.POLICY_EXTRA_KEY_TEMPLATE.format(key)] = (
        create_spec_proto(spec)
    )

  return feature_specs_pb2.FeatureSpecs(
      observation=observation_spec,
      reward=reward_spec,
      discount=discount_spec,
      action=action_spec,
      policy_extra_output=policy_extra_spec,
  )


def create_spec_proto(
    spec: specs.Array,
    codec: codec_pb2.Codec = codec_pb2.CODEC_NONE,
) -> spec_pb2.Spec:
  """Creates a Spec proto from an array spec.

  Args:
    spec: The array spec.
    codec: The codec to use for the spec.

  Returns:
    The spec proto.
  """

  spec_proto = spec_pb2.Spec()
  spec_proto.shape.extend(spec.shape)

  if isinstance(spec, specs.BoundedArray):
    spec_proto.minimum_values.extend(convert_spec_bound(spec.minimum))
    spec_proto.maximum_values.extend(convert_spec_bound(spec.maximum))

  spec_proto.dtype = create_dtype_proto(spec.dtype)
  spec_proto.codec = codec
  return spec_proto


def convert_spec_bound(bound: float | int | np.ndarray) -> list[float]:
  """Converts a bound value (min or max) from a spec to a list of floats.

  This function converts a bound value (min or max) from a spec into a list of
  floats. It handles scalar values (float, int) and numpy arrays. Additionally,
  it replaces infinite values with the maximum representable float value,
  ensuring that the bounds can be correctly represented in the proto.

  Args:
    bound: The bound value, which can be a float, int, or a numpy array.

  Returns:
    A list of floats representing the processed bound values.

  Raises:
    ValueError: If the bound type is not supported.
  """
  if np.isscalar(bound):
    values = [bound]
  elif isinstance(bound, np.ndarray):
    values = np.asarray(bound).flatten().tolist()
  else:
    raise ValueError(f'Unsupported bound type {type(bound)}')

  processed_values = []
  for val in values:
    if np.isinf(val):
      if val > 0:
        processed_values.append(sys.float_info.max)
      else:
        processed_values.append(-sys.float_info.max)
    else:
      # Ensure that the value is a float.
      processed_values.append(float(val))
  return processed_values


def create_dtype_proto(
    dtype: np.dtype,
) -> dtype_pb2.Dtype:
  """Creates a Dtype proto from a numpy dtype.

  Args:
    dtype: The numpy dtype.

  Returns:
    The dtype proto.

  Raises:
    ValueError: If the dtype is not supported.
  """
  if dtype == np.uint8:
    return dtype_pb2.DTYPE_UINT8
  elif dtype == np.uint16:
    return dtype_pb2.DTYPE_UINT16
  elif dtype == np.int32:
    return dtype_pb2.DTYPE_INT32
  elif dtype == np.int64:
    return dtype_pb2.DTYPE_INT64
  elif dtype == np.float32:
    return dtype_pb2.DTYPE_FLOAT32
  elif dtype == np.float64:
    return dtype_pb2.DTYPE_FLOAT64
  elif dtype == np.str_ or dtype == np.object_:
    return dtype_pb2.DTYPE_STRING

  raise ValueError(f'Unsupported dtype {dtype} used in spec.')
