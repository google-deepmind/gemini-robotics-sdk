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

"""Utilities for deserializing Safari SDK spec protos to dm_env specs.

This module provides the inverse of the serialization functions in
session_metadata.py. It converts spec_pb2.Spec protos back to dm_env
specs.Array objects, and extracts typed specs from Session metadata protos.

Typical usage:
    session = mcap_parser_utils.read_session_proto_data(mcap_file)
    timestep_spec, action_spec, policy_extra_spec = specs_from_session(session)
"""

import sys
from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np
from safari_sdk.logging.python import constants
from safari_sdk.protos.logging import dtype_pb2
from safari_sdk.protos.logging import metadata_pb2
from safari_sdk.protos.logging import spec_pb2

# Mapping from proto dtype enum to numpy dtype.
# Inverse of session_metadata.create_dtype_proto().
DTYPE_FROM_PROTO: dict[int, np.dtype] = {
    dtype_pb2.DTYPE_FLOAT32: np.dtype(np.float32),
    dtype_pb2.DTYPE_FLOAT64: np.dtype(np.float64),
    dtype_pb2.DTYPE_INT32: np.dtype(np.int32),
    dtype_pb2.DTYPE_INT64: np.dtype(np.int64),
    dtype_pb2.DTYPE_UINT8: np.dtype(np.uint8),
    dtype_pb2.DTYPE_UINT16: np.dtype(np.uint16),
    dtype_pb2.DTYPE_STRING: np.dtype(np.str_),
}


def spec_from_proto(spec_proto: spec_pb2.Spec) -> specs.Array:
  """Converts a Spec proto to a dm_env specs.Array.

  Inverse of session_metadata.create_spec_proto().

  Args:
    spec_proto: The spec proto to convert.

  Returns:
    A dm_env specs.Array, specs.BoundedArray, or specs.StringArray.

  Raises:
    ValueError: If the dtype is not supported.
  """
  dtype = DTYPE_FROM_PROTO.get(spec_proto.dtype)
  if dtype is None:
    raise ValueError(
        f"Unsupported dtype in spec proto: {spec_proto.dtype}. "
        f"Supported dtypes: {list(DTYPE_FROM_PROTO.keys())}"
    )

  shape = tuple(spec_proto.shape)

  if dtype == np.dtype(np.str_):
    return specs.StringArray(shape=shape)

  if spec_proto.minimum_values and spec_proto.maximum_values:
    minimum_64 = np.array(spec_proto.minimum_values, dtype=np.float64)
    maximum_64 = np.array(spec_proto.maximum_values, dtype=np.float64)

    # Replace sys.float_info.max sentinel values with np.inf.
    # session_metadata.convert_spec_bound() replaces inf with float_info.max.
    minimum_64 = np.where(
        np.abs(minimum_64) >= sys.float_info.max * 0.99,
        np.copysign(np.inf, minimum_64),
        minimum_64,
    )
    maximum_64 = np.where(
        np.abs(maximum_64) >= sys.float_info.max * 0.99,
        np.copysign(np.inf, maximum_64),
        maximum_64,
    )

    minimum = minimum_64.astype(dtype)
    maximum = maximum_64.astype(dtype)

    if shape and minimum.size > 1:
      minimum = minimum.reshape(shape)
      maximum = maximum.reshape(shape)
    elif not shape and minimum.size == 1:
      minimum = minimum.squeeze()
      maximum = maximum.squeeze()

    return specs.BoundedArray(
        shape=shape,
        dtype=dtype,
        minimum=minimum,
        maximum=maximum,
    )

  return specs.Array(shape=shape, dtype=dtype)


def strip_prefix(key: str, prefix: str) -> str:
  """Strips a key prefix (e.g., 'observation/') from a spec key.

  Args:
    key: The full key (e.g., 'observation/joint_positions').
    prefix: The prefix to strip (e.g., 'observation').

  Returns:
    The key with the prefix and separator stripped.
  """
  full_prefix = prefix + "/"
  if key.startswith(full_prefix):
    return key[len(full_prefix) :]
  return key


def specs_from_session(
    session: metadata_pb2.Session,
) -> tuple[
    gdmr_types.TimeStepSpec,
    specs.Array | dict[str, specs.Array],
    dict[str, specs.Array],
]:
  """Extracts dm_env specs from a Session proto's feature specs.

  Args:
    session: The Session proto containing policy_environment_metadata.

  Returns:
    A tuple of (timestep_spec, action_spec, policy_extra_spec) where
    timestep_spec is a gdmr_types.TimeStepSpec.

  Raises:
    ValueError: If the session has no observation specs.
  """
  feature_specs = session.policy_environment_metadata.feature_specs

  # Build observation spec (strip 'observation/' prefix).
  observation_spec = {}
  for key, spec_proto in feature_specs.observation.items():
    stripped_key = strip_prefix(key, constants.OBSERVATION_KEY_PREFIX)
    observation_spec[stripped_key] = spec_from_proto(spec_proto)

  if not observation_spec:
    raise ValueError("Session has no observation specs in feature_specs.")

  # Build reward spec: single key 'reward' or dict with 'reward/{key}'.
  reward_spec: specs.Array | dict[str, specs.Array]
  if (
      len(feature_specs.reward) == 1
      and constants.REWARD_KEY in feature_specs.reward
  ):
    reward_spec = spec_from_proto(feature_specs.reward[constants.REWARD_KEY])
  else:
    reward_spec = {
        strip_prefix(k, constants.REWARD_KEY): spec_from_proto(v)
        for k, v in feature_specs.reward.items()
    }

  # Build discount spec: single key 'discount' or dict with 'discount/{key}'.
  discount_spec: specs.Array | dict[str, specs.Array]
  if (
      len(feature_specs.discount) == 1
      and constants.DISCOUNT_KEY in feature_specs.discount
  ):
    discount_spec = spec_from_proto(
        feature_specs.discount[constants.DISCOUNT_KEY]
    )
  else:
    discount_spec = {
        strip_prefix(k, constants.DISCOUNT_KEY): spec_from_proto(v)
        for k, v in feature_specs.discount.items()
    }

  # Combine into TimeStepSpec.
  timestep_spec = gdmr_types.TimeStepSpec(
      step_type=gdmr_types.STEP_TYPE_SPEC,
      reward=reward_spec,
      discount=discount_spec,
      observation=observation_spec,
  )

  # Build action spec: single key 'action' or dict with 'action/{key}'.
  action_spec: specs.Array | dict[str, specs.Array]
  if (
      len(feature_specs.action) == 1
      and constants.ACTION_KEY_PREFIX in feature_specs.action
  ):
    action_spec = spec_from_proto(
        feature_specs.action[constants.ACTION_KEY_PREFIX]
    )
  else:
    action_spec = {
        strip_prefix(k, constants.ACTION_KEY_PREFIX): spec_from_proto(v)
        for k, v in feature_specs.action.items()
    }

  # Build policy extra spec (strip 'extra/policy_extra/' prefix).
  policy_extra_spec = {}
  for key, spec_proto in feature_specs.policy_extra_output.items():
    stripped_key = strip_prefix(key, constants.POLICY_EXTRA_PREFIX)
    policy_extra_spec[stripped_key] = spec_from_proto(spec_proto)

  return timestep_spec, action_spec, policy_extra_spec
