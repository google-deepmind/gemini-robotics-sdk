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

"""Tests for spec_utils module."""

import sys

from dm_env import specs
import numpy as np

from absl.testing import absltest
from safari_sdk.logging.python import constants
from safari_sdk.logging.python import spec_utils
from safari_sdk.protos.logging import dtype_pb2
from safari_sdk.protos.logging import metadata_pb2
from safari_sdk.protos.logging import spec_pb2


class DtypeFromProtoTest(absltest.TestCase):
  """Tests for the DTYPE_FROM_PROTO mapping."""

  def test_all_expected_dtypes_present(self):
    """Verifies all expected dtypes are in the mapping."""
    self.assertIn(dtype_pb2.DTYPE_FLOAT32, spec_utils.DTYPE_FROM_PROTO)
    self.assertIn(dtype_pb2.DTYPE_FLOAT64, spec_utils.DTYPE_FROM_PROTO)
    self.assertIn(dtype_pb2.DTYPE_INT32, spec_utils.DTYPE_FROM_PROTO)
    self.assertIn(dtype_pb2.DTYPE_INT64, spec_utils.DTYPE_FROM_PROTO)
    self.assertIn(dtype_pb2.DTYPE_UINT8, spec_utils.DTYPE_FROM_PROTO)
    self.assertIn(dtype_pb2.DTYPE_UINT16, spec_utils.DTYPE_FROM_PROTO)
    self.assertIn(dtype_pb2.DTYPE_STRING, spec_utils.DTYPE_FROM_PROTO)

  def test_dtype_values_correct(self):
    """Verifies the dtype values are correct numpy dtypes."""
    self.assertEqual(
        spec_utils.DTYPE_FROM_PROTO[dtype_pb2.DTYPE_FLOAT32],
        np.dtype(np.float32),
    )
    self.assertEqual(
        spec_utils.DTYPE_FROM_PROTO[dtype_pb2.DTYPE_INT64],
        np.dtype(np.int64),
    )
    self.assertEqual(
        spec_utils.DTYPE_FROM_PROTO[dtype_pb2.DTYPE_UINT8],
        np.dtype(np.uint8),
    )


class SpecFromProtoTest(absltest.TestCase):
  """Tests for spec_from_proto()."""

  def test_basic_array(self):
    """Converts a basic Spec proto to specs.Array."""
    proto = spec_pb2.Spec()
    proto.shape.extend([3, 4])
    proto.dtype = dtype_pb2.DTYPE_FLOAT32
    result = spec_utils.spec_from_proto(proto)
    self.assertIsInstance(result, specs.Array)
    self.assertNotIsInstance(result, specs.BoundedArray)
    self.assertEqual(result.shape, (3, 4))
    self.assertEqual(result.dtype, np.float32)

  def test_bounded_array(self):
    """Converts a Spec proto with bounds to specs.BoundedArray."""
    proto = spec_pb2.Spec()
    proto.shape.extend([2])
    proto.dtype = dtype_pb2.DTYPE_FLOAT64
    proto.minimum_values.extend([-1.0, -2.0])
    proto.maximum_values.extend([1.0, 2.0])
    result = spec_utils.spec_from_proto(proto)
    self.assertIsInstance(result, specs.BoundedArray)
    self.assertEqual(result.shape, (2,))
    self.assertEqual(result.dtype, np.float64)
    np.testing.assert_array_equal(result.minimum, [-1.0, -2.0])
    np.testing.assert_array_equal(result.maximum, [1.0, 2.0])

  def test_string_array(self):
    """Converts a Spec proto with string dtype to specs.StringArray."""
    proto = spec_pb2.Spec()
    proto.shape.extend([1])
    proto.dtype = dtype_pb2.DTYPE_STRING
    result = spec_utils.spec_from_proto(proto)
    self.assertIsInstance(result, specs.StringArray)
    self.assertEqual(result.shape, (1,))

  def test_unsupported_dtype_raises(self):
    """Raises ValueError for unsupported dtype."""
    proto = spec_pb2.Spec()
    proto.shape.extend([1])
    proto.dtype = (
        dtype_pb2.DTYPE_UNSPECIFIED
    )  # Unsupported (not in DTYPE_FROM_PROTO)
    with self.assertRaises(ValueError):
      spec_utils.spec_from_proto(proto)

  def test_inf_sentinel_roundtrip(self):
    """Verifies sys.float_info.max sentinel values are converted back to inf."""
    proto = spec_pb2.Spec()
    proto.shape.extend([2])
    proto.dtype = dtype_pb2.DTYPE_FLOAT64
    proto.minimum_values.extend([-sys.float_info.max, -1.0])
    proto.maximum_values.extend([sys.float_info.max, 1.0])
    result = spec_utils.spec_from_proto(proto)
    self.assertIsInstance(result, specs.BoundedArray)
    self.assertEqual(result.minimum[0], -np.inf)
    self.assertEqual(result.minimum[1], -1.0)
    self.assertEqual(result.maximum[0], np.inf)
    self.assertEqual(result.maximum[1], 1.0)

  def test_inf_sentinel_float32(self):
    """Verifies float32 sentinel values are converted to inf."""
    proto = spec_pb2.Spec()
    proto.shape.extend([2])
    proto.dtype = dtype_pb2.DTYPE_FLOAT32
    proto.minimum_values.extend([-sys.float_info.max, -0.5])
    proto.maximum_values.extend([sys.float_info.max, 0.5])
    result = spec_utils.spec_from_proto(proto)
    self.assertIsInstance(result, specs.BoundedArray)
    self.assertEqual(result.dtype, np.float32)
    self.assertEqual(result.minimum[0], -np.inf)
    self.assertEqual(result.maximum[0], np.inf)

  def test_finite_value_near_sentinel_preserved(self):
    """Verifies finite values below float_info.max * 0.99 are NOT converted to inf."""
    proto = spec_pb2.Spec()
    proto.shape.extend([1])
    proto.dtype = dtype_pb2.DTYPE_FLOAT64
    finite_val = 1e30
    proto.minimum_values.extend([-finite_val])
    proto.maximum_values.extend([finite_val])
    result = spec_utils.spec_from_proto(proto)
    self.assertIsInstance(result, specs.BoundedArray)
    self.assertTrue(np.isfinite(result.minimum[0]))
    self.assertTrue(np.isfinite(result.maximum[0]))
    self.assertEqual(result.minimum[0], -finite_val)
    self.assertEqual(result.maximum[0], finite_val)

  def test_scalar_shape(self):
    """Handles empty (scalar) shape correctly."""
    proto = spec_pb2.Spec()
    proto.dtype = dtype_pb2.DTYPE_INT32
    result = spec_utils.spec_from_proto(proto)
    self.assertIsInstance(result, specs.Array)
    self.assertEqual(result.shape, ())

  def test_scalar_bounded_array(self):
    """Handles scalar BoundedArray correctly (shape = ())."""
    proto = spec_pb2.Spec()
    proto.dtype = dtype_pb2.DTYPE_FLOAT32
    proto.minimum_values.extend([-1.0])
    proto.maximum_values.extend([1.0])
    result = spec_utils.spec_from_proto(proto)
    self.assertIsInstance(result, specs.BoundedArray)
    self.assertEqual(result.shape, ())
    self.assertEqual(result.minimum.shape, ())
    self.assertEqual(result.maximum.shape, ())
    self.assertEqual(result.minimum, np.float32(-1.0))
    self.assertEqual(result.maximum, np.float32(1.0))

  def test_bounded_array_both_minimum_and_maximum_values_provided(self):
    """Converts a Spec proto to specs.BoundedArray when both min and max values are provided."""
    proto = spec_pb2.Spec()
    proto.shape.extend([3])
    proto.dtype = dtype_pb2.DTYPE_FLOAT32
    proto.minimum_values.extend([0.0, 1.0, 2.0])
    proto.maximum_values.extend([5.0, 6.0, 7.0])
    result = spec_utils.spec_from_proto(proto)
    self.assertIsInstance(result, specs.BoundedArray)
    self.assertEqual(result.shape, (3,))
    np.testing.assert_array_equal(result.minimum, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(result.maximum, [5.0, 6.0, 7.0])

  def test_partial_bounds_returns_unbounded_array(self):
    """Returns specs.Array if only minimum_values or only maximum_values is provided."""
    proto_min_only = spec_pb2.Spec()
    proto_min_only.shape.extend([2])
    proto_min_only.dtype = dtype_pb2.DTYPE_FLOAT32
    proto_min_only.minimum_values.extend([0.0, 1.0])
    result_min = spec_utils.spec_from_proto(proto_min_only)
    self.assertIsInstance(result_min, specs.Array)
    self.assertNotIsInstance(result_min, specs.BoundedArray)

    proto_max_only = spec_pb2.Spec()
    proto_max_only.shape.extend([2])
    proto_max_only.dtype = dtype_pb2.DTYPE_FLOAT32
    proto_max_only.maximum_values.extend([5.0, 6.0])
    result_max = spec_utils.spec_from_proto(proto_max_only)
    self.assertIsInstance(result_max, specs.Array)
    self.assertNotIsInstance(result_max, specs.BoundedArray)


class StripPrefixTest(absltest.TestCase):
  """Tests for strip_prefix()."""

  def test_strips_matching_prefix(self):
    """Strips a matching prefix correctly."""
    self.assertEqual(
        spec_utils.strip_prefix("observation/joint_positions", "observation"),
        "joint_positions",
    )

  def test_strips_action_prefix(self):
    """Strips action prefix."""
    self.assertEqual(
        spec_utils.strip_prefix("action/joint_command", "action"),
        "joint_command",
    )

  def test_no_match_returns_unchanged(self):
    """Returns key unchanged when prefix does not match."""
    self.assertEqual(
        spec_utils.strip_prefix("reward", "observation"),
        "reward",
    )

  def test_prefix_without_slash_no_match(self):
    """Does not strip if key starts with prefix but without separator."""
    self.assertEqual(
        spec_utils.strip_prefix("observation_key", "observation"),
        "observation_key",
    )


class SpecsFromSessionTest(absltest.TestCase):
  """Tests for specs_from_session()."""

  def _make_session_with_specs(
      self,
      observation_specs=None,
      action_specs=None,
      reward_specs=None,
      discount_specs=None,
      policy_extra_specs=None,
  ):
    """Helper to create a Session proto with given feature specs."""
    session = metadata_pb2.Session()

    if observation_specs:
      for key, proto in observation_specs.items():
        session.policy_environment_metadata.feature_specs.observation[
            key
        ].CopyFrom(proto)

    if action_specs:
      for key, proto in action_specs.items():
        session.policy_environment_metadata.feature_specs.action[key].CopyFrom(
            proto
        )

    if reward_specs:
      for key, proto in reward_specs.items():
        session.policy_environment_metadata.feature_specs.reward[key].CopyFrom(
            proto
        )

    if discount_specs:
      for key, proto in discount_specs.items():
        session.policy_environment_metadata.feature_specs.discount[
            key
        ].CopyFrom(proto)

    if policy_extra_specs:
      for key, proto in policy_extra_specs.items():
        session.policy_environment_metadata.feature_specs.policy_extra_output[
            key
        ].CopyFrom(proto)

    return session

  def _make_spec_proto(self, shape, dtype, minimum=None, maximum=None):
    """Helper to create a Spec proto."""
    proto = spec_pb2.Spec()
    proto.shape.extend(shape)
    proto.dtype = dtype
    if minimum is not None:
      proto.minimum_values.extend(minimum)
    if maximum is not None:
      proto.maximum_values.extend(maximum)
    return proto

  def test_full_session(self):
    """Extracts all specs from a complete session."""
    obs_proto = self._make_spec_proto([6], dtype_pb2.DTYPE_FLOAT32)
    action_proto = self._make_spec_proto(
        [6], dtype_pb2.DTYPE_FLOAT32, [-1.0] * 6, [1.0] * 6
    )
    reward_proto = self._make_spec_proto([1], dtype_pb2.DTYPE_FLOAT64)

    session = self._make_session_with_specs(
        observation_specs={"observation/joint_positions": obs_proto},
        action_specs={"action/joint_command": action_proto},
        reward_specs={"reward": reward_proto},
    )

    timestep_spec, action_spec, policy_extra_spec = (
        spec_utils.specs_from_session(session)
    )

    # Check observation.
    self.assertIn("joint_positions", timestep_spec.observation)
    self.assertEqual(timestep_spec.observation["joint_positions"].shape, (6,))

    # Check action.
    self.assertIn("joint_command", action_spec)
    self.assertIsInstance(action_spec["joint_command"], specs.BoundedArray)

    # Check reward.
    self.assertIsInstance(timestep_spec.reward, specs.Array)

    # Check policy_extra is empty.
    self.assertEmpty(policy_extra_spec)

  def test_multidimensional_bounded_array(self):
    """Converts a multidimensional Spec proto with bounds."""
    proto = spec_pb2.Spec()
    proto.shape.extend([2, 3])
    proto.dtype = dtype_pb2.DTYPE_FLOAT32
    proto.minimum_values.extend([-1.0] * 6)
    proto.maximum_values.extend([1.0] * 6)
    result = spec_utils.spec_from_proto(proto)
    self.assertIsInstance(result, specs.BoundedArray)
    self.assertEqual(result.shape, (2, 3))
    self.assertEqual(result.minimum.shape, (2, 3))
    self.assertEqual(result.maximum.shape, (2, 3))

  def test_no_observations_raises(self):
    """Raises ValueError if session has no observation specs."""
    session = self._make_session_with_specs()
    with self.assertRaises(ValueError):
      spec_utils.specs_from_session(session)

  def test_empty_action_spec(self):
    """Handles session with observations but no actions."""
    obs_proto = self._make_spec_proto([3], dtype_pb2.DTYPE_FLOAT32)
    session = self._make_session_with_specs(
        observation_specs={"observation/joints": obs_proto},
    )
    timestep_spec, action_spec, policy_extra_spec = (
        spec_utils.specs_from_session(session)
    )
    self.assertIn("joints", timestep_spec.observation)
    self.assertEmpty(action_spec)
    self.assertEmpty(policy_extra_spec)

  def test_dict_reward_discount_action_and_policy_extra(self):
    """Handles session with dict reward, discount, action, and policy_extra."""
    obs_proto = self._make_spec_proto([3], dtype_pb2.DTYPE_FLOAT32)
    reward_proto1 = self._make_spec_proto([1], dtype_pb2.DTYPE_FLOAT32)
    reward_proto2 = self._make_spec_proto([1], dtype_pb2.DTYPE_FLOAT32)
    discount_proto1 = self._make_spec_proto([1], dtype_pb2.DTYPE_FLOAT32)
    discount_proto2 = self._make_spec_proto([1], dtype_pb2.DTYPE_FLOAT32)
    action_proto1 = self._make_spec_proto([2], dtype_pb2.DTYPE_FLOAT32)
    action_proto2 = self._make_spec_proto([2], dtype_pb2.DTYPE_FLOAT32)
    extra_proto = self._make_spec_proto([4], dtype_pb2.DTYPE_FLOAT32)

    session = self._make_session_with_specs(
        observation_specs={"observation/joints": obs_proto},
        reward_specs={"reward/r1": reward_proto1, "reward/r2": reward_proto2},
        discount_specs={
            "discount/d1": discount_proto1,
            "discount/d2": discount_proto2,
        },
        action_specs={
            "action/a1": action_proto1,
            "action/a2": action_proto2,
        },
        policy_extra_specs={"extra/policy_extra/pe1": extra_proto},
    )
    timestep_spec, action_spec, policy_extra_spec = (
        spec_utils.specs_from_session(session)
    )

    self.assertIsInstance(timestep_spec.reward, dict)
    self.assertIn("r1", timestep_spec.reward)
    self.assertIn("r2", timestep_spec.reward)

    self.assertIsInstance(timestep_spec.discount, dict)
    self.assertIn("d1", timestep_spec.discount)
    self.assertIn("d2", timestep_spec.discount)

    self.assertIsInstance(action_spec, dict)
    self.assertIn("a1", action_spec)
    self.assertIn("a2", action_spec)

    self.assertIn("pe1", policy_extra_spec)

  def test_single_key_discount_spec(self):
    """Extracts discount spec as a single Array when key is constants.DISCOUNT_KEY."""
    obs_proto = self._make_spec_proto([3], dtype_pb2.DTYPE_FLOAT32)
    discount_proto = self._make_spec_proto([1], dtype_pb2.DTYPE_FLOAT32)
    session = self._make_session_with_specs(
        observation_specs={"observation/joints": obs_proto},
        discount_specs={constants.DISCOUNT_KEY: discount_proto},
    )
    timestep_spec, _, _ = spec_utils.specs_from_session(session)
    self.assertIsInstance(timestep_spec.discount, specs.Array)
    self.assertNotIsInstance(timestep_spec.discount, dict)
    self.assertEqual(timestep_spec.discount.shape, (1,))

  def test_single_key_action_spec(self):
    """Extracts action spec as a single Array when key is constants.ACTION_KEY_PREFIX."""
    obs_proto = self._make_spec_proto([3], dtype_pb2.DTYPE_FLOAT32)
    action_proto = self._make_spec_proto(
        [6], dtype_pb2.DTYPE_FLOAT32, [-1.0] * 6, [1.0] * 6
    )
    session = self._make_session_with_specs(
        observation_specs={"observation/joints": obs_proto},
        action_specs={constants.ACTION_KEY_PREFIX: action_proto},
    )
    _, action_spec, _ = spec_utils.specs_from_session(session)
    self.assertIsInstance(action_spec, specs.BoundedArray)
    self.assertNotIsInstance(action_spec, dict)
    self.assertEqual(action_spec.shape, (6,))


if __name__ == "__main__":
  absltest.main()
