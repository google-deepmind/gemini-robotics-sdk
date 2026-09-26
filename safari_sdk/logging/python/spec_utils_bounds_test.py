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

"""Round-trip tests for independently shaped lower and upper spec bounds."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_env import specs
import numpy as np
from safari_sdk.logging.python import data_validator
from safari_sdk.logging.python import session_metadata
from safari_sdk.logging.python import spec_utils
from safari_sdk.protos.logging import metadata_pb2
from safari_sdk.protos.logging import spec_pb2


class MixedSpecBoundsTest(parameterized.TestCase):

  @parameterized.product(
      shape=((3,), (2, 3), (2, 2, 3)),
      dtype=(np.float32, np.float64, np.int32, np.int64, np.uint8, np.uint16),
      scalar_minimum=(False, True),
  )
  def test_mixed_scalar_and_elementwise_bounds_roundtrip(
      self, shape, dtype, scalar_minimum
  ):
    entries = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    if scalar_minimum:
      minimum, maximum = 0, entries + 1
    else:
      minimum, maximum = entries, int(np.prod(shape)) + 5
    original = specs.BoundedArray(shape, dtype, minimum, maximum)
    minimum_before, maximum_before = original.minimum.copy(), original.maximum.copy()
    encoded = session_metadata.create_spec_proto(original).SerializeToString()
    proto = spec_pb2.Spec.FromString(encoded)
    restored = spec_utils.spec_from_proto(proto)
    self.assertIsInstance(restored, specs.BoundedArray)
    self.assertEqual(restored.shape, original.shape)
    self.assertEqual(restored.dtype, original.dtype)
    np.testing.assert_array_equal(
        np.broadcast_to(restored.minimum, shape),
        np.broadcast_to(original.minimum, shape),
    )
    np.testing.assert_array_equal(
        np.broadcast_to(restored.maximum, shape),
        np.broadcast_to(original.maximum, shape),
    )
    for bound in (original.minimum, original.maximum):
      restored.validate(np.broadcast_to(bound, shape).copy())
    np.testing.assert_array_equal(original.minimum, minimum_before)
    np.testing.assert_array_equal(original.maximum, maximum_before)
    self.assertEqual(proto.SerializeToString(), encoded)

  @parameterized.product(dtype=(np.float32, np.float64), scalar_minimum=(False, True))
  def test_scalar_infinity_and_finite_elementwise_bounds_roundtrip(
      self, dtype, scalar_minimum
  ):
    entries = np.arange(6, dtype=dtype).reshape(2, 3)
    original = specs.BoundedArray(
        (2, 3),
        dtype,
        -np.inf if scalar_minimum else entries,
        entries if scalar_minimum else np.inf,
    )
    proto = session_metadata.create_spec_proto(original)
    restored = spec_utils.spec_from_proto(proto)
    for actual, expected in (
        (restored.minimum, original.minimum),
        (restored.maximum, original.maximum),
    ):
      np.testing.assert_array_equal(
          np.broadcast_to(actual, original.shape),
          np.broadcast_to(expected, original.shape),
      )

  @parameterized.parameters(dict(shape=()), dict(shape=(1,)), dict(shape=(2, 3)))
  def test_matching_bound_shapes_keep_existing_roundtrip_behavior(self, shape):
    for per_element in (False, True):
      minimum = np.full(shape, -1.0) if per_element else -1.0
      maximum = np.full(shape, 1.0) if per_element else 1.0
      original = specs.BoundedArray(shape, np.float32, minimum, maximum)
      restored = spec_utils.spec_from_proto(
          session_metadata.create_spec_proto(original)
      )
      self.assertEqual(restored.shape, shape)
      np.testing.assert_array_equal(np.broadcast_to(restored.minimum, shape), minimum)
      np.testing.assert_array_equal(np.broadcast_to(restored.maximum, shape), maximum)

  def test_session_extraction_preserves_observation_and_action_bounds(self):
    observation = specs.BoundedArray((2, 3), np.float32, 0.0, np.ones((2, 3)))
    action = specs.BoundedArray((2, 3), np.float32, -np.ones((2, 3)), 2.0)
    session = metadata_pb2.Session()
    features = session.policy_environment_metadata.feature_specs
    features.observation['observation/positions'].CopyFrom(
        session_metadata.create_spec_proto(observation)
    )
    features.action['action'].CopyFrom(session_metadata.create_spec_proto(action))
    encoded = session.SerializeToString()
    timestep_spec, action_spec, _ = spec_utils.specs_from_session(
        metadata_pb2.Session.FromString(encoded)
    )
    self.assertEqual(timestep_spec.observation['positions'].shape, (2, 3))
    self.assertEqual(action_spec.shape, (2, 3))
    validator = data_validator.EpisodeValidator()
    self.assertEmpty(
        validator.validate_array(
            np.ones((2, 3), dtype=np.float32), action_spec, 0, 'action'
        )
    )
    errors = validator.validate_array(
        np.full((2, 3), 3.0, dtype=np.float32), action_spec, 0, 'action'
    )
    self.assertLen(errors, 1)
    self.assertEqual(errors[0].error_type, data_validator.ErrorType.BOUNDS_VIOLATION)
    self.assertEqual(session.SerializeToString(), encoded)


if __name__ == '__main__':
  absltest.main()
