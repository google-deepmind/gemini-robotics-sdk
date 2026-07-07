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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from safari_sdk.model import action_chunk_resampling


class ActionChunkResamplingTest(parameterized.TestCase):

  def test_linear_resample_even_length(self):
    # Action chunk with 6 steps, 1 action dimension: [0, 1, 2, 3, 4, 5]
    action_chunk = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    resampled = action_chunk_resampling.apply_linear_resampling(
        action_chunk, resample_factor=2.0
    )
    # With linspace, the actions are resampled to span exactly from index 0
    # to index 5 over 3 steps: np.linspace(0, 5, 3) = [0.0, 2.5, 5.0].
    expected = np.array([[0.0], [2.5], [5.0]])
    np.testing.assert_array_almost_equal(resampled, expected)

  def test_linear_resample_odd_length(self):
    action_chunk = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    resampled = action_chunk_resampling.apply_linear_resampling(
        action_chunk, resample_factor=2.0
    )
    # For odd length 5, np.linspace(0, 4, 3) = [0.0, 2.0, 4.0]
    expected = np.array([[0.0], [2.0], [4.0]])
    np.testing.assert_array_almost_equal(resampled, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name="empty_chunk",
          action_chunk=np.zeros((0, 3)),
      ),
      dict(
          testcase_name="single_element_chunk",
          action_chunk=np.array([[42.0, 43.0]]),
      ),
  )
  def test_resample_one_or_less(self, action_chunk):
    resampled = action_chunk_resampling.apply_linear_resampling(
        action_chunk, resample_factor=2.0
    )
    np.testing.assert_array_equal(resampled, action_chunk)

  def test_linear_resample_raises_if_not_2d(self):
    action_chunk = np.array([1.0, 2.0, 3.0])
    with self.assertRaisesRegex(ValueError, "action_chunk must be 2D"):
      action_chunk_resampling.apply_linear_resampling(
          action_chunk, resample_factor=2.0
      )

  def test_linear_resample_preserves_dtype(self):
    action_chunk = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    resampled = action_chunk_resampling.apply_linear_resampling(
        action_chunk, resample_factor=2.0
    )
    self.assertEqual(resampled.dtype, np.float32)

  @parameterized.named_parameters(
      dict(
          testcase_name="no_resampling",
          resampled_length=4,
          resample_factor=1.0,
          expected_input_length=4,
      ),
      dict(
          testcase_name="resample_by_2",
          resampled_length=3,
          resample_factor=2.0,
          expected_input_length=5,
      ),
      dict(
          testcase_name="resample_by_1_5",
          resampled_length=4,
          resample_factor=1.5,
          expected_input_length=5,
      ),
      dict(
          testcase_name="zero_length",
          resampled_length=0,
          resample_factor=2.0,
          expected_input_length=0,
      ),
  )
  def test_resampled_to_min_source_length(
      self, resampled_length, resample_factor, expected_input_length
  ):
    self.assertEqual(
        action_chunk_resampling.resampled_to_min_source_length(
            resampled_length, resample_factor
        ),
        expected_input_length,
    )


if __name__ == "__main__":
  absltest.main()
