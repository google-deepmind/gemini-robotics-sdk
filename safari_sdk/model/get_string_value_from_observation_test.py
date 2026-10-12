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
from safari_sdk.model import get_string_value_from_observation


class GetStringValueFromObservationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('string', 'do the task'),
      ('bytes', b'do the task'),
      ('scalar_string_array', np.array('do the task')),
      ('scalar_bytes_array', np.array(b'do the task')),
      ('scalar_object_array', np.array('do the task', dtype=np.object_)),
      (
          'scalar_string_dtype_array',
          np.array('do the task', dtype=np.dtypes.StringDType()),
      ),
      ('1d_string_array', np.array(['do the task'])),
      ('1d_bytes_array', np.array([b'do the task'])),
  )
  def test_get_string_value_from_observation(self, string_obs):
    result = (
        get_string_value_from_observation.get_string_value_from_observation(
            string_obs=string_obs,
            string_obs_name='instruction',
        )
    )
    self.assertEqual(result, 'do the task')

  def test_invalid_observation_type_raises_error(self):
    with self.assertRaisesRegex(
        ValueError,
        'Observation for key instruction is not a string, bytes, or numpy'
        ' array.',
    ):
      get_string_value_from_observation.get_string_value_from_observation(
          string_obs=1,  # pyrefly: ignore[bad-argument-type]
          string_obs_name='instruction',
      )

  def test_invalid_numpy_element_type_raises_error(self):
    with self.assertRaisesRegex(
        ValueError,
        'Value 123 for key instruction is not a string or bytes.',
    ):
      get_string_value_from_observation.get_string_value_from_observation(
          string_obs=np.array(123),
          string_obs_name='instruction',
      )


if __name__ == '__main__':
  absltest.main()
