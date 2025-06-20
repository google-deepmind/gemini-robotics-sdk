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

import tempfile

from numpy import testing as np_testing
import tensorflow as tf

from absl.testing import absltest
from absl.testing import parameterized
from safari_sdk.model import saved_model_policy
from safari_sdk.model import tf_agents_interface as tfa_interface


class FakeModel(tf.Module):

  def __init__(self, api_version: int | None = 0, name: str | None = None):
    super().__init__(name=name)
    self._api_version = api_version

  observation_spec = {
      'cam': tf.TensorSpec(shape=(1, 480, 640, 3), dtype=tf.uint8, name='cam'),
      'joints_pos': tf.TensorSpec(
          shape=(1, 32), dtype=tf.float32, name='joints_pos'
      ),
  }
  policy_state_spec = {
      'rng': tf.TensorSpec(shape=(2,), dtype=tf.uint32, name='rng'),
      'step_num': tf.TensorSpec(shape=(1,), dtype=tf.int32, name='step_num'),
  }

  @tf.function(
      input_signature=(
          tfa_interface.time_step_spec(observation_spec),
          policy_state_spec,
      )
  )
  def action(
      self,
      time_step: tfa_interface.TimeStep,
      policy_state: tfa_interface.NestedArray,
  ) -> tfa_interface.PolicyStep:
    del time_step
    new_policy_state = {
        'rng': policy_state['rng'] + 1,
        'step_num': policy_state['step_num'] + 1,
    }

    act = tf.cast(new_policy_state['step_num'], tf.float32) * tf.ones(
        shape=(32,), dtype=tf.float32, name='action'
    )
    return tfa_interface.PolicyStep(
        action=act,
        state=new_policy_state,
        info=(),
    )

  @tf.function
  def get_initial_state(self) -> tfa_interface.NestedArray:
    return tf.nest.map_structure(
        lambda x: tf.zeros(x.shape, x.dtype), self.policy_state_spec
    )

  @tf.function
  def get_metadata(self) -> dict[str, tf.Tensor]:
    metadata = {}
    if self._api_version is not None:
      metadata['api_version'] = tf.constant(self._api_version)

    return metadata


class SavedModelPolicyTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    model = FakeModel()
    self.temp_dir = tempfile.TemporaryDirectory()
    self.saved_model_path = self.temp_dir.name
    tf.saved_model.save(model, self.saved_model_path)

  def tearDown(self):
    super().tearDown()
    self.temp_dir.cleanup()

  def test_can_step(self):
    loaded_policy = saved_model_policy.SavedModelPolicy(
        path=self.saved_model_path,
    )
    loaded_policy.setup()
    observation_0 = loaded_policy.reset()
    for i in range(10):
      action = loaded_policy.step(observation_0)
      np_testing.assert_allclose(action, i + 1)

  @parameterized.parameters(
      None,
      saved_model_policy._MIN_SUPPORTED_API_VERSION - 1,
      saved_model_policy._MAX_SUPPORTED_API_VERSION + 1,
  )
  def test_rejects_missing_or_incompatible_api_version(self, api_version):
    with self.assertRaises(RuntimeError):
      model = FakeModel(api_version=api_version)
      with tempfile.TemporaryDirectory() as td:
        tf.saved_model.save(model, td)
        loaded_policy = saved_model_policy.SavedModelPolicy(
            path=td,
        )
        loaded_policy.setup()

  def test_must_call_setup(self):
    loaded_policy = saved_model_policy.SavedModelPolicy(
        path=self.saved_model_path,
    )

    with self.assertRaises(RuntimeError):
      _ = loaded_policy.observation_spec

    with self.assertRaises(RuntimeError):
      loaded_policy.reset()

    with self.assertRaises(RuntimeError):
      loaded_policy.step({})


if __name__ == '__main__':
  absltest.main()
