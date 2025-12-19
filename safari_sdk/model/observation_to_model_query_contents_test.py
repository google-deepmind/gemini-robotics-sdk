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

import json

import numpy as np

from absl.testing import absltest
from safari_sdk.model import constants
from safari_sdk.model import observation_to_model_query_contents


class ObservationToModelQueryContentsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.task_instruction_key = 'instruction'
    self.string_obs_key = 'string_obs'
    self.proprio_obs_key = 'proprio'
    self.image_obs_key = 'image'

  def _call_converter(
      self,
      observation,
      model_output=np.array([]),
      string_observations_keys=('instruction', 'string_obs'),
      task_instruction_key='instruction',
      proprioceptive_observation_keys=('proprio',),
      image_observation_keys=('image',),
      inference_mode=constants.InferenceMode.ASYNCHRONOUS,
  ):
    return observation_to_model_query_contents._observation_to_model_query_contents(
        observation,
        model_output,
        string_observations_keys,
        task_instruction_key,
        proprioceptive_observation_keys,
        image_observation_keys,
        inference_mode,
    )

  def test_async_with_model_output(self):
    observation = {
        self.task_instruction_key: np.array('do the task'),
        self.string_obs_key: np.array('some string'),
        self.proprio_obs_key: np.array([1.0, 2.0]),
        self.image_obs_key: np.zeros((64, 64, 3), dtype=np.uint8),
    }
    model_output = np.array([3.0, 4.0])

    images, encoded_obs = self._call_converter(
        observation,
        model_output=model_output,
        inference_mode=constants.InferenceMode.ASYNCHRONOUS,
    )

    self.assertLen(images, 1)
    np.testing.assert_array_equal(images[0], observation[self.image_obs_key])

    self.assertIn('conditioning', encoded_obs)
    self.assertEqual(encoded_obs['conditioning'], model_output.tolist())
    self.assertEqual(
        encoded_obs['task_instruction'],
        np.array_str(observation[self.task_instruction_key]),
    )
    self.assertEqual(
        encoded_obs[self.string_obs_key],
        np.array_str(observation[self.string_obs_key]),
    )
    self.assertEqual(
        encoded_obs[self.proprio_obs_key],
        observation[self.proprio_obs_key].tolist(),
    )
    self.assertEqual(
        encoded_obs[f'{constants.IMAGE_ENCODED_OBS_PREFIX}image'], 0
    )

  def test_sync_mode_ignores_model_output(self):
    observation = {
        self.task_instruction_key: np.array('do the task'),
        self.proprio_obs_key: np.array([1.0, 2.0]),
        self.image_obs_key: np.zeros((64, 64, 3), dtype=np.uint8),
    }
    model_output = np.array([3.0, 4.0])

    _, encoded_obs = self._call_converter(
        observation,
        model_output=model_output,
        string_observations_keys=[self.task_instruction_key],
        task_instruction_key=self.task_instruction_key,
        proprioceptive_observation_keys=[self.proprio_obs_key],
        image_observation_keys=[self.image_obs_key],
        inference_mode=constants.InferenceMode.SYNCHRONOUS,
    )

    self.assertNotIn('conditioning', encoded_obs)

  def test_async_with_empty_model_output(self):
    observation = {
        self.task_instruction_key: np.array('do the task'),
        self.proprio_obs_key: np.array([1.0, 2.0]),
        self.image_obs_key: np.zeros((64, 64, 3), dtype=np.uint8),
    }

    _, encoded_obs = self._call_converter(
        observation,
        model_output=np.array([]),
        string_observations_keys=[self.task_instruction_key],
        task_instruction_key=self.task_instruction_key,
        proprioceptive_observation_keys=[self.proprio_obs_key],
        image_observation_keys=[self.image_obs_key],
        inference_mode=constants.InferenceMode.ASYNCHRONOUS,
    )

    self.assertNotIn('conditioning', encoded_obs)

  def test_multiple_images(self):
    image_obs_key2 = 'image2'
    observation = {
        self.task_instruction_key: np.array('do the task'),
        self.proprio_obs_key: np.array([1.0, 2.0]),
        self.image_obs_key: np.zeros((64, 64, 3), dtype=np.uint8),
        image_obs_key2: np.ones((64, 64, 3), dtype=np.uint8),
    }

    images, encoded_obs = self._call_converter(
        observation,
        string_observations_keys=[self.task_instruction_key],
        image_observation_keys=[self.image_obs_key, image_obs_key2],
    )

    self.assertLen(images, 2)
    np.testing.assert_array_equal(images[0], observation[self.image_obs_key])
    np.testing.assert_array_equal(images[1], observation[image_obs_key2])
    self.assertEqual(
        encoded_obs[
            f'{constants.IMAGE_ENCODED_OBS_PREFIX}{self.image_obs_key}'
        ],
        0,
    )
    self.assertEqual(
        encoded_obs[f'{constants.IMAGE_ENCODED_OBS_PREFIX}{image_obs_key2}'], 1
    )

  def test_proprio_with_batch_dimension(self):
    observation = {
        self.proprio_obs_key: np.array([[1.0, 2.0]]),
    }
    _, encoded_obs = self._call_converter(
        observation,
        string_observations_keys=[],
        image_observation_keys=[],
    )
    self.assertEqual(encoded_obs[self.proprio_obs_key], [1.0, 2.0])

  def test_invalid_proprio_dims_raises_error(self):
    observation = {
        self.proprio_obs_key: np.zeros((1, 1, 1)),
    }
    with self.assertRaisesRegex(
        ValueError, 'Observation proprio has 3 dimensions, but should be 1.'
    ):
      self._call_converter(
          observation,
          string_observations_keys=[],
          image_observation_keys=[],
      )

  def test_image_with_batch_dimension(self):
    observation = {
        self.image_obs_key: np.zeros((1, 64, 64, 3), dtype=np.uint8),
    }
    images, _ = self._call_converter(
        observation,
        string_observations_keys=[],
        proprioceptive_observation_keys=[],
    )
    self.assertEqual(images[0].shape, (64, 64, 3))

  def test_invalid_image_dims_raises_error(self):
    observation = {
        self.image_obs_key: np.zeros((64, 64), dtype=np.uint8),
    }
    with self.assertRaisesRegex(
        ValueError, 'Image image has 2 dimensions, but should be 3.'
    ):
      self._call_converter(
          observation,
          string_observations_keys=[],
          proprioceptive_observation_keys=[],
      )

  def test_image_as_bytes(self):
    image_bytes = b'some_image_bytes'
    observation = {
        self.image_obs_key: image_bytes,
    }
    images, _ = self._call_converter(
        observation,
        string_observations_keys=[],
        proprioceptive_observation_keys=[],
    )
    self.assertLen(images, 1)
    self.assertEqual(images[0], image_bytes)

  def test_public_function(self):
    observation = {
        self.task_instruction_key: np.array('do the task'),
        self.proprio_obs_key: np.array([1.0, 2.0]),
        self.image_obs_key: np.zeros((64, 64, 3), dtype=np.uint8),
    }
    contents = (
        observation_to_model_query_contents.observation_to_model_query_contents(
            observation,
            model_output=np.array([]),
            string_observations_keys=[self.task_instruction_key],
            task_instruction_key=self.task_instruction_key,
            proprioceptive_observation_keys=[self.proprio_obs_key],
            image_observation_keys=[self.image_obs_key],
            inference_mode=constants.InferenceMode.SYNCHRONOUS,
        )
    )

    self.assertLen(contents, 2)
    self.assertIsInstance(contents[0], np.ndarray)
    self.assertIsInstance(contents[1], str)

    # Check if the string is valid json
    encoded_obs = json.loads(contents[1])
    self.assertEqual(
        encoded_obs['task_instruction'],
        np.array_str(observation[self.task_instruction_key]),
    )
    self.assertEqual(
        encoded_obs[self.proprio_obs_key],
        observation[self.proprio_obs_key].tolist(),
    )
    self.assertEqual(
        encoded_obs[f'{constants.IMAGE_ENCODED_OBS_PREFIX}image'], 0
    )


if __name__ == '__main__':
  absltest.main()
