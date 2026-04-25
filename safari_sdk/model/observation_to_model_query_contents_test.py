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

import jax
import numpy as np
import tensorflow as tf

from absl.testing import absltest
from absl.testing import parameterized
from safari_sdk.model import constants
from safari_sdk.model import observation_to_model_query_contents


class ObservationToModelQueryContentsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.task_instruction_key = 'instruction'
    self.string_obs_key = 'string_obs'
    self.proprio_obs_key = 'proprio'
    self.image_obs_key = 'image'

  def test_observation_serialization(self):
    observation = {
        self.task_instruction_key: np.array('do the task'),
        self.string_obs_key: np.array('some string'),
        self.proprio_obs_key: np.array([1.0, 2.0]),
        self.image_obs_key: np.zeros((64, 64, 3), dtype=np.uint8),
    }

    contents = (
        observation_to_model_query_contents.observation_to_model_query_contents(
            observation=observation,
            string_observations_keys=('instruction', 'string_obs'),
            task_instruction_key='instruction',
            proprioceptive_observation_keys=('proprio',),
            image_observation_keys=('image',),
        )
    )

    encoded_obs = json.loads(contents[-1])
    images = contents[:-1]

    self.assertLen(images, 1)
    np.testing.assert_array_equal(images[0], observation[self.image_obs_key])

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

  def test_multiple_images(self):
    image_obs_key2 = 'image2'
    observation = {
        self.task_instruction_key: np.array('do the task'),
        self.proprio_obs_key: np.array([1.0, 2.0]),
        self.image_obs_key: np.zeros((64, 64, 3), dtype=np.uint8),
        image_obs_key2: np.ones((64, 64, 3), dtype=np.uint8),
    }

    contents = (
        observation_to_model_query_contents.observation_to_model_query_contents(
            observation=observation,
            task_instruction_key='instruction',
            proprioceptive_observation_keys=('proprio',),
            string_observations_keys=[self.task_instruction_key],
            image_observation_keys=[self.image_obs_key, image_obs_key2],
        )
    )

    encoded_obs = json.loads(contents[-1])
    images = contents[:-1]

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

    contents = (
        observation_to_model_query_contents.observation_to_model_query_contents(
            observation=observation,
            task_instruction_key='instruction',
            proprioceptive_observation_keys=('proprio',),
            string_observations_keys=[],
            image_observation_keys=[],
        )
    )
    encoded_obs = json.loads(contents[-1])

    self.assertEqual(encoded_obs[self.proprio_obs_key], [1.0, 2.0])

  def test_invalid_proprio_dims_raises_error(self):
    observation = {
        self.proprio_obs_key: np.zeros((1, 1, 1)),
    }
    with self.assertRaisesRegex(
        ValueError, 'Observation proprio has 3 dimensions, but should be 1.'
    ):

      observation_to_model_query_contents.observation_to_model_query_contents(
          observation=observation,
          task_instruction_key='instruction',
          proprioceptive_observation_keys=('proprio',),
          string_observations_keys=[],
          image_observation_keys=[],
      )

  @parameterized.named_parameters(
      ('string', 'do the task'),
      ('numpy_array_of_strings', np.array(['do the task'])),
      ('scalar_list', np.array('do the task', dtype=np.dtypes.StringDType())),  # pytype: disable=module-attr
      ('bytes', b'do the task'),
  )
  def test_string_observation_formats(self, task_instruction):
    observation = {
        self.task_instruction_key: task_instruction,
    }
    contents = (
        observation_to_model_query_contents.observation_to_model_query_contents(
            observation=observation,
            string_observations_keys=('instruction',),
            task_instruction_key='instruction',
            proprioceptive_observation_keys=[],
            image_observation_keys=[],
        )
    )
    encoded_obs = json.loads(contents[-1])
    self.assertEqual(
        encoded_obs[constants.TASK_INSTRUCTION_ENCODED_OBS_KEY],
        'do the task',
    )

  def test_invalid_string_observation_formats_raises_error(self):
    observation = {
        self.task_instruction_key: 1,
    }
    with self.assertRaisesRegex(
        ValueError,
        f'Observation for key {self.task_instruction_key} is not a string,'
        ' bytes, or numpy array.',
    ):
      observation_to_model_query_contents.observation_to_model_query_contents(
          observation=observation,
          string_observations_keys=('instruction',),
          task_instruction_key='instruction',
          proprioceptive_observation_keys=[],
          image_observation_keys=[],
      )

  def test_encode_observations_with_conditioning(self):
    observation = {
        constants.CONDITIONING_ENCODED_OBS_KEY: (
            np.zeros((15, 30), dtype=np.float32).tolist()
        ),
    }

    contents = (
        observation_to_model_query_contents.observation_to_model_query_contents(
            observation=observation,
            string_observations_keys=[],
            task_instruction_key='instruction',
            image_observation_keys=[],
            proprioceptive_observation_keys=[],
        )
    )

    encoded_obs = json.loads(contents[-1])
    self.assertIn(constants.CONDITIONING_ENCODED_OBS_KEY, encoded_obs)
    self.assertEqual(
        encoded_obs[constants.CONDITIONING_ENCODED_OBS_KEY],
        np.zeros((15, 30), dtype=np.float32).tolist(),
    )

  @parameterized.named_parameters(
      ('jax', 'jax'),
      ('tensorflow', 'tensorflow'),
      ('numpy', 'numpy'),
  )
  def test_array_image_conversion(self, array_type):
    width = 64
    height = 64
    channels = 3
    # We create the appropriate array type based on the test parameter here to
    # avoid jax functions being called before initialization.
    if array_type == 'jax':
      image_array = jax.numpy.zeros(
          (height, width, channels), dtype=jax.numpy.uint8
      )
    elif array_type == 'tensorflow':
      image_array = tf.zeros((height, width, channels), dtype=tf.uint8)
    else:
      image_array = np.zeros((height, width, channels), dtype=np.uint8)

    observation = {
        self.image_obs_key: image_array,
    }
    contents = (
        observation_to_model_query_contents.observation_to_model_query_contents(
            observation=observation,
            string_observations_keys=[],
            task_instruction_key='instruction',
            image_observation_keys=('image',),
            proprioceptive_observation_keys=[],
        )
    )
    images = contents[:-1]
    self.assertLen(images, 1)
    # Check that it is converted to a numpy array
    self.assertIsInstance(images[0], np.ndarray)
    self.assertEqual(images[0].shape, (height, width, channels))

  @parameterized.named_parameters(
      ('jax', 'jax'),
      ('tensorflow', 'tensorflow'),
      ('numpy', 'numpy'),
  )
  def test_array_image_conversion_with_batch_dimension(self, array_type):
    batch_size = 2
    width = 64
    height = 64
    channels = 3
    # We create the appropriate array type based on the test parameter here to
    # avoid jax functions being called before initialization.
    if array_type == 'jax':
      image_array = jax.numpy.zeros(
          (batch_size, height, width, channels), dtype=jax.numpy.uint8
      )
    elif array_type == 'tensorflow':
      image_array = tf.zeros(
          (batch_size, height, width, channels), dtype=tf.uint8
      )
    else:
      image_array = np.zeros(
          (batch_size, height, width, channels), dtype=np.uint8
      )

    observation = {
        self.image_obs_key: image_array,
    }
    contents = (
        observation_to_model_query_contents.observation_to_model_query_contents(
            observation=observation,
            string_observations_keys=[],
            task_instruction_key='instruction',
            image_observation_keys=('image',),
            proprioceptive_observation_keys=[],
        )
    )
    images = contents[:-1]
    self.assertLen(images, 1)
    # Check that it is converted to a numpy array
    self.assertIsInstance(images[0], np.ndarray)
    # Check that the batch dimension is removed.
    self.assertEqual(images[0].shape, (height, width, channels))

  def test_image_as_bytes(self):
    image_bytes = b'some_image_bytes'
    observation = {
        self.image_obs_key: image_bytes,
    }

    contents = (
        observation_to_model_query_contents.observation_to_model_query_contents(
            observation=observation,
            string_observations_keys=[],
            task_instruction_key='instruction',
            image_observation_keys=('image',),
            proprioceptive_observation_keys=[],
        )
    )
    images = contents[:-1]

    self.assertLen(images, 1)
    self.assertEqual(images[0], image_bytes)

  def test_invalid_image_dims_raises_error(self):
    observation = {
        self.image_obs_key: np.zeros((64, 64), dtype=np.uint8),
    }
    with self.assertRaisesRegex(
        ValueError, 'Image image has 2 dimensions, but should be 3.'
    ):

      observation_to_model_query_contents.observation_to_model_query_contents(
          observation=observation,
          string_observations_keys=[],
          proprioceptive_observation_keys=[],
          task_instruction_key='instruction',
          image_observation_keys=('image',),
      )

if __name__ == '__main__':
  absltest.main()
