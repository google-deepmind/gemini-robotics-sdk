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

"""Gemini Robotics Policy."""

from concurrent import futures
import copy
import enum
import json
import logging
import threading
from typing import Any

import numpy as np
import tensorflow as tf

from safari_sdk.model import genai_robotics
from safari_sdk.model import tf_agents_interface as tfa_interface


class InferenceMode(enum.Enum):
  SYNCHRONOUS = enum.auto()
  ASYNCHRONOUS = enum.auto()


class GeminiRoboticsPolicy:
  """Policy which uses the Gemini Robotics API."""

  def __init__(
      self,
      serve_id: str,
      task_instruction: str,
      cameras: dict[str, tuple[int, int]],
      joints: dict[str, int],
      min_replan_interval: int = 15,
      inference_mode: InferenceMode = InferenceMode.ASYNCHRONOUS,
      robotics_api_connection: genai_robotics.RoboticsApiConnectionType = genai_robotics.RoboticsApiConnectionType.CLOUD,
  ):
    """Initializes the evaluation policy.

    Args:
      serve_id: The serve ID to use for the policy.
      task_instruction: The task instruction to use for the policy.
      cameras: A dict of camera names to (height, width) tuples.
      joints: A dict of joint names to the number of joints.
      min_replan_interval: The minimum number of steps to wait before replanning
        the task instruction.
      inference_mode: Whether to use an async or sync implementation of the
        policy.
      robotics_api_connection: Connection type for the Robotics API.
    """
    self._serve_id = serve_id
    self._task_instruction = task_instruction
    self._cameras = cameras
    self._joints = joints
    self._min_replan_interval = min_replan_interval

    self._model_output = np.array([])
    self._action_spec = None
    # Initialize the genai_robotics client
    self._client = genai_robotics.Client(
        use_robotics_api=True,  # Use the specific Robotics API endpoint
        robotics_api_connection=robotics_api_connection,
    )
    # Threading setup
    self._inference_mode = inference_mode
    if inference_mode == InferenceMode.ASYNCHRONOUS:
      self._executor = futures.ThreadPoolExecutor(max_workers=1)
      self._future: futures.Future[np.ndarray] | None = None
      self._model_output_lock = threading.Lock()
      self._actions_executed_during_inference = 0

  def setup(self):
    """Initializes the policy."""
    self._action = self._query_model(
        self._reset_sync(), self._task_instruction, np.array([])
    )
    self._action_spec = tf.TensorSpec(
        shape=self._action.shape, dtype=tf.float32
    )

  @property
  def action_spec(self) -> tf.TensorSpec:
    if self._action_spec is None:
      logging.warning('action_spec is None, calling setup()')
      self.setup()
    assert self._action_spec is not None, 'action_spec is None, setup failed'
    return self._action_spec

  @property
  def observation_spec(self) -> dict[str, tf.TensorSpec]:
    return {
        joint_name: tf.TensorSpec(shape=(num_joints,), dtype=tf.float32)
        for joint_name, num_joints in self._joints.items()
    } | {
        camera_name: tf.TensorSpec(
            shape=(camera_info[0], camera_info[1], 3),
            dtype=tf.uint8,
        )
        for camera_name, camera_info in self._cameras.items()
    }

  def _reset_sync(self) -> dict[str, np.ndarray]:
    """Resets the policy."""

    self._model_output = np.array([])
    observation_0 = tf.nest.map_structure(
        lambda x: tf.zeros(x.shape, x.dtype),
        self.observation_spec,
    )

    return observation_0

  def _reset_async(self) -> dict[str, np.ndarray]:
    """Resets the policy."""
    # Reset the model output
    with self._model_output_lock:
      self._model_output = np.array([])

    # Cancel any pending futures on reset
    if self._future and self._future.running():
      self._future.cancel()
    self._future = None
    return self._reset_sync()

  def reset(self) -> dict[str, np.ndarray]:
    """Resets the policy."""
    if self._inference_mode == InferenceMode.ASYNCHRONOUS:
      return self._reset_async()
    return self._reset_sync()

  def _should_replan(self) -> bool:
    """Returns whether the policy should replan."""
    actions_left = self._model_output.shape[0]
    total_actions = self.action_spec.shape[0]
    if (total_actions - actions_left) >= self._min_replan_interval:
      return True
    if actions_left == 0:
      return True
    return False

  def _step_sync(
      self, observation: dict[str, np.ndarray]
  ) -> tfa_interface.ActionType:
    """Computes an action from observations."""
    if self._should_replan():
      self._model_output = self._query_model(
          observation, self._task_instruction, self._model_output
      )
      assert self._model_output.shape[0] > 0

    action = self._model_output[0]
    self._model_output = self._model_output[1:]
    return action

  def set_task_instruction(self, task_instruction: str):
    """Sets the task instruction for the policy."""
    self._task_instruction = task_instruction

  def _step_async(
      self, observation: dict[str, np.ndarray]
  ) -> tfa_interface.ActionType:
    """Computes an action from the given observation.

    Method:
    1. If Gemini Returned an action chunk, update the action buffer.
    2. If no Gemini query is pending and the action buffer is less than the
       minimum replan interval, trigger a new query.
    3. If the action buffer is empty (first query) trigger a new query.
    4. If there is more than one action in the buffer, consume the first
       action and remove it from the buffer.
    5. If only one action is in the buffer, consume it without removing it (we
    will keep outputting this action until a new action is generated, this is an
    edge case that should not happen in practice). This results in a quasi-async
    implementation.

    Args:
        observation: A dictionary of observations from the environment.

    Returns:
        The next action to take.

    Raises:
        ValueError: If no actions are available and no future to generate them
        is present.
    """
    with self._model_output_lock:
      # If new model output is available, update the buffer.
      if self._future and self._future.done():
        new_model_output = self._future.result()
        # Remove the actions that were executed while the future was running.
        self._model_output = new_model_output[
            self._actions_executed_during_inference :
        ]
        self._future = None
      actions_left = self._model_output.shape[0]
      # If not enough actions left and not generating, trigger a replan.
      if self._should_replan() and self._future is None:
        self._future = self._executor.submit(
            self._query_model,
            copy.deepcopy(observation),
            copy.deepcopy(self._task_instruction),
            copy.deepcopy(self._model_output),
        )
        self._actions_executed_during_inference = 0

    # If no actions left (first query), block until the future is done.
    if actions_left == 0:
      if not self._future:
        raise ValueError('No actions left and no future to generate them.')
      result_from_blocking_wait = self._future.result()
      with self._model_output_lock:
        self._model_output = result_from_blocking_wait[
            self._actions_executed_during_inference :
        ]
        self._future = None

    # Consume the action.
    with self._model_output_lock:
      action = self._model_output[0]
      self._model_output = self._model_output[1:]
      self._actions_executed_during_inference += 1
    return action

  def step(
      self, observation: dict[str, np.ndarray]
  ) -> tfa_interface.ActionType:
    """Computes an action from the given observation."""
    if self._inference_mode == InferenceMode.ASYNCHRONOUS:
      return self._step_async(observation)
    return self._step_sync(observation)

  def _observation_to_contents(
      self,
      observation: dict[str, np.ndarray],
      task_instruction: str,
      model_output: np.ndarray,
  ) -> list[Any]:
    """Encodes the observation and task instruction as a GenerateRequest."""
    encoded_observation = {}
    encoded_observation['task_instruction'] = task_instruction
    # Conditioning on what the model has left to output.
    if model_output.size > 0:
      encoded_observation['conditioning'] = model_output.tolist()

    for joint_name in self._joints:
      joints = observation[joint_name]
      # Tolerate common mistake of having an extra batch dimension.
      if joints.ndim == 2:
        joints = joints[0]
      if joints.ndim != 1:
        raise ValueError(
            f'Joint {joint_name} has {joints.ndim} dimensions, but should be 1.'
        )
      encoded_observation[joint_name] = [float(j) for j in joints]

    images = []
    for i, camera_name in enumerate(self._cameras):
      encoded_observation[f'images/{camera_name}'] = i
      image = observation[camera_name]
      if isinstance(image, (np.ndarray, tf.Tensor)):
        # Tolerate common mistake of having an extra batch dimension.
        image_dim = image.ndim
        if image_dim == 4:
          image = image[0]
        if image.ndim != 3:
          raise ValueError(
              f'Image {camera_name} has {image_dim} dimensions, but should'
              ' be 3.'
          )
      elif isinstance(image, bytes):
        pass  # can directly take encoded image bytes.
      else:
        raise ValueError(
            f'Image {camera_name} is of type {type(image)}, but should be'
            ' np.ndarray, tf.Tensor or bytes.'
        )
      images.append(image)
    return [
        *images,
        json.dumps(encoded_observation),
    ]

  def _query_model(
      self,
      observation: dict[str, np.ndarray],
      task_instruction: str,
      model_output: np.ndarray,
  ) -> np.ndarray:
    """Queries the model with the given observation and task instruction."""
    contents = self._observation_to_contents(
        observation, task_instruction, model_output
    )

    response = self._client.models.generate_content(
        model=self._serve_id,
        contents=contents,
    )

    # Parse the response text (assuming its JSON containing the action)
    response_data = json.loads(response.text)
    # Assuming the structure is {'action_chunk': [...]}
    action_chunk = response_data.get('action_chunk')
    if action_chunk is None:
      raise ValueError("Response JSON does not contain 'action_chunk'")
    return np.array(action_chunk)

  @property
  def policy_type(self) -> str:
    if self._inference_mode == InferenceMode.ASYNCHRONOUS:
      return f'gemini_robotics_async[{self._serve_id}]'
    return f'gemini_robotics[{self._serve_id}]'
