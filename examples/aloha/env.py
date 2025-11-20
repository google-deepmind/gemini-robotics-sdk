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

"""Aloha Gym environment."""

import threading
import time
from typing import Callable

from aloha import robot_utils
import cv2
from gdm_robotics.adapters import gymnasium_env_to_gdmr_env_wrapper
from gdm_robotics.interfaces import environment as gdmr_env
import gymnasium as gym
from interbotix_common_modules.common_robot import robot
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
import numpy as np
from sensor_msgs.msg import CompressedImage


# Constants for camera topics
_COMPRESSED_CAM_OVERHEAD = '/camera_high/camera/color/image_rect_raw/compressed'
_COMPRESSED_CAM_WORMS_EYE = '/camera_low/camera/color/image_rect_raw/compressed'
_COMPRESSED_CAM_LEFT = (
    '/camera_wrist_left/camera/color/image_rect_raw/compressed'
)
_COMPRESSED_CAM_RIGHT = (
    '/camera_wrist_right/camera/color/image_rect_raw/compressed'
)
FPS = 50

# Gripper Range A
GRIPPER_OPEN = 1.5155
GRIPPER_CLOSED = -0.06135

# Gripper Range B
# GRIPPER_OPEN = 1.6214
# GRIPPER_CLOSED = 0.6197

MOTOR_REGISTER_VALUE = 300
HOME_POSITION = [[0.0, -0.96, 1.16, 0.0, -0.3, 0.0]]

INSTRUCTION_RESET_OPTION_KEY = 'instruction'
SHOULD_MOVE_TO_HOME_RESET_OPTION_KEY = 'should_move_to_home'
_MAX_INSTRUCTION_LENGTH = 1000

# Forward reset option structure.
ResetOptions = gymnasium_env_to_gdmr_env_wrapper.GymnasiumEnvResetOptions


def create_aloha_environment(
    robot_config_name: str,
    config_base_path: str,
    max_num_steps: int,
    default_instruction: str = 'do something useful',
    **kwargs
) -> gdmr_env.Environment:
  """Creates an Aloha environment."""
  env = _AlohaEnv(
      robot_config_name=robot_config_name,
      config_base_path=config_base_path,
      max_num_steps=max_num_steps,
      default_instruction=default_instruction,
      **kwargs,
  )
  return gymnasium_env_to_gdmr_env_wrapper.GymnasiumEnvToGdmrEnvWrapper(env)


class ImageController:
  """A controller to subscribe to multiple compressed image topics given a ROS node."""

  def __init__(self, node):
    self.subs = {
        'overhead_cam': node.create_subscription(
            CompressedImage,
            _COMPRESSED_CAM_OVERHEAD,
            self._callback_factory(_COMPRESSED_CAM_OVERHEAD),
            10,
        ),
        'worms_eye_cam': node.create_subscription(
            CompressedImage,
            _COMPRESSED_CAM_WORMS_EYE,
            self._callback_factory(_COMPRESSED_CAM_WORMS_EYE),
            10,
        ),
        'wrist_cam_left': node.create_subscription(
            CompressedImage,
            _COMPRESSED_CAM_LEFT,
            self._callback_factory(_COMPRESSED_CAM_LEFT),
            10,
        ),
        'wrist_cam_right': node.create_subscription(
            CompressedImage,
            _COMPRESSED_CAM_RIGHT,
            self._callback_factory(_COMPRESSED_CAM_RIGHT),
            10,
        ),
    }
    self._topic_images = {}
    self._topic_mutex = threading.Lock()

  def _callback_factory(self, key) -> Callable[[CompressedImage], None]:
    """Creates a callback function for a given image topic.

    Args:
      key: The key for the image topic.

    Returns:
      A callback function for the given image topic.
    """

    def _callback(msg):
      with self._topic_mutex:
        image_np = np.frombuffer(msg.data, dtype=np.uint8)
        bgr_image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        self._topic_images[key] = rgb_image

    return _callback

  def get_images(self) -> dict[any, any]:
    """Returns a copy of the images."""
    with self._topic_mutex:
      # Return a copy to prevent race conditions
      return self._topic_images.copy()

  def wait_for_images(self, timeout=5.0) -> bool:
    """Wait until all required camera images are available.

    Args:
      timeout: The maximum time to wait for all images to be available.

    Returns:
      True if all images are available, False otherwise.
    """
    required_keys = self.subs.keys()
    start = time.time()
    while time.time() - start < timeout:
      images = self.get_images()
      if all(key in images for key in required_keys):
        return True
    print('Warning: Timed out waiting for all camera images.')
    return False


class _AlohaEnv(gym.Env):
  """A Gymnasium environment for the ALOHA robot system."""

  def __init__(
      self,
      robot_config_name: str,
      config_base_path: str,
      default_instruction: str,
      max_num_steps: int = 5000,
      **kwargs
  ):
    super().__init__()

    # --- ROS and Robot Initialization ---
    self.node = robot.create_interbotix_global_node('aloha_env0', **kwargs)

    config = robot_utils.load_yaml_file(
        'robot', robot_config_name, config_base_path
    ).get('robot', {})
    self.dt = 1 / FPS
    self.last_get_obs = 0

    # Initialize image recorder
    self.image_node = ImageController(self.node)

    # Initialize dictionary for robot instances
    self.robots = {}
    for follower in config.get('follower_arms', []):
      robot_instance = InterbotixManipulatorXS(
          robot_model=follower['model'],
          robot_name=follower['name'],
          node=self.node,
          iterative_update_fk=False,
      )
      self.robots[follower['name']] = robot_instance

    robot.robot_startup(self.node)
    for follower_name, follower_bot in self.robots.items():
      print('setting operation modes for robot: ' + follower_name)
      follower_bot.core.robot_reboot_motors('single', 'gripper', True)
      follower_bot.core.robot_set_operating_modes('group', 'arm', 'position')
      follower_bot.core.robot_set_operating_modes(
          'single', 'gripper', 'current_based_position'
      )
      follower_bot.core.robot_set_motor_registers(
          'single', 'gripper', 'current_limit', MOTOR_REGISTER_VALUE
      )
      robot_utils.torque_on(follower_bot)

    print('Waiting for initial images...')
    self.image_node.wait_for_images()
    print('Robot and cameras are ready.')

    # --- Define Gym Spaces ---
    # Action space: 7-DoF for each arm (6 joints + 1 gripper)
    # You should refine these bounds based on the actual robot limits
    joint_low = np.full(7, -np.inf)
    joint_high = np.full(7, np.inf)
    self.action_space = gym.spaces.Box(
        low=np.concatenate([joint_low, joint_low]),
        high=np.concatenate([joint_high, joint_high]),
        dtype=np.float64,
    )

    self.observation_space = gym.spaces.Dict({
        'joints_pos': gym.spaces.Box(
            dtype=np.float64, shape=(1, 14), low=-np.inf, high=np.inf
        ),
        'overhead_cam': gym.spaces.Box(
            dtype=np.uint8, shape=(480, 848, 3), low=0, high=255
        ),
        'wrist_cam_left': gym.spaces.Box(
            dtype=np.uint8, shape=(480, 848, 3), low=0, high=255
        ),
        'wrist_cam_right': gym.spaces.Box(
            dtype=np.uint8, shape=(480, 848, 3), low=0, high=255
        ),
        'worms_eye_cam': gym.spaces.Box(
            dtype=np.uint8, shape=(480, 848, 3), low=0, high=255
        ),
        INSTRUCTION_RESET_OPTION_KEY: gym.spaces.Text(
            max_length=_MAX_INSTRUCTION_LENGTH
        ),
    })
    self._default_instruction = default_instruction
    self._instruction = default_instruction
    self._max_num_steps = max_num_steps
    self._num_steps = 0

  @property
  def instruction(self) -> str:
    return self._instruction

  @instruction.setter
  def instruction(self, instruction: str):
    self._instruction = instruction

  def _get_proprio(self) -> np.ndarray:
    """Returns the proprioceptive observations for the follower arms."""
    left_bot = self.robots['follower_left']
    right_bot = self.robots['follower_right']
    left_joints = left_bot.core.joint_states.position[:7]
    right_joints = right_bot.core.joint_states.position[:7]
    return np.concatenate([left_joints, right_joints])

  def _get_obs(self) -> dict[str, any]:
    """Returns the observations for the follower arms."""
    self.last_get_obs = time.time()
    images = self.image_node.get_images()
    obs = {
        'joints_pos': self._get_proprio().reshape(1, 14),
        'overhead_cam': images[_COMPRESSED_CAM_OVERHEAD],
        'wrist_cam_left': images[_COMPRESSED_CAM_LEFT],
        'wrist_cam_right': images[_COMPRESSED_CAM_RIGHT],
        'worms_eye_cam': images[_COMPRESSED_CAM_WORMS_EYE],
    }
    return obs

  def _set_follower(
      self, robot_instance: InterbotixManipulatorXS, arm_positions: np.ndarray
  ):
    """Sets the follower arm to the given arm positions.

    Args:
      robot_instance: The follower arm to set.
      arm_positions: The arm positions to set.
    """
    joints = arm_positions[:6]
    gripper = arm_positions[6]
    robot_instance.arm.set_joint_positions(joints, blocking=False)
    gripper_command = JointSingleCommand(name='gripper', cmd=gripper)
    robot_instance.gripper.core.pub_single.publish(gripper_command)

  def reset(self, seed=None, options=None):
    """Resets the environment.

    Args:
      seed: The seed for the random number generator.
      options: The options for the reset.

    Returns:
      The initial observation and info.
    """
    super().reset(seed=seed)
    if (
        options
        and SHOULD_MOVE_TO_HOME_RESET_OPTION_KEY in options
        and options[SHOULD_MOVE_TO_HOME_RESET_OPTION_KEY]
    ):
      print('Resetting environment: moving arms to home position.')
      # Move arms to a known home position
      robot_utils.move_arms(
          bot_list=self.robots.values(),
          dt=self.dt,
          target_pose_list=HOME_POSITION * 2,
          moving_time=2.0,
      )
      robot_utils.move_grippers(
          list(self.robots.values()), [1.5, 1.5], moving_time=1.0, dt=self.dt
      )  # Open grippers
      time.sleep(1.0)

    # The task instruction can be passed via options
    if options and INSTRUCTION_RESET_OPTION_KEY in options:
      self._instruction = options[INSTRUCTION_RESET_OPTION_KEY]
    else:
      self._instruction = self._default_instruction

    observation = self._get_obs()
    # Provide the task instruction as part of the observation.
    observation[INSTRUCTION_RESET_OPTION_KEY] = self._instruction
    info = {'instruction': self._instruction}

    self._num_steps = 0

    return observation, info

  def step(self, action: np.ndarray):
    """Steps the environment.

    Args:
      action: The action to take.

    Returns:
      The observation.
    """
    # Execute the action on the robot
    left_action = action[:7]
    right_action = action[7:]
    self._set_follower(self.robots['follower_left'], left_action)
    self._set_follower(self.robots['follower_right'], right_action)

    # Wait for consistent step time.
    time_to_sleep = max(0, self.dt - (time.time() - self.last_get_obs))
    time.sleep(time_to_sleep)
    observation = self._get_obs()
    # Provide the task instruction as part of the observation.
    observation[INSTRUCTION_RESET_OPTION_KEY] = self._instruction

    reward = 0
    terminated = False
    truncated = False
    info = {}

    self._num_steps += 1
    if self._num_steps >= self._max_num_steps:
      truncated = True

    return observation, reward, terminated, truncated, info

  def close(self, hard_shutdown=True):
    """Closes the environment and shuts down the robot.

    Args:
      hard_shutdown: Whether to hard shutdown the robot. The default is True
        meaning the robot will move to sleep position before shutting down. Pass
        hard_shutdown=False to skip the robot move to sleep and shutdown
        immediately.
    """
    print('Closing environment and shutting down robot.')
    print(list(self.robots.values()))
    if hard_shutdown:
      robot_utils.move_arms(
          bot_list=self.robots.values(),
          dt=self.dt,
          target_pose_list=HOME_POSITION * 2,
          moving_time=0.5,
      )
      robot_utils.move_grippers(
          list(self.robots.values()), [1.5, 1.5], moving_time=0.5, dt=self.dt
      )
      time.sleep(1.0)

    robot.robot_shutdown(self.node)
    time.sleep(5.0)

  def open_grippers(self):
    """Opens the grippers."""
    robot_utils.move_grippers(
        list(self.robots.values()), [1.5, 1.5], moving_time=1.0, dt=self.dt
    )

  def get_overhead_cam_image(self) -> np.ndarray:
    """Returns the overhead camera image."""
    return self.image_node.get_images()[_COMPRESSED_CAM_OVERHEAD]

  def get_worms_eye_cam_image(self) -> np.ndarray:
    """Returns the worms eye camera image."""
    return self.image_node.get_images()[_COMPRESSED_CAM_WORMS_EYE]

  def get_wrist_cam_left_image(self) -> np.ndarray:
    """Returns the wrist left camera image."""
    return self.image_node.get_images()[_COMPRESSED_CAM_LEFT]

  def get_wrist_cam_right_image(self) -> np.ndarray:
    """Returns the wrist right camera image."""
    return self.image_node.get_images()[_COMPRESSED_CAM_RIGHT]
