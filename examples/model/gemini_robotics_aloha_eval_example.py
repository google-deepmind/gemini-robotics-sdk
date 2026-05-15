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

"""Run Gemini Robotics Policy Eval on Aloha Robot."""


import select
import signal
import sys
import termios
import threading
import time
import tty

from absl import app
from absl import flags
from aloha import robot_utils
import cv2
import dm_env
from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
from interbotix_common_modules.common_robot import robot
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
import numpy as np
from sensor_msgs.msg import CompressedImage

from safari_sdk.model import constants
from safari_sdk.model import gemini_robotics_policy


_TASK_INSTRUCTION = flags.DEFINE_string(
    'task_instruction',
    'pick up banana and hand over',
    'Task instruction to use for the policy.',
)
_SERVE_ID = flags.DEFINE_string(
    'serve_id',
    None,
    'The serve ID to use.',
    required=True,
)

_DT = 0.02
_IMAGE_SIZE = (480, 848)
_IMAGE_OBSERVATION_KEYS = (
    'overhead_cam',
    'worms_eye_cam',
    'wrist_cam_left',
    'wrist_cam_right',
)
_PROPRIOCEPTIVE_OBSERVATION_KEYS = ('joints_pos',)
_TASK_INSTRUCTION_KEY = 'instruction'

_COMPRESSED_CAM_TOPICS = {
    'overhead_cam': '/camera_high/camera/color/image_rect_raw/compressed',
    'worms_eye_cam': '/camera_low/camera/color/image_rect_raw/compressed',
    'wrist_cam_left': (
        '/camera_wrist_left/camera/color/image_rect_raw/compressed'
    ),
    'wrist_cam_right': (
        '/camera_wrist_right/camera/color/image_rect_raw/compressed'
    ),
}

HOME_POSITION = [[0.0, -0.96, 1.16, 0.0, -0.3, 0.0]]
MOTOR_REGISTER_VALUE = 300
CONFIG_BASE_PATH = '/home/juggler/interbotix_ws/src/aloha/config/'
ROBOT_CONFIG_NAME = 'aloha_stationary'


def _build_timestep_spec() -> gdmr_types.TimeStepSpec:
  """Builds a TimeStepSpec for the Aloha robot."""
  observation_spec = {}
  for cam_name in _IMAGE_OBSERVATION_KEYS:
    observation_spec[cam_name] = specs.BoundedArray(
        shape=(_IMAGE_SIZE[0], _IMAGE_SIZE[1], 3),
        dtype=np.uint8,
        minimum=0,
        maximum=255,
    )
  observation_spec['joints_pos'] = specs.Array(shape=(1, 14), dtype=np.float32)
  observation_spec[_TASK_INSTRUCTION_KEY] = specs.StringArray(shape=())

  return gdmr_types.TimeStepSpec(
      step_type=gdmr_types.STEP_TYPE_SPEC,
      reward={},
      discount={},
      observation=observation_spec,
  )


class ImageController:
  """Subscribes to compressed image topics and stores the latest frames."""

  def __init__(self, node):
    self._topic_images = {}
    self._topic_mutex = threading.Lock()
    for cam_name, topic in _COMPRESSED_CAM_TOPICS.items():
      node.create_subscription(
          CompressedImage,
          topic,
          self._callback_factory(cam_name),
          10,
      )

  def _callback_factory(self, key):
    def _callback(msg):
      with self._topic_mutex:
        image_np = np.frombuffer(msg.data, dtype=np.uint8)
        bgr_image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        self._topic_images[key] = rgb_image

    return _callback

  def get_images(self) -> dict[str, np.ndarray]:
    with self._topic_mutex:
      return self._topic_images.copy()

  def wait_for_images(self, timeout=5.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
      images = self.get_images()
      if all(key in images for key in _IMAGE_OBSERVATION_KEYS):
        return True
    print('Warning: Timed out waiting for all camera images.')
    return False


def main(_):
  node = robot.create_interbotix_global_node('aloha_eval')
  config = robot_utils.load_yaml_file(
      'robot', ROBOT_CONFIG_NAME, CONFIG_BASE_PATH
  ).get('robot', {})

  robots = {}
  for follower in config.get('follower_arms', []):
    robot_instance = InterbotixManipulatorXS(
        robot_model=follower['model'],
        robot_name=follower['name'],
        node=node,
        iterative_update_fk=False,
    )
    robots[follower['name']] = robot_instance

  robot.robot_startup(node)
  for _, follower_bot in robots.items():
    follower_bot.core.robot_reboot_motors('single', 'gripper', True)
    follower_bot.core.robot_set_operating_modes('group', 'arm', 'position')
    follower_bot.core.robot_set_operating_modes(
        'single', 'gripper', 'current_based_position'
    )
    follower_bot.core.robot_set_motor_registers(
        'single', 'gripper', 'current_limit', MOTOR_REGISTER_VALUE
    )
    robot_utils.torque_on(follower_bot)

  image_controller = ImageController(node)
  print('Waiting for initial images...')
  image_controller.wait_for_images()
  print('Robot and cameras are ready.')

  policy = gemini_robotics_policy.GeminiRoboticsPolicy(
      serve_id=_SERVE_ID.value,
      task_instruction_key=_TASK_INSTRUCTION_KEY,
      image_observation_keys=_IMAGE_OBSERVATION_KEYS,
      proprioceptive_observation_keys=_PROPRIOCEPTIVE_OBSERVATION_KEYS,
      inference_mode=constants.InferenceMode.SYNCHRONOUS,
      robotics_api_connection=constants.RoboticsApiConnectionType.LOCAL,
  )

  timestep_spec = _build_timestep_spec()
  policy.step_spec(timestep_spec)

  def shutdown():
    print('Shutting down.')
    robot_utils.move_arms(
        bot_list=robots.values(),
        dt=_DT,
        target_pose_list=HOME_POSITION * 2,
        moving_time=0.5,
    )
    robot.robot_shutdown(node)
    sys.exit(0)

  def handler_fn(sig, frame):
    del sig, frame
    shutdown()

  unused_sigint_handler = SigintHandler(handler_fn)
  task_instruction = TaskInstruction(_TASK_INSTRUCTION.value)

  def set_follower(robot_instance, arm_positions):
    joints = arm_positions[:6]
    gripper = arm_positions[6]
    robot_instance.arm.set_joint_positions(joints, blocking=False)
    gripper_command = JointSingleCommand(name='gripper', cmd=gripper)
    robot_instance.gripper.core.pub_single.publish(gripper_command)

  def run_episode():
    print('Homing...')
    robot_utils.move_arms(
        bot_list=robots.values(),
        dt=_DT,
        target_pose_list=HOME_POSITION * 2,
        moving_time=2.0,
    )

    state = policy.initial_state()

    task = task_instruction.get_user_input()
    print('Task instruction: ', task)

    with KeyDetect() as detector:
      print('Running policy... Press "q" to terminate episode.')
      while True:
        frame_start_time = time.time()

        images = image_controller.get_images()
        left_bot = robots['follower_left']
        right_bot = robots['follower_right']
        left_joints = left_bot.core.joint_states.position[:7]
        right_joints = right_bot.core.joint_states.position[:7]

        observation = {}
        observation.update(images)
        observation['joints_pos'] = (
            np.concatenate([left_joints, right_joints])
            .reshape(1, 14)
            .astype(np.float32)
        )
        observation[_TASK_INSTRUCTION_KEY] = np.array(task, dtype=np.object_)

        timestep = dm_env.transition(
            reward=0.0,
            discount=1.0,
            observation=observation,
        )
        (action, _), state = policy.step(timestep, state)

        set_follower(robots['follower_left'], action[:7])
        set_follower(robots['follower_right'], action[7:])

        if detector.is_down('q'):
          print('Episode terminated.')
          detector.clear()
          break

        frame_time = time.time() - frame_start_time
        time.sleep(max(0, _DT - frame_time))

  while True:
    run_episode()


class TaskInstruction:
  """Task instruction for the policy."""

  def __init__(self, task_instruction: str):
    self._task_instruction = task_instruction

  def get_user_input(self) -> str:
    new_instruction = input(
        f'Input task instruction [{self._task_instruction}]:'
    )
    if new_instruction:
      self._task_instruction = new_instruction
    return self._task_instruction

  def __str__(self):
    return self._task_instruction


class SigintHandler:
  """Lightweight utility to call a function on SIGINT.

  The SIGINT handling will be removed once this object is deleted.
  """

  def __init__(self, handler_fn):
    self._prev_sigint_signal = signal.signal(
        signal.SIGINT,
        handler_fn,
    )

  def __del__(self):
    if self._prev_sigint_signal:
      signal.signal(signal.SIGINT, self._prev_sigint_signal)


class KeyDetect:
  """A non-blocking key detection class."""

  def __init__(self):
    self._original_settings = None
    self._key_buffer = set()
    self._lock = threading.Lock()
    self._event = threading.Event()
    self._stop_flag = False
    self._thread = None

  def __enter__(self):
    """Enters the context, setting up non-blocking input."""
    self._original_settings = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin.fileno())
    self._start_listening()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    """Exits the context, restoring terminal settings and stopping listener."""
    self._stop_listening()
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._original_settings)

  def _start_listening(self):
    """Starts a background thread to listen for key presses."""
    self._stop_flag = False
    self._thread = threading.Thread(target=self._listen)
    self._thread.daemon = True
    self._thread.start()

  def _stop_listening(self):
    """Sets the stop flag and waits for the background thread to finish."""
    self._stop_flag = True
    self._event.set()  # Wake up the thread if it's waiting
    if self._thread and self._thread.is_alive():
      self._thread.join()

  def _listen(self):
    """Listens for key presses and updates the key buffer."""
    while not self._stop_flag:
      if select.select([sys.stdin], [], [], 0.1)[
          0
      ]:  # Check for input with a timeout
        try:
          key = sys.stdin.read(1)
          with self._lock:
            self._key_buffer.add(key)
        except BlockingIOError:
          pass  # No input available
      self._event.wait(0.01)  # Small delay to reduce CPU usage
      self._event.clear()

  def is_down(self, key):
    """Checks if a specific key is currently pressed."""
    with self._lock:
      return key in self._key_buffer

  def get_pressed(self):
    """Returns a set of all currently pressed keys."""
    with self._lock:
      return set(self._key_buffer)

  def clear(self):
    """Clears the buffer of currently pressed keys."""
    with self._lock:
      self._key_buffer.clear()


if __name__ == '__main__':
  app.run(main)
