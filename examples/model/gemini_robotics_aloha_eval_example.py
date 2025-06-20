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

# This is not a google3-compatible python file. It is intended to be run on a
# non-corp machine.

import select
import signal
import sys
import termios
import threading
import time
import tty

from absl import app
from absl import flags
# TODO: Remove dependency on hostbot.aloha and use interbotix instead.
from hostbot.aloha import aloha_ros_robot_client
import rclpy
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
_ALOHA_CAMERAS = {
    'overhead_cam': _IMAGE_SIZE,
    'worms_eye_cam': _IMAGE_SIZE,
    'wrist_cam_left': _IMAGE_SIZE,
    'wrist_cam_right': _IMAGE_SIZE,
}
_ALOHA_JOINTS = {'joints_pos': 14}


def main(_):
  rclpy.init()
  robot_client = aloha_ros_robot_client.AlohaROSRobotClient(
      include_leaders=False,
      subscribe_to_raw_images=False,
  )
  robot_client.prep_robots()

  def shutdown():
    print('Shutting down.')
    robot_client.move_to_rest_poses()
    robot_client.close()
    rclpy.try_shutdown()
    sys.exit(0)

  def handler_fn(sig, frame):
    del sig, frame
    shutdown()

  unused_sigint_handler = SigintHandler(handler_fn)

  task_instruction = TaskInstruction(_TASK_INSTRUCTION.value)
  model_client = gemini_robotics_policy.GeminiRoboticsPolicy(
      serve_id=_SERVE_ID.value,
      task_instruction=str(task_instruction),
      cameras=_ALOHA_CAMERAS,
      joints=_ALOHA_JOINTS,
  )

  def run_episode():
    print('Homing...')
    robot_client.move_to_home()

    print('Policy reset')
    model_client.reset()

    # Get new task instruction from user.
    task = task_instruction.get_user_input()
    print('Task instruction: ', task)
    model_client._task_instrution = task  # pylint: disable=protected-access

    with KeyDetect() as detector:
      print('Running policy... Press "q" to terminate episode.')
      while True:
        frame_start_time = time.time()
        obs = {
            camera_name: bytes(robot_client.get_image_jpeg(camera_name))
            for camera_name, _ in _ALOHA_CAMERAS.items()
        } | {
            joint_name: robot_client.get_follower_joints_pos()
            for joint_name, _ in _ALOHA_JOINTS.items()
        }
        gemini_actions = model_client.step(obs)
        cmd = aloha_ros_robot_client.robot_client.RobotCommand(
            left_arm_joint_target=gemini_actions[:6],
            right_arm_joint_target=gemini_actions[7:13],
            left_gripper_joint_target=gemini_actions[6:7],
            right_gripper_joint_target=gemini_actions[13:],
        )
        robot_client.step(cmd)

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
