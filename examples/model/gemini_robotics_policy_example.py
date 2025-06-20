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

"""Example script to test the GeminiRoboticsPolicy."""

from collections.abc import Sequence

from absl import app
from absl import flags
import numpy as np
from safari_sdk.model import gemini_robotics_policy
import tensorflow as tf


_SERVE_ID = flags.DEFINE_string(
    "serve_id",
    None,
    "The serve ID to use.",
    required=True,
)
_TASK_INSTRUCTION = flags.DEFINE_string(
    "task_instruction",
    "Pick up the red block.",
    "The task instruction for the policy.",
)
_ROBOT_TYPE = flags.DEFINE_enum(
    "robot_type",
    "aloha",
    ["aloha", "atari"],
    "The robot type to use.",
)

# Aloha
_ALOHA_IMAGE_SIZE = (480, 848)
_ALOHA_CAMERAS = {
    "overhead_cam": _ALOHA_IMAGE_SIZE,
    "worms_eye_cam": _ALOHA_IMAGE_SIZE,
    "wrist_cam_left": _ALOHA_IMAGE_SIZE,
    "wrist_cam_right": _ALOHA_IMAGE_SIZE,
}
_ALOHA_JOINTS = {"joints_pos": 14}

# Atari
_ATARI_STEREOLAB_HEADCAM_IMAGE_SIZE = (1200, 1920)
_ATARI_WRISTCAM_IMAGE_SIZE = (480, 640)
_ATARI_CAMERAS = {
    "stereolab_headcam0": _ATARI_STEREOLAB_HEADCAM_IMAGE_SIZE,
    "left_wrist_cam": _ATARI_WRISTCAM_IMAGE_SIZE,
    "right_wrist_cam": _ATARI_WRISTCAM_IMAGE_SIZE,
}
_ATARI_JOINTS = {
    "left_arm_joint_pos": 7,
    "right_arm_joint_pos": 7,
    "left_hand_command": 6,
    "right_hand_command": 6,
    "neck_joint_pos": 3,
    "torso_joint_pos": 3,
}

_CAMERAS = {"aloha": _ALOHA_CAMERAS, "atari": _ATARI_CAMERAS}
_JOINTS = {"aloha": _ALOHA_JOINTS, "atari": _ATARI_JOINTS}


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Instantiate the policy
  try:
    policy = gemini_robotics_policy.GeminiRoboticsPolicy(
        serve_id=_SERVE_ID.value,
        task_instruction=_TASK_INSTRUCTION.value,
        cameras=_CAMERAS[_ROBOT_TYPE.value],
        joints=_JOINTS[_ROBOT_TYPE.value],
    )
    policy.setup()  # Initialize the policy
    print("GeminiRoboticsPolicy initialized successfully.")
  except ValueError as e:
    print(f"Error initializing policy: {e}")
    return
  except Exception as e:  # pylint: disable=broad-except
    print(f"An unexpected error occurred during initialization: {e}")
    return

  # Create a dummy observation based on the observation_spec
  dummy_observation = {}
  for key, spec in policy.observation_spec.items():
    if spec.dtype == tf.string:
      # Use the provided task instruction for the 'instruction' field
      dummy_observation[key] = np.array(_TASK_INSTRUCTION.value, dtype=object)
    else:
      # Create dummy data (zeros) for other specs (like images)
      dummy_observation[key] = np.zeros(
          spec.shape, dtype=spec.dtype.as_numpy_dtype
      )

  print("\nCreated dummy observation:")
  for key, value in dummy_observation.items():
    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")

  # Run 100 steps
  try:
    for i in range(100):
      print(f"\nCalling policy.step() step {i}...")
      action = policy.step(dummy_observation)
      print("\nReceived action from policy:")
      print(action)
  except Exception as e:  # pylint: disable=broad-except
    # Catch broad exceptions as API calls can fail in various ways
    print(f"\nAn error occurred during policy.step(): {e}")
    print("Please check your API key, serve ID, and network connection.")


if __name__ == "__main__":
  app.run(main)
