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
import dm_env
from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np

from safari_sdk.model import constants
from safari_sdk.model import gemini_robotics_policy


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
    ["aloha", "trossen", "franka_duo"],
    "The robot type to use.",
)

# Aloha (Trossen Aloha Stationary) — checkpoint_type=aloha
_ALOHA_IMAGE_SIZE = (480, 848)
_ALOHA_CAMERAS = {
    "overhead_cam": _ALOHA_IMAGE_SIZE,
    "worms_eye_cam": _ALOHA_IMAGE_SIZE,
    "wrist_cam_left": _ALOHA_IMAGE_SIZE,
    "wrist_cam_right": _ALOHA_IMAGE_SIZE,
}
_ALOHA_JOINTS = {"joints_pos": 14}

# Franka Duo (dual Franka FR3 arms) — checkpoint_type=franka_duo
_FRANKA_DUO_IMAGE_SIZE = (480, 848)
_FRANKA_DUO_CAMERAS = {
    "head_cam_left_rgb": _FRANKA_DUO_IMAGE_SIZE,
    "head_cam_right_rgb": _FRANKA_DUO_IMAGE_SIZE,
}
_FRANKA_DUO_JOINTS = {
    "joint_position_left_arm": 7,
    "joint_position_right_arm": 7,
    "primitive_grasp_left_hand": 1,
    "primitive_grasp_right_hand": 1,
}


_CAMERAS = {
    "aloha": _ALOHA_CAMERAS,
    "trossen": _ALOHA_CAMERAS,
    "franka_duo": _FRANKA_DUO_CAMERAS,

}
_JOINTS = {
    "aloha": _ALOHA_JOINTS,
    "trossen": _ALOHA_JOINTS,
    "franka_duo": _FRANKA_DUO_JOINTS,

}

_TASK_INSTRUCTION_KEY = "instruction"


def _build_timestep_spec(
    cameras: dict[str, tuple[int, int]],
    joints: dict[str, int],
) -> gdmr_types.TimeStepSpec:
  """Builds a TimeStepSpec for the given camera and joint configuration."""
  observation_spec = {}
  for cam_name, (height, width) in cameras.items():
    observation_spec[cam_name] = specs.BoundedArray(
        shape=(height, width, 3),
        dtype=np.uint8,
        minimum=0,
        maximum=255,
    )
  for joint_name, dim in joints.items():
    observation_spec[joint_name] = specs.Array(shape=(1, dim), dtype=np.float32)
  observation_spec[_TASK_INSTRUCTION_KEY] = specs.StringArray(shape=())

  return gdmr_types.TimeStepSpec(
      step_type=gdmr_types.STEP_TYPE_SPEC,
      reward={},
      discount={},
      observation=observation_spec,
  )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  cameras = _CAMERAS[_ROBOT_TYPE.value]
  joints = _JOINTS[_ROBOT_TYPE.value]

  policy = gemini_robotics_policy.GeminiRoboticsPolicy(
      serve_id=_SERVE_ID.value,
      task_instruction_key=_TASK_INSTRUCTION_KEY,
      image_observation_keys=tuple(cameras.keys()),
      proprioceptive_observation_keys=tuple(joints.keys()),
      inference_mode=constants.InferenceMode.SYNCHRONOUS,
      robotics_api_connection=constants.RoboticsApiConnectionType.LOCAL,
  )

  timestep_spec = _build_timestep_spec(cameras, joints)
  policy.step_spec(timestep_spec)
  state = policy.initial_state()
  print("GeminiRoboticsPolicy initialized successfully.")

  # Create dummy data (zeros) for other specs (like images)
  dummy_observation = {}
  for cam_name, (height, width) in cameras.items():
    dummy_observation[cam_name] = np.zeros((height, width, 3), dtype=np.uint8)
  for joint_name, dim in joints.items():
    dummy_observation[joint_name] = np.zeros((1, dim), dtype=np.float32)
  # Use the provided task instruction for the 'instruction' field
  dummy_observation[_TASK_INSTRUCTION_KEY] = np.array(
      _TASK_INSTRUCTION.value, dtype=np.object_
  )

  print("\nCreated dummy observation:")
  for key, value in dummy_observation.items():
    print(f"  {key}: shape={np.shape(value)}, dtype={np.asarray(value).dtype}")

  try:
    for i in range(100):
      print(f"\nCalling policy.step() step {i}...")
      timestep = dm_env.transition(
          reward=0.0,
          discount=1.0,
          observation=dummy_observation,
      )
      (action, _), state = policy.step(timestep, state)
      print("\nReceived action from policy:")
      print(action)
  except Exception as e:  # pylint: disable=broad-except
    print(f"\nAn error occurred during policy.step(): {e}")
    print("Please check your API key, serve ID, and network connection.")


if __name__ == "__main__":
  app.run(main)
