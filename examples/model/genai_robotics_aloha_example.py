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

"""Example of using genai_robotics.py."""

from collections.abc import Sequence
import json
import time

from absl import app
from absl import flags
import numpy as np

from safari_sdk.model import genai_robotics

_SERVE_ID = flags.DEFINE_string(
    "serve_id",
    None,
    "The ID of the model to use. Required for Cloud-based inference.",
)
_USE_ROBOTICS_API = flags.DEFINE_boolean(
    "use_robotics_api",
    True,
    "Whether to use the specific Robotics API endpoint.",
)
_ROBOTICS_API_CONNECTION = flags.DEFINE_enum(
    "robotics_api_connection",
    "local",
    ["cloud", "local"],
    "The robotics API connection type to use.",
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if not _USE_ROBOTICS_API.value:
    raise ValueError("Only robotics API is supported.")

  # 1. Initialize the client
  client = genai_robotics.Client(
      use_robotics_api=_USE_ROBOTICS_API.value,
      robotics_api_connection=genai_robotics.RoboticsApiConnectionType(
          _ROBOTICS_API_CONNECTION.value
      ),
  )

  # 2. Prepare sample input data
  # Gemini Robotics
  test_img = np.random.randint(0, 255, (480, 848, 3), dtype=np.uint8)
  # Gemini Robotics Nano
  # test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
  obs = {
      "images/overhead_cam": 0,
      "images/wrist_cam_left": 1,
      "images/wrist_cam_right": 2,
      "images/worms_eye_cam": 3,
      "task_instruction": "make a fox shaped origami",
      "joints_pos": np.random.randn(14).astype(np.float32).tolist(),
  }
  obs_json = json.dumps(obs)

  # 5. Call 5 times and print the average time
  print("Calling model 100 times...")
  times = []
  for _ in range(100):
    start = time.time()
    response = client.models.generate_content(
        model=_SERVE_ID.value,
        contents=[
            test_img,
            test_img,
            test_img,
            test_img,
            obs_json,
        ],
    )
    print(response.text)
    end = time.time()
    times.append(end - start)
    print("Inference time (s): ", end - start)
    del response
  print("times: ", times)
  print("Average time:", np.mean(times[10:]))  # Skip the first 10 calls.


if __name__ == "__main__":
  app.run(main)
