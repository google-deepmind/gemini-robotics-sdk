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
# import cv2
import numpy as np

from safari_sdk.model import constants
from safari_sdk.model import genai_robotics

_CONNECTION = constants.RoboticsApiConnectionType

_SERVE_ID = flags.DEFINE_string(
    "serve_id",
    None,
    "The ID of the model to use. Required for Cloud-based inference.",
)
_ROBOTICS_API_CONNECTION = flags.DEFINE_enum_class(
    "robotics_api_connection",
    _CONNECTION.LOCAL,
    _CONNECTION,
    "The robotics API connection type to use.",
)
_SERVER_BASE_URL = flags.DEFINE_string(
    "server_base_url",
    None,
    "The server URL to use. None means use the default.",
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # 1. Initialize the client
  http_options = (
      genai_robotics.types.HttpOptions(base_url=_SERVER_BASE_URL.value)
      if _SERVER_BASE_URL.value
      else None
  )
  robotics_api_connection = _CONNECTION(_ROBOTICS_API_CONNECTION.value)
  client = genai_robotics.Client(
      robotics_api_connection=robotics_api_connection,
      http_options=http_options,
  )

  # 2. Prepare sample input data
  # Gemini Robotics
  test_img_gemini_robotics = np.random.randint(
      0, 255, (480, 848, 3), dtype=np.uint8
  )
  # Gemini Robotics Nano
  test_img_robotics_nano = np.random.randint(
      0, 255, (224, 224, 3), dtype=np.uint8
  )

  obs = {
      "images/overhead_cam": 0,
      "images/wrist_cam_left": 1,
      "images/wrist_cam_right": 2,
      "images/worms_eye_cam": 3,
      "task_instruction": "make a fox shaped origami",
      # "joints_pos": np.random.randn(14).astype(np.float32).tolist(),
      "joints_pos": [-np.inf] * 14,
  }
  obs_json = json.dumps(obs)

  match _ROBOTICS_API_CONNECTION.value:
    case _CONNECTION.CLOUD | _CONNECTION.CLOUD_GENAI:
      content = [
          test_img_gemini_robotics,
          test_img_gemini_robotics,
          test_img_gemini_robotics,
          test_img_gemini_robotics,
          obs_json,
      ]
      if _ROBOTICS_API_CONNECTION.value == _CONNECTION.CLOUD_GENAI:
        content = genai_robotics.update_robotics_content_to_genai_format(
            contents=content
        )
    case _CONNECTION.LOCAL:
      content = [
          test_img_robotics_nano,
          test_img_robotics_nano,
          test_img_robotics_nano,
          test_img_robotics_nano,
          obs_json,
      ]
    case _:
      raise ValueError(
          "Unsupported robotics_api_connection:"
          f" {_ROBOTICS_API_CONNECTION.value}."
      )

  # 3. Call 20 times and print the average time
  num_calls = 20
  print(f"Calling model {num_calls} times...")
  times = []
  for _ in range(num_calls):
    start = time.time()
    response = client.models.generate_content(
        model=_SERVE_ID.value,
        contents=content,
    )
    match _ROBOTICS_API_CONNECTION.value:
      case _CONNECTION.CLOUD_GENAI:
        action_chunk = np.array(
            json.loads(
                response.candidates[0].content.parts[0].inline_data.data
            )["action_chunk"]
        )[0]
        print("action_chunk: ", action_chunk[0])
      case _:
        print(response.text)
    end = time.time()
    times.append(end - start)
    print("Inference time (s): ", end - start)
    del response
  print("times: ", times)
  print("Average time:", np.mean(times[10:]))  # Skip the first 10 calls.


if __name__ == "__main__":
  app.run(main)
