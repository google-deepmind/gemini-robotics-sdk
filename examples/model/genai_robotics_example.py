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
    "serve_id", None, "The ID of the model to use.", required=True
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # 1. Initialize the client
  client = genai_robotics.Client()

  # 2. Prepare sample input data
  # Sample image (e.g., from a camera)
  sample_image = np.zeros(
      (64, 64, 3), dtype=np.uint8
  )  # Example: 64x64 RGB image

  # Sample observations, including image index and other data
  observations = {
      "images/overhead_cam": 0,  # Index 0 refers to sample_image
      "task_instruction": "Pick up the red block.",
      "joints_pos": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
  }
  observations_json = json.dumps(observations)

  # 3. Call the generate_content method
  print(f"Calling model {_SERVE_ID.value}...")
  try:
    response = client.models.generate_content(
        model=_SERVE_ID.value,
        contents=[
            sample_image,  # Can be np.array, tf.Tensor, bytes, or types.Part
            observations_json,
        ],
    )

    # 4. Print the response
    print("Model Response:", response.text)

  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"An error occurred: {e}")

  # 5. Call 5 times and print the average time
  print("Calling model 10 times...")
  times = []
  for _ in range(10):
    start = time.time()
    response = client.models.generate_content(
        model=_SERVE_ID.value,
        contents=[
            sample_image,  # Can be np.array, tf.Tensor, bytes, or types.Part
            observations_json,
        ],
    )
    end = time.time()
    times.append(end - start)
    del response
  print("times: ", times)
  print("Average time:", np.mean(times))


if __name__ == "__main__":
  app.run(main)
