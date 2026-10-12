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

"""Constants for Gemini Robotics Policy."""

import enum


class RoboticsApiConnectionType(enum.Enum):
  """Connection types for the Robotics API."""

  CLOUD = "cloud"  # Connects to a Google Cloud-based server.
  LOCAL = "local"  # Connects to a local server.
  CLOUD_GENAI = "cloud_genai"  # Connects via GenAI API.


class InferenceMode(enum.Enum):
  SYNCHRONOUS = "synchronous"
  ASYNCHRONOUS = "asynchronous"


# Prefix for image observation keys in the encoded observation.
IMAGE_ENCODED_OBS_PREFIX = "images/"

# Keys for the encoded observation.
CONDITIONING_ENCODED_OBS_KEY = "conditioning"

# Key for the task instruction in the encoded observation.
TASK_INSTRUCTION_ENCODED_OBS_KEY = "task_instruction"

# Key for the action chunk in the response.
ACTION_CHUNK_RESPONSE_KEY = "action_chunk"

DTYPE_RESPONSE_KEY = "dtype"

RNG_KEY_RESPONSE_KEY = "rng_key"

RNG_KEY_ENCODED_OBS_KEY = "rng_key"

# The SAX custom method the bidirectional streaming backend exposes. It is the
# only method `ModelServing.CmCustomBidi` accepts, so it is the default for
# every bidi client; the unary path keeps using `sample_actions_json_flat`.
SAMPLE_ACTIONS_STREAM_METHOD_NAME = "sample_actions_stream"

# DSCP 32 corresponds to CS4 (Class Selector 4) for high-priority QoS traffic.
DEFAULT_INFERENCE_DSCP = 32

