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

"""Canonical FastAPI endpoints."""

import enum

import fastapi

from safari_sdk.agent.framework.embodiments import fast_api_endpoint


FastApiEndpoint = fast_api_endpoint.FastApiEndpoint


@enum.unique
class FastApiEndpoints(enum.Enum):
  """Canonical FastAPI endpoints."""

  # Sends an instruction to the robot.
  RUN_INSTRUCTION = FastApiEndpoint(path="/run/")

  # Reset the robot to the default position.
  RESET = FastApiEndpoint(path="/reset/")

  # Stop the robot.
  STOP_INSTRUCTION = FastApiEndpoint(path="/stop/")

  # Open robot grippers.
  OPEN_GRIPPERS = FastApiEndpoint(path="/open_grippers/")

  # Put the robot in a safe resting position for shutdown.
  SLEEP = FastApiEndpoint(path="/sleep/")

  # Set the agent session ID.
  SET_AGENT_SESSION_ID = FastApiEndpoint(path="/set_agent_session_id/")

  # Update inference config.
  UPDATE_INFERENCE_CONFIG = FastApiEndpoint(path="/update_inference_config/")

  # Get the health status of the robot backend.
  GET_HEALTH_STATUS = FastApiEndpoint(path="/get_health_status/")

  # Get the camera stream.
  OVERHEAD_CAMERA_STREAM = FastApiEndpoint(
      path="/overhead-camera-stream/",
      response_class=fastapi.responses.StreamingResponse,
  )

  LEFT_WRIST_CAMERA_STREAM = FastApiEndpoint(
      path="/left-wrist-camera-stream/",
      response_class=fastapi.responses.StreamingResponse,
  )

  RIGHT_WRIST_CAMERA_STREAM = FastApiEndpoint(
      path="/right-wrist-camera-stream/",
      response_class=fastapi.responses.StreamingResponse,
  )

  WORMS_EYE_CAMERA_STREAM = FastApiEndpoint(
      path="/worms-eye-camera-stream/",
      response_class=fastapi.responses.StreamingResponse,
  )
