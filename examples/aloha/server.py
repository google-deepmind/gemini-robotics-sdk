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

r"""FastAPI server for Aloha Robot.

To run the server, use the following command:

```bash
python3 server.py \
--api_key=<API_KEY> \
--serve_id="gemini_robotics_on_device" \
--inference_mode=SYNCHRONOUS \
--robotics_api_connection=LOCAL \
--logging.robot_id=<ROBOT_ID>
```
"""

import asyncio
import contextlib
import io
import logging
import time

from absl import app
from absl import flags
import actor as aloha_actor
import fastapi
from fastapi.middleware import cors
import numpy as np
from PIL import Image
import uvicorn

from safari_sdk.agent.framework.embodiments import aloha_fast_api_endpoints
from safari_sdk.model import constants


CORSMiddleware = cors.CORSMiddleware
_ENDPOINTS = aloha_fast_api_endpoints.FastApiEndpoints

_DRY_RUN = flags.DEFINE_bool(
    "dry_run",
    False,
    "Whether to run the server in dry run mode. If True, the actor will not"
    " start the runloop.",
)
_SERVE_ID = flags.DEFINE_string(
    "serve_id",
    "gemini_robotics_on_device",
    "The serve id to use for the Gemini Robotics Policy.",
)
_INFERENCE_MODE = flags.DEFINE_enum_class(
    "inference_mode",
    constants.InferenceMode.SYNCHRONOUS,
    constants.InferenceMode,
    "The inference mode to use for the Gemini Robotics Policy.",
)
_ROBOTS_API_CONNECTION = flags.DEFINE_enum_class(
    "robotics_api_connection",
    constants.RoboticsApiConnectionType.LOCAL,
    constants.RoboticsApiConnectionType,
    "The robotics API connection type to use.",
)


asynccontextmanager = contextlib.asynccontextmanager

_actor_singleton = None


def create_default_actor() -> aloha_actor.Actor:
  global _actor_singleton
  _actor_singleton = aloha_actor.Actor(
      serve_id=_SERVE_ID.value,
      inference_mode=_INFERENCE_MODE.value,
      robotics_api_connection=_ROBOTS_API_CONNECTION.value,
  )
  return _actor_singleton


def create_actor(
    serve_id: str,
    inference_mode: constants.InferenceMode,
    robotics_api_connection: constants.RoboticsApiConnectionType,
) -> aloha_actor.Actor:
  """Creates and sets a global Aloha Actor instance.

  Shuts down any existing actor before creating a new one.

  Args:
    serve_id: The serve id to use for the Gemini Robotics Policy.
    inference_mode: The inference mode for the Gemini Robotics Policy.
    robotics_api_connection: The robotics API connection type.

  Returns:
    The created aloha_actor.Actor instance.
  """
  global _actor_singleton
  _actor_singleton = aloha_actor.Actor(
      serve_id=serve_id,
      inference_mode=inference_mode,
      robotics_api_connection=robotics_api_connection,
  )
  return _actor_singleton


def get_actor() -> aloha_actor.Actor | None:
  return _actor_singleton


def shutdown_actor():
  global _actor_singleton
  if _actor_singleton is not None:
    _actor_singleton.shutdown()
  time.sleep(0.5)
  _actor_singleton = None


@asynccontextmanager
async def lifespan(fast_api_app: fastapi.FastAPI):  # pylint: disable=unused-argument
  """Lifespan of the server. https://fastapi.tiangolo.com/advanced/events/."""
  logging.info("Starting the server.")

  robot_actor_creation_start_time = time.time()
  actor_instance = get_actor()
  if actor_instance is None:
    actor_instance = create_default_actor()
    actor_instance.reset()
  robot_actor_creation_end_time = time.time()
  logging.info(
      "Created the robot actor after %s seconds.",
      robot_actor_creation_end_time - robot_actor_creation_start_time,
  )
  logging.info("Started the server.")
  try:
    yield
  except asyncio.CancelledError:
    logging.info("Server shutdown request received (asyncio.CancelledError).")
  finally:
    logging.info("Shutting down the server.")
    actor_instance = get_actor()
    if actor_instance is not None:
      actor_instance.shutdown()
    logging.info("Shut down the server.")


fastapi_app = fastapi.FastAPI(lifespan=lifespan)
background_tasks = fastapi.BackgroundTasks()

# Additional origins that can connect to the backend
origins = [
    "http://localhost:3000",
]

# Add CORSMiddleware to the application
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of origins allowed
    allow_credentials=True,  # Allow cookies
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


@fastapi_app.get(**_ENDPOINTS.RUN_INSTRUCTION.value.args)
async def run(
    instruction: str,
):
  """Runs an instruction.

  Args:
    instruction: The instruction to run.

  Returns:
    A dictionary with the message "running instruction: {instruction}".
  """
  if _DRY_RUN.value:
    logging.info("[DRY RUN] running instruction: %s", instruction)
    return {"message": "[DRY RUN] running instruction: %s" % instruction}

  logging.info("Setting instruction: %s", instruction)
  actor_instance = get_actor()
  if actor_instance:
    actor_instance.stop_runloop()
    actor_instance.set_instruction(instruction)
    actor_instance.start_runloop()
  return {"message": "running instruction: %s" % instruction}


@fastapi_app.get(**_ENDPOINTS.RESET.value.args)
async def reset():
  """Resets the robot."""
  logging.info("Resetting the robot.")
  actor_instance = get_actor()
  if actor_instance:
    actor_instance.reset()
  return {"message": "reset"}


@fastapi_app.get(**_ENDPOINTS.STOP_INSTRUCTION.value.args)
async def stop():
  """Stops the instruction."""
  logging.info("Stopping the instruction.")
  actor_instance = get_actor()
  if actor_instance:
    actor_instance.stop_runloop()
  return {"message": "stopped"}


@fastapi_app.get(**_ENDPOINTS.OPEN_GRIPPERS.value.args)
async def open_grippers():
  """Opens the grippers."""
  logging.info("Opening grippers.")
  actor_instance = get_actor()
  if actor_instance:
    actor_instance.environment.wrapped_env.open_grippers()
  return {"message": "opened grippers"}


@fastapi_app.get(**_ENDPOINTS.SLEEP.value.args)
async def sleep():
  """Shuts down the robot."""
  logging.info("Shutting down the robot")
  actor_instance = get_actor()
  if actor_instance:
    actor_instance.shutdown()
  return {"message": "sleep"}


@fastapi_app.get(**_ENDPOINTS.SET_AGENT_SESSION_ID.value.args)
async def set_agent_session_id(agent_session_id: str):
  """Sets the agent session ID."""
  logging.info("Setting agent session ID: %s", agent_session_id)
  actor_instance = get_actor()
  if actor_instance:
    actor_instance.set_agent_session_id(agent_session_id)
  return {"message": "set agent session ID: %s" % agent_session_id}


@fastapi_app.get(**_ENDPOINTS.UPDATE_INFERENCE_CONFIG.value.args)
async def update_inference_config(
    serve_id: str,
    inference_mode: constants.InferenceMode,
    robotics_api_connection: constants.RoboticsApiConnectionType,
):
  """Resets the actor."""
  logging.info(
      "Updating inference config: serve_id: %s, inference_mode: %s,"
      " robots_api_connection: %s",
      serve_id,
      inference_mode,
      robotics_api_connection,
  )
  actor_instance = get_actor()
  if actor_instance:
    actor_instance.reset_from(serve_id, inference_mode, robotics_api_connection)
  return {"message": "updated inference config"}


@fastapi_app.get(**_ENDPOINTS.GET_HEALTH_STATUS.value.args)
async def get_health_status():
  """Returns the health status of the robot backend."""
  logging.info("Getting health status.")
  return {"health_status": "healthy"}


def convert_image_to_bytes(image_array: np.ndarray) -> bytes:
  """Converts a numpy array image to a bytes object."""
  image = Image.fromarray(image_array)
  img_buffer = io.BytesIO()
  image.save(img_buffer, format="JPEG")
  return img_buffer.getvalue()


async def generate_frames(cam_name: str):
  """Generates frames from the specified camera.

  Args:
    cam_name: The name of the camera to stream.

  Yields:
    Bytes of the frame.
  """
  while True:
    actor_instance = get_actor()
    if actor_instance is None:
      logging.info("Actor not found, sleeping for 0.1 seconds.")
      await asyncio.sleep(0.1)
      continue
    if cam_name == "overhead_cam":
      image_array = get_actor().environment.wrapped_env.get_overhead_cam_image()
    elif cam_name == "worms_eye_cam":
      image_array = (
          get_actor().environment.wrapped_env.get_worms_eye_cam_image()
      )
    elif cam_name == "wrist_left_cam":
      image_array = (
          get_actor().environment.wrapped_env.get_wrist_cam_left_image()
      )
    elif cam_name == "wrist_right_cam":
      image_array = (
          get_actor().environment.wrapped_env.get_wrist_cam_right_image()
      )
    else:
      raise fastapi.HTTPException(status_code=404, detail="Camera not found")
    frame = convert_image_to_bytes(image_array)

    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    await asyncio.sleep(0.1)  # Control frame rate


@fastapi_app.get(**_ENDPOINTS.OVERHEAD_CAMERA_STREAM.value.args)
async def overhead_camera_stream():
  """Streams the camera overhead camera video."""
  return fastapi.responses.StreamingResponse(
      generate_frames("overhead_cam"),
      media_type="multipart/x-mixed-replace; boundary=frame",
  )


@fastapi_app.get(**_ENDPOINTS.WORMS_EYE_CAMERA_STREAM.value.args)
async def worms_eye_camera_stream():
  """Streams the camera worms eye camera video."""
  return fastapi.responses.StreamingResponse(
      generate_frames("worms_eye_cam"),
      media_type="multipart/x-mixed-replace; boundary=frame",
  )


@fastapi_app.get(**_ENDPOINTS.LEFT_WRIST_CAMERA_STREAM.value.args)
async def wrist_left_camera_stream():
  """Streams the camera wrist left camera video."""
  return fastapi.responses.StreamingResponse(
      generate_frames("wrist_left_cam"),
      media_type="multipart/x-mixed-replace; boundary=frame",
  )


@fastapi_app.get(**_ENDPOINTS.RIGHT_WRIST_CAMERA_STREAM.value.args)
async def wrist_right_camera_stream():
  """Streams the camera wrist right camera video."""
  return fastapi.responses.StreamingResponse(
      generate_frames("wrist_right_cam"),
      media_type="multipart/x-mixed-replace; boundary=frame",
  )


def main(argv):
  if len(argv) > 1:
    raise ValueError("Too many command-line arguments.")
  uvicorn.run(fastapi_app, host="0.0.0.0", port=8888)


if __name__ == "__main__":
  app.run(main)
