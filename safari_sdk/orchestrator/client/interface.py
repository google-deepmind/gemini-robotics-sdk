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

"""Central interface to access external API to Orchestrator API functions."""

import random
import threading
from safari_sdk import auth
from safari_sdk.orchestrator.client.dataclass import api_response
from safari_sdk.orchestrator.client.libs import current_robot
from safari_sdk.orchestrator.client.libs import robot_job
from safari_sdk.orchestrator.client.libs import robot_job_work_unit
from safari_sdk.orchestrator.client.libs import visual_overlay

JOB_TYPE = robot_job.JobType

WORK_UNIT = robot_job_work_unit.WORK_UNIT
WORK_UNIT_OUTCOME = robot_job_work_unit.WORK_UNIT_OUTCOME

ACCEPTED_IMAGE_TYPES = visual_overlay.AcceptedImageTypes
IMAGE_FORMAT = visual_overlay.ImageFormat
DRAW_CIRCLE_ICON = visual_overlay.visual_overlay_icon.DrawCircleIcon
DRAW_ARROW_ICON = visual_overlay.visual_overlay_icon.DrawArrowIcon
DRAW_SQUARE_ICON = visual_overlay.visual_overlay_icon.DrawSquareIcon
DRAW_TRIANGLE_ICON = visual_overlay.visual_overlay_icon.DrawTriangleIcon
DRAW_CONTAINER = visual_overlay.visual_overlay_icon.DrawContainer

RESPONSE = api_response.OrchestratorAPIResponse
_SUCCESS = RESPONSE(success=True)

_ERROR_NO_ACTIVE_CONNECTION = (
    "OrchestratorInterface: No active connection. Please call connect() first."
)
_ERROR_NO_WORK_UNIT_CONTEXT = (
    "OrchestratorInterface: No context data found in current work unit."
)
_ERROR_NO_SCENE_PRESET_DETAILS = (
    "OrchestratorInterface: No scene preset details found in current work unit."
)
_ERROR_NO_REFERENCE_IMAGES = (
    "OrchestratorInterface: No reference images data found in current work"
    " unit."
)
_ERROR_NO_RENDERER_FOUND = (
    "OrchestratorInterface: No visual overlay renderer found for the given key."
)
_ERROR_RENDERER_ALREADY_EXISTS = (
    "OrchestratorInterface: Visual overlay renderer already exists for the"
    " given key."
)


class OrchestratorInterface:
  """Central interface for Orchestrator API calls."""

  def __init__(
      self,
      *,
      robot_id: str,
      job_type: JOB_TYPE,
  ):
    self._robot_id = robot_id
    self._job_type = job_type

    self._rpc_lock = threading.Lock()

    self._connection = None
    self._robot_job_lib = None
    self._robot_job_work_unit_lib = None
    self._visual_overlay = {}

  def connect(self) -> RESPONSE:
    """Create connection to the orchestrator server."""
    try:
      self._connection = auth.get_service()
    except ValueError as e:
      return RESPONSE(error_message=str(e))

    self._current_robot_lib = current_robot.OrchestratorCurrentRobotInfo(
        connection=self._connection,
        robot_id=self._robot_id,
    )
    self._robot_job_lib = robot_job.OrchestratorRobotJob(
        connection=self._connection,
        robot_id=self._robot_id,
        job_type=self._job_type,
    )
    self._robot_job_work_unit_lib = (
        robot_job_work_unit.OrchestratorRobotJobWorkUnit(
            connection=self._connection,
            robot_id=self._robot_id,
        )
    )
    return _SUCCESS

  def disconnect(self) -> None:
    """Disconnects from the orchestrator server."""
    self._connection = None

    if self._robot_job_work_unit_lib:
      self._robot_job_work_unit_lib.disconnect()
      self._robot_job_work_unit_lib = None

    if self._robot_job_lib:
      self._robot_job_lib.disconnect()
      self._robot_job_lib = None

  def get_current_connection(self) -> RESPONSE:
    """Gets the current active connection to the orchestrator server."""
    if self._connection is None:
      return RESPONSE(error_message=_ERROR_NO_ACTIVE_CONNECTION)
    return RESPONSE(
        success=True,
        server_connection=self._connection,
        robot_id=self._robot_id,
    )

  def get_current_robot_info(self) -> RESPONSE:
    """Gets the current robot information."""
    if self._connection is None or self._current_robot_lib is None:
      return RESPONSE(error_message=_ERROR_NO_ACTIVE_CONNECTION)

    with self._rpc_lock:
      return self._current_robot_lib.get_current_robot_info()

  def set_current_robot_operator_id(self, operator_id: str) -> RESPONSE:
    """Set the current operator ID for the robot."""
    if self._connection is None or self._current_robot_lib is None:
      return RESPONSE(error_message=_ERROR_NO_ACTIVE_CONNECTION)

    with self._rpc_lock:
      return self._current_robot_lib.set_current_robot_operator_id(
          operator_id=operator_id
      )

  def request_robot_job_work_unit(self) -> RESPONSE:
    """Requests for a work unit to start working on."""
    if (
        self._connection is None
        or self._robot_job_lib is None
        or self._robot_job_work_unit_lib is None
    ):
      return RESPONSE(error_message=_ERROR_NO_ACTIVE_CONNECTION)

    with self._rpc_lock:
      robot_job_response = self._robot_job_lib.request_robot_job()
    if not robot_job_response.success:
      return robot_job_response
    if robot_job_response.success and robot_job_response.no_more_robot_job:
      return robot_job_response

    self._robot_job_work_unit_lib.set_robot_job_id(
        robot_job_id=robot_job_response.robot_job_id
    )
    with self._rpc_lock:
      return self._robot_job_work_unit_lib.request_work_unit()

  def get_current_robot_job_work_unit(self) -> RESPONSE:
    """Get the currently assigned work unit's information."""
    if (
        self._connection is None
        or self._robot_job_lib is None
        or self._robot_job_work_unit_lib is None
    ):
      return RESPONSE(error_message=_ERROR_NO_ACTIVE_CONNECTION)

    return self._robot_job_work_unit_lib.get_current_work_unit()

  def is_visual_overlay_in_current_work_unit(self) -> RESPONSE:
    """Checks if the current work unit has visual overlay information."""
    if (self._connection is None or self._robot_job_work_unit_lib is None):
      return RESPONSE(error_message=_ERROR_NO_ACTIVE_CONNECTION)

    response = self._robot_job_work_unit_lib.get_current_work_unit()
    if not response.success:
      return response

    work_unit = response.work_unit
    if not work_unit.context:
      return RESPONSE(success=True, is_visual_overlay_found=False)
    if not work_unit.context.scenePresetDetails:
      return RESPONSE(success=True, is_visual_overlay_found=False)
    if not work_unit.context.scenePresetDetails.referenceImages:
      return RESPONSE(success=True, is_visual_overlay_found=False)

    return RESPONSE(success=True, is_visual_overlay_found=True)

  def create_visual_overlays_for_current_work_unit(self) -> RESPONSE:
    """Creates visual overlay renderers based on the current work unit."""
    if (self._connection is None or self._robot_job_work_unit_lib is None):
      return RESPONSE(error_message=_ERROR_NO_ACTIVE_CONNECTION)

    response = self._robot_job_work_unit_lib.get_current_work_unit()
    if not response.success:
      return response

    work_unit = response.work_unit
    if not work_unit.context:
      return RESPONSE(error_message=_ERROR_NO_WORK_UNIT_CONTEXT)
    if not work_unit.context.scenePresetDetails:
      return RESPONSE(error_message=_ERROR_NO_SCENE_PRESET_DETAILS)
    if not work_unit.context.scenePresetDetails.referenceImages:
      return RESPONSE(error_message=_ERROR_NO_REFERENCE_IMAGES)

    self._visual_overlay.clear()
    ref_imgs = work_unit.context.scenePresetDetails.referenceImages

    for ref_img in ref_imgs:
      if ref_img.sourceTopic in self._visual_overlay:
        continue
      self._visual_overlay[ref_img.sourceTopic] = (
          visual_overlay.OrchestratorRenderer(
              scene_reference_image_data=ref_img
          )
      )
      self._visual_overlay[
          ref_img.sourceTopic
      ].load_scene_objects_from_work_unit(
          scene_objects=work_unit.context.scenePresetDetails.sceneObjects
      )
    return _SUCCESS

  def list_visual_overlay_renderer_keys(self) -> RESPONSE:
    """Lists index key name for all visual overlay renderers."""
    if self._visual_overlay:
      return RESPONSE(
          success=True,
          visual_overlay_renderer_keys=list(self._visual_overlay),
      )
    else:
      return RESPONSE(error_message=_ERROR_NO_RENDERER_FOUND)

  def render_visual_overlay(
      self,
      renderer_key: str,
      new_image: ACCEPTED_IMAGE_TYPES | None = None,
  ) -> RESPONSE:
    """Renders the visual overlay for the given renderer ID."""
    if renderer_key not in self._visual_overlay:
      return RESPONSE(error_message=_ERROR_NO_RENDERER_FOUND)

    return self._visual_overlay[renderer_key].render_overlay(
        new_image=new_image
    )

  def get_visual_overlay_image_as_pil_image(
      self,
      renderer_key: str,
  ) -> RESPONSE:
    """Returns the visual overlay image as PIL image."""
    if renderer_key not in self._visual_overlay:
      return RESPONSE(error_message=_ERROR_NO_RENDERER_FOUND)

    return self._visual_overlay[renderer_key].get_image_as_pil_image()

  def get_visual_overlay_image_as_np_array(
      self,
      renderer_key: str,
  ) -> RESPONSE:
    """Returns the visual overlay image as numpy array."""
    if renderer_key not in self._visual_overlay:
      return RESPONSE(error_message=_ERROR_NO_RENDERER_FOUND)

    return self._visual_overlay[renderer_key].get_image_as_np_array()

  def get_visual_overlay_image_as_bytes(
      self,
      renderer_key: str,
      img_format: IMAGE_FORMAT = IMAGE_FORMAT.JPEG,
  ) -> RESPONSE:
    """Returns the visual overlay image as bytes in the specified format."""
    if renderer_key not in self._visual_overlay:
      return RESPONSE(error_message=_ERROR_NO_RENDERER_FOUND)

    return self._visual_overlay[renderer_key].get_image_as_bytes(
        img_format=img_format
    )

  def reset_visual_overlay_renderer(
      self, renderer_key: str, reset_all_renderers: bool = False
  ) -> RESPONSE:
    """Resets specific or all visual overlay renderers."""
    if reset_all_renderers:
      for renderer in self._visual_overlay.values():
        renderer.reset_all_object_settings()
      return _SUCCESS

    if renderer_key not in self._visual_overlay:
      return RESPONSE(error_message=_ERROR_NO_RENDERER_FOUND)

    return self._visual_overlay[renderer_key].reset_all_object_settings()

  def create_single_visual_overlay_renderer(
      self,
      renderer_key: str,
      image_pixel_width: int,
      image_pixel_height: int,
      overlay_bg_color: str = "#444444",
  ) -> RESPONSE:
    """Manually create a single visual overlay renderer."""
    if renderer_key in self._visual_overlay:
      return RESPONSE(error_message=_ERROR_RENDERER_ALREADY_EXISTS)

    scene_reference_image_data = visual_overlay.work_unit.SceneReferenceImage(
        artifactId=(
            "manual_overlay_renderer_" + str(random.randint(1000000, 9999999))
        ),
        sourceTopic=renderer_key,
        rawImageWidth=image_pixel_width,
        rawImageHeight=image_pixel_height,
        renderedCanvasWidth=image_pixel_width,
        renderedCanvasHeight=image_pixel_height,
    )
    manual_renderer = visual_overlay.OrchestratorRenderer(
        scene_reference_image_data=scene_reference_image_data,
        overlay_bg_color=overlay_bg_color,
    )
    self._visual_overlay[renderer_key] = manual_renderer
    return _SUCCESS

  def add_single_overlay_object_to_visual_overlay(
      self,
      renderer_key: str,
      overlay_object: (
          DRAW_CIRCLE_ICON
          | DRAW_ARROW_ICON
          | DRAW_SQUARE_ICON
          | DRAW_TRIANGLE_ICON
          | DRAW_CONTAINER
      ),
  ) -> RESPONSE:
    """Adds a single overlay object to the visual overlay renderer."""
    if renderer_key not in self._visual_overlay:
      return RESPONSE(error_message=_ERROR_NO_RENDERER_FOUND)

    return self._visual_overlay[renderer_key].add_single_object(
        overlay_object=overlay_object
    )

  def robot_job_work_unit_start_software_asset_prep(self) -> RESPONSE:
    """Sets the current work unit's stage as software asset prep."""
    if (
        self._connection is None
        or self._robot_job_lib is None
        or self._robot_job_work_unit_lib is None
    ):
      return RESPONSE(error_message=_ERROR_NO_ACTIVE_CONNECTION)

    with self._rpc_lock:
      return self._robot_job_work_unit_lib.start_work_unit_software_asset_prep()

  def robot_job_work_unit_start_scene_prep(self) -> RESPONSE:
    """Starts the current work unit's stage as scene prep."""
    if (
        self._connection is None
        or self._robot_job_lib is None
        or self._robot_job_work_unit_lib is None
    ):
      return RESPONSE(error_message=_ERROR_NO_ACTIVE_CONNECTION)

    with self._rpc_lock:
      return self._robot_job_work_unit_lib.start_work_unit_scene_prep()

  def robot_job_work_unit_start_execution(self) -> RESPONSE:
    """Set the current work unit's stage as executing."""
    if (
        self._connection is None
        or self._robot_job_lib is None
        or self._robot_job_work_unit_lib is None
    ):
      return RESPONSE(error_message=_ERROR_NO_ACTIVE_CONNECTION)

    with self._rpc_lock:
      return self._robot_job_work_unit_lib.start_work_unit_execution()

  def robot_job_work_unit_complete_work_unit(
      self, outcome: robot_job_work_unit.WORK_UNIT_OUTCOME, note: str
  ) -> RESPONSE:
    """Sets the current work unit's stage as completed."""
    if (
        self._connection is None
        or self._robot_job_lib is None
        or self._robot_job_work_unit_lib is None
    ):
      return RESPONSE(error_message=_ERROR_NO_ACTIVE_CONNECTION)

    with self._rpc_lock:
      return self._robot_job_work_unit_lib.complete_work_unit(
          outcome=outcome, note=note
      )
