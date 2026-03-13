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

import functools
import random
import threading

from safari_sdk import auth
from safari_sdk.orchestrator.client.dataclass import api_response
from safari_sdk.orchestrator.client.libs import artifact
from safari_sdk.orchestrator.client.libs import current_robot
from safari_sdk.orchestrator.client.libs import operator_event
from safari_sdk.orchestrator.client.libs import robot_job
from safari_sdk.orchestrator.client.libs import robot_job_work_unit
from safari_sdk.orchestrator.client.libs import rui_workcell_state
from safari_sdk.orchestrator.client.libs import visual_overlay

JOB_TYPE = robot_job.JobType

WORK_UNIT = robot_job_work_unit.WORK_UNIT
WORK_UNIT_OUTCOME = robot_job_work_unit.WORK_UNIT_OUTCOME
WORK_UNIT_QUESTION = robot_job_work_unit.WORK_UNIT_QUESTION
QUESTION_CONDITION = robot_job_work_unit.QUESTION_CONDITION
QUESTION_ANSWER_TYPE = robot_job_work_unit.QUESTION_ANSWER_TYPE

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
_ERROR_OBSERVER_MODE_ONLY = (
    "OrchestratorInterface: This method can only be called in observer mode."
)
_ERROR_IN_OBSERVER_MODE = (
    "OrchestratorInterface: This method is restricted when in observer mode."
)


def _check_orchestrator_interface_requirements(
    require_connection: bool = False,
    observer_check: str | None = None,
    required_libs: list[str] | None = None,
):
  """Decorator to check for observer mode and active connection."""

  def decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
      if require_connection and self._connection is None:  # pylint: disable=protected-access
        return RESPONSE(error_message=_ERROR_NO_ACTIVE_CONNECTION)

      if observer_check == "disallow" and self._observer_mode:  # pylint: disable=protected-access
        return RESPONSE(error_message=_ERROR_IN_OBSERVER_MODE)
      if observer_check == "require" and not self._observer_mode:  # pylint: disable=protected-access
        return RESPONSE(error_message=_ERROR_OBSERVER_MODE_ONLY)

      if require_connection and required_libs:
        for lib in required_libs:
          if getattr(self, lib, None) is None:
            return RESPONSE(error_message=_ERROR_NO_ACTIVE_CONNECTION)
      return func(self, *args, **kwargs)

    return wrapper

  return decorator


class OrchestratorInterface:
  """Central interface for Orchestrator API calls."""

  def __init__(
      self,
      *,
      robot_id: str,
      job_type: JOB_TYPE,
      hostname: str | None = None,
      observer_mode: bool = False,
  ):
    self._robot_id = robot_id
    self._job_type = job_type
    self._hostname = hostname
    self._observer_mode = observer_mode

    self._rpc_lock = threading.Lock()

    self._connection = None
    self._current_robot_lib = None
    self._robot_job_lib = None
    self._robot_job_work_unit_lib = None
    self._operator_event_lib = None
    self._artifact_lib = None
    self._rui_workcell_state_lib = None
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
        hostname=self._hostname,
    )
    response = self._current_robot_lib.get_current_robot_info()
    if not response.success:
      self._connection = None
      self._current_robot_lib = None
      id_or_hostname = self._robot_id or self._hostname
      return RESPONSE(
          error_message=(
              "Failed to validate connection to orchestrator server with"
              f" {id_or_hostname}. Validation failed with error:"
              f" {response.error_message}"
          ),
          robot_id=self._robot_id,
      )

    if not self._robot_id:
      self._robot_id = response.robot_id

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
    self._operator_event_lib = operator_event.OrchestratorOperatorEvent(
        connection=self._connection,
        robot_id=self._robot_id,
    )
    self._artifact_lib = artifact.OrchestratorArtifact(
        connection=self._connection,
    )
    self._rui_workcell_state_lib = (
        rui_workcell_state.OrchestratorRuiWorkcellState(
            connection=self._connection,
        )
    )
    return RESPONSE(
        success=True,
        robot_id=self._robot_id,
        latest_robot_release_configs=response.latest_robot_release_configs,
    )

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

  @_check_orchestrator_interface_requirements(
      require_connection=True,
      required_libs=["_current_robot_lib"],
  )
  def get_current_robot_info(self) -> RESPONSE:
    """Gets the current robot information."""
    assert self._current_robot_lib is not None
    with self._rpc_lock:
      return self._current_robot_lib.get_current_robot_info()

  @_check_orchestrator_interface_requirements(
      observer_check="disallow",
      require_connection=True,
      required_libs=["_current_robot_lib"],
  )
  def set_current_robot_operator_id(self, operator_id: str) -> RESPONSE:
    """Set the current operator ID for the robot."""
    assert self._current_robot_lib is not None
    with self._rpc_lock:
      return self._current_robot_lib.set_current_robot_operator_id(
          operator_id=operator_id
      )

  @_check_orchestrator_interface_requirements(
      observer_check="disallow",
      require_connection=True,
      required_libs=["_operator_event_lib"],
  )
  def add_operator_event(
      self,
      operator_event_str: str,
      operator_id: str,
      event_timestamp: int,
      resetter_id: str,
      event_note: str,
  ) -> RESPONSE:
    """Records an operator event."""
    assert self._operator_event_lib is not None
    with self._rpc_lock:
      return self._operator_event_lib.add_operator_event(
          operator_event_str=operator_event_str,
          operator_id=operator_id,
          event_timestamp=event_timestamp,
          resetter_id=resetter_id,
          event_note=event_note,
      )

  @_check_orchestrator_interface_requirements(
      observer_check="disallow",
      require_connection=True,
      required_libs=["_robot_job_lib", "_robot_job_work_unit_lib"],
  )
  def request_robot_job_work_unit(self) -> RESPONSE:
    """Requests for a work unit to start working on."""
    assert self._robot_job_lib is not None
    assert self._robot_job_work_unit_lib is not None
    with self._rpc_lock:
      robot_job_response = self._robot_job_lib.request_robot_job()
    if not robot_job_response.success:
      return robot_job_response
    if robot_job_response.success and robot_job_response.no_more_robot_job:
      return robot_job_response

    self._robot_job_work_unit_lib.set_robot_job_info(
        robot_job_id=robot_job_response.robot_job_id,
        launch_command=robot_job_response.launch_command,
    )
    with self._rpc_lock:
      return self._robot_job_work_unit_lib.request_work_unit()

  @_check_orchestrator_interface_requirements(
      observer_check="disallow",
      require_connection=True,
      required_libs=["_robot_job_lib", "_robot_job_work_unit_lib"],
  )
  def get_current_robot_job_work_unit(self) -> RESPONSE:
    """Get the currently assigned work unit's information."""
    assert self._robot_job_work_unit_lib is not None
    return self._robot_job_work_unit_lib.get_current_work_unit()

  @_check_orchestrator_interface_requirements(
      observer_check="disallow",
      require_connection=True,
      required_libs=["_robot_job_work_unit_lib"],
  )
  def is_visual_overlay_in_current_work_unit(self) -> RESPONSE:
    """Checks if the current work unit has visual overlay information."""
    assert self._robot_job_work_unit_lib is not None
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

  @_check_orchestrator_interface_requirements(
      observer_check="disallow",
      require_connection=True,
      required_libs=["_robot_job_work_unit_lib"],
  )
  def create_visual_overlays_for_current_work_unit(self) -> RESPONSE:
    """Creates visual overlay renderers based on the current work unit."""
    assert self._robot_job_work_unit_lib is not None
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

  @_check_orchestrator_interface_requirements(observer_check="disallow")
  def list_visual_overlay_renderer_keys(self) -> RESPONSE:
    """Lists index key name for all visual overlay renderers."""
    if self._visual_overlay:
      return RESPONSE(
          success=True,
          visual_overlay_renderer_keys=list(self._visual_overlay),
      )
    else:
      return RESPONSE(error_message=_ERROR_NO_RENDERER_FOUND)

  @_check_orchestrator_interface_requirements(observer_check="disallow")
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

  @_check_orchestrator_interface_requirements(observer_check="disallow")
  def get_visual_overlay_image_as_pil_image(
      self,
      renderer_key: str,
  ) -> RESPONSE:
    """Returns the visual overlay image as PIL image."""
    if renderer_key not in self._visual_overlay:
      return RESPONSE(error_message=_ERROR_NO_RENDERER_FOUND)

    return self._visual_overlay[renderer_key].get_image_as_pil_image()

  @_check_orchestrator_interface_requirements(observer_check="disallow")
  def get_visual_overlay_image_as_np_array(
      self,
      renderer_key: str,
  ) -> RESPONSE:
    """Returns the visual overlay image as numpy array."""
    if renderer_key not in self._visual_overlay:
      return RESPONSE(error_message=_ERROR_NO_RENDERER_FOUND)

    return self._visual_overlay[renderer_key].get_image_as_np_array()

  @_check_orchestrator_interface_requirements(observer_check="disallow")
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

  @_check_orchestrator_interface_requirements(observer_check="disallow")
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

  @_check_orchestrator_interface_requirements(observer_check="disallow")
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

  @_check_orchestrator_interface_requirements(observer_check="disallow")
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

  @_check_orchestrator_interface_requirements(
      observer_check="disallow",
      require_connection=True,
      required_libs=["_robot_job_lib", "_robot_job_work_unit_lib"],
  )
  def robot_job_work_unit_start_software_asset_prep(self) -> RESPONSE:
    """Sets the current work unit's stage as software asset prep."""
    assert self._robot_job_work_unit_lib is not None
    with self._rpc_lock:
      return self._robot_job_work_unit_lib.start_work_unit_software_asset_prep()

  @_check_orchestrator_interface_requirements(
      observer_check="disallow",
      require_connection=True,
      required_libs=["_robot_job_lib", "_robot_job_work_unit_lib"],
  )
  def robot_job_work_unit_start_scene_prep(self) -> RESPONSE:
    """Starts the current work unit's stage as scene prep."""
    assert self._robot_job_work_unit_lib is not None
    with self._rpc_lock:
      return self._robot_job_work_unit_lib.start_work_unit_scene_prep()

  @_check_orchestrator_interface_requirements(
      observer_check="disallow",
      require_connection=True,
      required_libs=["_robot_job_lib", "_robot_job_work_unit_lib"],
  )
  def robot_job_work_unit_start_execution(self) -> RESPONSE:
    """Set the current work unit's stage as executing."""
    assert self._robot_job_work_unit_lib is not None
    with self._rpc_lock:
      return self._robot_job_work_unit_lib.start_work_unit_execution()

  @_check_orchestrator_interface_requirements(
      observer_check="disallow",
      require_connection=True,
      required_libs=["_robot_job_lib", "_robot_job_work_unit_lib"],
  )
  def robot_job_work_unit_insert_session_info(
      self,
      session_log_type: str,
      session_start_time_ns: int,
      session_end_time_ns: int,
      session_note: str,
  ) -> RESPONSE:
    """Insert session info for the current work unit."""
    assert self._robot_job_work_unit_lib is not None
    with self._rpc_lock:
      return self._robot_job_work_unit_lib.insert_session_info(
          session_log_type=session_log_type,
          session_start_time_ns=session_start_time_ns,
          session_end_time_ns=session_end_time_ns,
          session_note=session_note,
      )

  @_check_orchestrator_interface_requirements(
      observer_check="disallow",
      require_connection=True,
      required_libs=["_robot_job_lib", "_robot_job_work_unit_lib"],
  )
  def robot_job_work_unit_complete_work_unit(
      self,
      outcome: robot_job_work_unit.WORK_UNIT_OUTCOME,
      success_score: float | None,
      success_score_definition: str | None,
      session_start_time_ns: int | None,
      session_end_time_ns: int | None,
      session_log_type: str | None,
      response_to_questions: (
          list[robot_job_work_unit.WORK_UNIT_QUESTION] | None
      ),
      note: str,
  ) -> RESPONSE:
    """Sets the current work unit's stage as completed."""
    assert self._robot_job_work_unit_lib is not None
    with self._rpc_lock:
      return self._robot_job_work_unit_lib.complete_work_unit(
          outcome=outcome,
          success_score=success_score,
          success_score_definition=success_score_definition,
          session_start_time_ns=session_start_time_ns,
          session_end_time_ns=session_end_time_ns,
          session_log_type=session_log_type,
          response_to_questions=response_to_questions,
          note=note,
      )

  @_check_orchestrator_interface_requirements(
      require_connection=True,
      required_libs=["_artifact_lib"],
  )
  def get_artifact(self, artifact_id: str) -> RESPONSE:
    """Gets the artifact's download URI."""
    assert self._artifact_lib is not None
    with self._rpc_lock:
      return self._artifact_lib.get_artifact(artifact_id=artifact_id)

  @_check_orchestrator_interface_requirements(
      require_connection=True,
      required_libs=["_artifact_lib"],
  )
  def get_artifact_uri(self, artifact_id: str) -> RESPONSE:
    """Gets the artifact's download URI."""
    assert self._artifact_lib is not None
    with self._rpc_lock:
      return self._artifact_lib.get_artifact_uri(artifact_id=artifact_id)

  @_check_orchestrator_interface_requirements(
      require_connection=True,
      required_libs=["_artifact_lib"],
  )
  def upload_text_log_artifact(
      self,
      source_file_name: str,
      text_file_bytes: bytes,
  ) -> RESPONSE:
    """Uploads a text log artifact."""
    assert self._artifact_lib is not None
    assert self._robot_job_lib is not None
    with self._rpc_lock:
      robot_job_response = self._robot_job_lib.get_current_robot_job()
    if not robot_job_response.success:
      return robot_job_response
    robot_job_id = robot_job_response.robot_job_id
    with self._rpc_lock:
      return self._artifact_lib.upload_text_log_artifact(
          robot_job_id=robot_job_id,
          robot_id=self._robot_id,
          source_file_name=source_file_name,
          text_file_bytes=text_file_bytes,
      )

  @_check_orchestrator_interface_requirements(
      observer_check="disallow",
      require_connection=True,
      required_libs=["_rui_workcell_state_lib"],
  )
  def load_rui_workcell_state(self, robot_id: str) -> RESPONSE:
    """Loads the RUI workcell state for the given robot."""
    assert self._rui_workcell_state_lib is not None
    with self._rpc_lock:
      return self._rui_workcell_state_lib.load_rui_workcell_state(
          robot_id=robot_id
      )

  @_check_orchestrator_interface_requirements(
      observer_check="disallow",
      require_connection=True,
      required_libs=["_rui_workcell_state_lib"],
  )
  def set_rui_workcell_state(
      self, robot_id: str, workcell_state: str
  ) -> RESPONSE:
    """Sets the RUI workcell state for the given robot."""
    assert self._rui_workcell_state_lib is not None
    with self._rpc_lock:
      return self._rui_workcell_state_lib.set_rui_workcell_state(
          robot_id=robot_id, workcell_state=workcell_state
      )

  @_check_orchestrator_interface_requirements(
      observer_check="require",
      require_connection=True,
      required_libs=["_robot_job_work_unit_lib"],
  )
  def observe_latest_work_unit(self) -> RESPONSE:
    """Observes the latest work unit."""
    assert self._robot_job_work_unit_lib is not None
    with self._rpc_lock:
      return self._robot_job_work_unit_lib.observe_latest_work_unit(
          job_type=self._job_type
      )
