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

"""Primary Orchestrator Client API interface for integration into run binaries.

In order to make it easy to use the Orchestrator client API, the user only needs
to use this module to do all interactions with the Orchestrator server. The
expected flow for integration and using this module is as follows:

Integration:

1. Import this module into your binary.
2. Create an instance of this OrchestratorHelper in the user's robot binary.
3. Call connect() to establish a connection to the Orchestrator server.

Primary use of this OrchestratorHelper instance:

1. Call request_work_unit() to request for a work unit from the server.
2. Call get_current_work_unit() to get currently assigned work unit's
   information.
3. Call start_work_unit_software_asset_prep() to both acknowledge accepting this
   work unit and update the work unit's stage to software asset prep, which
   allows the binary to start doing software asset preparation work for this
   work unit.
4. Once the software asset preparation is done, the user should call
   start_work_unit_scene_prep() to signal the server that the binary is moving
   to do physical scene preparation.
5. Once the scene preparation is done and the user is ready to start the episode
   in question, the user should call start_work_unit_execution() to signal the
   server that the binary is starting the episode execution.
6. Once the episode is determined to be completed by the user, the user should
   call complete_work_unit() to signal the server that this work unit is
   completed.
7. Repeat steps 1-6 for as many work units as needed.
8. Call disconnect() to close the connection to the Orchestrator server.

Additional use of this OrchestratorHelper instance:

1. Get the current active connection to the Orchestrator server by calling
   get_current_connection().
2. Call get_current_robot_info() to get the current robot information.
   This includes the currently set operator ID for the robot (if any), as well
   as the currently assigned work unit's information (if any).
3. Call set_current_robot_operator_id() to set the current operator ID for the
   robot. To clear the current operator ID, pass in an empty string.
4. Work Units can now contain additional information about the visual overlay
   system. This system allows the user to apply explicit overlay objects drawing
   to an image provided during binary execution. Please see the below section
   for usage details.

Using the visual overlay renderers as part of a work unit:

1. After you have successfully requested a work unit, you can have this helper
   library automatically create the visual overlay renderers by calling
   create_visual_overlays_for_current_work_unit().
2. Since there can be multiple image sources at the same time, it is important
   to call list_visual_overlay_renderer_keys() to get the list of all visual
   overlay renderers. These index keys names are the same as the sourceTopic
   field within each of the reference images.
3. To apply the visual overlay onto the current image, call
   render_visual_overlay() with the correct renderer key and the new image
   to apply the overlay to.
4. Retrieve the newly overlaid image as a PIL image, numpy array, or bytes
   object using the appropriate getter functions:
      get_visual_overlay_image_as_pil_image()
      get_visual_overlay_image_as_np_array()
      get_visual_overlay_image_as_bytes()
5. You can repeat steps 3 and 4 continuously to generate a psuedo-live stream
   view if your image is not static.

[OPTIONAL]
   If at any point you want to clear the data within a specific visual overlay
   renderers, you can call reset_visual_overlay_renderer() with the matching
   renderer key.

   To clear the data within all visual overlay renderers, just set the
   "reset_all_renderers" parameter to True when calling
   reset_visual_overlay_renderer().

   By default, calling create_visual_overlays_for_current_work_unit() will
   automatically clear all previously created renderers and their data.

Using the visual overlay renderers without a work unit:

1. Create a visual overlay renderer by calling
   create_single_visual_overlay_renderer().
2. Add overlay objects to the visual overlay renderer by calling
   add_single_overlay_object_to_visual_overlay() as many times as needed.
3. Render the visual overlay onto an image by calling render_visual_overlay()
   with the previously created renderer key and the new image to apply the
   overlay to.
4. Retrieve the newly overlaid image as a PIL image, numpy array, or bytes
   object using the appropriate getter functions:
      get_visual_overlay_image_as_pil_image()
      get_visual_overlay_image_as_np_array()
      get_visual_overlay_image_as_bytes()
5. You can repeat steps 3 and 4 continuously to generate a psuedo-live stream
   view if your image is not static.

[OPTIONAL]
   If at any point you want to clear the data within a specific visual overlay
   renderers, you can call reset_visual_overlay_renderer() with the matching
   renderer key.

   To clear the data within all visual overlay renderers, just set the
   "reset_all_renderers" parameter to True when calling
   reset_visual_overlay_renderer().

For an example on how to integrate and use this module, please see the example
mock binary at:
  orchestrator/example_client_sdk_integration.py.
  orchestrator/example_client_sdk_robot_and_operator_info.py.
"""

from safari_sdk.orchestrator.client import interface

RESPONSE = interface.RESPONSE
JOB_TYPE = interface.JOB_TYPE
WORK_UNIT = interface.WORK_UNIT
WORK_UNIT_OUTCOME = interface.WORK_UNIT_OUTCOME
ACCEPTED_IMAGE_TYPES = interface.ACCEPTED_IMAGE_TYPES
IMAGE_FORMAT = interface.IMAGE_FORMAT
DRAW_CIRCLE_ICON = interface.DRAW_CIRCLE_ICON
DRAW_ARROW_ICON = interface.DRAW_ARROW_ICON
DRAW_SQUARE_ICON = interface.DRAW_SQUARE_ICON
DRAW_TRIANGLE_ICON = interface.DRAW_TRIANGLE_ICON
DRAW_CONTAINER = interface.DRAW_CONTAINER

_ERROR_NO_ACTIVE_CONNECTION = (
    "OrchestratorHelper: No active connection. Please call connect() first."
)


class OrchestratorHelper:
  """Helper to simplify usage of Orchestrator Client API calls."""

  def __init__(
      self,
      *,
      robot_id: str,
      job_type: JOB_TYPE,
      raise_error: bool = False,
  ):
    self._interface: interface.OrchestratorInterface = None
    self._robot_id = robot_id
    self._job_type = job_type
    self._raise_error = raise_error

  def connect(self) -> RESPONSE:
    """Connects to the orchestrator server."""
    if self._interface is not None:
      self._interface.disconnect()

    self._interface = interface.OrchestratorInterface(
        robot_id=self._robot_id,
        job_type=self._job_type,
    )
    response = self._interface.connect()
    if not response.success:
      self._interface = None
      if self._raise_error:
        raise ValueError(response.error_message)
    return response

  def disconnect(self) -> None:
    """Disconnects from the orchestrator server."""
    if self._interface is not None:
      self._interface.disconnect()
      self._interface = None

  def get_current_connection(self) -> RESPONSE:
    """Gets the current active connection to the orchestrator server."""
    if self._interface is None:
      if self._raise_error:
        raise ValueError(_ERROR_NO_ACTIVE_CONNECTION)
      return RESPONSE(error_message=_ERROR_NO_ACTIVE_CONNECTION)

    return self._interface.get_current_connection()

  def get_current_robot_info(self) -> RESPONSE:
    """Gets the current robot information."""
    if self._interface is None:
      if self._raise_error:
        raise ValueError(_ERROR_NO_ACTIVE_CONNECTION)
      return RESPONSE(error_message=_ERROR_NO_ACTIVE_CONNECTION)

    return self._interface.get_current_robot_info()

  def set_current_robot_operator_id(self, operator_id: str) -> RESPONSE:
    """Set the current operator ID for the robot."""
    if self._interface is None:
      if self._raise_error:
        raise ValueError(_ERROR_NO_ACTIVE_CONNECTION)
      return RESPONSE(error_message=_ERROR_NO_ACTIVE_CONNECTION)

    return self._interface.set_current_robot_operator_id(operator_id)

  def request_work_unit(self) -> RESPONSE:
    """Requests for a work unit to execute on the robot."""
    if self._interface is None:
      if self._raise_error:
        raise ValueError(_ERROR_NO_ACTIVE_CONNECTION)
      return RESPONSE(error_message=_ERROR_NO_ACTIVE_CONNECTION)

    return self._interface.request_robot_job_work_unit()

  def get_current_work_unit(self) -> RESPONSE:
    """Get currently assigned work unit's information."""
    if self._interface is None:
      if self._raise_error:
        raise ValueError(_ERROR_NO_ACTIVE_CONNECTION)
      return RESPONSE(error_message=_ERROR_NO_ACTIVE_CONNECTION)

    return self._interface.get_current_robot_job_work_unit()

  def is_visual_overlay_in_current_work_unit(self) -> RESPONSE:
    """Checks if the current work unit has visual overlay information."""
    if self._interface is None:
      if self._raise_error:
        raise ValueError(_ERROR_NO_ACTIVE_CONNECTION)
      return RESPONSE(error_message=_ERROR_NO_ACTIVE_CONNECTION)

    return self._interface.is_visual_overlay_in_current_work_unit()

  def create_visual_overlays_for_current_work_unit(self) -> RESPONSE:
    """Creates visual overlay renderers based on the current work unit."""
    if self._interface is None:
      if self._raise_error:
        raise ValueError(_ERROR_NO_ACTIVE_CONNECTION)
      return RESPONSE(error_message=_ERROR_NO_ACTIVE_CONNECTION)

    return self._interface.create_visual_overlays_for_current_work_unit()

  def list_visual_overlay_renderer_keys(self) -> RESPONSE:
    """Lists index key name for all visual overlay renderers."""
    if self._interface is None:
      if self._raise_error:
        raise ValueError(_ERROR_NO_ACTIVE_CONNECTION)
      return RESPONSE(error_message=_ERROR_NO_ACTIVE_CONNECTION)

    return self._interface.list_visual_overlay_renderer_keys()

  def render_visual_overlay(
      self,
      renderer_key: str,
      new_image: ACCEPTED_IMAGE_TYPES | None = None,
  ) -> RESPONSE:
    """Renders the visual overlay for the given renderer ID."""
    if self._interface is None:
      if self._raise_error:
        raise ValueError(_ERROR_NO_ACTIVE_CONNECTION)
      return RESPONSE(error_message=_ERROR_NO_ACTIVE_CONNECTION)

    return self._interface.render_visual_overlay(
        renderer_key=renderer_key, new_image=new_image
    )

  def get_visual_overlay_image_as_pil_image(
      self,
      renderer_key: str,
  ) -> RESPONSE:
    """Returns the visual overlay image as PIL image."""
    if self._interface is None:
      if self._raise_error:
        raise ValueError(_ERROR_NO_ACTIVE_CONNECTION)
      return RESPONSE(error_message=_ERROR_NO_ACTIVE_CONNECTION)

    return self._interface.get_visual_overlay_image_as_pil_image(
        renderer_key=renderer_key
    )

  def get_visual_overlay_image_as_np_array(
      self,
      renderer_key: str,
  ) -> RESPONSE:
    """Returns the visual overlay image as numpy array."""
    if self._interface is None:
      if self._raise_error:
        raise ValueError(_ERROR_NO_ACTIVE_CONNECTION)
      return RESPONSE(error_message=_ERROR_NO_ACTIVE_CONNECTION)

    return self._interface.get_visual_overlay_image_as_np_array(
        renderer_key=renderer_key
    )

  def get_visual_overlay_image_as_bytes(
      self,
      renderer_key: str,
      img_format: IMAGE_FORMAT = IMAGE_FORMAT.JPEG,
  ) -> RESPONSE:
    """Returns the visual overlay image as bytes in the specified format."""
    if self._interface is None:
      if self._raise_error:
        raise ValueError(_ERROR_NO_ACTIVE_CONNECTION)
      return RESPONSE(error_message=_ERROR_NO_ACTIVE_CONNECTION)

    return self._interface.get_visual_overlay_image_as_bytes(
        renderer_key=renderer_key, img_format=img_format
    )

  def reset_visual_overlay_renderer(
      self, renderer_key: str, reset_all_renderers: bool = False
  ) -> RESPONSE:
    """Resets specific or all visual overlay renderers."""
    if self._interface is None:
      if self._raise_error:
        raise ValueError(_ERROR_NO_ACTIVE_CONNECTION)
      return RESPONSE(error_message=_ERROR_NO_ACTIVE_CONNECTION)

    return self._interface.reset_visual_overlay_renderer(
        renderer_key=renderer_key, reset_all_renderers=reset_all_renderers
    )

  def create_single_visual_overlay_renderer(
      self,
      renderer_key: str,
      image_pixel_width: int,
      image_pixel_height: int,
      overlay_bg_color: str = "#444444",
  ) -> RESPONSE:
    """Manually create a single visual overlay renderer."""
    if self._interface is None:
      if self._raise_error:
        raise ValueError(_ERROR_NO_ACTIVE_CONNECTION)
      return RESPONSE(error_message=_ERROR_NO_ACTIVE_CONNECTION)

    return self._interface.create_single_visual_overlay_renderer(
        renderer_key=renderer_key,
        image_pixel_width=image_pixel_width,
        image_pixel_height=image_pixel_height,
        overlay_bg_color=overlay_bg_color,
    )

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
    if self._interface is None:
      if self._raise_error:
        raise ValueError(_ERROR_NO_ACTIVE_CONNECTION)
      return RESPONSE(error_message=_ERROR_NO_ACTIVE_CONNECTION)

    return self._interface.add_single_overlay_object_to_visual_overlay(
        renderer_key=renderer_key, overlay_object=overlay_object
    )

  def start_work_unit_software_asset_prep(self) -> RESPONSE:
    """Sets the current work unit's stage as software asset prep."""
    if self._interface is None:
      if self._raise_error:
        raise ValueError(_ERROR_NO_ACTIVE_CONNECTION)
      return RESPONSE(error_message=_ERROR_NO_ACTIVE_CONNECTION)

    return self._interface.robot_job_work_unit_start_software_asset_prep()

  def start_work_unit_scene_prep(self) -> RESPONSE:
    """Sets the current work unit's stage as scene prep."""
    if self._interface is None:
      if self._raise_error:
        raise ValueError(_ERROR_NO_ACTIVE_CONNECTION)
      return RESPONSE(error_message=_ERROR_NO_ACTIVE_CONNECTION)

    return self._interface.robot_job_work_unit_start_scene_prep()

  def start_work_unit_execution(self) -> RESPONSE:
    """Sets the current work unit's stage as executing."""
    if self._interface is None:
      if self._raise_error:
        raise ValueError(_ERROR_NO_ACTIVE_CONNECTION)
      return RESPONSE(error_message=_ERROR_NO_ACTIVE_CONNECTION)

    return self._interface.robot_job_work_unit_start_execution()

  def complete_work_unit(
      self, outcome: WORK_UNIT_OUTCOME, note: str
  ) -> RESPONSE:
    """Sets the current work unit's stage as completed."""
    if self._interface is None:
      if self._raise_error:
        raise ValueError(_ERROR_NO_ACTIVE_CONNECTION)
      return RESPONSE(error_message=_ERROR_NO_ACTIVE_CONNECTION)

    return self._interface.robot_job_work_unit_complete_work_unit(
        outcome=outcome, note=note
    )
