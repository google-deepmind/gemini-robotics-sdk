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

"""Visual overlay renderer library for Orchestrator WorkUnits."""

import enum
import io
import math
from typing import Union

import cv2
import numpy as np
from PIL import Image

from safari_sdk.orchestrator.client.dataclass import api_response
from safari_sdk.orchestrator.client.dataclass import visual_overlay_icon
from safari_sdk.orchestrator.client.dataclass import work_unit

AcceptedImageTypes = Union[Image.Image, np.ndarray, bytes]
_RESPONSE = api_response.OrchestratorAPIResponse

_ERROR_NO_SCENE_OBJECTS_FOUND = (
    "OrchestratorRenderer: No scene objects found in work unit context."
)
_ERROR_NO_SCENE_OBJECTS_FOR_REFERENCE_IMAGE = (
    "OrchestratorRenderer: No scene objects for this reference image."
)
_ERROR_UNDEFINED_OBJECT_TYPE = "OrchestratorRenderer: Undefined object type:"

_DEFAULT_FONT_SCALE = 0.75
_DEFAULT_THICKNESS = 2
_DEFAULT_MARKER_SIZE = 15
_DEFAULT_ARROW_LENGTH_SCALE_FACTOR = 1.0
_DEFAULT_DOT_SIZE = 6


class ImageFormat(enum.Enum):
  JPEG = "JPEG"
  PNG = "PNG"


class OrchestratorRenderer:
  """Renderer for processing and rendering visual overlay objects."""

  def __init__(
      self,
      scene_reference_image_data: work_unit.SceneReferenceImage,
      overlay_bg_color: str = "#444444",
  ):
    """Initializes the renderer."""
    self._scene_reference_image_data = scene_reference_image_data
    self._overlay_bg_color = overlay_bg_color

    self._reference_image_artifact_id = scene_reference_image_data.artifactId
    self._source_topic = scene_reference_image_data.sourceTopic
    self._overlay_x_size = scene_reference_image_data.rawImageWidth
    self._overlay_y_size = scene_reference_image_data.rawImageHeight
    self._ratio_ref_x = self._compute_image_conversion_ratio(
        initial_size=scene_reference_image_data.renderedCanvasWidth,
        desired_size=self._overlay_x_size,
    )
    self._ratio_ref_y = self._compute_image_conversion_ratio(
        initial_size=scene_reference_image_data.renderedCanvasHeight,
        desired_size=self._overlay_y_size,
    )
    # Assuming that the reference image used is the same size as the camera.
    # This will be rechecked when the user provides an base image to render.
    self._ratio_camera_x = 1.0
    self._ratio_camera_y = 1.0

    self._overlay_image = Image.new(
        mode="RGB",
        size=(self._overlay_x_size, self._overlay_y_size),
        color=self._overlay_bg_color,
    )
    self._overlay_image_np = np.array(self._overlay_image)
    self._workunit_objects: list[work_unit.SceneObject] = []
    self._overlay_objects: list[
        visual_overlay_icon.DrawCircleIcon
        | visual_overlay_icon.DrawArrowIcon
        | visual_overlay_icon.DrawSquareIcon
        | visual_overlay_icon.DrawTriangleIcon
        | visual_overlay_icon.DrawContainer
    ] = []

    self._custom_thickness = None
    self._custom_font_size = None
    self._custom_marker_size = None
    self._custom_dot_size = None
    self._custom_arrow_length_scale_factor = None
    self._adjust_overlay_scale_to_reference_image_size()

  def _compute_image_conversion_ratio(
      self, initial_size: int, desired_size: int
  ) -> float:
    """Compute ratio to transfrom image from intital size to desired size."""
    return float(desired_size) / float(initial_size)

  def reset_all_object_settings(self) -> _RESPONSE:
    """Resets all object settings."""
    self._overlay_image = Image.new(
        mode="RGB",
        size=(self._overlay_x_size, self._overlay_y_size),
        color=self._overlay_bg_color,
    )
    self._overlay_image_np = np.array(self._overlay_image)
    self._workunit_objects: list[work_unit.SceneObject] = []
    self._overlay_objects: list[
        visual_overlay_icon.DrawCircleIcon
        | visual_overlay_icon.DrawArrowIcon
        | visual_overlay_icon.DrawSquareIcon
        | visual_overlay_icon.DrawTriangleIcon
        | visual_overlay_icon.DrawContainer
    ] = []
    self._custom_thickness = None
    self._custom_font_size = None
    return _RESPONSE(success=True)

  def set_custom_thickness(self, thickness: int) -> _RESPONSE:
    """Sets the custom thickness for all overlay objects."""
    self._custom_thickness = thickness
    return _RESPONSE(success=True)

  def set_custom_font_size(self, font_size: float) -> _RESPONSE:
    """Sets the custom font size for all overlay objects."""
    self._custom_font_size = font_size
    return _RESPONSE(success=True)

  def _adjust_overlay_scale_to_reference_image_size(self) -> None:
    """Sets whether to auto scale the overlay to the reference image size."""
    pixel_size = self._overlay_x_size * self._overlay_y_size
    scale_factor = pixel_size / 1000000.0  # Scaling by megapixels.

    if scale_factor <= 1.0:
      # No scaling needed.
      return None

    scale_factor -= 1.0  # Remove the baseline scaling factor.
    custom_font_size = _DEFAULT_FONT_SCALE + (scale_factor * 0.15)
    custom_thickness = _DEFAULT_THICKNESS + int(scale_factor * 0.5)

    self._custom_arrow_length_scale_factor = (
        _DEFAULT_ARROW_LENGTH_SCALE_FACTOR + int(scale_factor * 0.2)
    )
    self._custom_marker_size = _DEFAULT_MARKER_SIZE + int(scale_factor * 1.25)
    self._custom_dot_size = self._custom_marker_size - 9

    self.set_custom_font_size(font_size=custom_font_size)
    self.set_custom_thickness(thickness=custom_thickness)

    return None

  def get_image_as_pil_image(self) -> _RESPONSE:
    """Returns the overlay image as a PIL image."""
    return _RESPONSE(success=True, visual_overlay_image=self._overlay_image)

  def get_image_as_np_array(self) -> _RESPONSE:
    """Returns the overlay image as a numpy array."""
    return _RESPONSE(success=True, visual_overlay_image=self._overlay_image_np)

  def get_image_as_bytes(
      self, img_format: ImageFormat = ImageFormat.JPEG
  ) -> _RESPONSE:
    """Returns the overlay image as bytes in the specified format."""
    img_byte_arr = io.BytesIO()
    overlay_img = self._overlay_image.convert("RGB")
    overlay_img.save(img_byte_arr, format=img_format.value)
    return _RESPONSE(success=True, visual_overlay_image=img_byte_arr.getvalue())

  def load_scene_objects_from_work_unit(
      self, scene_objects: list[work_unit.SceneObject]
  ) -> _RESPONSE:
    """Load scene objects from Orchestrator WorkUnit."""
    if not scene_objects:
      return _RESPONSE(error_message=_ERROR_NO_SCENE_OBJECTS_FOUND)

    self._workunit_objects = scene_objects
    found_scene_objects = False
    for obj in self._workunit_objects:
      if obj.sceneReferenceImageArtifactId == self._reference_image_artifact_id:
        self._process_scene_object(obj=obj)
        found_scene_objects = True

    if found_scene_objects:
      return _RESPONSE(success=True)
    else:
      return _RESPONSE(
          error_message=_ERROR_NO_SCENE_OBJECTS_FOR_REFERENCE_IMAGE
      )

  def add_single_object(
      self,
      overlay_object: (
          visual_overlay_icon.DrawCircleIcon
          | visual_overlay_icon.DrawArrowIcon
          | visual_overlay_icon.DrawSquareIcon
          | visual_overlay_icon.DrawTriangleIcon
          | visual_overlay_icon.DrawContainer
      ),
  ) -> _RESPONSE:
    """Manually add a single scene object to object list."""
    self._overlay_objects.append(overlay_object)
    return _RESPONSE(success=True)

  def _convert_color_to_tuple(self, color_string: str) -> tuple[int, int, int]:
    assert len(color_string) == 6
    return (
        int(color_string[0:2], 16),
        int(color_string[2:4], 16),
        int(color_string[4:6], 16),
    )

  def _generate_xy_position_with_no_overruns(
      self,
      x: int,
      y: int,
      x_offset: int,
      y_offset: int,
      image_width: int,
      image_height: int,
      text_width: int,
      text_height: int,
      text_baseline: int,
  ) -> tuple[int, int]:
    """Generates text x and y position with no overruns."""

    ideal_x = x + x_offset
    ideal_y = y + y_offset
    is_overrun_x_left = ideal_x <= 5
    is_overrun_x_right = ideal_x + text_width > image_width
    is_overrun_y_top = ideal_y - text_height <= 5
    is_overrun_y_bottom = ideal_y > image_height

    if is_overrun_x_left and not is_overrun_x_right:
      ideal_x = 5
    elif is_overrun_x_right and not is_overrun_x_left:
      ideal_x = image_width - text_width - 5

    if is_overrun_y_top and not is_overrun_y_bottom:
      ideal_y = 5 + text_height
    elif is_overrun_y_bottom and not is_overrun_y_top:
      ideal_y = image_height - text_baseline - 5

    return ideal_x, ideal_y

  def _find_ideal_text_position(
      self,
      text: str,
      font_scale: float,
      thickness: int,
      icon_object: (
          visual_overlay_icon.DrawCircleIcon
          | visual_overlay_icon.DrawArrowIcon
          | visual_overlay_icon.DrawSquareIcon
          | visual_overlay_icon.DrawTriangleIcon
          | visual_overlay_icon.DrawContainer
      ),
      radius: int | None = None,
      arrow_end_point: tuple[int, int] | None = None,
  ) -> tuple[int, int]:
    """Finds the ideal position for the text to be drawn."""
    # Find the size of text box.
    (text_width, text_height), text_baseline = cv2.getTextSize(
        text=text,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        thickness=thickness,
    )
    image_width, image_height = self._overlay_image.size
    obj_type = type(icon_object)

    match obj_type:
      case visual_overlay_icon.DrawCircleIcon:
        if self._custom_dot_size:
          x_shift = self._custom_dot_size + 2
        else:
          x_shift = 14
        return self._generate_xy_position_with_no_overruns(
            x=icon_object.x,
            y=icon_object.y,
            x_offset=x_shift,
            y_offset=5,
            image_width=image_width,
            image_height=image_height,
            text_width=text_width,
            text_height=text_height,
            text_baseline=text_baseline,
        )
      case visual_overlay_icon.DrawArrowIcon:
        assert isinstance(icon_object, visual_overlay_icon.DrawArrowIcon)
        assert arrow_end_point is not None
        end_x, end_y = arrow_end_point
        angle = icon_object.rad if icon_object.rad else 0.0
        angle = math.atan2(math.sin(angle), math.cos(angle))

        if angle <= 0 and angle >= -0.436332:
          x_shift = 10
          y_shift = -text_baseline  # text_baseline + 10
        elif angle >= 0 and angle <= 0.436332:
          x_shift = 10
          y_shift = text_height + text_baseline
        elif angle <= math.pi and angle >= math.pi - 0.436332:
          x_shift = -text_width - 10
          y_shift = text_height + text_baseline
        elif angle >= -math.pi and angle <= -math.pi + 0.436332:
          x_shift = -text_width - 10
          y_shift = -text_baseline
        elif angle > 0:
          x_shift = int(-text_width / 2)
          y_shift = text_height + text_baseline
        else:
          x_shift = int(-text_width / 2)
          y_shift = -(text_baseline * 2)

        return self._generate_xy_position_with_no_overruns(
            x=end_x,
            y=end_y,
            x_offset=x_shift,
            y_offset=y_shift,
            image_width=image_width,
            image_height=image_height,
            text_width=text_width,
            text_height=text_height,
            text_baseline=text_baseline,
        )
      case visual_overlay_icon.DrawSquareIcon:
        return self._generate_xy_position_with_no_overruns(
            x=icon_object.x,
            y=icon_object.y,
            x_offset=self._custom_marker_size or 10,
            y_offset=5,
            image_width=image_width,
            image_height=image_height,
            text_width=text_width,
            text_height=text_height,
            text_baseline=text_baseline,
        )
      case visual_overlay_icon.DrawTriangleIcon:
        return self._generate_xy_position_with_no_overruns(
            x=icon_object.x,
            y=icon_object.y,
            x_offset=self._custom_marker_size or 10,
            y_offset=5,
            image_width=image_width,
            image_height=image_height,
            text_width=text_width,
            text_height=text_height,
            text_baseline=text_baseline,
        )
      case visual_overlay_icon.DrawContainer:
        # Rectanglular bounding box.
        if radius is None:
          assert isinstance(icon_object, visual_overlay_icon.DrawContainer)
          assert icon_object.w is not None and icon_object.h is not None
          upper_left_x = icon_object.x
          upper_left_y = icon_object.y
          if icon_object.w < 0:
            upper_left_x += icon_object.w
          if icon_object.h < 0:
            upper_left_y += icon_object.h
          return self._generate_xy_position_with_no_overruns(
              x=upper_left_x,
              y=upper_left_y,
              x_offset=0,
              y_offset=-12,
              image_width=image_width,
              image_height=image_height,
              text_width=text_width,
              text_height=text_height,
              text_baseline=text_baseline,
          )
        else:
          # Circular bounding box.
          return self._generate_xy_position_with_no_overruns(
              x=icon_object.x,
              y=icon_object.y,
              x_offset=0,
              y_offset=radius + 12,
              image_width=image_width,
              image_height=image_height,
              text_width=text_width,
              text_height=text_height,
              text_baseline=text_baseline,
          )
      case _:
        return icon_object.x, icon_object.y

  def render_overlay(
      self, new_image: AcceptedImageTypes | None = None
  ) -> _RESPONSE:
    """Renders the overlay."""
    if new_image is not None:
      self._update_image(image=new_image)

    for obj in self._overlay_objects:
      obj_type = type(obj)
      match obj_type:
        case visual_overlay_icon.DrawCircleIcon:
          self._draw_overlay_circle(
              x=obj.x,
              y=obj.y,
              radius=self._custom_dot_size or _DEFAULT_DOT_SIZE,
              color=self._convert_color_to_tuple(obj.rgb_hex_color_value),
              # thickness=self._custom_thickness or -1,
              thickness=-1,
          )
          text_x, text_y = self._find_ideal_text_position(
              text=obj.overlay_text_label,
              font_scale=self._custom_font_size or _DEFAULT_FONT_SCALE,
              thickness=self._custom_thickness or _DEFAULT_THICKNESS,
              icon_object=obj,
          )
          self._write_text(
              x=text_x,
              y=text_y,
              display_text=obj.overlay_text_label,
              color=(255, 255, 255),
              font_scale=self._custom_font_size or _DEFAULT_FONT_SCALE,
              thickness=self._custom_thickness or _DEFAULT_THICKNESS,
          )
        case visual_overlay_icon.DrawArrowIcon:
          assert isinstance(obj, visual_overlay_icon.DrawArrowIcon)
          arrow_end_x, arrow_end_y = self._draw_overlay_arrow(
              x=obj.x,
              y=obj.y,
              angle=obj.rad if obj.rad else 0.0,
              color=self._convert_color_to_tuple(obj.rgb_hex_color_value),
              thickness=self._custom_thickness or 4,
          )

          text_x, text_y = self._find_ideal_text_position(
              text=obj.overlay_text_label,
              font_scale=self._custom_font_size or _DEFAULT_FONT_SCALE,
              thickness=self._custom_thickness or _DEFAULT_THICKNESS,
              icon_object=obj,
              arrow_end_point=(arrow_end_x, arrow_end_y),
          )
          # Need to dynamically set XY position based on angle...
          self._write_text(
              x=text_x,
              y=text_y,
              display_text=obj.overlay_text_label,
              color=(255, 255, 255),
              font_scale=self._custom_font_size or _DEFAULT_FONT_SCALE,
              thickness=self._custom_thickness or _DEFAULT_THICKNESS,
          )
        case visual_overlay_icon.DrawSquareIcon:
          self._draw_overlay_square(
              x=obj.x,
              y=obj.y,
              color=self._convert_color_to_tuple(obj.rgb_hex_color_value),
              marker_size=self._custom_marker_size or _DEFAULT_MARKER_SIZE,
              thickness=self._custom_thickness or _DEFAULT_THICKNESS,
          )
          text_x, text_y = self._find_ideal_text_position(
              text=obj.overlay_text_label,
              font_scale=self._custom_font_size or _DEFAULT_FONT_SCALE,
              thickness=self._custom_thickness or _DEFAULT_THICKNESS,
              icon_object=obj,
          )
          self._write_text(
              x=text_x,
              y=text_y,
              display_text=obj.overlay_text_label,
              color=(255, 255, 255),
              font_scale=self._custom_font_size or _DEFAULT_FONT_SCALE,
              thickness=self._custom_thickness or _DEFAULT_THICKNESS,
          )
        case visual_overlay_icon.DrawTriangleIcon:
          self._draw_overlay_triangle(
              x=obj.x,
              y=obj.y,
              color=self._convert_color_to_tuple(obj.rgb_hex_color_value),
              marker_size=self._custom_marker_size or _DEFAULT_MARKER_SIZE,
              thickness=self._custom_thickness or _DEFAULT_THICKNESS,
          )
          text_x, text_y = self._find_ideal_text_position(
              text=obj.overlay_text_label,
              font_scale=self._custom_font_size or _DEFAULT_FONT_SCALE,
              thickness=self._custom_thickness or _DEFAULT_THICKNESS,
              icon_object=obj,
          )
          self._write_text(
              x=text_x,
              y=text_y,
              display_text=obj.overlay_text_label,
              color=(255, 255, 255),
              font_scale=self._custom_font_size or _DEFAULT_FONT_SCALE,
              thickness=self._custom_thickness or _DEFAULT_THICKNESS,
          )
        case visual_overlay_icon.DrawContainer:
          assert isinstance(obj, visual_overlay_icon.DrawContainer)
          if obj.w is not None and obj.h is not None:
            self._draw_overlay_box(
                x=obj.x,
                y=obj.y,
                w=obj.w,
                h=obj.h,
                color=self._convert_color_to_tuple(obj.rgb_hex_color_value),
                thickness=self._custom_thickness or _DEFAULT_THICKNESS,
            )
            text_x, text_y = self._find_ideal_text_position(
                text=obj.overlay_text_label,
                font_scale=self._custom_font_size or _DEFAULT_FONT_SCALE,
                thickness=self._custom_thickness or _DEFAULT_THICKNESS,
                icon_object=obj,
            )
            self._write_text(
                x=text_x,
                y=text_y,
                display_text=obj.overlay_text_label,
                color=(255, 255, 255),
                font_scale=self._custom_font_size or _DEFAULT_FONT_SCALE,
                thickness=self._custom_thickness or _DEFAULT_THICKNESS,
            )
          elif obj.radius is not None:
            self._draw_overlay_circle(
                x=obj.x,
                y=obj.y,
                radius=obj.radius,
                color=self._convert_color_to_tuple(obj.rgb_hex_color_value),
                thickness=self._custom_thickness or _DEFAULT_THICKNESS,
            )
            text_x, text_y = self._find_ideal_text_position(
                text=obj.overlay_text_label,
                font_scale=self._custom_font_size or _DEFAULT_FONT_SCALE,
                thickness=self._custom_thickness or _DEFAULT_THICKNESS,
                icon_object=obj,
                radius=obj.radius,
            )
            self._write_text(
                x=text_x,
                y=text_y,
                display_text=obj.overlay_text_label,
                color=(255, 255, 255),
                font_scale=self._custom_font_size or _DEFAULT_FONT_SCALE,
                thickness=self._custom_thickness or _DEFAULT_THICKNESS,
            )
        case _:
          _RESPONSE(error_message=f"{_ERROR_UNDEFINED_OBJECT_TYPE} {obj_type}")
    return _RESPONSE(success=True)

  def _update_image(self, image: AcceptedImageTypes) -> None:
    """Updates the underlaying image to be drawn on."""
    current_image_size = self._overlay_image.size
    if isinstance(image, Image.Image):
      self._overlay_image = image
      self._overlay_image_np = np.array(self._overlay_image)
    elif isinstance(image, np.ndarray):
      self._overlay_image = Image.fromarray(image)
      self._overlay_image_np = image
    elif isinstance(image, bytes):
      self._overlay_image = Image.open(io.BytesIO(image))
      self._overlay_image_np = np.array(self._overlay_image)
    else:
      raise ValueError(f"Unsupported image type: {type(image)}")

    updated_image_size = self._overlay_image.size
    if current_image_size != updated_image_size:
      self._ratio_camera_x = self._compute_image_conversion_ratio(
          initial_size=self._overlay_x_size,
          desired_size=updated_image_size[0],
      )
      self._ratio_camera_y = self._compute_image_conversion_ratio(
          initial_size=self._overlay_y_size,
          desired_size=updated_image_size[1],
      )
      self._overlay_objects.clear()
      for obj in self._workunit_objects:
        if (
            obj.sceneReferenceImageArtifactId
            == self._reference_image_artifact_id
        ):
          self._process_scene_object(obj=obj)

  def _draw_overlay_arrow(
      self,
      x: int,
      y: int,
      angle: float,
      color: tuple[int, int, int] = (255, 0, 0),
      thickness: int = 4,
  ) -> tuple[int, int]:
    """Draw arrow."""
    arrow_length_scale_factor = (
        self._custom_arrow_length_scale_factor
        or _DEFAULT_ARROW_LENGTH_SCALE_FACTOR
    )
    arrow_end = (
        int(x + (50.0 * arrow_length_scale_factor) * (math.cos(angle))),
        int(y + (50.0 * arrow_length_scale_factor) * (math.sin(angle))),
    )
    self._overlay_image_np = cv2.arrowedLine(
        img=self._overlay_image_np,
        pt1=arrow_end,
        pt2=(x, y),
        color=color,
        thickness=thickness,
        tipLength=0.5,
    )
    self._overlay_image = Image.fromarray(self._overlay_image_np)
    return arrow_end

  def _draw_overlay_circle(
      self,
      x: int,
      y: int,
      radius: int = 25,
      color: tuple[int, int, int] = (255, 0, 0),
      thickness: int = -1,
  ) -> None:
    """Draw circle."""
    self._overlay_image_np = cv2.circle(
        img=self._overlay_image_np,
        center=(x, y),
        radius=radius,
        color=color,
        thickness=thickness,
    )
    self._overlay_image = Image.fromarray(self._overlay_image_np)

  def _draw_overlay_box(
      self,
      x: int,
      y: int,
      w: int,
      h: int,
      color: tuple[int, int, int] = (255, 0, 0),
      thickness: int = _DEFAULT_THICKNESS,
  ) -> None:
    """Draw box."""
    self._overlay_image_np = cv2.rectangle(
        img=self._overlay_image_np,
        pt1=(x, y),
        pt2=(x + w, y + h),
        color=color,
        thickness=thickness,
    )
    self._overlay_image = Image.fromarray(self._overlay_image_np)

  def _draw_overlay_triangle(
      self,
      x: int,
      y: int,
      color: tuple[int, int, int] = (255, 0, 0),
      marker_size: int = _DEFAULT_MARKER_SIZE,
      thickness: int = _DEFAULT_THICKNESS,
  ) -> None:
    """Draw triangle."""
    self._draw_icon(
        x=x,
        y=y,
        icon_type=cv2.MARKER_TRIANGLE_UP,
        color=color,
        marker_size=marker_size,
        thickness=thickness,
    )

  def _draw_overlay_square(
      self,
      x: int,
      y: int,
      color: tuple[int, int, int] = (255, 0, 0),
      marker_size: int = _DEFAULT_MARKER_SIZE,
      thickness: int = _DEFAULT_THICKNESS,
  ) -> None:
    """Draw square."""
    self._draw_icon(
        x=x,
        y=y,
        icon_type=cv2.MARKER_SQUARE,
        color=color,
        marker_size=marker_size,
        thickness=thickness,
    )

  def _draw_icon(
      self,
      x: int,
      y: int,
      icon_type: int,
      color: tuple[int, int, int] = (255, 0, 0),
      marker_size: int = _DEFAULT_MARKER_SIZE,
      thickness: int = _DEFAULT_THICKNESS,
  ) -> None:
    """Draw requested native CV2 icons."""
    self._overlay_image_np = cv2.drawMarker(
        img=self._overlay_image_np,
        position=(x, y),
        color=color,
        markerType=icon_type,
        markerSize=marker_size,
        thickness=thickness,
    )
    self._overlay_image = Image.fromarray(self._overlay_image_np)

  def _write_text(
      self,
      x: int,  # Lower left of text box.
      y: int,  # Lower left of text box.
      display_text: str,
      color: tuple[int, int, int] = (255, 0, 0),
      font_scale: float = _DEFAULT_FONT_SCALE,
      thickness: int = _DEFAULT_THICKNESS,
  ) -> None:
    """Draw text into image via CV2."""
    (text_width, text_height), text_baseline = cv2.getTextSize(
        text=display_text,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        thickness=thickness,
    )
    black_background_color = (0, 0, 0)
    self._overlay_image_np = cv2.rectangle(
        img=self._overlay_image_np,
        pt1=(x - 2, y - text_height - 2),
        pt2=(x + text_width + 2, y + text_baseline + 2),
        color=black_background_color,
        thickness=-1,
    )
    self._overlay_image_np = cv2.putText(
        img=self._overlay_image_np,
        text=display_text,
        org=(x, y),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        color=color,
        thickness=thickness,
    )
    self._overlay_image = Image.fromarray(self._overlay_image_np)

  def _scale_to_image_size(
      self, value: int, ratio_ui_to_ref: float, ratio_ref_to_camera: float
  ) -> int:
    """Scale coordinate to image size."""
    return int(value * ratio_ui_to_ref * ratio_ref_to_camera)

  def _process_scene_object(self, obj: work_unit.SceneObject) -> None:
    """Processes a single scene object."""

    object_id = obj.objectId
    overlay_text_label = ""
    if obj.overlayTextLabels and obj.overlayTextLabels.labels:
      overlay_text_label = obj.overlayTextLabels.labels[0].text
    overlay_icon = obj.evaluationLocation.overlayIcon
    layer_order = (
        obj.evaluationLocation.layerOrder
        if obj.evaluationLocation.layerOrder
        else 0
    )
    rgb_hex_color_value = (
        obj.evaluationLocation.rgbHexColorValue
        if obj.evaluationLocation.rgbHexColorValue
        else "FF0000"
    )

    match overlay_icon:
      case work_unit.OverlayObjectIcon.OVERLAY_OBJECT_ICON_CIRCLE:
        x = self._scale_to_image_size(
            value=obj.evaluationLocation.location.coordinate.x,
            ratio_ui_to_ref=self._ratio_ref_x,
            ratio_ref_to_camera=self._ratio_camera_x,
        )
        y = self._scale_to_image_size(
            value=obj.evaluationLocation.location.coordinate.y,
            ratio_ui_to_ref=self._ratio_ref_y,
            ratio_ref_to_camera=self._ratio_camera_y,
        )
        self._overlay_objects.append(
            visual_overlay_icon.DrawCircleIcon(
                object_id=object_id,
                overlay_text_label=overlay_text_label,
                rgb_hex_color_value=rgb_hex_color_value,
                layer_order=layer_order,
                x=x,
                y=y,
            )
        )
      case work_unit.OverlayObjectIcon.OVERLAY_OBJECT_ICON_ARROW:
        x = self._scale_to_image_size(
            value=obj.evaluationLocation.location.coordinate.x,
            ratio_ui_to_ref=self._ratio_ref_x,
            ratio_ref_to_camera=self._ratio_camera_x,
        )
        y = self._scale_to_image_size(
            value=obj.evaluationLocation.location.coordinate.y,
            ratio_ui_to_ref=self._ratio_ref_y,
            ratio_ref_to_camera=self._ratio_camera_y,
        )
        self._overlay_objects.append(
            visual_overlay_icon.DrawArrowIcon(
                object_id=object_id,
                overlay_text_label=overlay_text_label,
                rgb_hex_color_value=rgb_hex_color_value,
                layer_order=layer_order,
                x=x,
                y=y,
                rad=obj.evaluationLocation.location.direction.rad,
            )
        )
      case work_unit.OverlayObjectIcon.OVERLAY_OBJECT_ICON_SQUARE:
        x = self._scale_to_image_size(
            value=obj.evaluationLocation.location.coordinate.x,
            ratio_ui_to_ref=self._ratio_ref_x,
            ratio_ref_to_camera=self._ratio_camera_x,
        )
        y = self._scale_to_image_size(
            value=obj.evaluationLocation.location.coordinate.y,
            ratio_ui_to_ref=self._ratio_ref_y,
            ratio_ref_to_camera=self._ratio_camera_y,
        )
        self._overlay_objects.append(
            visual_overlay_icon.DrawSquareIcon(
                object_id=object_id,
                overlay_text_label=overlay_text_label,
                rgb_hex_color_value=rgb_hex_color_value,
                layer_order=layer_order,
                x=x,
                y=y,
            )
        )
      case work_unit.OverlayObjectIcon.OVERLAY_OBJECT_ICON_TRIANGLE:
        x = self._scale_to_image_size(
            value=obj.evaluationLocation.location.coordinate.x,
            ratio_ui_to_ref=self._ratio_ref_x,
            ratio_ref_to_camera=self._ratio_camera_x,
        )
        y = self._scale_to_image_size(
            value=obj.evaluationLocation.location.coordinate.y,
            ratio_ui_to_ref=self._ratio_ref_y,
            ratio_ref_to_camera=self._ratio_camera_y,
        )
        self._overlay_objects.append(
            visual_overlay_icon.DrawTriangleIcon(
                object_id=object_id,
                overlay_text_label=overlay_text_label,
                rgb_hex_color_value=rgb_hex_color_value,
                layer_order=layer_order,
                x=x,
                y=y,
            )
        )
      case work_unit.OverlayObjectIcon.OVERLAY_OBJECT_ICON_CONTAINER:
        if obj.evaluationLocation.containerArea.circle:
          x = self._scale_to_image_size(
              value=obj.evaluationLocation.containerArea.circle.center.x,
              ratio_ui_to_ref=self._ratio_ref_x,
              ratio_ref_to_camera=self._ratio_camera_x,
          )
          y = self._scale_to_image_size(
              value=obj.evaluationLocation.containerArea.circle.center.y,
              ratio_ui_to_ref=self._ratio_ref_y,
              ratio_ref_to_camera=self._ratio_camera_y,
          )
          radius = self._scale_to_image_size(
              value=obj.evaluationLocation.containerArea.circle.radius,
              ratio_ui_to_ref=max(self._ratio_ref_x, self._ratio_ref_y),
              ratio_ref_to_camera=max(
                  self._ratio_camera_x, self._ratio_camera_y
              ),

          )
          self._overlay_objects.append(
              visual_overlay_icon.DrawContainer(
                  object_id=object_id,
                  overlay_text_label=overlay_text_label,
                  rgb_hex_color_value=rgb_hex_color_value,
                  layer_order=layer_order,
                  x=x,
                  y=y,
                  radius=radius,
              )
          )
        elif obj.evaluationLocation.containerArea.box:
          x = self._scale_to_image_size(
              value=obj.evaluationLocation.containerArea.box.x,
              ratio_ui_to_ref=self._ratio_ref_x,
              ratio_ref_to_camera=self._ratio_camera_x,
          )
          y = self._scale_to_image_size(
              value=obj.evaluationLocation.containerArea.box.y,
              ratio_ui_to_ref=self._ratio_ref_y,
              ratio_ref_to_camera=self._ratio_camera_y,
          )
          w = self._scale_to_image_size(
              value=obj.evaluationLocation.containerArea.box.w,
              ratio_ui_to_ref=self._ratio_ref_x,
              ratio_ref_to_camera=self._ratio_camera_x,
          )
          h = self._scale_to_image_size(
              value=obj.evaluationLocation.containerArea.box.h,
              ratio_ui_to_ref=self._ratio_ref_y,
              ratio_ref_to_camera=self._ratio_camera_y,
          )
          self._overlay_objects.append(
              visual_overlay_icon.DrawContainer(
                  object_id=object_id,
                  overlay_text_label=overlay_text_label,
                  rgb_hex_color_value=rgb_hex_color_value,
                  layer_order=layer_order,
                  x=x,
                  y=y,
                  w=w,
                  h=h,
              )
          )
        else:
          raise ValueError(f"No container area specified for: {object_id}")
      case _:
        raise ValueError(
            f"Undefined object icon type [{overlay_icon}] for {object_id}"
        )
