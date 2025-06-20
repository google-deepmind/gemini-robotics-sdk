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
    self._ratio_x = self._compute_ui_to_image_ratio(
        size_ui=scene_reference_image_data.renderedCanvasWidth,
        size_image=self._overlay_x_size,
    )
    self._ratio_y = self._compute_ui_to_image_ratio(
        size_ui=scene_reference_image_data.renderedCanvasHeight,
        size_image=self._overlay_y_size,
    )
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

  def _compute_ui_to_image_ratio(self, size_ui: int, size_image: int) -> float:
    """Compute ratio of UI to image size."""
    return float(size_image) / float(size_ui)

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
    return _RESPONSE(success=True)

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

  def render_overlay(
      self, new_image: AcceptedImageTypes | None = None
  ) -> _RESPONSE:
    """Renders the overlay."""
    if new_image is not None:
      self._update_image(image=new_image)

    for obj in self._overlay_objects:
      if isinstance(obj, visual_overlay_icon.DrawCircleIcon):
        self._draw_overlay_circle(
            x=obj.x,
            y=obj.y,
            radius=4,
            color=self._convert_color_to_tuple(obj.rgb_hex_color_value),
        )
        self._write_text(
            x=obj.x + 10,
            y=obj.y + 5,
            display_text=obj.overlay_text_label,
            color=(255, 255, 255),
        )
      elif isinstance(obj, visual_overlay_icon.DrawArrowIcon):
        self._draw_overlay_arrow(
            x=obj.x,
            y=obj.y,
            angle=obj.rad if obj.rad else 0.0,
            color=self._convert_color_to_tuple(obj.rgb_hex_color_value),
        )
        # Need to dynamically set XY position based on angle...
        self._write_text(
            x=obj.x + 10,
            y=obj.y + 10,
            display_text=obj.overlay_text_label,
            color=(255, 255, 255),
        )
      elif isinstance(obj, visual_overlay_icon.DrawSquareIcon):
        self._draw_overlay_square(
            x=obj.x,
            y=obj.y,
            color=self._convert_color_to_tuple(obj.rgb_hex_color_value),
        )
        self._write_text(
            x=obj.x + 10,
            y=obj.y + 5,
            display_text=obj.overlay_text_label,
            color=(255, 255, 255),
        )
      elif isinstance(obj, visual_overlay_icon.DrawTriangleIcon):
        self._draw_overlay_triangle(
            x=obj.x,
            y=obj.y,
            color=self._convert_color_to_tuple(obj.rgb_hex_color_value),
        )
        self._write_text(
            x=obj.x + 10,
            y=obj.y + 5,
            display_text=obj.overlay_text_label,
            color=(255, 255, 255),
        )
      elif isinstance(obj, visual_overlay_icon.DrawContainer):
        if obj.w is not None and obj.h is not None:
          self._draw_overlay_box(
              x=obj.x,
              y=obj.y,
              w=obj.w,
              h=obj.h,
              color=self._convert_color_to_tuple(obj.rgb_hex_color_value),
              thickness=1,
          )
          self._write_text(
              x=obj.x,
              y=obj.y - 8,
              display_text=obj.overlay_text_label,
              color=(255, 255, 255),
          )
        elif obj.radius is not None:
          self._draw_overlay_circle(
              x=obj.x,
              y=obj.y,
              radius=obj.radius,
              color=self._convert_color_to_tuple(obj.rgb_hex_color_value),
              thickness=1,
          )
          self._write_text(
              x=obj.x,
              y=obj.y + obj.radius + 10,
              display_text=obj.overlay_text_label,
              color=(255, 255, 255),
          )
      else:
        _RESPONSE(error_message=f"{_ERROR_UNDEFINED_OBJECT_TYPE} {type(obj)}")
    return _RESPONSE(success=True)

  def _update_image(self, image: AcceptedImageTypes) -> None:
    """Updates the underlaying image to be drawn on."""
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

  def _draw_overlay_arrow(
      self,
      x: int,
      y: int,
      angle: float,
      color: tuple[int, int, int] = (255, 0, 0),
      thickness: int = 4,
  ) -> None:
    """Draw arrow."""
    arrow_end = (
        int(x + 50.0 * (math.cos(angle))),
        int(y + 50.0 * (math.sin(angle))),
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
      thickness: int = 4,
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
  ) -> None:
    """Draw triangle."""
    self._draw_icon(
        x=x,
        y=y,
        icon_type=cv2.MARKER_TRIANGLE_UP,
        color=color,
    )

  def _draw_overlay_square(
      self,
      x: int,
      y: int,
      color: tuple[int, int, int] = (255, 0, 0),
  ) -> None:
    """Draw square."""
    self._draw_icon(
        x=x,
        y=y,
        icon_type=cv2.MARKER_SQUARE,
        color=color,
    )

  def _draw_icon(
      self,
      x: int,
      y: int,
      icon_type: int,
      color: tuple[int, int, int] = (255, 0, 0),
  ) -> None:
    """Draw requested native CV2 icons."""
    self._overlay_image_np = cv2.drawMarker(
        img=self._overlay_image_np,
        position=(x, y),
        color=color,
        markerType=icon_type,
        markerSize=15,
        thickness=2,
    )
    self._overlay_image = Image.fromarray(self._overlay_image_np)

  def _write_text(
      self,
      x: int,  # Lower left of text box.
      y: int,  # Lower left of text box.
      display_text: str,
      color: tuple[int, int, int] = (255, 0, 0),
      font_scale: float = 0.5,
      thickness: int = 1,
  ) -> None:
    """Draw text into image via CV2."""
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

  def _scale_to_image_size(self, value: int, ratio: float) -> int:
    """Scale coordinate to image size."""
    return int(value * ratio)

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
            ratio=self._ratio_x,
        )
        y = self._scale_to_image_size(
            value=obj.evaluationLocation.location.coordinate.y,
            ratio=self._ratio_y,
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
            ratio=self._ratio_x,
        )
        y = self._scale_to_image_size(
            value=obj.evaluationLocation.location.coordinate.y,
            ratio=self._ratio_y,
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
            ratio=self._ratio_x,
        )
        y = self._scale_to_image_size(
            value=obj.evaluationLocation.location.coordinate.y,
            ratio=self._ratio_y,
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
            ratio=self._ratio_x,
        )
        y = self._scale_to_image_size(
            value=obj.evaluationLocation.location.coordinate.y,
            ratio=self._ratio_y,
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
              ratio=self._ratio_x,
          )
          y = self._scale_to_image_size(
              value=obj.evaluationLocation.containerArea.circle.center.y,
              ratio=self._ratio_y,
          )
          radius = self._scale_to_image_size(
              value=obj.evaluationLocation.containerArea.circle.radius,
              ratio=max(self._ratio_x, self._ratio_y),
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
              ratio=self._ratio_x,
          )
          y = self._scale_to_image_size(
              value=obj.evaluationLocation.containerArea.box.y,
              ratio=self._ratio_y,
          )
          w = self._scale_to_image_size(
              value=obj.evaluationLocation.containerArea.box.w,
              ratio=self._ratio_x,
          )
          h = self._scale_to_image_size(
              value=obj.evaluationLocation.containerArea.box.h,
              ratio=self._ratio_y,
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
