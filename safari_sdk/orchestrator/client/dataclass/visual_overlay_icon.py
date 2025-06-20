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

"""Visual overlay icon information for overlay rendering."""

import dataclasses


@dataclasses.dataclass
class DrawCircleIcon:
  """Required information for visual overlay renderer to draw a circle.

  Attributes:
    object_id: The object id of the overlay object.
    overlay_text_label: The overlay text label of the overlay object.
    rgb_hex_color_value: The rgb hex color value of the overlay object.
    layer_order: The layer order of the overlay object.
    x: The x pixel coordinate of the circle's center.
    y: The y pixel coordinate of the cicrle's center.
  """

  object_id: str
  overlay_text_label: str
  rgb_hex_color_value: str
  layer_order: int
  x: int
  y: int


@dataclasses.dataclass
class DrawArrowIcon:
  """Required information for visual overlay renderer to draw an arrow.

  Attributes:
    object_id: The object id of the overlay object.
    overlay_text_label: The overlay text label of the overlay object.
    rgb_hex_color_value: The rgb hex color value of the overlay object.
    layer_order: The layer order of the overlay object.
    x: The x pixel coordinate of the arrow's head.
    y: The y pixel coordinate of the arrow's head.
    rad: The direction of the arrow in radians. Radian of 0 is right, pi/2 is
      up, pi or -pi is left, and -pi/2 is down.
  """

  object_id: str
  overlay_text_label: str
  rgb_hex_color_value: str
  layer_order: int
  x: int
  y: int
  rad: float


@dataclasses.dataclass
class DrawSquareIcon:
  """Required information for visual overlay renderer to draw a square.

  Attributes:
    object_id: The object id of the overlay object.
    overlay_text_label: The overlay text label of the overlay object.
    rgb_hex_color_value: The rgb hex color value of the overlay object.
    layer_order: The layer order of the overlay object.
    x: The x pixel coordinate of the square's center.
    y: The y pixel coordinate of the square's center.
  """

  object_id: str
  overlay_text_label: str
  rgb_hex_color_value: str
  layer_order: int
  x: int
  y: int


@dataclasses.dataclass
class DrawTriangleIcon:
  """Required information for visual overlay renderer to draw a triangle.

  Attributes:
    object_id: The object id of the overlay object.
    overlay_text_label: The overlay text label of the overlay object.
    rgb_hex_color_value: The rgb hex color value of the overlay object.
    layer_order: The layer order of the overlay object.
    x: The x pixel coordinate of the triangle's center.
    y: The y pixel coordinate of the triangle's center.
  """

  object_id: str
  overlay_text_label: str
  rgb_hex_color_value: str
  layer_order: int
  x: int
  y: int


@dataclasses.dataclass
class DrawContainer:
  """Required information for visual overlay renderer to draw a container.

  Attributes:
    object_id: The object id of the overlay object.
    overlay_text_label: The overlay text label of the overlay object.
    rgb_hex_color_value: The rgb hex color value of the overlay object.
    layer_order: The layer order of the overlay object.
    x: The x pixel coordinate of the container. If used with "w" and "h", this
      is the x pixel coordinate of the container's top left corner. If used with
      "radius", this is the x pixel coordinate of the container's center.
    y: The y pixel coordinate of the container. If used with "w" and "h", this
      is the y pixel coordinate of the container's top left corner. If used with
      "radius", this is the y pixel coordinate of the container's center.
    w: The width of the container. If used, then this indicates that this
      container is a rectangle and "h" must also be specified.
    h: The height of the container. If used, then this indicates that this
      container is a rectangle and "w" must also be specified.
    radius: The radius of the container. If used, then this indicates that this
      container is a circle.
  """

  object_id: str
  overlay_text_label: str
  rgb_hex_color_value: str
  layer_order: int
  x: int
  y: int
  w: int | None = None
  h: int | None = None
  radius: int | None = None
