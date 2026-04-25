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

"""Utility functions for image processing."""

from collections.abc import Sequence
import io
import itertools
import math
from typing import Any

from google.genai import types
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


_DEFAULT_CELL_WIDTH = 640
_DEFAULT_CELL_HEIGHT = 480
_LABEL_FONT_SIZE_MIN = 16
_LABEL_FONT_SIZE_MAX = 48
_LABEL_FONT_SIZE_RATIO = 0.04
_LABEL_PADDING_RATIO = 0.015
_LABEL_BG_COLOR = (0, 0, 0, 180)
_LABEL_TEXT_COLOR = (255, 255, 255)


def convert_bytes_to_image(prompt_elems: Sequence[Any]) -> Sequence[Any]:
  """Converts bytes to image in the prompt elements."""
  new_prompt_elems = []
  for prompt_elem in prompt_elems:
    if isinstance(prompt_elem, bytes):
      new_elem = types.Part.from_bytes(
          data=prompt_elem,
          mime_type="image/jpeg",
      )
    else:
      new_elem = prompt_elem
    new_prompt_elems.append(new_elem)
  return new_prompt_elems


def _calculate_grid_dimensions(num_images: int) -> tuple[int, int]:
  """Calculates optimal grid dimensions for the given number of images.

  Args:
    num_images: Number of images to arrange in a grid.

  Returns:
    Tuple of (rows, cols) for the grid layout.
  """
  if num_images <= 0:
    return (0, 0)
  if num_images == 1:
    return (1, 1)
  if num_images == 2:
    return (1, 2)
  if num_images == 3:
    return (1, 3)
  if num_images == 4:
    return (2, 2)

  cols = math.ceil(math.sqrt(num_images))
  rows = math.ceil(num_images / cols)
  return (rows, cols)


def _resize_image_to_cell(
    image: Image.Image, cell_width: int, cell_height: int
) -> Image.Image:
  """Resizes an image to fit within a cell, maintaining aspect ratio.

  Args:
    image: The PIL Image to resize.
    cell_width: Target cell width.
    cell_height: Target cell height.

  Returns:
    Resized PIL Image with letterboxing if needed.
  """
  original_width, original_height = image.size

  width_ratio = cell_width / original_width
  height_ratio = cell_height / original_height
  scaling_factor = min(width_ratio, height_ratio)

  new_width = int(original_width * scaling_factor)
  new_height = int(original_height * scaling_factor)

  resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

  cell = Image.new("RGB", (cell_width, cell_height), (0, 0, 0))
  paste_x = (cell_width - new_width) // 2
  paste_y = (cell_height - new_height) // 2
  cell.paste(resized, (paste_x, paste_y))

  return cell


def _add_label_to_image(
    image: Image.Image, label: str, use_background: bool = False
) -> Image.Image:
  """Adds a text label overlay to the top-left corner of an image.

  Args:
    image: The PIL Image to add label to.
    label: The text label to display.
    use_background: If True, draws a semi-transparent background box behind the
      label. If False, draws the text with a dark outline for visibility.

  Returns:
    PIL Image with label overlay.
  """
  if not label:
    return image

  min_dim = min(image.size)
  font_size = int(min_dim * _LABEL_FONT_SIZE_RATIO)
  font_size_clamped = max(
      _LABEL_FONT_SIZE_MIN, min(font_size, _LABEL_FONT_SIZE_MAX)
  )
  padding = max(5, int(min_dim * _LABEL_PADDING_RATIO))
  try:
    font = ImageFont.load_default(size=font_size_clamped)
  except TypeError:
    try:
      font = ImageFont.truetype(
          "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size_clamped
      )
    except (OSError, IOError):
      font = ImageFont.load_default()
  if use_background:
    result = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(result)

    x1, y1, x2, y2 = draw.textbbox((0, 0), label, font=font)
    text_width = x2 - x1
    text_height = y2 - y1

    bg_x1 = padding
    bg_y1 = padding
    bg_x2 = bg_x1 + text_width + 2 * padding
    bg_y2 = bg_y1 + text_height + 2 * padding

    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=_LABEL_BG_COLOR)
    result_with_overlay = Image.alpha_composite(result, overlay)

    draw = ImageDraw.Draw(result_with_overlay)
    draw.text(
        (bg_x1 + padding, bg_y1 + padding),
        label,
        font=font,
        fill=_LABEL_TEXT_COLOR,
    )
    return result_with_overlay.convert("RGB")
  else:
    result = image.copy().convert("RGB")
    draw = ImageDraw.Draw(result)

    text_x = padding
    text_y = padding
    outline_color = (0, 0, 0)
    outline_width = max(1, font_size_clamped // 12)

    for dx, dy in itertools.product(
        range(-outline_width, outline_width + 1),
        range(-outline_width, outline_width + 1),
    ):
      if dx != 0 or dy != 0:
        draw.text(
            (text_x + dx, text_y + dy), label, font=font, fill=outline_color
        )

    draw.text((text_x, text_y), label, font=font, fill=_LABEL_TEXT_COLOR)
    return result


def _get_image_size(image_data: bytes | Image.Image) -> tuple[int, int]:
  """Gets the size of an image without fully loading bytes images."""
  if isinstance(image_data, Image.Image):
    return image_data.size
  with Image.open(io.BytesIO(image_data)) as img:
    return img.size


def stitch_images(
    images: dict[str, bytes | Image.Image],
    labels: dict[str, str] | None = None,
    cell_width: int | None = None,
    cell_height: int | None = None,
    show_labels: bool = True,
    label_background: bool = False,
    border_color: str | tuple[int, int, int] | None = None,
    image_order: Sequence[str] | None = None,
    border_spacing: int = 4,
) -> bytes:
  """Stitches multiple camera images into a single grid image.

  Args:
    images: Dictionary mapping camera names to JPEG image bytes or PIL Images.
    labels: Optional dictionary mapping camera names to display labels. If None,
      camera names are used as labels.
    cell_width: Width of each cell in the grid. If None, uses max width of all
      input images.
    cell_height: Height of each cell in the grid. If None, uses max height of
      all input images.
    show_labels: Whether to display camera name labels on each cell.
    label_background: If True, labels have a semi-transparent background box. If
      False (default), labels have a dark outline for visibility.
    border_color: Optional color for borders around each cell. Can be a color
      name ('black', 'white') or RGB tuple. If None, no borders are drawn.
    image_order: Optional sequence of image keys specifying the order to arrange
      images in the grid (left-to-right, top-to-bottom). If None, uses
      dictionary insertion order.
    border_spacing: Spacing between images in pixels when border_color is set.
      Defaults to 4.

  Returns:
    JPEG bytes of the stitched image.

  Raises:
    ValueError: If images dict is empty.
  """
  if not images:
    raise ValueError("Cannot stitch empty images dictionary")

  if cell_width is None or cell_height is None:
    max_width = 0
    max_height = 0
    for image_data in images.values():
      w, h = _get_image_size(image_data)
      max_width = max(max_width, w)
      max_height = max(max_height, h)
    if cell_width is None:
      cell_width = max_width
    if cell_height is None:
      cell_height = max_height

  if labels is None:
    labels = {name: name for name in images.keys()}

  num_images = len(images)
  rows, cols = _calculate_grid_dimensions(num_images)

  spacing = border_spacing if border_color is not None else 0
  bg_color = border_color if border_color is not None else (0, 0, 0)

  grid_width = cols * cell_width + (cols + 1) * spacing
  grid_height = rows * cell_height + (rows + 1) * spacing
  grid = Image.new("RGB", (grid_width, grid_height), bg_color)

  ordered_keys = image_order if image_order is not None else list(images.keys())
  for idx, camera_name in enumerate(ordered_keys):
    image_data = images[camera_name]
    row = idx // cols
    col = idx % cols

    if isinstance(image_data, Image.Image):
      pil_image: Image.Image = image_data
    else:
      with Image.open(io.BytesIO(image_data)) as img:
        pil_image = img.copy()
    if pil_image.mode != "RGB":
      pil_image = pil_image.convert("RGB")

    cell = _resize_image_to_cell(pil_image, cell_width, cell_height)

    if show_labels:
      label = labels.get(camera_name, camera_name)
      cell = _add_label_to_image(cell, label, use_background=label_background)

    x = spacing + col * (cell_width + spacing)
    y = spacing + row * (cell_height + spacing)
    grid.paste(cell, (x, y))

  output = io.BytesIO()
  grid.save(output, format="JPEG", quality=85)
  return output.getvalue()


def get_stitched_image_dimensions(
    num_images: int,
    cell_width: int = _DEFAULT_CELL_WIDTH,
    cell_height: int = _DEFAULT_CELL_HEIGHT,
) -> tuple[int, int]:
  """Returns the dimensions of a stitched image for a given number of images.

  Args:
    num_images: Number of images to stitch.
    cell_width: Width of each cell.
    cell_height: Height of each cell.

  Returns:
    Tuple of (width, height) for the stitched image.
  """
  rows, cols = _calculate_grid_dimensions(num_images)
  return (cols * cell_width, rows * cell_height)
