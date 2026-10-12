# Copyright 2026 Google LLC
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

"""Saves Episode data to disk."""

import datetime
import os
import textwrap
from typing import List, Optional

from absl import logging
import cv2
import dm_env
from dm_env import specs
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from typing_extensions import override

from safari_sdk.model import additional_observations_provider


class VideoSaver(
    additional_observations_provider.AdditionalObservationsProvider
):
  """Saves videos and text logs to disk."""

  # Constants for filenames and annotation settings
  _ORIGINAL_VIDEO_FILENAME = 'episode.mp4'
  _ANNOTATED_VIDEO_FILENAME = 'episode_with_thinking.mp4'
  _TEXT_LOG_FILENAME = 'thinking_text.txt'
  _FONT_SIZE = 40

  def __init__(
      self,
      output_dir: str,
      task_instruction_key: str,
      image_observation_key: str,
      fps: float = 30.0,
  ):
    """Initializes the logger.

    Args:
      output_dir: The base directory where episode subdirectories will be
        created.
      task_instruction_key: The observation key for the task instruction string.
      image_observation_key: The observation key for the image observation.
      fps: The frames per second for the output videos.
    """
    self._output_dir = output_dir
    self._task_instruction_key = task_instruction_key
    self._image_observation_key = image_observation_key
    self._fps = fps

    # Buffers to store episode data in memory
    self._frame_buffer: List[np.ndarray] = []
    self._thinking_buffer: List[tuple[str, datetime.datetime]] = []
    self._task_instruction: Optional[str] = None

    try:
      self._font = ImageFont.load_default(size=self._FONT_SIZE)
    except TypeError:
      self._font = ImageFont.load_default()

  def _get_image_from_timestep(self, timestep: dm_env.TimeStep) -> np.ndarray:
    """Extracts and correctly shapes the image array from the observation."""
    image_rgb = timestep.observation[self._image_observation_key].copy()
    if image_rgb.ndim == 4:
      image_rgb = np.squeeze(image_rgb, axis=0)
    return image_rgb

  def _create_episode_subdir(self, task_instruction: str) -> str:
    """Creates a unique, timestamped subdirectory for an episode's artifacts."""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    task_str = (
        task_instruction.replace(' ', '_').replace('/', '_')
        if task_instruction
        else 'no_task_instruction'
    )
    # Ensure task string is a valid directory name component
    safe_task_str = ''.join(
        c for c in task_str if c.isalnum() or c in ('_', '-')
    )[:50]

    subdir_name = f'{safe_task_str}_at_{timestamp}'
    video_dir = os.path.join(self._output_dir, subdir_name)
    os.makedirs(video_dir, exist_ok=True)
    return video_dir

  def _annotate_image_array(self, image: np.ndarray, text: str) -> np.ndarray:
    """Creates and returns an annotated image array with the given text."""
    # This logic remains the same, but is now called in batch from reset()
    img = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(img)

    wrapper = textwrap.TextWrapper(width=80)
    lines = wrapper.wrap(text=text)

    y_text = 10
    text_spacing = self._FONT_SIZE + 5
    for line in lines:
      bbox = draw.textbbox((10, y_text), line, font=self._font)
      draw.rectangle(bbox, fill=(0, 0, 0, 128))
      draw.text((10, y_text), line, (255, 255, 255), font=self._font)
      y_text += text_spacing
    return np.array(img)

  def _process_and_save_episode(self):
    """Processes in-memory buffers to write video and text files to disk."""
    if not self._frame_buffer or self._task_instruction is None:
      logging.warning('Attempted to save episode with no data or instruction.')
      return

    video_dir = self._create_episode_subdir(self._task_instruction)
    annotated_video_path = os.path.join(
        video_dir, self._ANNOTATED_VIDEO_FILENAME
    )
    video_path = os.path.join(video_dir, self._ORIGINAL_VIDEO_FILENAME)
    text_path = os.path.join(video_dir, self._TEXT_LOG_FILENAME)

    first_image = self._frame_buffer[0]
    height, width, _ = first_image.shape
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    cam_writer = None
    cam_thinking_writer = None
    thinking_writer = None

    try:
      cam_thinking_writer = cv2.VideoWriter(
          annotated_video_path, fourcc, self._fps, frame_size
      )
      cam_writer = cv2.VideoWriter(video_path, fourcc, self._fps, frame_size)
      thinking_writer = open(text_path, 'w')
      logging.info(
          'Processing episode... Saving videos and text to: %s', video_dir
      )

      # Now loop through the buffered data and write everything
      num_frames = len(self._frame_buffer)
      for i, (original_image_rgb, (thinking_text, timestamp)) in enumerate(
          zip(self._frame_buffer, self._thinking_buffer)
      ):
        original_image_bgr = cv2.cvtColor(original_image_rgb, cv2.COLOR_RGB2BGR)
        cam_writer.write(original_image_bgr)
        annotated_image_rgb = self._annotate_image_array(
            original_image_rgb, thinking_text
        )
        annotated_image_bgr = cv2.cvtColor(
            annotated_image_rgb, cv2.COLOR_RGB2BGR
        )
        cam_thinking_writer.write(annotated_image_bgr)
        thinking_writer.write(f'Frame {i} ({timestamp}): {thinking_text}\n')
        logging.info('Wrote frame %d of %d.', i, num_frames)

    except Exception as e:  # pylint: disable=broad-except
      logging.error('Failed during batch write of episode data: %s', e)
    finally:
      # Ensure all writers are closed safely
      if cam_thinking_writer:
        cam_thinking_writer.release()
      if cam_writer:
        cam_writer.release()
      if thinking_writer:
        thinking_writer.close()
      logging.info('Finished writing artifact to %s', video_dir)

  @override
  def reset(self):
    """Processes and saves the completed episode data, then clears buffers."""
    self._process_and_save_episode()
    self.dump()

  def dump(self):
    """Clears buffers without saving episode data."""
    logging.info('Clearing buffers for episode data...')
    self._frame_buffer = []
    self._thinking_buffer = []
    self._task_instruction = None

  @override
  def get_additional_observations_spec(self) -> dict[str, specs.Array]:
    """Returns the spec for the additional observations (none)."""
    return {}

  @override
  def get_additional_observations(
      self,
      timestep: dm_env.TimeStep,
      should_replan: bool,
  ) -> dict[str, np.ndarray]:
    """Buffers the frame and thinking text for batch processing on reset."""
    self.get_additional_frame(
        task_instruction=str(timestep.observation[self._task_instruction_key]),
        thinking_text=str(timestep.observation['thinking']),
        cam_frame=self._get_image_from_timestep(timestep),
    )

    return {}

  def get_additional_frame(
      self, task_instruction: str, thinking_text: str, cam_frame
  ) -> dict[str, specs.Array]:
    """Returns the spec for the additional observations (none)."""
    self._task_instruction = task_instruction
    self._frame_buffer.append(cam_frame)
    self._thinking_buffer.append((thinking_text, datetime.datetime.now()))
    return {}
