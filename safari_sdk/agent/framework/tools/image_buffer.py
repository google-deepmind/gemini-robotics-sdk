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

"""Image buffer to store and manage image events."""

import collections
import datetime
from typing import Sequence

from safari_sdk.agent.framework import config as framework_config
from safari_sdk.agent.framework import constants
from safari_sdk.agent.framework import types as framework_types
from safari_sdk.agent.framework.event_bus import event_bus

Deque = collections.deque


class ImageBuffer:
  """Stores image events from the event bus."""

  def __init__(
      self,
      camera_endpoint_names: Sequence[str],
      config: framework_config.AgentFrameworkConfig,
  ):
    # The latest image events from the event bus.
    # - The key is the camera stream name.
    # - The value is a queue of image events, sorted by timestamp ASC.
    self._latest_images_map = {}
    # The start image events from the event bus.
    # - The key is the camera stream name.
    # - The value is the start image event.
    self._start_image_map = {}
    # The camera endpoint names.
    self._camera_endpoint_names = camera_endpoint_names
    self._config = config

  def get_camera_endpoint_names(self) -> Sequence[str]:
    """Returns the camera endpoint names."""
    return self._camera_endpoint_names

  def get_latest_image_timestamp(self) -> datetime.datetime | None:
    """Returns the timestamp of the latest image event."""
    latest_image_timestamp = None
    if not self._latest_images_map:
      return latest_image_timestamp
    for image_events in self._latest_images_map.values():
      if not image_events:
        continue
      image_timestamp = image_events[-1].timestamp
      if latest_image_timestamp is None:
        latest_image_timestamp = image_timestamp
      else:
        latest_image_timestamp = max(latest_image_timestamp, image_timestamp)
    return latest_image_timestamp

  def get_latest_images_map(
      self,
  ) -> dict[str, Deque[event_bus.Event]]:
    """Returns the latest images."""
    return self._latest_images_map

  def reset_latest_images_map(self):
    """Resets the latest images map."""
    self._latest_images_map = {}

  def get_start_images_map(
      self,
  ) -> dict[str, event_bus.Event]:
    """Returns the start images."""
    return self._start_image_map

  def reset_start_images_map(self):
    """Resets the start images map."""
    self._start_image_map = {}

  def handle_model_image_input_event(self, event: event_bus.Event):
    """Handle the `MODEL_IMAGE_INPUT` event."""
    if event.type != framework_types.EventType.MODEL_IMAGE_INPUT:
      return
    camera_endpoint_name = event.metadata[constants.STREAM_NAME_METADATA_KEY]
    if camera_endpoint_name not in self._camera_endpoint_names:
      return
    if camera_endpoint_name not in self._start_image_map:
      self._start_image_map[camera_endpoint_name] = event
    if camera_endpoint_name not in self._latest_images_map:
      self._latest_images_map[camera_endpoint_name] = Deque()
      self._latest_images_map[camera_endpoint_name].append(event)
      return
    lastest_images = self._latest_images_map[camera_endpoint_name]
    if (
        event.timestamp - lastest_images[-1].timestamp
    ).total_seconds() < self._config.sd_history_interval_s:
      return
    lastest_images.append(event)
    while len(lastest_images) - 1 > self._config.sd_num_history_frames:
      lastest_images.popleft()
