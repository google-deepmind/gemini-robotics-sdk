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

"""Scene description tool."""

import copy
from typing import Any, Sequence

from absl import logging
from google import genai
from google.genai import types

from safari_sdk.agent.framework import config as framework_config
from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.tools import genai_client
from safari_sdk.agent.framework.tools import image_buffer
from safari_sdk.agent.framework.tools import tool


_SCENE_DESCRIPTION_PROMPT = """
You are an expert at describing scenes based on images from one or more cameras.
For each camera, detect objects in the scene and describe their properties such as color, shape, material, and size.
Merge the object descriptions from all cameras into a single, coherent scene description.
If there are duplicate objects in the scene, use natural language spatial references to disambiguate between them, such as 'the red large bowl on the left' and 'the red large bowl on the right'.
Limit your response to {num_words} words.
"""


class SceneDescriptionTool(tool.Tool):
  """Returns the detailed scene description of the robot's environment."""

  def __init__(
      self,
      bus: event_bus.EventBus,
      config: framework_config.AgentFrameworkConfig,
      api_key: str,
      camera_endpoint_names: Sequence[str],
  ):
    declaration = types.FunctionDeclaration(
        name="describe_scene",
        description=(
            "Returns the detailed scene description of the robot's"
            " environment using all the cameras."
        ),
        behavior=types.Behavior.NON_BLOCKING,
    )
    super().__init__(
        fn=self.describe_scene,
        declaration=declaration,
        bus=bus,
    )
    self._call_id = None
    self._config = config
    # Gemini client for scene description.
    self._client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(base_url=config.base_url),
    )
    self._gemini_wrapper = genai_client.GeminiClientWrapper(
        bus=bus,
        client=self._client,
        model_name=config.scene_description_model_name,
        config=types.GenerateContentConfig(
            temperature=0.0,
            thinking_config=types.ThinkingConfig(
                include_thoughts=False,
                thinking_budget=config.scene_description_thinking_budget,
            ),
        ),
        print_raw_response=False,
    )
    # Subscribe to the model input image events.
    self._image_buffer = image_buffer.ImageBuffer(
        camera_endpoint_names=camera_endpoint_names, config=config
    )
    bus.subscribe(
        event_types=[event_bus.EventType.MODEL_IMAGE_INPUT],
        handler=self._image_buffer.handle_model_image_input_event,
    )

  async def describe_scene(self, call_id: str) -> types.FunctionResponse:
    """Describes the scene using the latest images from the cameras."""
    self._call_id = call_id
    prompt = self._build_prompt()
    try:
      response = await self._gemini_wrapper.generate_content(contents=prompt)
      return types.FunctionResponse(
          response={
              "scene_description": response.text,  # pylint: disable=attribute-error
              "Do not call this tool again.": True,
          },
          will_continue=False,
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception("Scene Descriptor Error: %s", e)
      return types.FunctionResponse(
          response={
              "Failed to describe scene": True,
          },
          will_continue=False,
      )

  def _build_prompt(self) -> list[Any]:
    """Builds the prompt for scene description."""
    prompt = []
    prompt.append(
        "The images below show what the robot sees now (the robot may have more"
        " than one camera):"
    )
    image_buffer_copy = copy.deepcopy(self._image_buffer)
    latest_images_map = image_buffer_copy.get_latest_images_map()
    for stream_name in image_buffer_copy.get_camera_endpoint_names():
      if (
          stream_name in latest_images_map
          and latest_images_map[stream_name] is not None
      ):
        prompt.append(stream_name.replace("/", "").strip() + ":")
        prompt.append(latest_images_map[stream_name][-1].data)
    prompt.append(
        _SCENE_DESCRIPTION_PROMPT.format(
            num_words=self._config.scene_description_num_output_words
        )
    )
    return prompt
