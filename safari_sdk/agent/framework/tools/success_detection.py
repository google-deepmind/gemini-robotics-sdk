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

"""A collection of success detection tools."""

import abc
import asyncio
import copy
import datetime
import time
from typing import Any, Optional, Sequence

from absl import logging
from google import genai
from google.genai import types

from safari_sdk.agent.framework import config as framework_config
from safari_sdk.agent.framework import constants
from safari_sdk.agent.framework import types as framework_types
from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.tools import genai_client
from safari_sdk.agent.framework.tools import image_buffer
from safari_sdk.agent.framework.tools import tool

_TASK_SUCCESS_KEY = "task_success"
_OVERALL_TASK_SUCCESS_KEY = "overall_task_success"
_TIMEOUT_KEY = "timeout"
_REASON_KEY = "reason"

_GUIDED_THINKING_SD_QUESTIONS = """
  Q1 Tell me about the state of task relevent objects at each time by combining all views to form a complete picture.
  Q2 What are the relationships between the relevant objects at each time?
  Q3 What state should the objects be in if the robot successfully completed the task?
  Q4 Are the objects currently in the state consistent with the state where the robot successfully completed at the current time? What are the differences? Tell me what views and times do you base your answer on first.

  Example answers for the questions above for the robot task: put the spoon in the compost bin:
  Q1:
    1) The spoon is on the table now and at start, based on left and right wrist and overhead cameras.
    2) The compost bin is empty with the lid open now and at start, based on the overhead cameras.
  Q2:
    1) the spoon is outside of the compost bin now and at start
  Q3:
    1) The spoon should be inside the compost bin. The overhead camera should show the spoon in the compost bin.
  Q4:
    1) No, the spoon is on the table instead of in the compost bin based on the overhead camera and answers to Q1.

  After you answered these questions, tell me if the robot successfully completed the task.
  Use the most linient way to intrepret task completion. If you think the robot partially completed or it may have completed the task (cannot be determined based on the current views), count it as completed.
  If the task specifies a quantity (e.g., one, a, two), the minimum requirement is met as soon as at least that number of items is in the correct final state.
  The state of any other similar items that were not part of the task does not matter.

  If you think the robot finished the task, answer: 'FINAL ANSWER: yes'
  If you think the roboy did not finish the task, answer: 'FINAL ANSWER: no'.
  Do not mention the word 'FINAL ANSWER' anywhere else."""


class AbstractSuccessDetectionTool(tool.Tool, abc.ABC):
  """Abstract class for success signal tools."""

  def __init__(
      self,
      bus: event_bus.EventBus,
      declaration: types.FunctionDeclaration,
      fn: framework_types.AsyncFunction,
  ):
    super().__init__(
        bus=bus,
        fn=fn,
        declaration=declaration,
    )
    # The ID of the latest call to this tool, to track cancellation. Note that
    # if the tool is queried twice, only the newest call can be cancelled!
    self._call_id = None

    # The timeout in seconds for the success detection.
    self._timeout_seconds = None

    # Subscribe to the external success signal events.
    self._external_success_signal = None
    bus.subscribe(
        event_types=[event_bus.EventType.SUCCESS_SIGNAL],
        handler=self._handle_external_success_signal_event,
    )

  def set_timeout_seconds(self, timeout_seconds: Optional[float]):
    """Set the time limit for the success detection."""
    self._timeout_seconds = timeout_seconds

  def get_timeout_seconds(self) -> float:
    """Get the time limit for the success detection."""
    if self._timeout_seconds is None:
      return float("inf")  # Effectively no timeout.
    return self._timeout_seconds

  def _handle_tool_call_cancellation_events(self, event: event_bus.Event):
    """Handle the subscribed events."""
    # Subscribed to the `TOOL_CALL_CANCELLATION` event in the superclass.
    if self._call_id in event.data.ids:
      self._tool_call_cancelled = True
      logging.info(
          "Cancelled success detection for call_id: %s.",
          self._call_id,
      )

  def _handle_external_success_signal_event(self, event: event_bus.Event):
    """Handle the success signal events."""
    self._external_success_signal = event.data
    logging.info(
        "External success signal received: %s",
        self._external_success_signal,
    )


class VisionSuccessDetectionTool(AbstractSuccessDetectionTool):
  """Base class for success detection tools."""

  def __init__(
      self,
      bus: event_bus.EventBus,
      declaration: types.FunctionDeclaration,
      fn: framework_types.AsyncFunction,
      config: framework_config.AgentFrameworkConfig,
      api_key: str,
      camera_endpoint_names: Sequence[str],
  ):
    super().__init__(
        bus=bus,
        fn=fn,
        declaration=declaration,
    )
    self._config = config
    # Gemini client for success detection.
    self._client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(base_url=config.base_url),
    )

    # Subscribe to the model input image events.
    self._image_buffer = image_buffer.ImageBuffer(camera_endpoint_names, config)
    bus.subscribe(
        event_types=[event_bus.EventType.MODEL_IMAGE_INPUT],
        handler=self._image_buffer.handle_model_image_input_event,
    )


class SubtaskSuccessDetectorV4(VisionSuccessDetectionTool):
  """Success detector for one-shot tasks based on state verification."""

  def __init__(
      self,
      bus: event_bus.EventBus,
      config: framework_config.AgentFrameworkConfig,
      api_key: str,
      sd_camera_endpoint_names: Optional[Sequence[str]] = None,
  ):
    super().__init__(
        bus=bus,
        fn=self.detect_success,
        declaration=types.FunctionDeclaration(
            name="detect_success",
            description=(
                "Detects subtask success as frequent as possible. This is a"
                " non-blocking tool. It will detect success and only return"
                ' {"task_success": True} when the subtask is successful.'
                " Although not part of the parameters, this function requires"
                " at least one video stream from a camera endpoint. But will"
                " likely work better with more cameras to provide different"
                " angles."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "subtask": types.Schema(
                        type=types.Type.STRING,
                        description=(
                            "A short, clear subtask for the robot (e.g. 'pick"
                            " up the cup', 'place on the table', 'move left')"
                        ),
                    ),
                },
                required=["subtask"],
            ),
            behavior=types.Behavior.NON_BLOCKING,
        ),
        config=config,
        api_key=api_key,
        camera_endpoint_names=sd_camera_endpoint_names,
    )
    self._gemini_wrapper = genai_client.GeminiClientWrapper(
        bus=bus,
        client=self._client,
        model_name=config.sd_model_name,
        config=types.GenerateContentConfig(
            temperature=config.sd_temperature,
            thinking_config=types.ThinkingConfig(
                include_thoughts=False,
                thinking_budget=config.sd_thinking_budget,
            ),
        ),
        print_raw_response=config.sd_print_raw_sd_response,
    )
    self._task = None
    self._image_buffer.reset_start_images_map()
    self._image_buffer.reset_latest_images_map()
    self._success_signal = False
    self._success_signal_images_time = None
    self._external_success_signal = None
    self.set_timeout_seconds(config.sd_timeout_seconds)

  async def detect_success(
      self, subtask: str, call_id: str
  ) -> types.FunctionResponse:
    """Returns the success detection result for a given subtask.

    Args:
      subtask: The subtask to be evaluated for success.
      call_id: The ID of the tool call.
    """
    # Record the id of the tool call and mark it as running.
    self._call_id = call_id
    self._tool_call_cancelled = False

    # Reset the task success properties.
    self._task = subtask
    self._success_signal = False
    self._success_signal_images_time = None
    self._external_success_signal = None
    self._image_buffer.reset_start_images_map()
    self._image_buffer.reset_latest_images_map()
    await asyncio.sleep(0.5)  # wait for the image buffer to be ready.

    # Set a timeout for the task to be successful.
    timeout_seconds = self.get_timeout_seconds()
    end_time = time.time() + timeout_seconds

    # Check if the task is successful.
    background_tasks = set()
    response = types.FunctionResponse(
        response={_TASK_SUCCESS_KEY: True, _TIMEOUT_KEY: False},
        will_continue=False,
    )
    while not self._success_signal:
      # Check if the tool call is cancelled.
      if self._tool_call_cancelled:
        response = types.FunctionResponse(
            response={_TASK_SUCCESS_KEY: False, _TIMEOUT_KEY: False},
            will_continue=False,
        )
        break
      # Check if we received an external success signal.
      if self._external_success_signal is not None:
        response = types.FunctionResponse(
            response={
                _TASK_SUCCESS_KEY: self._external_success_signal,
                _TIMEOUT_KEY: False,
            },
            will_continue=False,
        )
        break
      # Check if the time limit is reached.
      if time.time() > end_time:
        logging.info("SD: Timed out after %s seconds.", timeout_seconds)
        response = types.FunctionResponse(
            response={_TASK_SUCCESS_KEY: False, _TIMEOUT_KEY: True},
            will_continue=False,
        )
        break
      # If dry run is enabled, sleep for a short time and continue.
      # This skips the actual SD check and still allows the timeout to work.
      if self._config.sd_dry_run:
        await asyncio.sleep(0.5)
        continue
      # Check the current task success.
      task = asyncio.create_task(self._query_success_signal(subtask))
      background_tasks.add(task)
      task.add_done_callback(background_tasks.discard)
      await asyncio.sleep(self._config.sd_async_sd_interval_s)
    for background_task in background_tasks:
      background_task.cancel()
    return response

  async def _query_success_signal(
      self,
      subtask: str,
  ) -> None:
    """Queries the success signal for a given subtask."""
    # Step 0: Check if there are any images available.
    current_images_timestamp = self._image_buffer.get_latest_image_timestamp()
    if not current_images_timestamp:
      logging.warning("[SD]: No images available.")
      return
    # Step 1: Build the prompt and query the model.
    query_success_signal_start_time = time.time()
    prompt = self._build_prompt(subtask=subtask)
    try:
      response = await self._gemini_wrapper.generate_content(contents=prompt)
      if response.text is None:
        logging.warning("[SD] No response text from the model.")
        success_signal = False
      else:
        success_signal = (
            "FINAL ANSWER: yes" in response.text
            or "$\boxed{yes}$" in response.text
            or "$\\boxed{yes}$" in response.text
            or "**Final Answer:** yes" in response.text
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception("[SD] Model failed to respond:\n%s", e)
      success_signal = False
    query_success_signal_end_time = time.time()
    logging.info(
        "[SD] TASK: %s, RAW SIGNAL: %s, QUERY TIME: %s.",
        subtask,
        success_signal,
        query_success_signal_end_time - query_success_signal_start_time,
    )
    # Step 2: Update the success signal.
    # The subtask is no longer the current task.
    if self._task != subtask:
      return
    # The first time we receive a success signal for the current task.
    if not self._success_signal_images_time:
      self._success_signal_images_time = current_images_timestamp
      self._success_signal = success_signal
      return
    # The success signal based on stale images.
    if current_images_timestamp <= self._success_signal_images_time:
      return
    self._success_signal_images_time = current_images_timestamp
    self._success_signal = success_signal

  def _build_prompt(
      self,
      subtask: str,
  ) -> list[Any]:
    """Builds the prompt for success detection.

    Args:
      subtask: The subtask to be evaluated for success.

    Returns:
      A list containing the prompt elements (text and image data).
    """
    prompt = []
    image_buffer_copy = copy.deepcopy(self._image_buffer)
    if self._config.sd_use_start_images:
      prompt.append(
          "The images below show what the robot saw at the start of the task"
          " (the robot may have more than one camera):"
      )
      start_images_map = image_buffer_copy.get_start_images_map()
      for stream_name in image_buffer_copy.get_camera_endpoint_names():
        if (
            stream_name in start_images_map
            and start_images_map[stream_name] is not None
        ):
          prompt.append(stream_name.replace("/", "").strip() + ":")
          prompt.append(start_images_map[stream_name].data)
    if self._config.sd_num_history_frames > 0:
      prompt.append(
          "The images below show what the robot saw recently (the robot may"
          " have more than one camera):"
      )
      latest_images_map = image_buffer_copy.get_latest_images_map()
      now = datetime.datetime.now()
      for stream_name in image_buffer_copy.get_camera_endpoint_names():
        if (
            stream_name in latest_images_map
            and latest_images_map[stream_name] is not None
        ):
          image_events = latest_images_map[stream_name]
          for i in range(len(image_events) - 1):
            event = image_events[i]
            elapsed_time = (now - event.timestamp).total_seconds()
            endpoint_name = event.metadata[constants.STREAM_NAME_METADATA_KEY]
            prompt.append(
                endpoint_name.replace("/", "").strip()
                + f" from {elapsed_time:.1f} seconds ago:"
            )
            prompt.append(event.data)
    prompt.append(
        "The images below show what the robot sees now (the robot may have more"
        " than one camera):"
    )
    latest_images_map = image_buffer_copy.get_latest_images_map()
    for stream_name in image_buffer_copy.get_camera_endpoint_names():
      if (
          stream_name in latest_images_map
          and latest_images_map[stream_name] is not None
      ):
        prompt.append(stream_name.replace("/", "").strip() + ":")
        prompt.append(latest_images_map[stream_name][-1].data)
    prompt.append(f"The robot's task is: {subtask}")
    if self._config.sd_use_explicit_thinking:
      prompt.append(
          "Given the information above, answer the following questions first:"
      )
    else:
      prompt.append(
          "Given the information above, think silently about the following"
          " questions first (only if you have thinking budget):"
      )
    prompt.append(_GUIDED_THINKING_SD_QUESTIONS)

    if self._config.sd_guided_thinking_word_limit is not None:
      prompt.append(
          "Please limit your response to"
          f" {self._config.sd_guided_thinking_word_limit} words."
      )
    if self._config.sd_print_final_prompt:
      prompt_string = self._prompt_to_single_string(prompt)
      logging.info("Full prompt:\n%s", prompt_string)
    return prompt

  def _prompt_to_single_string(self, prompt: list[Any]) -> str:
    """Converts a prompt to a single string with non-text elements redacted.

    Args:
      prompt: A list containing prompt elements (text and image data).

    Returns:
      A single string representation of the prompt where non-text elements
      are replaced with "<REDACTED>".
    """
    result_parts = []
    for element in prompt:
      if isinstance(element, str):
        result_parts.append(element)
      else:
        result_parts.append("<REDACTED>")
    return "\n".join(result_parts)
