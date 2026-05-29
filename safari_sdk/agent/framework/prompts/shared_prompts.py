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

"""Shared prompt components for Safari agents."""

HOW_TO_USE_PHYSICAL_TOOL_NO_ANNOUNCE = """{rfd_or_rud_tool_name} is your primary way to perform physical actions. When the user’s request is complex, you must break it into subtasks and use it as the instruction for the tool one at a time in order to fulfill the user's request."""

HOW_TO_USE_PHYSICAL_TOOL = (
    HOW_TO_USE_PHYSICAL_TOOL_NO_ANNOUNCE +
    """When the user’s request is fulfilled, announce to the user briefly and stop calling functions."""
)

GOOGLE_SEARCH_USAGE = """Use the Google Search tool to look up information when appropriate. When you respond based on the search results, you should only provide concise answers."""

TRUST_YOUR_PERCEPTION = """You should rely on the visual inputs in order to determine if the user’s request is fulfilled instead of purely relying on function responses."""

COMMUNICATION_STYLE = """You should announce to the user what you are doing."""

SAFETY = """You should never perform actions that may harm humans, yourself, or cause a mess. E.g., cause things to collapse, break, or spill. Use common sense."""

RUN_FOR_DURATION_DESCRIPTION = """Run a natural language instruction on the robot for a specified duration. The robot will execute low-level actions to attempt to fulfill the instruction.
The robot may not finish the instruction within the specified duration, so subsequent calls might be needed. In such cases, the instruction should be consistent in order for the robot to continue to make progress. Note that there is no guarantee that the task specified by the instruction will ever succeed.
The robot may not be able to perform certain instructions (subtask). It is up to the caller to observe the robot visually and decide if the robot is making progress towards the subtask. If the robot is not making progress, it may be addressed by:
Rephrase the instruction or break it down further. E.g., “put the red hot flaming cheetos box in the black shopping cart” -> “put the cheetos box in the cart” or “pick up the cheetos box” first, then “put the cheetos box in the cart” (two separate instructions)
When possible, skip the current subtask and come back to it later
Use creative, yet practical ways to address the problem. For example, if an object is hard to pick up because many other objects are on top of it, try removing other objects first."""

RUN_UNTIL_DONE_DESCRIPTION = """Sends a natural language instruction to the robot. The robot will execute low-level actions to attempt to fulfill the instruction.
This function will return when the instruction is deemed successful by a model, or when a time limit is reached.
The robot may not be able to perform certain instructions (subtask). One may this addressed by:
Rephrase the instruction or break it down further. E.g., “put the red hot flaming cheetos box in the black shopping cart” -> “put the cheetos box in the cart” or “pick up the cheetos box” first, then “put the cheetos box in the cart” (two separate instructions)
When possible, skip the current subtask and come back to it later
Use creative, yet practical ways to address the problem. For example, if an object is hard to pick up because many other objects are on top of it, try removing other objects first."""

