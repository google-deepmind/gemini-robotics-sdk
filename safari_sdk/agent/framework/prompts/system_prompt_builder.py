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

"""System prompt builder for Safari agents."""


def build_system_prompt(
    identity: str,
    how_to_use_vla: str,
    how_to_use_embodiment_tools: str,
    how_to_use_search: str,
    perception: str,
    communication_style: str,
    safety: str,
    scene_task: str = "",
    misc: str = "",
) -> str:
  """Builds a system prompt from structured components.

  Args:
    identity: Identity, persona, hardware capability. (Agent specific)
    how_to_use_vla: How to use RfD or RuD (VLA). (Shared or formatted)
    how_to_use_embodiment_tools: How to use other embodiment tools. (Shared)
    how_to_use_search: How to use google search. (Shared)
    perception: Trust your perception. (Shared)
    communication_style: Communication style. (Shared)
    safety: Safety. (Shared)
    scene_task: Scene and task description. (Agent specific, optional)
    misc: Misc. (Agent specific, optional)

  Returns:
    The final system prompt string.

  Raises:
    ValueError: If any required argument is empty.
  """
  if not identity:
    raise ValueError("identity is required")
  if not how_to_use_vla:
    raise ValueError("how_to_use_vla is required")
  if not how_to_use_embodiment_tools:
    raise ValueError("how_to_use_embodiment_tools is required")

  if not perception:
    raise ValueError("perception is required")
  if not communication_style:
    raise ValueError("communication_style is required")
  if not safety:
    raise ValueError("safety is required")

  components = [
      identity,
      how_to_use_vla,
      how_to_use_embodiment_tools,
      how_to_use_search,
      perception,
      communication_style,
      safety,
      scene_task,
      misc,
  ]
  # Filter out empty components
  non_empty = [c for c in components if c]
  return "\n\n".join(non_empty)
