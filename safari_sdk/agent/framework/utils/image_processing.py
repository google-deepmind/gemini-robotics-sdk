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
from typing import Any
from google.genai import types


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
