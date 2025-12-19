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

"""Interface for FastAPI endpoints."""

import dataclasses
from typing import Any

import fastapi


@dataclasses.dataclass(frozen=True)
class FastApiEndpoint:
  """Definition of a canonical FastAPI endpoint.

  This class encapsulates endpoint configuration for FastAPI routes, allowing
  consistent endpoint definitions to be shared between server and client code.

  Attributes:
    path: The path of the endpoint (e.g., "/run/"). Should start and end with
      forward slashes.
    response_class: The FastAPI response class to use for this endpoint. Should
      be a subclass of fastapi.responses.Response, or None to use the default
      JSON response.
  """

  path: str
  response_class: type[fastapi.responses.Response] | None = None

  def __post_init__(self):
    """Validates the endpoint configuration."""
    if not self.path.startswith('/') or not self.path.endswith('/'):
      raise ValueError(
          f'Endpoint path must start and end with "/", got: {self.path}'
      )

  @property
  def args(self) -> dict[str, Any]:
    """Returns FastAPI route arguments as a dictionary.

    The path is guaranteed to be first in the returned dictionary, as required
    by FastAPI's positional argument convention. Only non-None fields are
    included.

    Returns:
      A dictionary of arguments suitable for passing to FastAPI route
      decorators (e.g., app.get(**endpoint.args)).
    """
    return {
        'path': self.path,
        **{
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name != 'path' and getattr(self, field.name) is not None
        },
    }
