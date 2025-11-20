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

"""Orchestrator artifact information."""

import dataclasses
import enum

import dataclasses_json


class ArtifactObjectType(enum.Enum):
  """Orchestrator artifact object type."""

  ARTIFACT_OBJECT_TYPE_UNSPECIFIED = "ARTIFACT_OBJECT_TYPE_UNSPECIFIED"
  ARTIFACT_OBJECT_TYPE_IMAGE = "ARTIFACT_OBJECT_TYPE_IMAGE"
  ARTIFACT_OBJECT_TYPE_VIDEO = "ARTIFACT_OBJECT_TYPE_VIDEO"
  ARTIFACT_OBJECT_TYPE_AUDIO = "ARTIFACT_OBJECT_TYPE_AUDIO"
  ARTIFACT_OBJECT_TYPE_TEXT = "ARTIFACT_OBJECT_TYPE_TEXT"
  ARTIFACT_OBJECT_TYPE_JSON = "ARTIFACT_OBJECT_TYPE_JSON"
  ARTIFACT_OBJECT_TYPE_PROTOBUF = "ARTIFACT_OBJECT_TYPE_PROTOBUF"
  ARTIFACT_OBJECT_TYPE_DOCKER = "ARTIFACT_OBJECT_TYPE_DOCKER"
  ARTIFACT_OBJECT_TYPE_BYTE = "ARTIFACT_OBJECT_TYPE_BYTE"
  ARTIFACT_OBJECT_TYPE_OTHER = "ARTIFACT_OBJECT_TYPE_OTHER"


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class Artifact:
  """Represents an artifact."""

  # pylint: disable=invalid-name
  uri: str | None = None
  artifactId: str | None = None
  name: str | None = None
  desc: str | None = None
  artifactObjectType: ArtifactObjectType | None = None
  commitTime: str | None = None
  tags: list[str] | None = None
  version: str | None = None
  isZipped: bool | None = None

  def __post_init__(self):
    if self.artifactObjectType is None:
      self.artifactObjectType = (
          ArtifactObjectType.ARTIFACT_OBJECT_TYPE_UNSPECIFIED
      )
  # pylint: enable=invalid-name


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class LoadArtifactResponse:
  """Orchestrator artifact information."""

  artifact: Artifact
