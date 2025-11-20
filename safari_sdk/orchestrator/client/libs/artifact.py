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

"""Artifact APIs interacting with the orchestrator server."""

import json

from googleapiclient import discovery
from googleapiclient import errors

from safari_sdk.orchestrator.client.dataclass import api_response
from safari_sdk.orchestrator.client.dataclass import artifact

_RESPONSE = api_response.OrchestratorAPIResponse

_ERROR_NO_ORCHESTRATOR_CONNECTION = (
    "OrchestratorArtifact: Orchestrator connection is invalid."
)
_ERROR_GET_ARTIFACT = "OrchestratorArtifact: Error in requesting artifact.\n"
_ERROR_EMPTY_RESPONSE = (
    "OrchestratorArtifact: Received empty response for get artifact request."
)


class OrchestratorArtifact:
  """Artifact API client for interacting with the orchestrator server."""

  def __init__(
      self,
      *,
      connection: discovery.Resource,
  ):
    """Initializes the robot job handler."""
    self._connection = connection

  def disconnect(self) -> None:
    """Clears current connection to the orchestrator server."""
    self._connection = None

  def get_artifact(self, artifact_id: str) -> _RESPONSE:
    """Gets detailed artifact information."""
    if self._connection is None:
      return _RESPONSE(error_message=_ERROR_NO_ORCHESTRATOR_CONNECTION)

    body = {"artifact_id": artifact_id}

    try:
      response = (
          self._connection.orchestrator().loadArtifact(body=body).execute()
      )
    except errors.HttpError as e:
      return _RESPONSE(
          error_message=(
              _ERROR_GET_ARTIFACT
              + f"Reason: {e.reason}\nDetail: {e.error_details}"
          )
      )

    if not response or "artifact" not in response:
      return _RESPONSE(error_message=_ERROR_EMPTY_RESPONSE)

    as_json = json.dumps(response)
    artifact_response = artifact.LoadArtifactResponse.from_json(as_json)

    artifact_obj = artifact_response.artifact
    if not artifact_obj or not artifact_obj.uri:
      return _RESPONSE(error_message=_ERROR_EMPTY_RESPONSE)

    return _RESPONSE(success=True, artifact=artifact_obj)

  def get_artifact_uri(self, artifact_id: str) -> _RESPONSE:
    """Gets the artifact's download URI."""
    if not artifact_id:
      return _RESPONSE(error_message="Artifact ID is empty.")

    if self._connection is None:
      return _RESPONSE(error_message=_ERROR_NO_ORCHESTRATOR_CONNECTION)

    body = {"artifact_id": artifact_id}

    try:
      response = (
          self._connection.orchestrator().loadArtifact(body=body).execute()
      )
    except errors.HttpError as e:
      return _RESPONSE(
          error_message=(
              _ERROR_GET_ARTIFACT
              + f"Reason: {e.reason}\nDetail: {e.error_details}"
          )
      )

    if not response:
      return _RESPONSE(error_message=_ERROR_EMPTY_RESPONSE)

    as_json = json.dumps(response)
    artifact_response = artifact.LoadArtifactResponse.from_json(as_json)

    if not artifact_response or not artifact_response.artifact:
      return _RESPONSE(error_message=_ERROR_EMPTY_RESPONSE)

    artifact_obj = artifact_response.artifact

    return _RESPONSE(success=True, artifact_uri=artifact_obj.uri)
