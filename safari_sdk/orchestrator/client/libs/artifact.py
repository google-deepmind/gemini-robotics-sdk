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

import base64
import json
import time

from googleapiclient import discovery
from googleapiclient import errors

from safari_sdk.orchestrator.client.dataclass import api_response
from safari_sdk.orchestrator.client.dataclass import artifact

_RESPONSE = api_response.OrchestratorAPIResponse

_ERROR_NO_ORCHESTRATOR_CONNECTION = (
    "OrchestratorArtifact: Orchestrator connection is invalid."
)
_ERROR_GET_ARTIFACT = "OrchestratorArtifact: Error in requesting artifact.\n"
_ERROR_UPLOAD_TEXT_LOG_ARTIFACT = (
    "OrchestratorArtifact: Error in uploading text log artifact.\n"
)
_ERROR_EMPTY_RESPONSE = (
    "OrchestratorArtifact: Received empty response for get artifact request. "
)

_ERROR_EMPTY_ARTIFACT_ID = "OrchestratorArtifact: Artifact ID is empty."
_ERROR_EMPTY_ROBOT_JOB_ID = "OrchestratorArtifact: Robot job ID is empty."
_ERROR_EMPTY_ROBOT_ID = "OrchestratorArtifact: Robot ID is empty."
_ERROR_EMPTY_SOURCE_FILE_NAME = (
    "OrchestratorArtifact: Source file name is empty."
)
_ERROR_EMPTY_TEXT_FILE_BYTES = (
    "OrchestratorArtifact: Text file bytes are empty."
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

    tracer = time.time_ns()
    error_id = f"[Error ID: {tracer}]"
    body = {"artifact_id": artifact_id, "tracer": tracer}

    try:
      response = (
          self._connection.orchestrator().loadArtifact(body=body).execute()
      )
    except errors.HttpError as e:
      return _RESPONSE(
          error_message=(
              _ERROR_GET_ARTIFACT
              + f"{error_id} Reason: {e.reason}\nDetail: {e.error_details}"
          )
      )

    if not response or "artifact" not in response:
      return _RESPONSE(error_message=_ERROR_EMPTY_RESPONSE + error_id)

    as_json = json.dumps(response)
    artifact_response = artifact.LoadArtifactResponse.from_json(as_json)

    artifact_obj = artifact_response.artifact
    if not artifact_obj or not artifact_obj.uri:
      return _RESPONSE(error_message=_ERROR_EMPTY_RESPONSE + error_id)

    return _RESPONSE(success=True, artifact=artifact_obj)

  def get_artifact_uri(self, artifact_id: str) -> _RESPONSE:
    """Gets the artifact's download URI."""
    if not artifact_id:
      return _RESPONSE(error_message=_ERROR_EMPTY_ARTIFACT_ID)

    if self._connection is None:
      return _RESPONSE(error_message=_ERROR_NO_ORCHESTRATOR_CONNECTION)

    tracer = time.time_ns()
    error_id = f"[Error ID: {tracer}]"
    body = {"artifact_id": artifact_id, "tracer": tracer}

    try:
      response = (
          self._connection.orchestrator().loadArtifact(body=body).execute()
      )
    except errors.HttpError as e:
      return _RESPONSE(
          error_message=(
              _ERROR_GET_ARTIFACT
              + f"{error_id} Reason: {e.reason}\nDetail: {e.error_details}"
          )
      )

    if not response:
      return _RESPONSE(error_message=_ERROR_EMPTY_RESPONSE + error_id)

    as_json = json.dumps(response)
    artifact_response = artifact.LoadArtifactResponse.from_json(as_json)

    if not artifact_response or not artifact_response.artifact:
      return _RESPONSE(error_message=_ERROR_EMPTY_RESPONSE + error_id)

    artifact_obj = artifact_response.artifact

    return _RESPONSE(success=True, artifact_uri=artifact_obj.uri)

  def upload_text_log_artifact(
      self,
      robot_job_id: str,
      robot_id: str,
      source_file_name: str,
      text_file_bytes: bytes,
  ) -> _RESPONSE:
    """Uploads a text log artifact."""
    if not robot_job_id:
      return _RESPONSE(error_message=_ERROR_EMPTY_ROBOT_JOB_ID)

    if not robot_id:
      return _RESPONSE(error_message=_ERROR_EMPTY_ROBOT_ID)

    if not source_file_name:
      return _RESPONSE(error_message=_ERROR_EMPTY_SOURCE_FILE_NAME)

    if not text_file_bytes:
      return _RESPONSE(error_message=_ERROR_EMPTY_TEXT_FILE_BYTES)

    if self._connection is None:
      return _RESPONSE(error_message=_ERROR_NO_ORCHESTRATOR_CONNECTION)

    tracer = time.time_ns()
    error_id = f"[Error ID: {tracer}]"
    body = {
        "robot_job_id": robot_job_id,
        "robot_id": robot_id,
        "source_file_name": source_file_name,
        "text_file_bytes": base64.b64encode(text_file_bytes).decode("utf-8"),
        "tracer": tracer,
    }

    try:
      response = (
          self._connection.orchestrator()
          .uploadTextLogArtifact(body=body)
          .execute()
      )
    except errors.HttpError as e:
      return _RESPONSE(
          error_message=(
              _ERROR_UPLOAD_TEXT_LOG_ARTIFACT
              + f"{error_id} Reason: {e.reason}\nDetail: {e.error_details}"
          )
      )

    if not response:
      return _RESPONSE(error_message=_ERROR_EMPTY_RESPONSE + error_id)

    as_json = json.dumps(response)
    artifact_response = artifact.UploadTextLogArtifactResponse.from_json(
        as_json
    )

    if not artifact_response or not artifact_response.artifactId:
      return _RESPONSE(error_message=_ERROR_EMPTY_RESPONSE + error_id)

    return _RESPONSE(success=True, artifact_id=artifact_response.artifactId)
