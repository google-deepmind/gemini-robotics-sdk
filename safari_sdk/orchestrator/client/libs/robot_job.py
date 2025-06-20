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

"""Robot job APIs interacting with the orchestrator server."""

import enum
import json

from googleapiclient import discovery
from googleapiclient import errors

from safari_sdk.orchestrator.client.dataclass import api_response
from safari_sdk.orchestrator.client.dataclass import robot_job

_RESPONSE = api_response.OrchestratorAPIResponse

_ERROR_ROBOT_JOB_NOT_ACQUIRED = (
    "OrchestratorRobotJob: No active robot job. Please call request_robot_job()"
    " first."
)
_ERROR_NO_ORCHESTRATOR_CONNECTION = (
    "OrchestratorRobotJob: Orchestrator connection is invalid."
)
_ERROR_GET_ROBOT_JOB = "OrchestratorRobotJob: Error in requesting robot job.\n"

_ERROR_EMPTY_RESPONSE = (
    "OrchestratorRobotJob: Received empty response for robot job request."
)
_ERROR_EMPTY_ROBOT_JOB_ID = (
    "OrchestratorRobotJob: Received empty robot job ID in response for robot"
    " job request."
)


class JobType(enum.Enum):
  """Type of robot job."""
  # This "ALL" enum value is an unique usage, where it maps to the default proto
  # value of "UNSPECIFIED" in Orchestrator, which is treated as "all job types".
  ALL = 0  # All job types.
  COLLECTION = 1  # Collection job only.
  EVALUATION = 2  # Evaluation job only.


class OrchestratorRobotJob:
  """Robot job API client for interacting with the orchestrator server."""

  def __init__(
      self,
      *,
      connection: discovery.Resource,
      robot_id: str,
      job_type: JobType,
  ):
    """Initializes the robot job handler."""
    self._connection = connection
    self._robot_id = robot_id
    self._job_type = job_type

    self._current_robot_job: robot_job.RobotJob | None = None

  def disconnect(self) -> None:
    """Clears current connection to the orchestrator server."""
    self._connection = None

  def get_current_robot_job(self) -> _RESPONSE:
    """Gets the current robot job."""
    if self._current_robot_job is None:
      return _RESPONSE(error_message=_ERROR_ROBOT_JOB_NOT_ACQUIRED)
    else:
      return _RESPONSE(success=True, robot_job=self._current_robot_job)

  def request_robot_job(self) -> _RESPONSE:
    """Request orchestrator server for next available robot job to execute."""
    if self._connection is None:
      return _RESPONSE(error_message=_ERROR_NO_ORCHESTRATOR_CONNECTION)

    body = {
        "robot_id": self._robot_id,
        "type": self._job_type.value,
    }

    try:
      response = (
          self._connection.orchestrator().allocateRobotJob(body=body).execute()
      )
    except errors.HttpError as e:
      return _RESPONSE(
          error_message=(
              _ERROR_GET_ROBOT_JOB
              + f"Reason: {e.reason}\nDetail: {e.error_details}"
          )
      )

    as_json = json.dumps(response)
    self._current_robot_job = robot_job.RobotJobResponse.from_json(as_json)

    if not self._current_robot_job:
      self._current_robot_job = None
      return _RESPONSE(error_message=_ERROR_EMPTY_RESPONSE)

    if self._current_robot_job.robotJob.robotJobId is None:
      self._current_robot_job = None
      return _RESPONSE(
          success=True,
          no_more_robot_job=True,
          error_message=_ERROR_EMPTY_RESPONSE
      )

    if not self._current_robot_job.robotJob.robotJobId:
      self._current_robot_job = None
      return _RESPONSE(error_message=_ERROR_EMPTY_ROBOT_JOB_ID)

    self._current_robot_job = self._current_robot_job.robotJob
    return _RESPONSE(
        success=True,
        robot_id=self._robot_id,
        robot_job_id=self._current_robot_job.robotJobId,
    )
