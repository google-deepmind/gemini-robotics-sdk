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

"""Robot job work unit APIs for interacting with the orchestrator server."""

import json

from googleapiclient import discovery
from googleapiclient import errors

from safari_sdk.orchestrator.client.dataclass import api_response
from safari_sdk.orchestrator.client.dataclass import work_unit

WORK_UNIT = work_unit.WorkUnit
WORK_UNIT_OUTCOME = work_unit.WorkUnitOutcome
_RESPONSE = api_response.OrchestratorAPIResponse

_ERROR_NO_ORCHESTRATOR_CONNECTION = (
    "OrchestratorRobotJobWorkUnit: Orchestrator connection is invalid."
)
_ERROR_WORK_UNIT_NOT_ACQUIRED = (
    "OrchestratorRobotJobWorkUnit: No active work unit."
)
_ERROR_EMPTY_ROBOT_JOB_ID = (
    "OrchestratorRobotJobWorkUnit: No robot job ID is set to request work unit."
    " Please call set_robot_job_id() first."
)
_ERROR_GET_WORK_UNIT = (
    "OrchestratorRobotJobWorkUnit: Error in requesting work unit.\n"
)
_ERROR_EMPTY_RESPONSE = (
    "OrchestratorRobotJobWorkUnit: Received empty response for work unit"
    " request."
)
_ERROR_EMPTY_WORK_UNIT = (
    "OrchestratorRobotJobWorkUnit: Received empty work unit in response for"
    " work unit request."
)
_ERROR_WORK_UNIT_SOFTWARE_ASSET_PREP = (
    "OrchestratorRobotJobWorkUnit: Error in starting work unit software asset"
    " prep."
)
_ERROR_WORK_UNIT_SCENE_PREP = (
    "OrchestratorRobotJobWorkUnit: Error in starting work unit scene prep."
)
_ERROR_WORK_UNIT_EXECUTION = (
    "OrchestratorRobotJobWorkUnit: Error in starting work unit execution."
)
_ERROR_WORK_UNIT_COMPLETED = (
    "OrchestratorRobotJobWorkUnit: Error in completing work unit."
)


class OrchestratorRobotJobWorkUnit:
  """Robot job work unit client for interacting with the orchestrator server."""

  def __init__(
      self, *, connection: discovery.Resource, robot_id: str
  ):
    """Initializes the work unit handler."""

    self._connection = connection
    self._robot_id = robot_id

    self._current_work_unit: work_unit.WorkUnit | None = None
    self._robot_job_id: str | None = None

  def disconnect(self) -> None:
    """Clears current connection to the orchestrator server."""
    self._connection = None

  def set_robot_job_id(self, robot_job_id: str | None) -> _RESPONSE:
    """Sets the robot job ID."""
    self._robot_job_id = robot_job_id
    return _RESPONSE(success=True, robot_job_id=robot_job_id)

  def get_current_work_unit(self) -> _RESPONSE:
    if self._current_work_unit is None:
      return _RESPONSE(error_message=_ERROR_WORK_UNIT_NOT_ACQUIRED)
    else:
      return _RESPONSE(
          success=True,
          robot_job_id=self._current_work_unit.robotJobId,
          work_unit_id=self._current_work_unit.workUnitId,
          work_unit=self._current_work_unit
      )

  def request_work_unit(self) -> _RESPONSE:
    """Gets the next work item to execute."""
    if self._connection is None:
      return _RESPONSE(error_message=_ERROR_NO_ORCHESTRATOR_CONNECTION)

    if not self._robot_job_id:
      return _RESPONSE(error_message=_ERROR_EMPTY_ROBOT_JOB_ID)

    body = {
        "robot_id": self._robot_id,
        "robot_job_id": self._robot_job_id,
    }

    try:
      response = (
          self._connection.orchestrator().allocateWorkUnit(body=body).execute()
      )
    except errors.HttpError as e:
      return _RESPONSE(
          error_message=(
              _ERROR_GET_WORK_UNIT
              + f"Reason: {e.reason}\nDetail: {e.error_details}"
          )
      )

    if not response:
      return _RESPONSE(
          success=True,
          no_more_work_unit=True,
          error_message=_ERROR_EMPTY_RESPONSE
      )

    as_json = json.dumps(response)
    self._current_work_unit = work_unit.WorkUnitResponse.from_json(as_json)

    if not self._current_work_unit.workUnit:
      self._current_work_unit = None
      return _RESPONSE(error_message=_ERROR_EMPTY_WORK_UNIT)

    self._current_work_unit = self._current_work_unit.workUnit
    return _RESPONSE(
        success=True,
        robot_id=self._robot_id,
        robot_job_id=self._current_work_unit.robotJobId,
        work_unit_id=self._current_work_unit.workUnitId,
        work_unit=self._current_work_unit
    )

  def start_work_unit_software_asset_prep(self) -> _RESPONSE:
    """Sets the current work unit's stageas software asset prep."""
    if self._connection is None:
      return _RESPONSE(error_message=_ERROR_NO_ORCHESTRATOR_CONNECTION)

    if not self._robot_job_id:
      return _RESPONSE(error_message=_ERROR_EMPTY_ROBOT_JOB_ID)

    work_unit_response = self.get_current_work_unit()
    if not work_unit_response.success:
      return work_unit_response

    body = {
        "robot_id": self._robot_id,
        "work_unit_id": work_unit_response.work_unit_id,
    }

    try:
      (
          self._connection.orchestrator()
          .startWorkUnitSoftwareAssetPrep(body=body).execute()
      )
    except errors.HttpError as e:
      return _RESPONSE(
          error_message=(
              _ERROR_WORK_UNIT_SOFTWARE_ASSET_PREP
              + f"Reason: {e.reason}\nDetail: {e.error_details}"
          )
      )

    return _RESPONSE(
        success=True,
        robot_id=self._robot_id,
        robot_job_id=work_unit_response.robot_job_id,
        work_unit_id=work_unit_response.work_unit_id,
        work_unit=self._current_work_unit
    )

  def start_work_unit_scene_prep(self) -> _RESPONSE:
    """Sets the current work unit's stage as scene prep."""
    if self._connection is None:
      return _RESPONSE(error_message=_ERROR_NO_ORCHESTRATOR_CONNECTION)

    if not self._robot_job_id:
      return _RESPONSE(error_message=_ERROR_EMPTY_ROBOT_JOB_ID)

    work_unit_response = self.get_current_work_unit()
    if not work_unit_response.success:
      return work_unit_response

    body = {
        "robot_id": self._robot_id,
        "work_unit_id": work_unit_response.work_unit_id,
    }

    try:
      (
          self._connection.orchestrator().startWorkUnitScenePrep(body=body)
          .execute()
      )
    except errors.HttpError as e:
      return _RESPONSE(
          error_message=(
              _ERROR_WORK_UNIT_SCENE_PREP
              + f"Reason: {e.reason}\nDetail: {e.error_details}"
          )
      )

    return _RESPONSE(
        success=True,
        robot_id=self._robot_id,
        robot_job_id=work_unit_response.robot_job_id,
        work_unit_id=work_unit_response.work_unit_id,
        work_unit=self._current_work_unit
    )

  def start_work_unit_execution(self) -> _RESPONSE:
    """Sets the current work unit's stage as executing."""
    if self._connection is None:
      return _RESPONSE(error_message=_ERROR_NO_ORCHESTRATOR_CONNECTION)

    if not self._robot_job_id:
      return _RESPONSE(error_message=_ERROR_EMPTY_ROBOT_JOB_ID)

    work_unit_response = self.get_current_work_unit()
    if not work_unit_response.success:
      return work_unit_response

    body = {
        "robot_id": self._robot_id,
        "work_unit_id": work_unit_response.work_unit_id,
    }

    try:
      (
          self._connection.orchestrator().startWorkUnitExecution(body=body)
          .execute()
      )
    except errors.HttpError as e:
      return _RESPONSE(
          error_message=(
              _ERROR_WORK_UNIT_EXECUTION
              + f"Reason: {e.reason}\nDetail: {e.error_details}"
          )
      )

    return _RESPONSE(
        success=True,
        robot_id=self._robot_id,
        robot_job_id=work_unit_response.robot_job_id,
        work_unit_id=work_unit_response.work_unit_id,
        work_unit=self._current_work_unit
    )

  def complete_work_unit(
      self, outcome: work_unit.WorkUnitOutcome, note: str
  ) -> _RESPONSE:
    """Set the current work unit's stage as completed."""
    if self._connection is None:
      return _RESPONSE(error_message=_ERROR_NO_ORCHESTRATOR_CONNECTION)

    if not self._robot_job_id:
      return _RESPONSE(error_message=_ERROR_EMPTY_ROBOT_JOB_ID)

    work_unit_response = self.get_current_work_unit()
    if not work_unit_response.success:
      return work_unit_response

    body = {
        "robot_id": self._robot_id,
        "work_unit_id": work_unit_response.work_unit_id,
        "outcome": outcome.num_value(),
        "note": note,
    }

    try:
      self._connection.orchestrator().completeWorkUnit(body=body).execute()
    except errors.HttpError as e:
      return _RESPONSE(
          error_message=(
              _ERROR_WORK_UNIT_COMPLETED
              + f"Reason: {e.reason}\nDetail: {e.error_details}"
          )
      )

    return _RESPONSE(
        success=True,
        robot_id=self._robot_id,
        robot_job_id=work_unit_response.robot_job_id,
        work_unit_id=work_unit_response.work_unit_id,
        work_unit=self._current_work_unit
    )
