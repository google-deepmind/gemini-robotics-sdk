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

"""Current robot info APIs for interacting with the orchestrator server."""

import json

from googleapiclient import discovery
from googleapiclient import errors

from safari_sdk.orchestrator.client.dataclass import api_response
from safari_sdk.orchestrator.client.dataclass import current_robot_info

_RESPONSE = api_response.OrchestratorAPIResponse

_ERROR_NO_ORCHESTRATOR_CONNECTION = (
    "OrchestratorCurrentRobotInfo: Orchestrator connection is invalid."
)
_ERROR_GET_CURRENT_ROBOT_INFO = (
    "OrchestratorCurrentRobotInfo: Error in requesting current robot info.\n"
)
_ERROR_SET_CURRENT_ROBOT_OPERATOR_ID = (
    "OrchestratorCurrentRobotInfo: Error in setting robot operator ID.\n"
)


class OrchestratorCurrentRobotInfo:
  """Current robot info API client for interacting with orchestrator server."""

  def __init__(
      self, *, connection: discovery.Resource, robot_id: str
  ):
    """Initializes the robot job handler."""
    self._connection = connection
    self._robot_id = robot_id

  def disconnect(self) -> None:
    """Clears current connection to the orchestrator server."""
    self._connection = None

  def get_current_robot_info(self) -> _RESPONSE:
    """Gets the current robot job."""

    if self._connection is None:
      return _RESPONSE(error_message=_ERROR_NO_ORCHESTRATOR_CONNECTION)

    body = {
        "robot_id": self._robot_id,
    }

    # assert server_connection_response.server_connection is not None
    try:
      response = (
          self._connection.orchestrator().currentRobotInfo(body=body).execute()
      )
    except errors.HttpError as e:
      return _RESPONSE(
          error_message=(
              _ERROR_GET_CURRENT_ROBOT_INFO
              + f"Reason: {e.reason}\nDetail: {e.error_details}"
          )
      )

    as_json = json.dumps(response)
    info = current_robot_info.CurrentRobotInfoResponse.from_json(as_json)

    return _RESPONSE(
        success=True,
        robot_id=self._robot_id,
        robot_job_id=info.robotJobId,
        work_unit_id=info.workUnitId,
        work_unit_stage=info.stage,
        operator_id=info.operatorId,
        is_operational=info.isOperational,
    )

  def set_current_robot_operator_id(self, operator_id: str) -> _RESPONSE:
    """Set the current operator ID for the robot."""

    if self._connection is None:
      return _RESPONSE(error_message=_ERROR_NO_ORCHESTRATOR_CONNECTION)

    body = {
        "robot_id": self._robot_id,
        "operator_id": operator_id,
    }

    try:
      (
          self._connection.orchestrator().currentRobotSetOperatorId(body=body)
          .execute()
      )
    except errors.HttpError as e:
      return _RESPONSE(
          error_message=(
              _ERROR_SET_CURRENT_ROBOT_OPERATOR_ID
              + f"Reason: {e.reason}\nDetail: {e.error_details}"
          )
      )

    return _RESPONSE(
        success=True,
        robot_id=self._robot_id,
        operator_id=operator_id,
    )
