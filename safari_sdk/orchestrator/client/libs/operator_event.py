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

"""Operator Event API for interacting with the spanner database via the orchestrator server."""

import json
import time

from googleapiclient import discovery
from googleapiclient import errors

from safari_sdk.orchestrator.client.dataclass import api_response
from safari_sdk.orchestrator.client.dataclass import operator_event

_RESPONSE = api_response.OrchestratorAPIResponse

_ERROR_NO_ORCHESTRATOR_CONNECTION = (
    "OrchestratorCurrentRobotInfo: Orchestrator connection is invalid."
)

_ERROR_RECORD_OPERATOR_EVENT = (
    "OrchestratorOperatorEvent: Error in recording operator event.\n"
)


class OrchestratorOperatorEvent:
  """Operator Event API client for interacting with the spanner database via the orchestrator server."""

  def __init__(
      self, *, connection: discovery.Resource, robot_id: str,
  ):
    """Initializes the robot job handler."""
    self._connection = connection
    self._robot_id = robot_id

  def disconnect(self) -> None:
    """Clears current connection to the orchestrator server."""
    self._connection = None  # pyrefly: ignore[bad-assignment]

  def add_operator_event(
      self,
      operator_event_type: int | None,
      operator_id: str,
      event_timestamp: int,
      resetter_id: str,
      event_note: str,
  ) -> _RESPONSE:
    """Records an operator event."""

    if self._connection is None:
      return _RESPONSE(error_message=_ERROR_NO_ORCHESTRATOR_CONNECTION)

    if operator_event_type is None:
      return _RESPONSE(
          error_message=(
              _ERROR_RECORD_OPERATOR_EVENT
              + "operator_event_type must be provided."
          )
      )

    tracer = time.time_ns()
    error_id = f"[Error ID: {tracer}]"
    body = {
        "operator_event": {
            "robotId": self._robot_id,
            "eventType": operator_event_type,
            "eventEpochMicros": event_timestamp,
            "operatorId": operator_id,
            "resetterId": resetter_id,
            "note": event_note,
        },
        "tracer": tracer,
    }

    try:
      response = (
          self._connection.orchestrator()
          .addOperatorEvent(body=body)
          .execute()
      )
    except errors.HttpError as e:
      return _RESPONSE(
          error_message=(
              _ERROR_RECORD_OPERATOR_EVENT
              + f"{error_id} Reason: {e.reason}\nDetail: {e.error_details}"
          )
      )

    as_json = json.dumps(response)
    add_operator_event_response = (
        operator_event.AddOperatorEventResponse.from_json(as_json)  # pyrefly: ignore[missing-attribute]
    )

    if not add_operator_event_response.success:
      return _RESPONSE(
          success=False,
          error_message=(
              f"{error_id} Failed to record operator event"
              f" [{operator_event_type}] for [{self._robot_id}] at"
              f" [{event_timestamp}]."
          ),
      )

    return _RESPONSE(success=True)
