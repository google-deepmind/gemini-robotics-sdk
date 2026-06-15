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

"""Robot Event API for creating config-driven robot events via the orchestrator server."""

import datetime
import json
import time
from typing import Any

from googleapiclient import discovery
from googleapiclient import errors

from safari_sdk.orchestrator.client.dataclass import api_response
from safari_sdk.orchestrator.client.dataclass import robot_event

_RESPONSE = api_response.OrchestratorAPIResponse

_ERROR_NO_ORCHESTRATOR_CONNECTION = (
    "OrchestratorRobotEvent: Orchestrator connection is invalid."
)

_ERROR_RECORD_ROBOT_EVENT = (
    "OrchestratorRobotEvent: Error in recording robot event.\n"
)


class OrchestratorRobotEvent:
  """Robot Event API client for creating config-driven events via the orchestrator server.

  This replaces the legacy enum-based OperatorEvent pattern with a string-based,
  config-driven event type system. Event types and property schemas are
  validated
  server-side against the partner config.
  """

  def __init__(
      self,
      *,
      connection: discovery.Resource,
      robot_id: str,
  ):
    """Initializes the robot event handler.

    Args:
      connection: Google API Client discovery resource for the orchestrator.
      robot_id: The robot ID to associate events with.
    """
    self._connection = connection
    self._robot_id = robot_id

  def disconnect(self) -> None:
    """Clears current connection to the orchestrator server."""
    self._connection = None

  def add_robot_event(
      self,
      event_type: str,
      payload: dict[str, Any],
      event_timestamp: datetime.datetime | str | None = None,
  ) -> _RESPONSE:
    """Creates a new robot event with generic properties via Orca 1P API.

    The event_type and payload properties are validated server-side against the
    partner's allowed_robot_event_types config. The payload dictionary is
    serialized to JSON and sent as payload_json in the RobotEvent proto.

    Args:
      event_type: Config-driven event type string.
      payload: Dictionary of event properties. Keys and value types are
        validated against the partner config schema. Maps directly to
        google.protobuf.Struct on the backend.
      event_timestamp: Optional timestamp of the event. Can be a
        datetime.datetime object or an RFC 3339 string. If not set, the server
        will use the commit time.

    Returns:
      OrchestratorAPIResponse with success=True and event_id on success,
      or success=False with error_message on failure.
    """
    if self._connection is None:
      return _RESPONSE(error_message=_ERROR_NO_ORCHESTRATOR_CONNECTION)

    tracer = time.time_ns()
    error_id = f"[Error ID: {tracer}]"

    body = {
        "event": {
            "robotId": self._robot_id,
            "eventType": event_type,
            "payloadJson": json.dumps(payload),
        },
        "tracer": tracer,
    }

    if isinstance(event_timestamp, datetime.datetime):
      if event_timestamp.tzinfo is None:
        event_timestamp = event_timestamp.replace(tzinfo=datetime.timezone.utc)
      body["event"]["eventTimestamp"] = event_timestamp.isoformat()
    elif isinstance(event_timestamp, str):
      body["event"]["eventTimestamp"] = event_timestamp

    try:
      response = (
          self._connection.orchestrator().createRobotEvent(body=body).execute()
      )
    except errors.HttpError as e:
      return _RESPONSE(
          error_message=(
              _ERROR_RECORD_ROBOT_EVENT
              + f"{error_id} Reason: {e.reason}\nDetail: {e.error_details}"
          )
      )

    as_json = json.dumps(response)
    create_robot_event_response = (
        robot_event.CreateRobotEventResponse.from_json(as_json)
    )
    if not create_robot_event_response.eventId:
      return _RESPONSE(
          success=False,
          error_message=(
              f"{error_id} Failed to record robot event"
              f" [{event_type}] for [{self._robot_id}]."
          ),
      )

    return _RESPONSE(success=True, event_id=create_robot_event_response.eventId)
