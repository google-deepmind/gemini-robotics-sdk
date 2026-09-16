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

"""RuiWorkcellState APIs interacting with the orchestrator server."""

import json
import time

from googleapiclient import discovery
from googleapiclient import errors

from safari_sdk.orchestrator.client.dataclass import api_response
from safari_sdk.orchestrator.client.dataclass import rui_workcell_state

_RESPONSE = api_response.OrchestratorAPIResponse

_ERROR_NO_ORCHESTRATOR_CONNECTION = (
    "OrchestratorRuiWorkcellState: Orchestrator connection is invalid."
)
_ERROR_LOAD_RUI_WORKCELL_STATE = (
    "OrchestratorRuiWorkcellState: Error in loading RUI workcell state. "
)
_ERROR_SET_RUI_WORKCELL_STATE = (
    "OrchestratorRuiWorkcellState: Error in setting RUI workcell state. "
)


class OrchestratorRuiWorkcellState:
  """RuiWorkcellState API client for interacting with the orchestrator server."""

  def __init__(
      self,
      *,
      connection: discovery.Resource,
  ):
    self._connection = connection

  def disconnect(self) -> None:
    """Clears current connection to the orchestrator server."""
    self._connection = None  # pyrefly: ignore[bad-assignment]

  def load_rui_workcell_state(self, robot_id: str) -> _RESPONSE:
    """Load the RUI workcell state for the given robot."""

    if self._connection is None:
      return _RESPONSE(error_message=_ERROR_NO_ORCHESTRATOR_CONNECTION)

    tracer = time.time_ns()
    error_id = f"[Error ID: {tracer}]"
    body = {"robot_id": robot_id, "tracer": tracer}

    try:
      response = (
          self._connection.orchestrator()
          .loadRuiWorkcellState(body=body)
          .execute()
      )
    except errors.HttpError as e:
      return _RESPONSE(
          error_message=(
              _ERROR_LOAD_RUI_WORKCELL_STATE
              + f"{error_id} Reason: {e.reason}\nDetail: {e.error_details}"
          )
      )

    as_json = json.dumps(response)
    response = rui_workcell_state.LoadRuiWorkcellStateResponse.from_json(  # pyrefly: ignore[missing-attribute]
        as_json
    )

    return _RESPONSE(success=True, workcell_state=response.workcellState)

  def set_rui_workcell_state(
      self, robot_id: str, workcell_state_type: int
  ) -> _RESPONSE:
    """Set the RUI workcell state for the given robot."""
    if self._connection is None:
      return _RESPONSE(error_message=_ERROR_NO_ORCHESTRATOR_CONNECTION)

    tracer = time.time_ns()
    error_id = f"[Error ID: {tracer}]"
    body = {
        "robot_id": robot_id,
        "workcell_state": workcell_state_type,
        "tracer": tracer,
    }

    try:
      self._connection.orchestrator().setRuiWorkcellState(body=body).execute()
    except errors.HttpError as e:
      return _RESPONSE(
          error_message=(
              _ERROR_SET_RUI_WORKCELL_STATE
              + f"{error_id} Reason: {e.reason}\nDetail: {e.error_details}"
          )
      )

    return _RESPONSE(success=True)
