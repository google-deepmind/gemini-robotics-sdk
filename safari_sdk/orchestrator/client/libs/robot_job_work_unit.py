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
import time

from googleapiclient import discovery
from googleapiclient import errors

from safari_sdk.orchestrator.client.dataclass import api_response
from safari_sdk.orchestrator.client.dataclass import work_unit
from safari_sdk.orchestrator.client.libs import robot_job

WORK_UNIT = work_unit.WorkUnit
WORK_UNIT_OUTCOME = work_unit.WorkUnitOutcome
WORK_UNIT_QUESTION = work_unit.Question
QUESTION_CONDITION = work_unit.QuestionCondition
QUESTION_ANSWER_TYPE = work_unit.AnswerType
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
    " request. "
)
_ERROR_EMPTY_WORK_UNIT = (
    "OrchestratorRobotJobWorkUnit: Received empty work unit in response for"
    " work unit request. "
)
_ERROR_WORK_UNIT_SOFTWARE_ASSET_PREP = (
    "OrchestratorRobotJobWorkUnit: Error in starting work unit software asset"
    " prep. "
)
_ERROR_WORK_UNIT_SCENE_PREP = (
    "OrchestratorRobotJobWorkUnit: Error in starting work unit scene prep. "
)
_ERROR_WORK_UNIT_EXECUTION = (
    "OrchestratorRobotJobWorkUnit: Error in starting work unit execution. "
)
_ERROR_WORK_UNIT_INSERT_SESSION_INFO = (
    "OrchestratorRobotJobWorkUnit: Error in inserting session info. "
)
_ERROR_WORK_UNIT_COMPLETED = (
    "OrchestratorRobotJobWorkUnit: Error in completing work unit. "
)
_ERROR_WORK_UNIT_IN_EXECUTION_STAGE = (
    "OrchestratorRobotJobWorkUnit: Work unit is not in execution stage."
)
_ERROR_BAD_SESSION_START_TIME = (
    "OrchestratorRobotJobWorkUnit: Session start time is not a valid"
    " nanoseconds timestamp."
)
_ERROR_BAD_SESSION_END_TIME = (
    "OrchestratorRobotJobWorkUnit: Session end time is not a valid"
    " nanoseconds timestamp."
)
_ERROR_MISSING_SESSION_START_TIME = (
    "OrchestratorRobotJobWorkUnit: Session start time is not provided."
)
_ERROR_MISSING_SESSION_END_TIME = (
    "OrchestratorRobotJobWorkUnit: Session end time is not provided."
)
_ERROR_MISSING_SESSION_LOG_TYPE = (
    "OrchestratorRobotJobWorkUnit: Session log type is not provided."
)
_ERROR_OBSERVE_LATEST_WORK_UNIT = (
    "OrchestratorRobotJobWorkUnit: Error in observing latest work unit.\n"
)
_ERROR_EMPTY_OBSERVE_LATEST_WORK_UNIT_RESPONSE = (
    "OrchestratorRobotJobWorkUnit: Received empty response for observe latest"
    " work unit request. "
)
_ERROR_EMPTY_OBSERVE_LATEST_WORK_UNIT = (
    "OrchestratorRobotJobWorkUnit: Received empty work unit in response for"
    " observe latest work unit request. "
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
    self._launch_command: str | None = None

    self._work_unit_in_execution_stage: bool = False

  def disconnect(self) -> None:
    """Clears current connection to the orchestrator server."""
    self._connection = None

  def set_robot_job_info(
      self, robot_job_id: str | None, launch_command: str | None
  ) -> _RESPONSE:
    """Sets the robot job information to be used for work unit requests."""
    self._robot_job_id = robot_job_id
    self._launch_command = launch_command
    return _RESPONSE(
        success=True,
        robot_job_id=robot_job_id,
        launch_command=launch_command,
    )

  def get_current_work_unit(self) -> _RESPONSE:
    if self._current_work_unit is None:
      return _RESPONSE(error_message=_ERROR_WORK_UNIT_NOT_ACQUIRED)
    else:
      return _RESPONSE(
          success=True,
          robot_job_id=self._current_work_unit.robotJobId,
          launch_command=self._launch_command,
          work_unit_id=self._current_work_unit.workUnitId,
          work_unit=self._current_work_unit
      )

  def request_work_unit(self) -> _RESPONSE:
    """Gets the next work item to execute."""
    if self._connection is None:
      return _RESPONSE(error_message=_ERROR_NO_ORCHESTRATOR_CONNECTION)

    if not self._robot_job_id:
      return _RESPONSE(error_message=_ERROR_EMPTY_ROBOT_JOB_ID)

    self._work_unit_in_execution_stage = False

    tracer = time.time_ns()
    error_id = f"[Error ID: {tracer}]"
    body = {
        "robot_id": self._robot_id,
        "robot_job_id": self._robot_job_id,
        "tracer": tracer,
    }

    try:
      response = (
          self._connection.orchestrator().allocateWorkUnit(body=body).execute()
      )
    except errors.HttpError as e:
      return _RESPONSE(
          error_message=(
              _ERROR_GET_WORK_UNIT
              + f"{error_id} Reason: {e.reason}\nDetail: {e.error_details}"
          )
      )

    if not response:
      return _RESPONSE(
          success=True,
          no_more_work_unit=True,
          robot_job_id=self._robot_job_id,
          launch_command=self._launch_command,
          error_message=_ERROR_EMPTY_RESPONSE + error_id
      )

    as_json = json.dumps(response)
    self._current_work_unit = work_unit.WorkUnitResponse.from_json(as_json)

    if not self._current_work_unit.workUnit:
      self._current_work_unit = None
      return _RESPONSE(error_message=_ERROR_EMPTY_WORK_UNIT + error_id)

    self._current_work_unit = self._current_work_unit.workUnit
    return _RESPONSE(
        success=True,
        robot_id=self._robot_id,
        robot_job_id=self._current_work_unit.robotJobId,
        launch_command=self._launch_command,
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

    tracer = time.time_ns()
    error_id = f"[Error ID: {tracer}]"
    body = {
        "robot_id": self._robot_id,
        "work_unit_id": work_unit_response.work_unit_id,
        "tracer": tracer,
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
              + f"{error_id} Reason: {e.reason}\nDetail: {e.error_details}"
          )
      )

    self._work_unit_in_execution_stage = False
    return _RESPONSE(
        success=True,
        robot_id=self._robot_id,
        robot_job_id=work_unit_response.robot_job_id,
        launch_command=self._launch_command,
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

    tracer = time.time_ns()
    error_id = f"[Error ID: {tracer}]"
    body = {
        "robot_id": self._robot_id,
        "work_unit_id": work_unit_response.work_unit_id,
        "tracer": tracer,
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
              + f"{error_id} Reason: {e.reason}\nDetail: {e.error_details}"
          )
      )

    self._work_unit_in_execution_stage = False
    return _RESPONSE(
        success=True,
        robot_id=self._robot_id,
        robot_job_id=work_unit_response.robot_job_id,
        launch_command=self._launch_command,
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

    tracer = time.time_ns()
    error_id = f"[Error ID: {tracer}]"
    body = {
        "robot_id": self._robot_id,
        "work_unit_id": work_unit_response.work_unit_id,
        "tracer": tracer,
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
              + f"{error_id} Reason: {e.reason}\nDetail: {e.error_details}"
          )
      )

    self._work_unit_in_execution_stage = True
    return _RESPONSE(
        success=True,
        robot_id=self._robot_id,
        robot_job_id=work_unit_response.robot_job_id,
        launch_command=self._launch_command,
        work_unit_id=work_unit_response.work_unit_id,
        work_unit=self._current_work_unit
    )

  def insert_session_info(
      self,
      session_log_type: str,
      session_start_time_ns: int,
      session_end_time_ns: int,
      session_note: str,
  ) -> _RESPONSE:
    """Insert session info for the current work unit."""
    if self._connection is None:
      return _RESPONSE(error_message=_ERROR_NO_ORCHESTRATOR_CONNECTION)

    work_unit_response = self.get_current_work_unit()
    if not work_unit_response.success:
      return work_unit_response
    assert self._current_work_unit is not None

    if not self._work_unit_in_execution_stage:
      return _RESPONSE(
          error_message=_ERROR_WORK_UNIT_IN_EXECUTION_STAGE
      )
    if not session_log_type:
      return _RESPONSE(error_message=_ERROR_MISSING_SESSION_LOG_TYPE)
    if session_start_time_ns and len(str(session_start_time_ns)) != 19:
      return _RESPONSE(error_message=_ERROR_BAD_SESSION_START_TIME)
    if session_end_time_ns and len(str(session_end_time_ns)) != 19:
      return _RESPONSE(error_message=_ERROR_BAD_SESSION_END_TIME)

    tracer = time.time_ns()
    error_id = f"[Error ID: {tracer}]"
    body = {
        "robot_id": self._robot_id,
        "work_unit_id": self._current_work_unit.workUnitId,
        "session_log_type": session_log_type,
        "session_start_time_ns": session_start_time_ns,
        "session_end_time_ns": session_end_time_ns,
        "session_note": session_note,
        "tracer": tracer,
    }

    try:
      self._connection.orchestrator().insertSessionInfo(body=body).execute()
    except errors.HttpError as e:
      return _RESPONSE(
          error_message=(
              _ERROR_WORK_UNIT_INSERT_SESSION_INFO
              + f"{error_id} Reason: {e.reason}\nDetail: {e.error_details}"
          )
      )

    return _RESPONSE(
        success=True,
        robot_id=self._robot_id,
        robot_job_id=work_unit_response.robot_job_id,
        work_unit_id=work_unit_response.work_unit_id,
        work_unit=self._current_work_unit,
    )

  def complete_work_unit(
      self,
      outcome: work_unit.WorkUnitOutcome,
      success_score: float | None,
      success_score_definition: str | None,
      session_start_time_ns: int | None,
      session_end_time_ns: int | None,
      session_log_type: str | None,
      session_note: str | None,
      response_to_questions: list[work_unit.Question] | None,
      note: str,
      request_retry_bypass: bool,
  ) -> _RESPONSE:
    """Set the current work unit's stage as completed."""
    if self._connection is None:
      return _RESPONSE(error_message=_ERROR_NO_ORCHESTRATOR_CONNECTION)

    if not self._robot_job_id:
      return _RESPONSE(error_message=_ERROR_EMPTY_ROBOT_JOB_ID)

    if session_start_time_ns or session_end_time_ns or session_log_type:
      if not session_start_time_ns:
        return _RESPONSE(error_message=_ERROR_MISSING_SESSION_START_TIME)
      if not session_end_time_ns:
        return _RESPONSE(error_message=_ERROR_MISSING_SESSION_END_TIME)
      if not session_log_type:
        return _RESPONSE(error_message=_ERROR_MISSING_SESSION_LOG_TYPE)

    if session_start_time_ns and len(str(session_start_time_ns)) != 19:
      return _RESPONSE(error_message=_ERROR_BAD_SESSION_START_TIME)
    if session_end_time_ns and len(str(session_end_time_ns)) != 19:
      return _RESPONSE(error_message=_ERROR_BAD_SESSION_END_TIME)

    work_unit_response = self.get_current_work_unit()
    if not work_unit_response.success:
      return work_unit_response

    tracer = time.time_ns()
    error_id = f"[Error ID: {tracer}]"
    body = {
        "robot_id": self._robot_id,
        "work_unit_id": work_unit_response.work_unit_id,
        "outcome": outcome.num_value(),
        "note": note,
        "tracer": tracer,
    }
    if success_score is not None:
      body["success_score"] = {
          "score": success_score,
          "definition": (
              success_score_definition if success_score_definition else ""
          ),
      }
    if session_start_time_ns is not None:
      body["session_start_time_ns"] = session_start_time_ns
    if session_end_time_ns is not None:
      body["session_end_time_ns"] = session_end_time_ns
    if session_log_type is not None:
      body["session_log_type"] = session_log_type
    if session_note is not None:
      body["session_note"] = session_note
    body["request_retry_bypass"] = request_retry_bypass

    if response_to_questions is not None:
      assert self._current_work_unit is not None
      assert self._current_work_unit.context is not None
      self._current_work_unit.context.questions = response_to_questions
      responses = []
      for response in response_to_questions:
        assert response.question is not None
        assert response.whenToAsk is not None
        assert response.answerFormat is not None
        assert response.allowedAnswers is not None
        assert response.userAnswers is not None
        assert response.wasDisplayed is not None
        question = {
            "question": response.question,
            "when_to_ask": [
                condition.num_value() for condition in response.whenToAsk
            ],
            "answer_format": response.answerFormat.num_value(),
            "allowed_answers": response.allowedAnswers,
            "user_answers": response.userAnswers,
            "was_displayed": response.wasDisplayed,
        }
        responses.append(question)
      body["questions"] = responses

    try:
      self._connection.orchestrator().completeWorkUnit(body=body).execute()
    except errors.HttpError as e:
      return _RESPONSE(
          error_message=(
              _ERROR_WORK_UNIT_COMPLETED
              + f"{error_id} Reason: {e.reason}\nDetail: {e.error_details}"
          )
      )

    self._work_unit_in_execution_stage = False
    return _RESPONSE(
        success=True,
        robot_id=self._robot_id,
        robot_job_id=work_unit_response.robot_job_id,
        launch_command=self._launch_command,
        work_unit_id=work_unit_response.work_unit_id,
        work_unit=self._current_work_unit,
    )

  def observe_latest_work_unit(self, job_type: robot_job.JobType) -> _RESPONSE:
    """Observes the latest work unit."""
    if self._connection is None:
      return _RESPONSE(error_message=_ERROR_NO_ORCHESTRATOR_CONNECTION)

    tracer = time.time_ns()
    error_id = f"[Error ID: {tracer}]"
    body = {
        "job_type": job_type.value,
        "robot_id": self._robot_id,
        "tracer": tracer,
    }

    try:
      response = (
          self._connection.orchestrator()
          .observeCurrentWorkUnit(body=body)
          .execute()
      )
    except errors.HttpError as e:
      return _RESPONSE(
          error_message=(
              _ERROR_OBSERVE_LATEST_WORK_UNIT
              + f"{error_id} Reason: {e.reason}\nDetail: {e.error_details}"
          )
      )

    if not response:
      return _RESPONSE(
          success=True,
          no_more_work_unit=True,
          error_message=(
              _ERROR_EMPTY_OBSERVE_LATEST_WORK_UNIT_RESPONSE + error_id
          ),
      )

    as_json = json.dumps(response)
    current_work_unit = work_unit.WorkUnitResponse.from_json(as_json)

    if not current_work_unit.workUnit:
      return _RESPONSE(
          error_message=_ERROR_EMPTY_OBSERVE_LATEST_WORK_UNIT + error_id
      )

    return _RESPONSE(
        success=True,
        robot_id=self._robot_id,
        robot_job_id=current_work_unit.workUnit.robotJobId,
        work_unit_id=current_work_unit.workUnit.workUnitId,
        work_unit=current_work_unit.workUnit,
    )
