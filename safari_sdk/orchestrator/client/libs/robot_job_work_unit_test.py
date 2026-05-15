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

"""Unit tests for robot_job.py."""

from unittest import mock

from absl.testing import absltest
from googleapiclient import errors

from safari_sdk.orchestrator.client.libs import robot_job
from safari_sdk.orchestrator.client.libs import robot_job_work_unit


class RobotJobWorkUnitTest(absltest.TestCase):

  def test_get_current_work_unit_good(self):

    mock_connection = mock.MagicMock()
    work_unit_lib = robot_job_work_unit.OrchestratorRobotJobWorkUnit(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    work_unit_lib._current_work_unit = mock.MagicMock(
        spec=robot_job_work_unit.work_unit.WorkUnit
    )

    response = work_unit_lib.get_current_work_unit()
    self.assertTrue(response.success)
    self.assertIsInstance(
        response.work_unit, robot_job_work_unit.work_unit.WorkUnit
    )

  def test_get_current_work_unit_bad(self):

    mock_connection = mock.MagicMock()
    work_unit_lib = robot_job_work_unit.OrchestratorRobotJobWorkUnit(
        connection=mock_connection,
        robot_id="test_robot_id",
    )

    response = work_unit_lib.get_current_work_unit()
    self.assertFalse(response.success)
    self.assertIn(
        robot_job_work_unit._ERROR_WORK_UNIT_NOT_ACQUIRED,
        response.error_message,
    )

  def test_request_work_unit_good(self):

    mock_connection = mock.MagicMock()
    mock_response = {
        "workUnit": {
            "robotJobId": "test_robot_job_id",
            "workUnitId": "test_work_unit_id",
            "context": {
                "scenePresetId": "test_scene_preset",
                "sceneEpisodeIndex": "1",
                "scenePresetDetails": {
                    "setupInstructions": "test_setup_instructions",
                    "parameters": [{
                        "key": "scene_key_1",
                        "value": {
                            "stringValue": "scene_value_1"
                        },
                        "type": "KV_MSG_VALUE_TYPE_STRING"
                    }]
                },
                "policyDetails": {
                    "name": "test_policy_name",
                    "parameters": [{
                        "key": "policy_key_1",
                        "value": {
                            "stringValue": "policy_value_1"
                        },
                        "type": "KV_MSG_VALUE_TYPE_STRING"
                    }]
                },
                "questions": [{
                    "question": "test_question_1",
                    "whenToAsk": [
                        "QUESTION_CONDITION_ONLY_FAILED_EPISODE",
                        "QUESTION_CONDITION_ESTOP_NOT_PRESSED"
                    ],
                    "answerFormat": "ANSWER_TYPE_YES_NO",
                    "allowedAnswers": ["allowed_answer_1", "allowed_answer_2"],
                }]
            },
            "stage": "WORK_UNIT_STAGE_QUEUED_TO_ROBOT",
            "outcome": "WORK_UNIT_OUTCOME_UNSPECIFIED",
            "note": "Work Unit queued to robot."
        }
    }
    mock_connection.orchestrator().allocateWorkUnit().execute.return_value = (
        mock_response
    )

    work_unit_lib = robot_job_work_unit.OrchestratorRobotJobWorkUnit(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    work_unit_lib.set_robot_job_info(
        robot_job_id="test_robot_job_id", launch_command="test_launch_command"
    )

    response = work_unit_lib.request_work_unit()
    self.assertTrue(response.success)
    self.assertEqual(response.robot_id, "test_robot_id")
    self.assertEqual(response.robot_job_id, "test_robot_job_id")
    self.assertEqual(response.work_unit_id, "test_work_unit_id")
    self.assertIsInstance(
        response.work_unit, robot_job_work_unit.work_unit.WorkUnit
    )
    self.assertEqual(
        response.work_unit.stage,
        robot_job_work_unit.work_unit.WorkUnitStage.WORK_UNIT_STAGE_QUEUED_TO_ROBOT,
    )
    self.assertEqual(
        response.work_unit.outcome,
        robot_job_work_unit.work_unit.WorkUnitOutcome.WORK_UNIT_OUTCOME_UNSPECIFIED,
    )
    self.assertEqual(response.work_unit.note, "Work Unit queued to robot.")
    questions = response.work_unit.context.questions
    self.assertLen(questions, 1)
    self.assertEqual(
        questions[0].question, "test_question_1"
    )
    self.assertEqual(
        questions[0].whenToAsk,
        [
            robot_job_work_unit.work_unit.QuestionCondition.QUESTION_CONDITION_ONLY_FAILED_EPISODE,
            robot_job_work_unit.work_unit.QuestionCondition.QUESTION_CONDITION_ESTOP_NOT_PRESSED,
        ],
    )
    self.assertEqual(
        questions[0].answerFormat,
        robot_job_work_unit.work_unit.AnswerType.ANSWER_TYPE_YES_NO,
    )
    self.assertEqual(
        questions[0].allowedAnswers,
        ["allowed_answer_1", "allowed_answer_2"],
    )
    self.assertEmpty(questions[0].userAnswers)
    self.assertFalse(questions[0].wasDisplayed)

  def test_request_work_unit_no_robot_job_id(self):

    mock_connection = mock.MagicMock()
    work_unit_lib = robot_job_work_unit.OrchestratorRobotJobWorkUnit(
        connection=mock_connection,
        robot_id="test_robot_id",
    )

    response = work_unit_lib.request_work_unit()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, robot_job_work_unit._ERROR_EMPTY_ROBOT_JOB_ID
    )

  def test_request_work_unit_bad_server_call(self):

    class MockHttpError:

      def __init__(self):
        self.status = "Mock status"
        self.reason = "Mock reason"
        self.error_details = "Mock error details"

    def raise_error_side_effect():
      raise errors.HttpError(MockHttpError(), "Mock failed HTTP call.".encode())

    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().allocateWorkUnit().execute.side_effect = (
        raise_error_side_effect
    )

    work_unit_lib = robot_job_work_unit.OrchestratorRobotJobWorkUnit(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    work_unit_lib.set_robot_job_info(
        robot_job_id="test_robot_job_id", launch_command="test_launch_command"
    )

    response = work_unit_lib.request_work_unit()
    self.assertFalse(response.success)
    self.assertIn(
        robot_job_work_unit._ERROR_GET_WORK_UNIT, response.error_message
    )

  @mock.patch("time.time_ns")
  def test_request_work_unit_no_more_work_unit(self, mock_time):
    mock_time.return_value = 123456789
    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().allocateWorkUnit().execute.return_value = ""

    work_unit_lib = robot_job_work_unit.OrchestratorRobotJobWorkUnit(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    work_unit_lib.set_robot_job_info(
        robot_job_id="test_robot_job_id", launch_command="test_launch_command"
    )

    response = work_unit_lib.request_work_unit()
    self.assertTrue(response.success)
    self.assertTrue(response.no_more_work_unit)
    self.assertEqual(response.robot_job_id, "test_robot_job_id")
    self.assertEqual(
        response.error_message,
        robot_job_work_unit._ERROR_EMPTY_RESPONSE + "[Error ID: 123456789]",
    )

  def test_start_work_unit_software_asset_prep_good(self):

    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().startWorkUnitSoftwareAssetPrep().execute.return_value = (
        ""
    )

    work_unit_lib = robot_job_work_unit.OrchestratorRobotJobWorkUnit(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    work_unit_lib.set_robot_job_info(
        robot_job_id="test_robot_job_id", launch_command="test_launch_command"
    )
    work_unit_lib._current_work_unit = robot_job_work_unit.work_unit.WorkUnit(
        robotJobId="test_robot_job_id",
        workUnitId="test_work_unit_id",
        context=robot_job_work_unit.work_unit.WorkUnitContext(
            scenePresetId="test_scene_preset",
            sceneEpisodeIndex=1,
            scenePresetDetails=robot_job_work_unit.work_unit.ScenePresetDetails(
                setupInstructions="test_setup_instructions",
                parameters=[
                    robot_job_work_unit.work_unit.KvMsg(
                        key="test_key_1",
                        type=robot_job_work_unit.work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_STRING,
                        value=robot_job_work_unit.work_unit.KvMsgValue(
                            stringValue="test_value_1"
                        ),
                    )
                ],
            ),
            policyDetails=robot_job_work_unit.work_unit.PolicyDetails(
                name="test_policy_name",
                parameters=[
                    robot_job_work_unit.work_unit.KvMsg(
                        key="test_key_1",
                        type=robot_job_work_unit.work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_STRING,
                        value=robot_job_work_unit.work_unit.KvMsgValue(
                            stringValue="test_value_1"
                        ),
                    )
                ],
            ),
        ),
        stage=robot_job_work_unit.work_unit.WorkUnitStage.WORK_UNIT_STAGE_QUEUED_TO_ROBOT,
        outcome=robot_job_work_unit.work_unit.WorkUnitOutcome.WORK_UNIT_OUTCOME_UNSPECIFIED,
        note="Work Unit queued to robot.",
    )

    response = work_unit_lib.start_work_unit_software_asset_prep()
    self.assertTrue(response.success)
    self.assertEqual(response.robot_id, "test_robot_id")
    self.assertEqual(response.robot_job_id, "test_robot_job_id")
    self.assertEqual(response.work_unit_id, "test_work_unit_id")
    self.assertIsInstance(
        response.work_unit, robot_job_work_unit.work_unit.WorkUnit
    )

  def test_start_work_unit_software_asset_prep_bad_server_call(self):

    class MockHttpError:

      def __init__(self):
        self.status = "Mock status"
        self.reason = "Mock reason"
        self.error_details = "Mock error details"

    def raise_error_side_effect():
      raise errors.HttpError(MockHttpError(), "Mock failed HTTP call.".encode())

    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().startWorkUnitSoftwareAssetPrep().execute.side_effect = (
        raise_error_side_effect
    )

    work_unit_lib = robot_job_work_unit.OrchestratorRobotJobWorkUnit(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    work_unit_lib.set_robot_job_info(
        robot_job_id="test_robot_job_id", launch_command="test_launch_command"
    )
    work_unit_lib._current_work_unit = robot_job_work_unit.work_unit.WorkUnit(
        robotJobId="test_robot_job_id",
        workUnitId="test_work_unit_id",
        context=robot_job_work_unit.work_unit.WorkUnitContext(
            scenePresetId="test_scene_preset",
            sceneEpisodeIndex=1,
            scenePresetDetails=robot_job_work_unit.work_unit.ScenePresetDetails(
                setupInstructions="test_setup_instructions",
                parameters=[
                    robot_job_work_unit.work_unit.KvMsg(
                        key="test_key_1",
                        type=robot_job_work_unit.work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_STRING,
                        value=robot_job_work_unit.work_unit.KvMsgValue(
                            stringValue="test_value_1"
                        ),
                    )
                ],
            ),
            policyDetails=robot_job_work_unit.work_unit.PolicyDetails(
                name="test_policy_name",
                parameters=[
                    robot_job_work_unit.work_unit.KvMsg(
                        key="test_key_1",
                        type=robot_job_work_unit.work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_STRING,
                        value=robot_job_work_unit.work_unit.KvMsgValue(
                            stringValue="test_value_1"
                        ),
                    )
                ],
            ),
        ),
        stage=robot_job_work_unit.work_unit.WorkUnitStage.WORK_UNIT_STAGE_QUEUED_TO_ROBOT,
        outcome=robot_job_work_unit.work_unit.WorkUnitOutcome.WORK_UNIT_OUTCOME_UNSPECIFIED,
        note="Work Unit queued to robot.",
    )

    response = work_unit_lib.start_work_unit_software_asset_prep()
    self.assertFalse(response.success)
    self.assertIn(
        robot_job_work_unit._ERROR_WORK_UNIT_SOFTWARE_ASSET_PREP,
        response.error_message,
    )

  def test_start_work_unit_scene_prep_good(self):

    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().startWorkUnitScenePrep().execute.return_value = (
        ""
    )

    work_unit_lib = robot_job_work_unit.OrchestratorRobotJobWorkUnit(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    work_unit_lib.set_robot_job_info(
        robot_job_id="test_robot_job_id", launch_command="test_launch_command"
    )
    work_unit_lib._current_work_unit = robot_job_work_unit.work_unit.WorkUnit(
        robotJobId="test_robot_job_id",
        workUnitId="test_work_unit_id",
        context=robot_job_work_unit.work_unit.WorkUnitContext(
            scenePresetId="test_scene_preset",
            sceneEpisodeIndex=1,
            scenePresetDetails=robot_job_work_unit.work_unit.ScenePresetDetails(
                setupInstructions="test_setup_instructions",
                parameters=[
                    robot_job_work_unit.work_unit.KvMsg(
                        key="test_key_1",
                        type=robot_job_work_unit.work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_STRING,
                        value=robot_job_work_unit.work_unit.KvMsgValue(
                            stringValue="test_value_1"
                        ),
                    )
                ],
            ),
            policyDetails=robot_job_work_unit.work_unit.PolicyDetails(
                name="test_policy_name",
                parameters=[
                    robot_job_work_unit.work_unit.KvMsg(
                        key="test_key_1",
                        type=robot_job_work_unit.work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_STRING,
                        value=robot_job_work_unit.work_unit.KvMsgValue(
                            stringValue="test_value_1"
                        ),
                    )
                ],
            ),
        ),
        stage=robot_job_work_unit.work_unit.WorkUnitStage.WORK_UNIT_STAGE_ROBOT_SOFTWARE_ASSETS_PREP,
        outcome=robot_job_work_unit.work_unit.WorkUnitOutcome.WORK_UNIT_OUTCOME_UNSPECIFIED,
        note="Work Unit queued to robot.",
    )

    response = work_unit_lib.start_work_unit_scene_prep()
    self.assertTrue(response.success)
    self.assertEqual(response.robot_id, "test_robot_id")
    self.assertEqual(response.robot_job_id, "test_robot_job_id")
    self.assertEqual(response.work_unit_id, "test_work_unit_id")
    self.assertIsInstance(
        response.work_unit, robot_job_work_unit.work_unit.WorkUnit
    )

  def test_start_work_unit_scene_prep_bad_server_call(self):

    class MockHttpError:

      def __init__(self):
        self.status = "Mock status"
        self.reason = "Mock reason"
        self.error_details = "Mock error details"

    def raise_error_side_effect():
      raise errors.HttpError(MockHttpError(), "Mock failed HTTP call.".encode())

    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().startWorkUnitScenePrep().execute.side_effect = (
        raise_error_side_effect
    )

    work_unit_lib = robot_job_work_unit.OrchestratorRobotJobWorkUnit(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    work_unit_lib.set_robot_job_info(
        robot_job_id="test_robot_job_id", launch_command="test_launch_command"
    )
    work_unit_lib._current_work_unit = robot_job_work_unit.work_unit.WorkUnit(
        robotJobId="test_robot_job_id",
        workUnitId="test_work_unit_id",
        context=robot_job_work_unit.work_unit.WorkUnitContext(
            scenePresetId="test_scene_preset",
            sceneEpisodeIndex=1,
            scenePresetDetails=robot_job_work_unit.work_unit.ScenePresetDetails(
                setupInstructions="test_setup_instructions",
                parameters=[
                    robot_job_work_unit.work_unit.KvMsg(
                        key="test_key_1",
                        type=robot_job_work_unit.work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_STRING,
                        value=robot_job_work_unit.work_unit.KvMsgValue(
                            stringValue="test_value_1"
                        ),
                    )
                ],
            ),
            policyDetails=robot_job_work_unit.work_unit.PolicyDetails(
                name="test_policy_name",
                parameters=[
                    robot_job_work_unit.work_unit.KvMsg(
                        key="test_key_1",
                        type=robot_job_work_unit.work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_STRING,
                        value=robot_job_work_unit.work_unit.KvMsgValue(
                            stringValue="test_value_1"
                        ),
                    )
                ],
            ),
        ),
        stage=robot_job_work_unit.work_unit.WorkUnitStage.WORK_UNIT_STAGE_ROBOT_SOFTWARE_ASSETS_PREP,
        outcome=robot_job_work_unit.work_unit.WorkUnitOutcome.WORK_UNIT_OUTCOME_UNSPECIFIED,
        note="Work Unit queued to robot.",
    )

    response = work_unit_lib.start_work_unit_scene_prep()
    self.assertFalse(response.success)
    self.assertIn(
        robot_job_work_unit._ERROR_WORK_UNIT_SCENE_PREP,
        response.error_message,
    )

  def test_start_work_unit_execution_good(self):

    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().startWorkUnitExecution().execute.return_value = (
        ""
    )

    work_unit_lib = robot_job_work_unit.OrchestratorRobotJobWorkUnit(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    work_unit_lib.set_robot_job_info(
        robot_job_id="test_robot_job_id", launch_command="test_launch_command"
    )
    work_unit_lib._current_work_unit = robot_job_work_unit.work_unit.WorkUnit(
        robotJobId="test_robot_job_id",
        workUnitId="test_work_unit_id",
        context=robot_job_work_unit.work_unit.WorkUnitContext(
            scenePresetId="test_scene_preset",
            sceneEpisodeIndex=1,
            scenePresetDetails=robot_job_work_unit.work_unit.ScenePresetDetails(
                setupInstructions="test_setup_instructions",
                parameters=[
                    robot_job_work_unit.work_unit.KvMsg(
                        key="test_key_1",
                        type=robot_job_work_unit.work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_STRING,
                        value=robot_job_work_unit.work_unit.KvMsgValue(
                            stringValue="test_value_1"
                        ),
                    )
                ],
            ),
            policyDetails=robot_job_work_unit.work_unit.PolicyDetails(
                name="test_policy_name",
                parameters=[
                    robot_job_work_unit.work_unit.KvMsg(
                        key="test_key_1",
                        type=robot_job_work_unit.work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_STRING,
                        value=robot_job_work_unit.work_unit.KvMsgValue(
                            stringValue="test_value_1"
                        ),
                    )
                ],
            ),
        ),
        stage=robot_job_work_unit.work_unit.WorkUnitStage.WORK_UNIT_STAGE_ROBOT_SCENE_PREP,
        outcome=robot_job_work_unit.work_unit.WorkUnitOutcome.WORK_UNIT_OUTCOME_UNSPECIFIED,
        note="Work Unit queued to robot.",
    )

    response = work_unit_lib.start_work_unit_execution()
    self.assertTrue(response.success)
    self.assertEqual(response.robot_id, "test_robot_id")
    self.assertEqual(response.robot_job_id, "test_robot_job_id")
    self.assertEqual(response.work_unit_id, "test_work_unit_id")
    self.assertIsInstance(
        response.work_unit, robot_job_work_unit.work_unit.WorkUnit
    )

  def test_start_work_unit_execution_bad_server_call(self):

    class MockHttpError:

      def __init__(self):
        self.status = "Mock status"
        self.reason = "Mock reason"
        self.error_details = "Mock error details"

    def raise_error_side_effect():
      raise errors.HttpError(MockHttpError(), "Mock failed HTTP call.".encode())

    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().startWorkUnitExecution().execute.side_effect = (
        raise_error_side_effect
    )

    work_unit_lib = robot_job_work_unit.OrchestratorRobotJobWorkUnit(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    work_unit_lib.set_robot_job_info(
        robot_job_id="test_robot_job_id", launch_command="test_launch_command"
    )
    work_unit_lib._current_work_unit = robot_job_work_unit.work_unit.WorkUnit(
        robotJobId="test_robot_job_id",
        workUnitId="test_work_unit_id",
        context=robot_job_work_unit.work_unit.WorkUnitContext(
            scenePresetId="test_scene_preset",
            sceneEpisodeIndex=1,
            scenePresetDetails=robot_job_work_unit.work_unit.ScenePresetDetails(
                setupInstructions="test_setup_instructions",
                parameters=[
                    robot_job_work_unit.work_unit.KvMsg(
                        key="test_key_1",
                        type=robot_job_work_unit.work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_STRING,
                        value=robot_job_work_unit.work_unit.KvMsgValue(
                            stringValue="test_value_1"
                        ),
                    )
                ],
            ),
            policyDetails=robot_job_work_unit.work_unit.PolicyDetails(
                name="test_policy_name",
                parameters=[
                    robot_job_work_unit.work_unit.KvMsg(
                        key="test_key_1",
                        type=robot_job_work_unit.work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_STRING,
                        value=robot_job_work_unit.work_unit.KvMsgValue(
                            stringValue="test_value_1"
                        ),
                    )
                ],
            ),
        ),
        stage=robot_job_work_unit.work_unit.WorkUnitStage.WORK_UNIT_STAGE_ROBOT_SCENE_PREP,
        outcome=robot_job_work_unit.work_unit.WorkUnitOutcome.WORK_UNIT_OUTCOME_UNSPECIFIED,
        note="Work Unit queued to robot.",
    )

    response = work_unit_lib.start_work_unit_execution()
    self.assertFalse(response.success)
    self.assertIn(
        robot_job_work_unit._ERROR_WORK_UNIT_EXECUTION,
        response.error_message,
    )

  def test_insert_session_info_good(self):

    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().insertSessionInfo().execute.return_value = ""

    work_unit_lib = robot_job_work_unit.OrchestratorRobotJobWorkUnit(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    work_unit_lib.set_robot_job_info(
        robot_job_id="test_robot_job_id", launch_command="test_launch_command"
    )
    work_unit_lib._current_work_unit = robot_job_work_unit.work_unit.WorkUnit(
        robotJobId="test_robot_job_id",
        workUnitId="test_work_unit_id",
        stage=robot_job_work_unit.work_unit.WorkUnitStage.WORK_UNIT_STAGE_ROBOT_EXECUTION,
    )
    work_unit_lib._work_unit_in_execution_stage = True

    response = work_unit_lib.insert_session_info(
        session_log_type="test_log_type",
        session_start_time_ns=1234567890123456789,
        session_end_time_ns=1234567890123456799,
        session_note="test_note",
    )

    self.assertTrue(response.success)
    self.assertEqual(response.robot_id, "test_robot_id")
    self.assertEqual(response.robot_job_id, "test_robot_job_id")
    self.assertEqual(response.work_unit_id, "test_work_unit_id")

  def test_insert_session_info_not_in_execution(self):

    mock_connection = mock.MagicMock()
    work_unit_lib = robot_job_work_unit.OrchestratorRobotJobWorkUnit(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    work_unit_lib._current_work_unit = robot_job_work_unit.work_unit.WorkUnit(
        robotJobId="test_robot_job_id",
        workUnitId="test_work_unit_id",
    )
    work_unit_lib._work_unit_in_execution_stage = False

    response = work_unit_lib.insert_session_info(
        session_log_type="test_log_type",
        session_start_time_ns=1234567890123456789,
        session_end_time_ns=1234567890123456799,
        session_note="test_note",
    )

    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message,
        robot_job_work_unit._ERROR_WORK_UNIT_IN_EXECUTION_STAGE,
    )

  def test_insert_session_info_missing_log_type(self):

    mock_connection = mock.MagicMock()
    work_unit_lib = robot_job_work_unit.OrchestratorRobotJobWorkUnit(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    work_unit_lib._current_work_unit = robot_job_work_unit.work_unit.WorkUnit(
        robotJobId="test_robot_job_id",
        workUnitId="test_work_unit_id",
    )
    work_unit_lib._work_unit_in_execution_stage = True

    response = work_unit_lib.insert_session_info(
        session_log_type="",
        session_start_time_ns=1234567890123456789,
        session_end_time_ns=1234567890123456799,
        session_note="test_note",
    )

    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message,
        robot_job_work_unit._ERROR_MISSING_SESSION_LOG_TYPE,
    )

  def test_insert_session_info_bad_start_time(self):

    mock_connection = mock.MagicMock()
    work_unit_lib = robot_job_work_unit.OrchestratorRobotJobWorkUnit(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    work_unit_lib._current_work_unit = robot_job_work_unit.work_unit.WorkUnit(
        robotJobId="test_robot_job_id",
        workUnitId="test_work_unit_id",
    )
    work_unit_lib._work_unit_in_execution_stage = True

    response = work_unit_lib.insert_session_info(
        session_log_type="test_log_type",
        session_start_time_ns=12345,
        session_end_time_ns=1234567890123456799,
        session_note="test_note",
    )

    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message,
        robot_job_work_unit._ERROR_BAD_SESSION_START_TIME,
    )

  def test_insert_session_info_bad_end_time(self):

    mock_connection = mock.MagicMock()
    work_unit_lib = robot_job_work_unit.OrchestratorRobotJobWorkUnit(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    work_unit_lib._current_work_unit = robot_job_work_unit.work_unit.WorkUnit(
        robotJobId="test_robot_job_id",
        workUnitId="test_work_unit_id",
    )
    work_unit_lib._work_unit_in_execution_stage = True

    response = work_unit_lib.insert_session_info(
        session_log_type="test_log_type",
        session_start_time_ns=1234567890123456789,
        session_end_time_ns=12345,
        session_note="test_note",
    )

    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message,
        robot_job_work_unit._ERROR_BAD_SESSION_END_TIME,
    )

  def test_insert_session_info_bad_server_call(self):

    class MockHttpError:

      def __init__(self):
        self.status = "Mock status"
        self.reason = "Mock reason"
        self.error_details = "Mock error details"

    def raise_error_side_effect():
      raise errors.HttpError(MockHttpError(), "Mock failed HTTP call.".encode())

    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().insertSessionInfo().execute.side_effect = (
        raise_error_side_effect
    )

    work_unit_lib = robot_job_work_unit.OrchestratorRobotJobWorkUnit(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    work_unit_lib._current_work_unit = robot_job_work_unit.work_unit.WorkUnit(
        robotJobId="test_robot_job_id",
        workUnitId="test_work_unit_id",
    )
    work_unit_lib._work_unit_in_execution_stage = True

    response = work_unit_lib.insert_session_info(
        session_log_type="test_log_type",
        session_start_time_ns=1234567890123456789,
        session_end_time_ns=1234567890123456799,
        session_note="test_note",
    )

    self.assertFalse(response.success)
    self.assertIn(
        robot_job_work_unit._ERROR_WORK_UNIT_INSERT_SESSION_INFO,
        response.error_message,
    )

  def test_complete_work_unit_good(self):

    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().completeWorkUnit().execute.return_value = (
        ""
    )

    work_unit_lib = robot_job_work_unit.OrchestratorRobotJobWorkUnit(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    work_unit_lib.set_robot_job_info(
        robot_job_id="test_robot_job_id", launch_command="test_launch_command"
    )
    work_unit_lib._current_work_unit = robot_job_work_unit.work_unit.WorkUnit(
        robotJobId="test_robot_job_id",
        workUnitId="test_work_unit_id",
        context=robot_job_work_unit.work_unit.WorkUnitContext(
            scenePresetId="test_scene_preset",
            sceneEpisodeIndex=1,
            scenePresetDetails=robot_job_work_unit.work_unit.ScenePresetDetails(
                setupInstructions="test_setup_instructions",
                parameters=[
                    robot_job_work_unit.work_unit.KvMsg(
                        key="test_key_1",
                        type=robot_job_work_unit.work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_STRING,
                        value=robot_job_work_unit.work_unit.KvMsgValue(
                            stringValue="test_value_1"
                        ),
                    )
                ],
            ),
            policyDetails=robot_job_work_unit.work_unit.PolicyDetails(
                name="test_policy_name",
                parameters=[
                    robot_job_work_unit.work_unit.KvMsg(
                        key="test_key_1",
                        type=robot_job_work_unit.work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_STRING,
                        value=robot_job_work_unit.work_unit.KvMsgValue(
                            stringValue="test_value_1"
                        ),
                    )
                ],
            ),
            questions=[
                robot_job_work_unit.work_unit.Question(
                    question="test_question_1",
                    whenToAsk=[
                        robot_job_work_unit.work_unit.QuestionCondition.QUESTION_CONDITION_ALWAYS,
                    ],
                    answerFormat=robot_job_work_unit.work_unit.AnswerType.ANSWER_TYPE_SINGLE_CHOICE,
                    allowedAnswers=[
                        "test_allowed_answer_1", "test_allowed_answer_2"
                    ],
                    userAnswers=["test_user_answer_1"],
                    wasDisplayed=True,
                ),
            ],
        ),
        stage=robot_job_work_unit.work_unit.WorkUnitStage.WORK_UNIT_STAGE_ROBOT_EXECUTION,
        outcome=robot_job_work_unit.work_unit.WorkUnitOutcome.WORK_UNIT_OUTCOME_UNSPECIFIED,
        note="Work Unit queued to robot.",
    )

    response = work_unit_lib.complete_work_unit(
        outcome=robot_job_work_unit.work_unit.WorkUnitOutcome.WORK_UNIT_OUTCOME_SUCCESS,
        success_score=0.5,
        success_score_definition="test_success_score_definition",
        note="Work Unit completed.",
        session_start_time_ns=1764547200000000001,
        session_end_time_ns=1764547210000000002,
        session_log_type="test_session_log_type",
        session_note="test_session_note",
        response_to_questions=[
            robot_job_work_unit.work_unit.Question(
                question="test_question_1",
                whenToAsk=[
                    robot_job_work_unit.work_unit.QuestionCondition.QUESTION_CONDITION_ALWAYS,
                ],
                answerFormat=robot_job_work_unit.work_unit.AnswerType.ANSWER_TYPE_SINGLE_CHOICE,
                allowedAnswers=[
                    "test_allowed_answer_1",
                    "test_allowed_answer_2",
                ],
                userAnswers=["test_user_answer_1"],
                wasDisplayed=True,
            ),
        ],
        request_retry_bypass=False,
        client_overrides=None,
    )
    self.assertTrue(response.success)
    self.assertEqual(response.robot_id, "test_robot_id")
    self.assertEqual(response.robot_job_id, "test_robot_job_id")
    self.assertEqual(response.work_unit_id, "test_work_unit_id")
    self.assertIsInstance(
        response.work_unit, robot_job_work_unit.work_unit.WorkUnit
    )

  def test_complete_work_unit_with_client_overrides(self):
    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().completeWorkUnit().execute.return_value = ""
    work_unit_lib = robot_job_work_unit.OrchestratorRobotJobWorkUnit(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    work_unit_lib.set_robot_job_info(
        robot_job_id="test_robot_job_id", launch_command="test_launch_command"
    )
    work_unit_lib._current_work_unit = robot_job_work_unit.work_unit.WorkUnit(
        robotJobId="test_robot_job_id",
        workUnitId="test_work_unit_id",
        context=robot_job_work_unit.work_unit.WorkUnitContext(),
        stage=robot_job_work_unit.work_unit.WorkUnitStage.WORK_UNIT_STAGE_ROBOT_EXECUTION,
    )
    work_unit_lib._work_unit_in_execution_stage = True

    response = work_unit_lib.complete_work_unit(
        outcome=robot_job_work_unit.work_unit.WorkUnitOutcome.WORK_UNIT_OUTCOME_SUCCESS,
        success_score=0.5,
        success_score_definition="test_success_score_definition",
        note="Work Unit completed.",
        session_start_time_ns=1764547200000000001,
        session_end_time_ns=1764547210000000002,
        session_log_type="test_session_log_type",
        session_note="test_session_note",
        response_to_questions=[],
        request_retry_bypass=False,
        client_overrides=[
            robot_job_work_unit.work_unit.KvMsg(
                key="test_override_key",
                type=robot_job_work_unit.work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_STRING,
                value=robot_job_work_unit.work_unit.KvMsgValue(
                    stringValue="test_override_value"
                ),
            )
        ],
    )
    self.assertTrue(response.success)

    calls = mock_connection.orchestrator().completeWorkUnit.call_args_list
    body_kwargs = [c.kwargs["body"] for c in calls if "body" in c.kwargs]
    self.assertLen(body_kwargs, 1)

    client_overrides = body_kwargs[0]["client_overrides"]
    self.assertLen(client_overrides, 1)
    self.assertEqual(client_overrides[0]["key"], "test_override_key")
    self.assertEqual(client_overrides[0]["type"], 1)
    self.assertEqual(
        client_overrides[0]["value"]["string_value"], "test_override_value"
    )

  def test_complete_work_unit_missing_session_start_time_ns(self):

    mock_connection = mock.MagicMock()
    work_unit_lib = robot_job_work_unit.OrchestratorRobotJobWorkUnit(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    work_unit_lib.set_robot_job_info(
        robot_job_id="test_robot_job_id", launch_command="test_launch_command"
    )

    response = work_unit_lib.complete_work_unit(
        outcome=robot_job_work_unit.work_unit.WorkUnitOutcome.WORK_UNIT_OUTCOME_SUCCESS,
        success_score=0.5,
        success_score_definition="test_success_score_definition",
        note="Work Unit completed.",
        session_start_time_ns=None,
        session_end_time_ns=1764547210000000002,
        session_log_type="test_session_log_type",
        session_note=None,
        response_to_questions=None,
        request_retry_bypass=False,
        client_overrides=None,
    )
    self.assertFalse(response.success)
    self.assertIn(
        robot_job_work_unit._ERROR_MISSING_SESSION_START_TIME,
        response.error_message,
    )

  def test_complete_work_unit_missing_session_end_time_ns(self):

    mock_connection = mock.MagicMock()
    work_unit_lib = robot_job_work_unit.OrchestratorRobotJobWorkUnit(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    work_unit_lib.set_robot_job_info(
        robot_job_id="test_robot_job_id", launch_command="test_launch_command"
    )

    response = work_unit_lib.complete_work_unit(
        outcome=robot_job_work_unit.work_unit.WorkUnitOutcome.WORK_UNIT_OUTCOME_SUCCESS,
        success_score=0.5,
        success_score_definition="test_success_score_definition",
        note="Work Unit completed.",
        session_start_time_ns=1764547210000000002,
        session_end_time_ns=None,
        session_log_type="test_session_log_type",
        session_note=None,
        response_to_questions=None,
        request_retry_bypass=False,
        client_overrides=None,
    )
    self.assertFalse(response.success)
    self.assertIn(
        robot_job_work_unit._ERROR_MISSING_SESSION_END_TIME,
        response.error_message,
    )

  def test_complete_work_unit_missing_session_log_type(self):

    mock_connection = mock.MagicMock()
    work_unit_lib = robot_job_work_unit.OrchestratorRobotJobWorkUnit(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    work_unit_lib.set_robot_job_info(
        robot_job_id="test_robot_job_id", launch_command="test_launch_command"
    )

    response = work_unit_lib.complete_work_unit(
        outcome=robot_job_work_unit.work_unit.WorkUnitOutcome.WORK_UNIT_OUTCOME_SUCCESS,
        success_score=0.5,
        success_score_definition="test_success_score_definition",
        note="Work Unit completed.",
        session_start_time_ns=1764547200000000001,
        session_end_time_ns=1764547210000000002,
        session_log_type=None,
        session_note=None,
        response_to_questions=None,
        request_retry_bypass=False,
        client_overrides=None,
    )
    self.assertFalse(response.success)
    self.assertIn(
        robot_job_work_unit._ERROR_MISSING_SESSION_LOG_TYPE,
        response.error_message,
    )

  def test_complete_work_unit_bad_session_start_time_ns(self):

    mock_connection = mock.MagicMock()
    work_unit_lib = robot_job_work_unit.OrchestratorRobotJobWorkUnit(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    work_unit_lib.set_robot_job_info(
        robot_job_id="test_robot_job_id", launch_command="test_launch_command"
    )

    response = work_unit_lib.complete_work_unit(
        outcome=robot_job_work_unit.work_unit.WorkUnitOutcome.WORK_UNIT_OUTCOME_SUCCESS,
        success_score=0.5,
        success_score_definition="test_success_score_definition",
        note="Work Unit completed.",
        session_start_time_ns=123456789,
        session_end_time_ns=1764547210000000002,
        session_log_type="test_session_log_type",
        session_note="test_session_note",
        response_to_questions=None,
        request_retry_bypass=False,
        client_overrides=None,
    )
    self.assertFalse(response.success)
    self.assertIn(
        robot_job_work_unit._ERROR_BAD_SESSION_START_TIME,
        response.error_message,
    )

  def test_complete_work_unit_bad_session_end_time_ns(self):

    mock_connection = mock.MagicMock()
    work_unit_lib = robot_job_work_unit.OrchestratorRobotJobWorkUnit(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    work_unit_lib.set_robot_job_info(
        robot_job_id="test_robot_job_id", launch_command="test_launch_command"
    )

    response = work_unit_lib.complete_work_unit(
        outcome=robot_job_work_unit.work_unit.WorkUnitOutcome.WORK_UNIT_OUTCOME_SUCCESS,
        success_score=0.5,
        success_score_definition="test_success_score_definition",
        note="Work Unit completed.",
        session_start_time_ns=1764547200000000001,
        session_end_time_ns=987654321,
        session_log_type="test_session_log_type",
        session_note="test_session_note",
        response_to_questions=None,
        request_retry_bypass=False,
        client_overrides=None,
    )
    self.assertFalse(response.success)
    self.assertIn(
        robot_job_work_unit._ERROR_BAD_SESSION_END_TIME,
        response.error_message,
    )

  def test_complete_work_unit_bad_server_call(self):

    class MockHttpError:

      def __init__(self):
        self.status = "Mock status"
        self.reason = "Mock reason"
        self.error_details = "Mock error details"

    def raise_error_side_effect():
      raise errors.HttpError(MockHttpError(), "Mock failed HTTP call.".encode())

    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().completeWorkUnit().execute.side_effect = (
        raise_error_side_effect
    )

    work_unit_lib = robot_job_work_unit.OrchestratorRobotJobWorkUnit(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    work_unit_lib.set_robot_job_info(
        robot_job_id="test_robot_job_id", launch_command="test_launch_command"
    )
    work_unit_lib._current_work_unit = robot_job_work_unit.work_unit.WorkUnit(
        robotJobId="test_robot_job_id",
        workUnitId="test_work_unit_id",
        context=robot_job_work_unit.work_unit.WorkUnitContext(
            scenePresetId="test_scene_preset",
            sceneEpisodeIndex=1,
            scenePresetDetails=robot_job_work_unit.work_unit.ScenePresetDetails(
                setupInstructions="test_setup_instructions",
                parameters=[
                    robot_job_work_unit.work_unit.KvMsg(
                        key="test_key_1",
                        type=robot_job_work_unit.work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_STRING,
                        value=robot_job_work_unit.work_unit.KvMsgValue(
                            stringValue="test_value_1"
                        ),
                    )
                ],
            ),
            policyDetails=robot_job_work_unit.work_unit.PolicyDetails(
                name="test_policy_name",
                parameters=[
                    robot_job_work_unit.work_unit.KvMsg(
                        key="test_key_1",
                        type=robot_job_work_unit.work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_STRING,
                        value=robot_job_work_unit.work_unit.KvMsgValue(
                            stringValue="test_value_1"
                        ),
                    )
                ],
            ),
        ),
        stage=robot_job_work_unit.work_unit.WorkUnitStage.WORK_UNIT_STAGE_ROBOT_EXECUTION,
        outcome=robot_job_work_unit.work_unit.WorkUnitOutcome.WORK_UNIT_OUTCOME_UNSPECIFIED,
        note="Work Unit queued to robot.",
    )

    response = work_unit_lib.complete_work_unit(
        outcome=robot_job_work_unit.work_unit.WorkUnitOutcome.WORK_UNIT_OUTCOME_SUCCESS,
        success_score=0.5,
        success_score_definition="test_success_score_definition",
        note="Work Unit completed.",
        session_start_time_ns=1764547200000000001,
        session_end_time_ns=1764547210000000002,
        session_log_type="test_session_log_type",
        session_note="test_session_note",
        response_to_questions=[
            robot_job_work_unit.work_unit.Question(
                question="test_question_1",
                whenToAsk=[
                    robot_job_work_unit.work_unit.QuestionCondition.QUESTION_CONDITION_ALWAYS,
                ],
                answerFormat=robot_job_work_unit.work_unit.AnswerType.ANSWER_TYPE_SINGLE_CHOICE,
                allowedAnswers=[
                    "test_allowed_answer_1",
                    "test_allowed_answer_2",
                ],
                userAnswers=["test_user_answer_1"],
                wasDisplayed=True,
            ),
        ],
        request_retry_bypass=False,
        client_overrides=None,
    )
    self.assertFalse(response.success)
    self.assertIn(
        robot_job_work_unit._ERROR_WORK_UNIT_COMPLETED,
        response.error_message,
    )

  def test_observe_latest_work_unit_good(self):

    mock_connection = mock.MagicMock()
    mock_response = {
        "workUnit": {
            "robotJobId": "test_robot_job_id",
            "workUnitId": "test_work_unit_id",
            "context": {
                "scenePresetId": "test_scene_preset",
                "sceneEpisodeIndex": "1",
                "scenePresetDetails": {
                    "setupInstructions": "test_setup_instructions",
                    "parameters": [{
                        "key": "scene_key_1",
                        "value": {"stringValue": "scene_value_1"},
                        "type": "KV_MSG_VALUE_TYPE_STRING",
                    }],
                },
                "policyDetails": {
                    "name": "test_policy_name",
                    "parameters": [{
                        "key": "policy_key_1",
                        "value": {"stringValue": "policy_value_1"},
                        "type": "KV_MSG_VALUE_TYPE_STRING",
                    }],
                },
                "questions": [{
                    "question": "test_question_1",
                    "whenToAsk": [
                        "QUESTION_CONDITION_ONLY_FAILED_EPISODE",
                        "QUESTION_CONDITION_ESTOP_NOT_PRESSED",
                    ],
                    "answerFormat": "ANSWER_TYPE_YES_NO",
                    "allowedAnswers": ["allowed_answer_1", "allowed_answer_2"],
                }],
            },
            "stage": "WORK_UNIT_STAGE_QUEUED_TO_ROBOT",
            "outcome": "WORK_UNIT_OUTCOME_UNSPECIFIED",
            "note": "Work Unit queued to robot.",
        }
    }
    mock_connection.orchestrator().observeCurrentWorkUnit().execute.return_value = (
        mock_response
    )
    work_unit_lib = robot_job_work_unit.OrchestratorRobotJobWorkUnit(
        connection=mock_connection,
        robot_id="test_robot_id",
    )

    response = work_unit_lib.observe_latest_work_unit(
        job_type=robot_job.JobType.EVALUATION
    )

    self.assertTrue(response.success)
    self.assertEqual(response.robot_id, "test_robot_id")
    self.assertEqual(response.robot_job_id, "test_robot_job_id")
    self.assertEqual(response.work_unit_id, "test_work_unit_id")
    self.assertIsInstance(
        response.work_unit, robot_job_work_unit.work_unit.WorkUnit
    )
    self.assertEqual(
        response.work_unit.stage,
        robot_job_work_unit.work_unit.WorkUnitStage.WORK_UNIT_STAGE_QUEUED_TO_ROBOT,
    )
    self.assertEqual(
        response.work_unit.outcome,
        robot_job_work_unit.work_unit.WorkUnitOutcome.WORK_UNIT_OUTCOME_UNSPECIFIED,
    )
    self.assertEqual(response.work_unit.note, "Work Unit queued to robot.")
    questions = response.work_unit.context.questions
    self.assertLen(questions, 1)
    self.assertEqual(questions[0].question, "test_question_1")
    self.assertEqual(
        questions[0].whenToAsk,
        [
            robot_job_work_unit.work_unit.QuestionCondition.QUESTION_CONDITION_ONLY_FAILED_EPISODE,
            robot_job_work_unit.work_unit.QuestionCondition.QUESTION_CONDITION_ESTOP_NOT_PRESSED,
        ],
    )
    self.assertEqual(
        questions[0].answerFormat,
        robot_job_work_unit.work_unit.AnswerType.ANSWER_TYPE_YES_NO,
    )
    self.assertEqual(
        questions[0].allowedAnswers,
        ["allowed_answer_1", "allowed_answer_2"],
    )
    self.assertEmpty(questions[0].userAnswers)
    self.assertFalse(questions[0].wasDisplayed)

  def test_observe_latest_work_unit_bad_server_call(self):

    class MockHttpError:

      def __init__(self):
        self.status = "Mock status"
        self.reason = "Mock reason"
        self.error_details = "Mock error details"

    def raise_error_side_effect():
      raise errors.HttpError(MockHttpError(), "Mock failed HTTP call.".encode())

    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().observeCurrentWorkUnit().execute.side_effect = (
        raise_error_side_effect
    )

    work_unit_lib = robot_job_work_unit.OrchestratorRobotJobWorkUnit(
        connection=mock_connection,
        robot_id="test_robot_id",
    )

    response = work_unit_lib.observe_latest_work_unit(
        job_type=robot_job.JobType.EVALUATION
    )
    self.assertFalse(response.success)
    self.assertIn(
        robot_job_work_unit._ERROR_OBSERVE_LATEST_WORK_UNIT,
        response.error_message,
    )

  @mock.patch("time.time_ns")
  def test_observe_latest_work_unit_no_more_work_unit(self, mock_time):
    mock_time.return_value = 123456789
    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().observeCurrentWorkUnit().execute.return_value = (
        ""
    )

    work_unit_lib = robot_job_work_unit.OrchestratorRobotJobWorkUnit(
        connection=mock_connection,
        robot_id="test_robot_id",
    )

    response = work_unit_lib.observe_latest_work_unit(
        job_type=robot_job.JobType.COLLECTION
    )
    self.assertTrue(response.success)
    self.assertTrue(response.no_more_work_unit)
    self.assertEqual(
        response.error_message,
        (
            robot_job_work_unit._ERROR_EMPTY_OBSERVE_LATEST_WORK_UNIT_RESPONSE
            + "[Error ID: 123456789]"
        ),
    )


if __name__ == "__main__":
  absltest.main()
