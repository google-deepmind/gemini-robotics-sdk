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
                }
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
    work_unit_lib.set_robot_job_id("test_robot_job_id")

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
    work_unit_lib.set_robot_job_id("test_robot_job_id")

    response = work_unit_lib.request_work_unit()
    self.assertFalse(response.success)
    self.assertIn(
        robot_job_work_unit._ERROR_GET_WORK_UNIT, response.error_message
    )

  def test_request_work_unit_no_more_work_unit(self):

    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().allocateWorkUnit().execute.return_value = ""

    work_unit_lib = robot_job_work_unit.OrchestratorRobotJobWorkUnit(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    work_unit_lib.set_robot_job_id("test_robot_job_id")

    response = work_unit_lib.request_work_unit()
    self.assertTrue(response.success)
    self.assertTrue(response.no_more_work_unit)
    self.assertEqual(
        response.error_message, robot_job_work_unit._ERROR_EMPTY_RESPONSE
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
    work_unit_lib.set_robot_job_id("test_robot_job_id")
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
    work_unit_lib.set_robot_job_id("test_robot_job_id")
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
    work_unit_lib.set_robot_job_id("test_robot_job_id")
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
    work_unit_lib.set_robot_job_id("test_robot_job_id")
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
    work_unit_lib.set_robot_job_id("test_robot_job_id")
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
    work_unit_lib.set_robot_job_id("test_robot_job_id")
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

  def test_complete_work_unit_good(self):

    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().completeWorkUnit().execute.return_value = (
        ""
    )

    work_unit_lib = robot_job_work_unit.OrchestratorRobotJobWorkUnit(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    work_unit_lib.set_robot_job_id("test_robot_job_id")
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
        note="Work Unit completed.",
    )
    self.assertTrue(response.success)
    self.assertEqual(response.robot_id, "test_robot_id")
    self.assertEqual(response.robot_job_id, "test_robot_job_id")
    self.assertEqual(response.work_unit_id, "test_work_unit_id")
    self.assertIsInstance(
        response.work_unit, robot_job_work_unit.work_unit.WorkUnit
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
    work_unit_lib.set_robot_job_id("test_robot_job_id")
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
        note="Work Unit completed.",
    )
    self.assertFalse(response.success)
    self.assertIn(
        robot_job_work_unit._ERROR_WORK_UNIT_COMPLETED,
        response.error_message,
    )


if __name__ == "__main__":
  absltest.main()
