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

"""Unit tests for current_robot.py."""

from unittest import mock

from absl.testing import absltest
from googleapiclient import errors

from safari_sdk.orchestrator.client.libs import current_robot


class CurrentRobotTest(absltest.TestCase):

  def test_get_current_robot_info_good(self):

    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().currentRobotInfo().execute.return_value = {
        "robotId": "test_robot_id",
        "isOperational": True,
        "operatorId": "test_operator_id",
        "robotJobId": "test_robot_job_id",
        "workUnitId": "test_work_unit_id",
        "stage": "WORK_UNIT_STAGE_QUEUED_TO_ROBOT",
    }

    current_robot_lib = current_robot.OrchestratorCurrentRobotInfo(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    response = current_robot_lib.get_current_robot_info()
    self.assertTrue(response.success)
    self.assertEqual(response.robot_id, "test_robot_id")
    self.assertEqual(response.robot_job_id, "test_robot_job_id")
    self.assertEqual(response.work_unit_id, "test_work_unit_id")
    self.assertEqual(
        response.work_unit_stage,
        current_robot.current_robot_info.work_unit.WorkUnitStage.WORK_UNIT_STAGE_QUEUED_TO_ROBOT,
    )
    self.assertEqual(response.operator_id, "test_operator_id")
    self.assertTrue(response.is_operational)

    mock_connection.orchestrator().currentRobotInfo().execute.return_value = {
        "robotId": "test_robot_id",
        "isOperational": False,
        "operatorId": "test_operator_id",
        "robotJobId": "test_robot_job_id",
        "workUnitId": "test_work_unit_id",
        "stage": "WORK_UNIT_STAGE_QUEUED_TO_ROBOT",
    }
    response = current_robot_lib.get_current_robot_info()
    self.assertTrue(response.success)
    self.assertFalse(response.is_operational)

  def test_request_robot_job_bad_server_call(self):

    class MockHttpError:

      def __init__(self):
        self.status = "Mock status"
        self.reason = "Mock reason"
        self.error_details = "Mock error details"

    def raise_error_side_effect():
      raise errors.HttpError(MockHttpError(), "Mock failed HTTP call.".encode())

    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().currentRobotInfo().execute.side_effect = (
        raise_error_side_effect
    )

    current_robot_lib = current_robot.OrchestratorCurrentRobotInfo(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    response = current_robot_lib.get_current_robot_info()

    self.assertFalse(response.success)
    self.assertIn(
        current_robot._ERROR_GET_CURRENT_ROBOT_INFO, response.error_message
    )

  def test_set_current_robot_operator_id_good(self):

    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().currentRobotSetOperatorId().execute.return_value = {
        "robotId": "test_robot_id",
        "operatorId": "test_operator_id",
    }

    current_robot_lib = current_robot.OrchestratorCurrentRobotInfo(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    response = current_robot_lib.set_current_robot_operator_id(
        operator_id="test_operator_id"
    )
    self.assertTrue(response.success)
    self.assertEqual(response.robot_id, "test_robot_id")
    self.assertEqual(response.operator_id, "test_operator_id")

  def test_set_current_robot_operator_id_bad_server_call(self):

    class MockHttpError:

      def __init__(self):
        self.status = "Mock status"
        self.reason = "Mock reason"
        self.error_details = "Mock error details"

    def raise_error_side_effect():
      raise errors.HttpError(MockHttpError(), "Mock failed HTTP call.".encode())

    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().currentRobotSetOperatorId().execute.side_effect = (
        raise_error_side_effect
    )

    current_robot_lib = current_robot.OrchestratorCurrentRobotInfo(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    response = current_robot_lib.set_current_robot_operator_id(
        operator_id="test_operator_id"
    )

    self.assertFalse(response.success)
    self.assertIn(
        current_robot._ERROR_SET_CURRENT_ROBOT_OPERATOR_ID,
        response.error_message,
    )


if __name__ == "__main__":
  absltest.main()
