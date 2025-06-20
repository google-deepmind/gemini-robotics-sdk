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


class RobotJobTest(absltest.TestCase):

  def test_get_current_robot_job_good(self):
    mock_connection = mock.MagicMock()
    robot_job_lib = robot_job.OrchestratorRobotJob(
        connection=mock_connection,
        robot_id="test_robot_id",
        job_type=robot_job.JobType.ALL,
    )
    robot_job_lib._current_robot_job = mock.MagicMock(
        spec=robot_job.robot_job.RobotJob
    )

    response = robot_job_lib.get_current_robot_job()
    self.assertTrue(response.success)
    self.assertIsInstance(response.robot_job, robot_job.robot_job.RobotJob)

  def test_get_current_robot_job_bad(self):
    mock_connection = mock.MagicMock()
    robot_job_lib = robot_job.OrchestratorRobotJob(
        connection=mock_connection,
        robot_id="test_robot_id",
        job_type=robot_job.JobType.COLLECTION,
    )

    response = robot_job_lib.get_current_robot_job()
    self.assertFalse(response.success)
    self.assertIn(
        robot_job._ERROR_ROBOT_JOB_NOT_ACQUIRED,
        response.error_message,
    )

  def test_request_robot_job_good(self):

    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().allocateRobotJob().execute.return_value = {
        "robotJob": {"robotJobId": "test_robot_job_id"}
    }

    robot_job_lib = robot_job.OrchestratorRobotJob(
        connection=mock_connection,
        robot_id="test_robot_id",
        job_type=robot_job.JobType.EVALUATION,
    )
    response = robot_job_lib.request_robot_job()
    self.assertTrue(response.success)
    self.assertEqual(response.robot_id, "test_robot_id")
    self.assertEqual(response.robot_job_id, "test_robot_job_id")

  def test_request_robot_job_bad_server_call(self):

    class MockHttpError:

      def __init__(self):
        self.status = "Mock status"
        self.reason = "Mock reason"
        self.error_details = "Mock error details"

    def raise_error_side_effect():
      raise errors.HttpError(MockHttpError(), "Mock failed HTTP call.".encode())

    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().allocateRobotJob().execute.side_effect = (
        raise_error_side_effect
    )

    robot_job_lib = robot_job.OrchestratorRobotJob(
        connection=mock_connection,
        robot_id="test_robot_id",
        job_type=robot_job.JobType.ALL,
    )
    response = robot_job_lib.request_robot_job()

    self.assertFalse(response.success)
    self.assertIn(robot_job._ERROR_GET_ROBOT_JOB, response.error_message)

  def test_request_robot_job_no_more_robot_job(self):

    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().allocateRobotJob().execute.return_value = {
        "robotJob": {}
    }

    robot_job_lib = robot_job.OrchestratorRobotJob(
        connection=mock_connection,
        robot_id="test_robot_id",
        job_type=robot_job.JobType.ALL,
    )
    response = robot_job_lib.request_robot_job()
    self.assertTrue(response.success)
    self.assertTrue(response.no_more_robot_job)
    self.assertEqual(response.error_message, robot_job._ERROR_EMPTY_RESPONSE)

  def test_request_robot_job_bad_response_robot_job_id(self):

    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().allocateRobotJob().execute.return_value = {
        "robotJob": {"robotJobId": ""}
    }

    robot_job_lib = robot_job.OrchestratorRobotJob(
        connection=mock_connection,
        robot_id="test_robot_id",
        job_type=robot_job.JobType.ALL,
    )
    response = robot_job_lib.request_robot_job()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, robot_job._ERROR_EMPTY_ROBOT_JOB_ID
    )


if __name__ == "__main__":
  absltest.main()
