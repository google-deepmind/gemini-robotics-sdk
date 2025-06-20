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

"""Unit tests for current_robot_info.py."""

from absl.testing import absltest

from safari_sdk.orchestrator.client.dataclass import current_robot_info


class CurrentRobotInfoTest(absltest.TestCase):

  def test_response_post_init_from_json_response(self):
    response = current_robot_info.CurrentRobotInfoResponse(
        robotId="test_robot_id",
        isOperational=True,
        operatorId="test_operator_id",
        robotJobId="test_robot_job_id",
        workUnitId="test_work_unit_id",
        stage="WORK_UNIT_STAGE_QUEUED_TO_ROBOT",
    )
    self.assertEqual(response.robotId, "test_robot_id")
    self.assertTrue(response.isOperational)
    self.assertEqual(response.operatorId, "test_operator_id")
    self.assertEqual(response.robotJobId, "test_robot_job_id")
    self.assertEqual(response.workUnitId, "test_work_unit_id")
    self.assertIsInstance(
        response.stage, current_robot_info.work_unit.WorkUnitStage
    )
    self.assertEqual(
        response.stage,
        current_robot_info.work_unit.WorkUnitStage.WORK_UNIT_STAGE_QUEUED_TO_ROBOT,
    )

  def test_response_post_init_as_enum(self):
    response = current_robot_info.CurrentRobotInfoResponse(
        robotId="test_robot_id",
        stage=current_robot_info.work_unit.WorkUnitStage.WORK_UNIT_STAGE_QUEUED_TO_ROBOT,
    )
    self.assertEqual(response.robotId, "test_robot_id")
    self.assertFalse(response.isOperational)
    self.assertIsNone(response.operatorId)
    self.assertIsNone(response.robotJobId)
    self.assertIsNone(response.workUnitId)
    self.assertIsInstance(
        response.stage, current_robot_info.work_unit.WorkUnitStage
    )
    self.assertEqual(
        response.stage,
        current_robot_info.work_unit.WorkUnitStage.WORK_UNIT_STAGE_QUEUED_TO_ROBOT,
    )

  def test_response_post_init_as_none(self):
    response = current_robot_info.CurrentRobotInfoResponse(
        robotId="test_robot_id",
    )
    self.assertEqual(response.robotId, "test_robot_id")
    self.assertFalse(response.isOperational)
    self.assertIsNone(response.operatorId)
    self.assertIsNone(response.robotJobId)
    self.assertIsNone(response.workUnitId)
    self.assertIsInstance(
        response.stage, current_robot_info.work_unit.WorkUnitStage
    )
    self.assertEqual(
        response.stage,
        current_robot_info.work_unit.WorkUnitStage.WORK_UNIT_STAGE_UNSPECIFIED,
    )


if __name__ == "__main__":
  absltest.main()
