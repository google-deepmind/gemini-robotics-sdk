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

"""Unit tests for work_unit.py."""

from absl.testing import absltest

from safari_sdk.orchestrator.client.dataclass import work_unit


class WorkUnitResponseTest(absltest.TestCase):

  def test_work_unit_outcome_num_value(self):
    outcome = work_unit.WorkUnitOutcome.WORK_UNIT_OUTCOME_SUCCESS
    self.assertEqual(outcome.num_value(), 1)
    outcome = work_unit.WorkUnitOutcome.WORK_UNIT_OUTCOME_FAILURE
    self.assertEqual(outcome.num_value(), 2)
    outcome = work_unit.WorkUnitOutcome.WORK_UNIT_OUTCOME_INVALID
    self.assertEqual(outcome.num_value(), 3)
    outcome = work_unit.WorkUnitOutcome.WORK_UNIT_OUTCOME_UNSPECIFIED
    self.assertEqual(outcome.num_value(), 0)

  def test_kv_msg_get_value_good(self):
    kv_msg = work_unit.KvMsg(
        key="test_key",
        type=work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_STRING,
        value=work_unit.KvMsgValue(
            stringValue="test_value",
            stringListValue=["test_value_1", "test_value_2"],
            intValue=1,
            intListValue=[2, 3],
            floatValue=4.5,
            floatListValue=[6.7, 8.9],
            boolValue=True,
            boolListValue=[False, True, False],
            jsonValue='{"json_key": "json_value"}',
        ),
    )
    self.assertEqual(kv_msg.get_value(), "test_value")

    kv_msg.type = work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_STRING_LIST
    self.assertSequenceEqual(
        kv_msg.get_value(), ["test_value_1", "test_value_2"]
    )

    kv_msg.type = work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_INT
    self.assertEqual(kv_msg.get_value(), 1)

    kv_msg.type = work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_INT_LIST
    self.assertSequenceEqual(kv_msg.get_value(), [2, 3])

    kv_msg.type = work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_FLOAT
    self.assertEqual(kv_msg.get_value(), 4.5)

    kv_msg.type = work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_FLOAT_LIST
    self.assertSequenceEqual(kv_msg.get_value(), [6.7, 8.9])

    kv_msg.type = work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_BOOL
    self.assertEqual(kv_msg.get_value(), True)

    kv_msg.type = work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_BOOL_LIST
    self.assertSequenceEqual(kv_msg.get_value(), [False, True, False])

    kv_msg.type = work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_JSON
    self.assertEqual(kv_msg.get_value(), '{"json_key": "json_value"}')

  def test_kv_msg_get_value_with_no_value(self):
    kv_msg = work_unit.KvMsg(
        key="test_key",
        type=work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_STRING,
        value=work_unit.KvMsgValue(),
    )
    self.assertIsNone(kv_msg.get_value())

    kv_msg.type = work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_STRING_LIST
    self.assertIsNone(kv_msg.get_value())

    kv_msg.type = work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_INT
    self.assertIsNone(kv_msg.get_value())

    kv_msg.type = work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_INT_LIST
    self.assertIsNone(kv_msg.get_value())

    kv_msg.type = work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_FLOAT
    self.assertIsNone(kv_msg.get_value())

    kv_msg.type = work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_FLOAT_LIST
    self.assertIsNone(kv_msg.get_value())

    kv_msg.type = work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_BOOL
    self.assertIsNone(kv_msg.get_value())

    kv_msg.type = work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_BOOL_LIST
    self.assertIsNone(kv_msg.get_value())

    kv_msg.type = work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_JSON
    self.assertIsNone(kv_msg.get_value())

  def test_scene_preset_details_get_all_parameters_good(self):
    scene_preset_details = work_unit.ScenePresetDetails(
        parameters=[
            work_unit.KvMsg(
                key="test_key_1",
                type=work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_STRING,
                value=work_unit.KvMsgValue(stringValue="test_value_1"),
            ),
            work_unit.KvMsg(
                key="test_key_2",
                type=work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_INT,
                value=work_unit.KvMsgValue(intValue=2),
            ),
        ]
    )
    params = scene_preset_details.get_all_parameters()
    self.assertLen(params, 2)
    self.assertSameElements(params.keys(), ["test_key_1", "test_key_2"])
    self.assertEqual(params["test_key_1"], "test_value_1")
    self.assertEqual(params["test_key_2"], 2)

  def test_scene_preset_details_get_all_parameters_with_no_parameters(self):
    scene_preset_details = work_unit.ScenePresetDetails()
    params = scene_preset_details.get_all_parameters()
    self.assertEmpty(params)

  def test_scene_preset_details_get_parameter_good(self):
    scene_preset_details = work_unit.ScenePresetDetails(
        parameters=[
            work_unit.KvMsg(
                key="test_key_1",
                type=work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_STRING,
                value=work_unit.KvMsgValue(stringValue="test_value_1"),
            ),
            work_unit.KvMsg(
                key="test_key_2",
                type=work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_INT,
                value=work_unit.KvMsgValue(intValue=2),
            ),
        ]
    )
    value = scene_preset_details.get_parameter_value(key="test_key_1")
    self.assertEqual(value, "test_value_1")

    value = scene_preset_details.get_parameter_value(key="test_key_2")
    self.assertEqual(value, 2)

    value = scene_preset_details.get_parameter_value(key="test_key_3")
    self.assertIsNone(value)

  def test_scene_preset_details_get_parameter_with_no_parameters(self):
    scene_preset_details = work_unit.ScenePresetDetails()
    value = scene_preset_details.get_parameter_value(key="test_key_1")
    self.assertIsNone(value)

    value = scene_preset_details.get_parameter_value(
        key="test_key_1", default_value="ERROR"
    )
    self.assertEqual(value, "ERROR")

  def test_policy_details_get_all_parameters_good(self):
    policy_details = work_unit.PolicyDetails(
        parameters=[
            work_unit.KvMsg(
                key="test_key_1",
                type=work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_STRING,
                value=work_unit.KvMsgValue(stringValue="test_value_1"),
            ),
            work_unit.KvMsg(
                key="test_key_2",
                type=work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_INT,
                value=work_unit.KvMsgValue(intValue=2),
            ),
        ]
    )
    params = policy_details.get_all_parameters()
    self.assertLen(params, 2)
    self.assertSameElements(params.keys(), ["test_key_1", "test_key_2"])
    self.assertEqual(params["test_key_1"], "test_value_1")
    self.assertEqual(params["test_key_2"], 2)

  def test_policy_details_get_all_parameters_with_no_parameters(self):
    policy_details = work_unit.PolicyDetails()
    params = policy_details.get_all_parameters()
    self.assertEmpty(params)

  def test_policy_details_get_parameter_good(self):
    policy_details = work_unit.PolicyDetails(
        parameters=[
            work_unit.KvMsg(
                key="test_key_1",
                type=work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_STRING,
                value=work_unit.KvMsgValue(stringValue="test_value_1"),
            ),
            work_unit.KvMsg(
                key="test_key_2",
                type=work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_INT,
                value=work_unit.KvMsgValue(intValue=2),
            ),
        ]
    )
    value = policy_details.get_parameter_value(key="test_key_1")
    self.assertEqual(value, "test_value_1")

    value = policy_details.get_parameter_value(key="test_key_2")
    self.assertEqual(value, 2)

    value = policy_details.get_parameter_value(key="test_key_3")
    self.assertIsNone(value)

  def test_policy_details_get_parameter_with_no_parameters(self):
    policy_details = work_unit.PolicyDetails()
    value = policy_details.get_parameter_value(key="test_key_1")
    self.assertIsNone(value)

    value = policy_details.get_parameter_value(
        key="test_key_1", default_value="ERROR"
    )
    self.assertEqual(value, "ERROR")

  def test_response_post_init_from_json_response(self):
    response = work_unit.WorkUnit(
        projectId="test_project_id",
        robotJobId="test_robot_job_id",
        workUnitId="test_work_unit_id",
        context=work_unit.WorkUnitContext(),
        stage="WORK_UNIT_STAGE_QUEUED_TO_ROBOT",
        outcome="WORK_UNIT_OUTCOME_UNSPECIFIED",
        note="test_note",
    )
    self.assertEqual(response.projectId, "test_project_id")
    self.assertEqual(response.robotJobId, "test_robot_job_id")
    self.assertEqual(response.workUnitId, "test_work_unit_id")
    self.assertIsInstance(response.context, work_unit.WorkUnitContext)
    self.assertIsInstance(response.stage, work_unit.WorkUnitStage)
    self.assertEqual(
        response.stage, work_unit.WorkUnitStage.WORK_UNIT_STAGE_QUEUED_TO_ROBOT
    )
    self.assertIsInstance(response.outcome, work_unit.WorkUnitOutcome)
    self.assertEqual(
        response.outcome,
        work_unit.WorkUnitOutcome.WORK_UNIT_OUTCOME_UNSPECIFIED,
    )
    self.assertEqual(response.note, "test_note")

  def test_response_post_init_as_enum(self):
    response = work_unit.WorkUnit(
        projectId="test_project_id",
        robotJobId="test_robot_job_id",
        workUnitId="test_work_unit_id",
        stage=work_unit.WorkUnitStage.WORK_UNIT_STAGE_QUEUED_TO_ROBOT,
        outcome=work_unit.WorkUnitOutcome.WORK_UNIT_OUTCOME_SUCCESS,
    )
    self.assertEqual(response.projectId, "test_project_id")
    self.assertEqual(response.robotJobId, "test_robot_job_id")
    self.assertEqual(response.workUnitId, "test_work_unit_id")
    self.assertIsNone(response.context)
    self.assertIsInstance(response.stage, work_unit.WorkUnitStage)
    self.assertEqual(
        response.stage, work_unit.WorkUnitStage.WORK_UNIT_STAGE_QUEUED_TO_ROBOT
    )
    self.assertIsInstance(response.outcome, work_unit.WorkUnitOutcome)
    self.assertEqual(
        response.outcome, work_unit.WorkUnitOutcome.WORK_UNIT_OUTCOME_SUCCESS,
    )
    self.assertIsNone(response.note)

  def test_response_post_init_as_none(self):
    response = work_unit.WorkUnit()

    self.assertIsNone(response.projectId)
    self.assertIsNone(response.robotJobId)
    self.assertIsNone(response.workUnitId)
    self.assertIsNone(response.context)
    self.assertIsInstance(response.stage, work_unit.WorkUnitStage)
    self.assertEqual(
        response.stage, work_unit.WorkUnitStage.WORK_UNIT_STAGE_UNSPECIFIED
    )
    self.assertIsInstance(response.outcome, work_unit.WorkUnitOutcome)
    self.assertEqual(
        response.outcome,
        work_unit.WorkUnitOutcome.WORK_UNIT_OUTCOME_UNSPECIFIED
    )
    self.assertIsNone(response.note)


if __name__ == "__main__":
  absltest.main()
