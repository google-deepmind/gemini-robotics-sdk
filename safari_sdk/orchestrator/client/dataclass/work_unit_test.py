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

  def test_pixel_location_post_init(self):
    pixel_location = work_unit.PixelLocation(x=1, y=2)
    self.assertEqual(pixel_location.x, 1)
    self.assertEqual(pixel_location.y, 2)

    pixel_location = work_unit.PixelLocation(x=1)
    self.assertEqual(pixel_location.x, 1)
    self.assertEqual(pixel_location.y, 0)

    pixel_location = work_unit.PixelLocation(y=2)
    self.assertEqual(pixel_location.x, 0)
    self.assertEqual(pixel_location.y, 2)

    pixel_location = work_unit.PixelLocation()
    self.assertIsNone(pixel_location.x)
    self.assertIsNone(pixel_location.y)

  def test_pixel_vector_post_init(self):
    pixel_vector = work_unit.PixelVector(
        coordinate=work_unit.PixelLocation(x=1, y=2),
        direction=work_unit.PixelDirection(rad=1.57)
    )
    self.assertEqual(pixel_vector.coordinate.x, 1)
    self.assertEqual(pixel_vector.coordinate.y, 2)
    self.assertEqual(pixel_vector.direction.rad, 1.57)

    pixel_vector = work_unit.PixelVector(
        coordinate=work_unit.PixelLocation(x=1, y=2)
    )
    self.assertEqual(pixel_vector.coordinate.x, 1)
    self.assertEqual(pixel_vector.coordinate.y, 2)
    self.assertEqual(pixel_vector.direction.rad, 0.0)

    pixel_vector = work_unit.PixelVector(
        direction=work_unit.PixelDirection(rad=1.57)
    )
    self.assertEqual(pixel_vector.coordinate.x, 0)
    self.assertEqual(pixel_vector.coordinate.y, 0)
    self.assertEqual(pixel_vector.direction.rad, 1.57)

    pixel_vector = work_unit.PixelVector()
    self.assertIsNone(pixel_vector.coordinate)
    self.assertIsNone(pixel_vector.direction)

  def test_shape_box_post_init(self):
    shape_box = work_unit.ShapeBox(x=1, y=2, w=3, h=4)
    self.assertEqual(shape_box.x, 1)
    self.assertEqual(shape_box.y, 2)
    self.assertEqual(shape_box.w, 3)
    self.assertEqual(shape_box.h, 4)

    shape_box = work_unit.ShapeBox(x=1, w=3, h=4)
    self.assertEqual(shape_box.x, 1)
    self.assertEqual(shape_box.y, 0)
    self.assertEqual(shape_box.w, 3)
    self.assertEqual(shape_box.h, 4)

    shape_box = work_unit.ShapeBox(x=1, y=2, h=4)
    self.assertEqual(shape_box.x, 1)
    self.assertEqual(shape_box.y, 2)
    self.assertEqual(shape_box.w, 0)
    self.assertEqual(shape_box.h, 4)

    shape_box = work_unit.ShapeBox(x=1, y=2, w=3)
    self.assertEqual(shape_box.x, 1)
    self.assertEqual(shape_box.y, 2)
    self.assertEqual(shape_box.w, 3)
    self.assertEqual(shape_box.h, 0)

    shape_box = work_unit.ShapeBox(w=3)
    self.assertEqual(shape_box.x, 0)
    self.assertEqual(shape_box.y, 0)
    self.assertEqual(shape_box.w, 3)
    self.assertEqual(shape_box.h, 0)

    shape_box = work_unit.ShapeBox()
    self.assertIsNone(shape_box.x)
    self.assertIsNone(shape_box.y)
    self.assertIsNone(shape_box.w)
    self.assertIsNone(shape_box.h)

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
    self.assertEmpty(kv_msg.get_value())

    kv_msg.type = work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_STRING_LIST
    self.assertEmpty(kv_msg.get_value())

    kv_msg.type = work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_INT
    self.assertEqual(kv_msg.get_value(), 0)

    kv_msg.type = work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_INT_LIST
    self.assertEmpty(kv_msg.get_value())

    kv_msg.type = work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_FLOAT
    self.assertEqual(kv_msg.get_value(), 0.0)

    kv_msg.type = work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_FLOAT_LIST
    self.assertEmpty(kv_msg.get_value())

    kv_msg.type = work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_BOOL
    self.assertFalse(kv_msg.get_value())

    kv_msg.type = work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_BOOL_LIST
    self.assertEmpty(kv_msg.get_value())

    kv_msg.type = work_unit.KvMsgValueType.KV_MSG_VALUE_TYPE_JSON
    self.assertEmpty(kv_msg.get_value())

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
        name="test_policy_name",
        description="test_policy_description",
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
        ],
        artifactIds=["test_artifact_id_1", "test_artifact_id_2"]
    )
    params = policy_details.get_all_parameters()
    self.assertEqual(policy_details.name, "test_policy_name")
    self.assertEqual(policy_details.description, "test_policy_description")
    self.assertLen(params, 2)
    self.assertSameElements(params.keys(), ["test_key_1", "test_key_2"])
    self.assertEqual(params["test_key_1"], "test_value_1")
    self.assertEqual(params["test_key_2"], 2)
    self.assertSequenceEqual(
        policy_details.artifactIds, ["test_artifact_id_1", "test_artifact_id_2"]
    )

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

  def test_policy_details_get_artifact_ids_good(self):
    policy_details = work_unit.PolicyDetails(
        artifactIds=["test_artifact_id_1", "test_artifact_id_2"]
    )
    self.assertSequenceEqual(
        policy_details.artifactIds,
        ["test_artifact_id_1", "test_artifact_id_2"],
    )

  def test_policy_details_get_artifact_ids_with_no_artifact_ids(self):
    policy_details = work_unit.PolicyDetails()
    self.assertIsNone(policy_details.artifactIds)

  def test_success_score_post_init_from_json_response(self):
    success_score = work_unit.SuccessScore(
        score=0.5,
        definition="test_definition",
    )
    self.assertEqual(success_score.score, 0.5)
    self.assertEqual(success_score.definition, "test_definition")

    success_score = work_unit.SuccessScore()
    self.assertEqual(success_score.score, 0.0)
    self.assertEqual(success_score.definition, "")

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
