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

from dm_env import specs
import numpy as np

from google.protobuf import struct_pb2
from absl.testing import absltest
from safari_sdk.logging.python import constants
from safari_sdk.logging.python import session_manager
from safari_sdk.logging.python import session_metadata as session_metadata_lib
from safari_sdk.protos import label_pb2
from safari_sdk.protos.logging import codec_pb2
from safari_sdk.protos.logging import dtype_pb2
from safari_sdk.protos.logging import orchestrator_info_pb2
from safari_sdk.protos.logging import policy_type_pb2

_TEST_TASK_ID = "test_task"


class TestPolicyTypeProvider:

  def __init__(self, policy_type: policy_type_pb2.PolicyType):
    self._policy_type = policy_type

  def set_policy_type(self, policy_type: policy_type_pb2.PolicyType):
    self._policy_type = policy_type

  def __call__(self) -> policy_type_pb2.PolicyType:
    return self._policy_type


class SessionManagerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._policy_environment_metadata_params = session_metadata_lib.PolicyEnvironmentMetadataParams(
        jpeg_compression_keys=[],
        observation_spec={
            "observation1": specs.Array(shape=(1, 2, 3), dtype=np.float32)
        },
        reward_spec=specs.Array(shape=(1,), dtype=np.float32),
        discount_spec=specs.Array(shape=(1,), dtype=np.float32),
        action_spec=specs.BoundedArray(
            shape=(1, 2, 3),
            dtype=np.float32,
            minimum=np.array([1.0, 2.0, 3.0]),
            maximum=np.array([4.0, 5.0, 6.0]),
        ),
        policy_extra_spec={},
        policy_type=policy_type_pb2.PolicyType.POLICY_TYPE_ROBOT_TELEOPERATION,
        control_timestep=0.04,
        embodiment_version="test_version",
    )
    self._session_metadata_config = session_metadata_lib.SessionMetadataConfig(
        policy_type=policy_type_pb2.PolicyType.POLICY_TYPE_ROBOT_TELEOPERATION,
        control_timestep_seconds=0.04,
        embodiment_version="test_version",
    )

  def test_init_with_valid_inputs(self):
    manager = session_manager.SessionManager(
        topics={"topic1", "topic2", "topic3"},
        required_topics={"topic1", "topic2"},
        policy_environment_metadata_params=self._policy_environment_metadata_params,
    )
    self.assertEqual(manager._topics, {"topic1", "topic2", "topic3"})
    self.assertEqual(manager._required_topics, {"topic1", "topic2"})
    self.assertFalse(manager.session_started)

  def test_init_with_invalid_required_topics(self):
    with self.assertRaisesRegex(
        ValueError, "required_topics must be a subset of topics"
    ):
      session_manager.SessionManager(
          topics={"topic1", "topic2"},
          required_topics={"topic1", "topic3"},
          policy_environment_metadata_params=self._policy_environment_metadata_params,
      )

  def test_start_session(self):
    manager = session_manager.SessionManager(
        topics={"topic1", "topic2", "topic3"},
        required_topics={"topic1", "topic2"},
        policy_environment_metadata_params=self._policy_environment_metadata_params,
    )
    start_nsec = 1234567890
    task_id = _TEST_TASK_ID
    manager.start_session(start_timestamp_nsec=start_nsec, task_id=task_id)
    self.assertTrue(manager.session_started)
    self.assertEqual(manager._session.interval.start_nsec, start_nsec)
    self.assertEqual(manager._session.task_id, _TEST_TASK_ID)
    self.assertLen(manager._session.streams, 3)
    for stream in manager._session.streams:
      self.assertEqual(stream.key_range.interval.start_nsec, start_nsec)
      self.assertEqual(
          stream.is_required, stream.key_range.topic in ["topic1", "topic2"]
      )

  def test_start_session_twice_raises_error(self):
    manager = session_manager.SessionManager(
        topics={"topic1", "topic2"},
        required_topics={"topic1"},
        policy_environment_metadata_params=self._policy_environment_metadata_params,
    )
    manager.start_session(start_timestamp_nsec=123, task_id=_TEST_TASK_ID)
    with self.assertRaisesRegex(ValueError, "Session has already been started"):
      manager.start_session(start_timestamp_nsec=456, task_id="test_task2")

  def test_add_session_label(self):
    manager = session_manager.SessionManager(
        topics={"topic1", "topic2"},
        required_topics={"topic1"},
        policy_environment_metadata_params=self._policy_environment_metadata_params,
    )
    manager.start_session(start_timestamp_nsec=123, task_id=_TEST_TASK_ID)
    label1 = label_pb2.LabelMessage(
        key="key1", label_value=struct_pb2.Value(string_value="value1")
    )
    label2 = label_pb2.LabelMessage(
        key="key2", label_value=struct_pb2.Value(string_value="value2")
    )
    manager.add_session_label(label1)
    manager.add_session_label(label2)
    self.assertLen(manager._session.labels, 2)
    self.assertEqual(manager._session.labels[0], label1)
    self.assertEqual(manager._session.labels[1], label2)

  def test_add_session_label_before_start_raises_error(self):
    manager = session_manager.SessionManager(
        topics={"topic1", "topic2"},
        required_topics={"topic1"},
        policy_environment_metadata_params=self._policy_environment_metadata_params,
    )
    label = label_pb2.LabelMessage(
        key="key1", label_value=struct_pb2.Value(string_value="value1")
    )
    with self.assertRaisesRegex(
        ValueError,
        "add_session_label is called before session has been started",
    ):
      manager.add_session_label(label)

  def test_orchestrator_info_provider(self):
    orchestrator_info = orchestrator_info_pb2.OrchestratorInfo(
        task_id="test_orca_task",
        robot_job_id="test_robot_job",
        work_unit_id="test_work_unit",
        scene_preset_id="test_scene_preset",
        task_instruction="pick up the block",
    )
    config = session_metadata_lib.SessionMetadataConfig(
        orchestrator_info_provider=lambda: orchestrator_info,
    )
    manager = session_manager.SessionManager(
        topics={"topic1"},
        required_topics={"topic1"},
        policy_environment_metadata_params=self._policy_environment_metadata_params,
        session_metadata_config=config,
    )
    manager.start_session(start_timestamp_nsec=123, task_id=_TEST_TASK_ID)
    session = manager.stop_session(stop_timestamp_nsec=456)

    self.assertTrue(session.HasField("orchestrator_info"))
    self.assertEqual(session.orchestrator_info.task_id, "test_orca_task")
    self.assertEqual(session.orchestrator_info.robot_job_id, "test_robot_job")
    self.assertEqual(session.orchestrator_info.work_unit_id, "test_work_unit")
    self.assertEqual(
        session.orchestrator_info.scene_preset_id, "test_scene_preset"
    )
    self.assertEqual(
        session.orchestrator_info.task_instruction, "pick up the block"
    )

  def test_orchestrator_info_not_set_by_default(self):
    manager = session_manager.SessionManager(
        topics={"topic1"},
        required_topics={"topic1"},
        policy_environment_metadata_params=self._policy_environment_metadata_params,
    )
    manager.start_session(start_timestamp_nsec=123, task_id=_TEST_TASK_ID)
    session = manager.stop_session(stop_timestamp_nsec=456)

    self.assertFalse(session.HasField("orchestrator_info"))

  def test_policy_config_callable_updates_before_stop_session(self):
    policy_type_provider = TestPolicyTypeProvider(
        policy_type_pb2.PolicyType.POLICY_TYPE_ROBOT_TELEOPERATION
    )
    config = session_metadata_lib.SessionMetadataConfig(
        policy_type=policy_type_provider,
    )
    manager = session_manager.SessionManager(
        topics={"topic1"},
        required_topics={"topic1"},
        policy_environment_metadata_params=self._policy_environment_metadata_params,
        session_metadata_config=config,
    )
    manager.start_session(start_timestamp_nsec=123, task_id=_TEST_TASK_ID)
    policy_type_provider.set_policy_type(
        policy_type_pb2.PolicyType.POLICY_TYPE_ROBOT_EVALUATION
    )
    session = manager.stop_session(stop_timestamp_nsec=456)
    self.assertEqual(
        session.policy_environment_metadata.policy_type,
        policy_type_pb2.PolicyType.POLICY_TYPE_ROBOT_EVALUATION,
    )

  def test_stop_session(self):
    config = session_metadata_lib.SessionMetadataConfig(
        policy_type=policy_type_pb2.PolicyType.POLICY_TYPE_ROBOT_TELEOPERATION,
        control_timestep_seconds=0.04,
        embodiment_version="test_version",
        fixed_tags=["tag1", "tag2"],
    )
    manager = session_manager.SessionManager(
        topics={"topic1", "topic2"},
        required_topics={"topic1"},
        policy_environment_metadata_params=self._policy_environment_metadata_params,
        session_metadata_config=config,
    )
    start_nsec = 123
    stop_nsec = 456
    manager.start_session(
        start_timestamp_nsec=start_nsec, task_id=_TEST_TASK_ID
    )
    session = manager.stop_session(stop_timestamp_nsec=stop_nsec)

    self.assertFalse(manager.session_started)
    self.assertEqual(session.interval.stop_nsec, stop_nsec)
    for stream in session.streams:
      self.assertEqual(stream.key_range.interval.stop_nsec, stop_nsec)

    feature_specs = session.policy_environment_metadata.feature_specs
    self.assertSequenceEqual(
        feature_specs.observation[
            f"{constants.OBSERVATION_KEY_PREFIX}/observation1"
        ].shape,
        [1, 2, 3],
    )
    self.assertEqual(
        feature_specs.observation[
            f"{constants.OBSERVATION_KEY_PREFIX}/observation1"
        ].dtype,
        dtype_pb2.DTYPE_FLOAT32,
    )
    self.assertEqual(
        feature_specs.observation[
            f"{constants.OBSERVATION_KEY_PREFIX}/observation1"
        ].codec,
        codec_pb2.CODEC_NONE,
    )
    self.assertSequenceEqual(
        feature_specs.reward[constants.REWARD_KEY].shape, [1]
    )
    self.assertEqual(
        feature_specs.reward[constants.REWARD_KEY].dtype,
        dtype_pb2.DTYPE_FLOAT32,
    )
    self.assertEqual(
        feature_specs.reward[constants.REWARD_KEY].codec, codec_pb2.CODEC_NONE
    )

    self.assertSequenceEqual(
        feature_specs.discount[constants.DISCOUNT_KEY].shape, [1]
    )
    self.assertEqual(
        feature_specs.discount[constants.DISCOUNT_KEY].dtype,
        dtype_pb2.DTYPE_FLOAT32,
    )
    self.assertEqual(
        feature_specs.discount[constants.DISCOUNT_KEY].codec,
        codec_pb2.CODEC_NONE,
    )

    self.assertSequenceEqual(
        feature_specs.action[constants.ACTION_KEY_PREFIX].shape, [1, 2, 3]
    )
    self.assertEqual(
        feature_specs.action[constants.ACTION_KEY_PREFIX].dtype,
        dtype_pb2.DTYPE_FLOAT32,
    )
    self.assertEqual(
        feature_specs.action[constants.ACTION_KEY_PREFIX].codec,
        codec_pb2.CODEC_NONE,
    )

    self.assertSequenceEqual(session.tags, ["tag1", "tag2"])

    self.assertEqual(
        session.policy_environment_metadata.policy_type,
        policy_type_pb2.PolicyType.POLICY_TYPE_ROBOT_TELEOPERATION,
    )
    self.assertEqual(session.policy_environment_metadata.control_timestep, 0.04)
    self.assertEqual(
        session.policy_environment_metadata.embodiment_version, "test_version"
    )

  def test_stop_session_with_fixed_and_dynamic_tags(self):
    config = session_metadata_lib.SessionMetadataConfig(
        fixed_tags=["tag1", "tag2"],
        dynamic_episode_taggers=[
            lambda: ["tag3", "tag4"],
            lambda: ["tag5", "tag6"],
        ],
    )
    manager = session_manager.SessionManager(
        topics={"topic1", "topic2"},
        required_topics={"topic1"},
        policy_environment_metadata_params=self._policy_environment_metadata_params,
        session_metadata_config=config,
    )
    manager.start_session(start_timestamp_nsec=123, task_id=_TEST_TASK_ID)
    session = manager.stop_session(stop_timestamp_nsec=456)
    self.assertSequenceEqual(
        session.tags, ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6"]
    )

  def test_stop_session_before_start_raises_error(self):
    manager = session_manager.SessionManager(
        topics={"topic1", "topic2"},
        required_topics={"topic1"},
        policy_environment_metadata_params=self._policy_environment_metadata_params,
    )
    with self.assertRaisesRegex(ValueError, "Session is not started"):
      manager.stop_session(stop_timestamp_nsec=456)


if __name__ == "__main__":
  absltest.main()
