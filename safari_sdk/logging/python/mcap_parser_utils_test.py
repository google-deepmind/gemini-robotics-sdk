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
from unittest import mock

import cv2
from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized
from safari_sdk.logging.python import mcap_parser_utils
from safari_sdk.protos.logging import metadata_pb2
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2


class McapParserUtilsTest(parameterized.TestCase):

  def test_get_mcap_file_paths_single_file(self):
    single_path = "/path/to/file_shard0.mcap"
    result = mcap_parser_utils.get_mcap_file_paths(single_path)
    self.assertEqual(result, [single_path])

  def test_get_mcap_file_paths_directory_and_sorting(self):
    temp_dir = self.create_tempdir()
    # Create directory structure expected by glob: */*/*/*.mcap
    sub_dir = temp_dir.mkdir("a").mkdir("b").mkdir("c")
    f3 = sub_dir.create_file("uuidA_shard10.mcap").full_path
    f1 = sub_dir.create_file("uuidA_shard0.mcap").full_path
    f2 = sub_dir.create_file("uuidA_shard1.mcap").full_path

    paths = mcap_parser_utils.get_mcap_file_paths(temp_dir.full_path)
    self.assertEqual(paths, [f1, f2, f3])

  def test_get_mcap_file_paths_empty_dir_raises_value_error(self):
    temp_dir = self.create_tempdir()
    with self.assertRaisesRegex(ValueError, "No mcap files found"):
      mcap_parser_utils.get_mcap_file_paths(temp_dir.full_path)

  def test_non_sharded_filename_with_shard_substring(self):
    tmp_dir = self.create_tempdir()
    nested_dir = tmp_dir.mkdir("a").mkdir("b").mkdir("c")
    f = nested_dir.create_file("episode_ashard_test.mcap")
    paths = mcap_parser_utils.get_mcap_file_paths(tmp_dir.full_path)
    self.assertEqual(paths, [f.full_path])

  def test_maybe_decode_image_rgb_jpeg(self):
    img = np.ones((20, 30, 3), dtype=np.uint8) * 100
    _, encoded = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    val = [encoded.tobytes()]

    decoded = mcap_parser_utils._maybe_decode_image(val, key="cam0")
    self.assertIsNotNone(decoded)
    self.assertEqual(decoded.shape, (20, 30, 3))
    self.assertEqual(decoded.dtype, np.uint8)

  def test_maybe_decode_image_rgb_png(self):
    img = np.ones((20, 30, 3), dtype=np.uint8) * 100
    _, encoded = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    val = [encoded.tobytes()]

    decoded = mcap_parser_utils._maybe_decode_image(val, key="cam0")
    self.assertIsNotNone(decoded)
    self.assertEqual(decoded.shape, (20, 30, 3))

  def test_maybe_decode_image_grayscale(self):
    img = np.ones((20, 30), dtype=np.uint8) * 150
    _, encoded = cv2.imencode(".jpg", img)
    val = [encoded.tobytes()]

    decoded = mcap_parser_utils._maybe_decode_image(val, key="gray")
    self.assertIsNotNone(decoded)
    self.assertEqual(decoded.shape, (20, 30, 1))

  def test_maybe_decode_image_alpha_rejection(self):
    img_rgba = np.ones((20, 30, 4), dtype=np.uint8) * 200
    _, encoded = cv2.imencode(".png", img_rgba)
    val = [encoded.tobytes()]

    with self.assertRaisesRegex(
        ValueError, "Alpha-channel images are not supported"
    ):
      mcap_parser_utils._maybe_decode_image(val, key="rgba_cam")

  def test_maybe_decode_image_corrupt_bytes(self):
    corrupt_bytes = [b"\xff\xd8\xff\xe0corruptdatahere"]
    with self.assertRaisesRegex(ValueError, "Failed to decode image bytes"):
      mcap_parser_utils._maybe_decode_image(corrupt_bytes, key="bad_cam")

  def test_maybe_decode_image_non_image_bytes(self):
    non_img = [b"12345"]
    self.assertIsNone(mcap_parser_utils._maybe_decode_image(non_img))
    self.assertIsNone(mcap_parser_utils._maybe_decode_image([b"a", b"b"]))

  def test_parse_and_match_spec_single_value_string(self):
    spec = specs.StringArray(shape=(), name="text")
    values = {"text": [b"hello world"]}
    res = mcap_parser_utils._parse_and_match_spec(spec, "text", values)
    self.assertEqual(res, "hello world")

  def test_parse_and_match_spec_dict_branch(self):
    spec_dict = {
        "cam0": specs.Array(shape=(20, 30, 3), dtype=np.uint8),
        "joint_pos": specs.Array(shape=(3,), dtype=np.float32),
    }
    img = np.ones((20, 30, 3), dtype=np.uint8) * 50
    _, encoded = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    values = {
        "obs/cam0": [encoded.tobytes()],
        "obs/joint_pos": [1.0, 2.0, 3.0],
    }

    res = mcap_parser_utils._parse_and_match_spec(spec_dict, "obs", values)
    self.assertIsInstance(res, dict)
    self.assertEqual(res["cam0"].shape, (20, 30, 3))
    np.testing.assert_array_almost_equal(res["joint_pos"], [1.0, 2.0, 3.0])

  def test_parse_and_match_spec_image_shape_mismatch_raises(self):
    spec = specs.Array(shape=(50, 50, 3), dtype=np.uint8)
    img = np.ones((20, 30, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", img)
    values = {"cam": [encoded.tobytes()]}

    with self.assertRaisesRegex(ValueError, "does not match spec shape"):
      mcap_parser_utils._parse_and_match_spec(spec, "cam", values)

  def test_python_value_from_example_feature(self):
    f_float = feature_pb2.Feature(
        float_list=feature_pb2.FloatList(value=[1.5, 2.5])
    )
    self.assertEqual(
        mcap_parser_utils._python_value_from_example_feature(f_float),
        [1.5, 2.5],
    )

    f_int = feature_pb2.Feature(
        int64_list=feature_pb2.Int64List(value=[10, 20])
    )
    self.assertEqual(
        mcap_parser_utils._python_value_from_example_feature(f_int), [10, 20]
    )

    f_bytes = feature_pb2.Feature(
        bytes_list=feature_pb2.BytesList(value=[b"foo"])
    )
    self.assertEqual(
        mcap_parser_utils._python_value_from_example_feature(f_bytes), [b"foo"]
    )

    f_empty = feature_pb2.Feature()
    with self.assertRaisesRegex(ValueError, "Unsupported feature type"):
      mcap_parser_utils._python_value_from_example_feature(f_empty)

  def test_parse_action_feature_count_mismatch_raises(self):
    ex = example_pb2.Example()
    ex.features.feature["act/a"].float_list.value.append(1.0)
    ex.features.feature["unexpected_key"].int64_list.value.append(5)

    spec = specs.BoundedArray(
        shape=(1,), dtype=np.float32, minimum=0.0, maximum=1.0
    )
    with self.assertRaisesRegex(ValueError, "more features than expected"):
      mcap_parser_utils._parse_action_from_example(ex, spec, "act")

  def test_parse_policy_extra_feature_count_mismatch_raises(self):
    ex = example_pb2.Example()
    ex.features.feature["extra/e"].float_list.value.append(1.0)
    ex.features.feature["unexpected_key"].int64_list.value.append(5)

    spec = specs.BoundedArray(
        shape=(1,), dtype=np.float32, minimum=0.0, maximum=1.0
    )
    with self.assertRaisesRegex(ValueError, "more features than expected"):
      mcap_parser_utils._parse_policy_extra_from_example(ex, spec, "extra")

  def test_get_mcap_file_paths_invalid_shard_number_fallback(self):
    temp_dir = self.create_tempdir()
    sub_dir = temp_dir.mkdir("a").mkdir("b").mkdir("c")
    f1 = sub_dir.create_file("uuidA_shardABC.mcap").full_path
    paths = mcap_parser_utils.get_mcap_file_paths(temp_dir.full_path)
    self.assertEqual(paths, [f1])

  @mock.patch.object(cv2, "imdecode")
  def test_maybe_decode_image_invalid_dtype_raises(self, mock_imdecode):
    mock_imdecode.return_value = np.ones((10, 10, 3), dtype=np.float32)
    val = [b"\xff\xd8\xff\xe0dummy"]
    with self.assertRaisesRegex(ValueError, "expected uint8"):
      mcap_parser_utils._maybe_decode_image(val, key="float_img")

  @mock.patch.object(cv2, "imdecode")
  def test_maybe_decode_image_unsupported_shape_raises(self, mock_imdecode):
    mock_imdecode.return_value = np.ones((10, 10, 5), dtype=np.uint8)
    val = [b"\xff\xd8\xff\xe0dummy"]
    with self.assertRaisesRegex(ValueError, "Unexpected decoded image shape"):
      mcap_parser_utils._maybe_decode_image(val, key="5ch_img")

  def test_parse_and_match_spec_string_not_bytes_raises(self):
    spec = specs.StringArray(shape=(), name="text")
    values = {"text": [123]}
    with self.assertRaisesRegex(ValueError, "Expected bytes but got"):
      mcap_parser_utils._parse_and_match_spec(spec, "text", values)

  def test_parse_and_match_spec_dict_string_not_bytes_raises(self):
    spec_dict = {"text": specs.StringArray(shape=(), name="text")}
    values = {"prefix/text": [123]}
    with self.assertRaisesRegex(ValueError, "Expected bytes but got"):
      mcap_parser_utils._parse_and_match_spec(spec_dict, "prefix", values)

  def test_parse_and_match_spec_scalar_squeeze(self):
    spec = specs.Array(shape=(), dtype=np.float32)
    values = {"val": [5.5]}
    res = mcap_parser_utils._parse_and_match_spec(spec, "val", values)
    self.assertIsInstance(res, np.ndarray)
    self.assertEqual(res.shape, ())
    self.assertEqual(res, 5.5)

    spec_dict = {"val": specs.Array(shape=(), dtype=np.float32)}
    values_dict = {"prefix/val": [5.5]}
    res_dict = mcap_parser_utils._parse_and_match_spec(
        spec_dict, "prefix", values_dict
    )
    self.assertIsInstance(res_dict, dict)
    self.assertEqual(res_dict["val"].shape, ())
    self.assertEqual(res_dict["val"], 5.5)

  def test_parse_and_match_spec_dict_string_bytes_type(self):
    spec_dict = {
        "text": specs.StringArray(shape=(), name="text", string_type=bytes)
    }
    values_dict = {"prefix/text": [b"hello"]}
    res = mcap_parser_utils._parse_and_match_spec(
        spec_dict, "prefix", values_dict
    )
    self.assertIsInstance(res, dict)
    self.assertTrue(np.issubdtype(res["text"].dtype, np.bytes_))

  @mock.patch.object(mcap_parser_utils, "get_mcap_file_paths")
  @mock.patch.object(mcap_parser_utils, "read_and_parse_mcap_messages")
  def test_read_session_proto_data(self, mock_parse, mock_paths):
    mock_paths.return_value = ["/dummy.mcap"]
    mock_parse.return_value = []
    with self.assertRaisesRegex(ValueError, "No session messages found"):
      mcap_parser_utils.read_session_proto_data("/dummy.mcap", "session_topic")

    session_msg = metadata_pb2.Session(task_id="task_123")
    mock_parse.return_value = [session_msg]
    result = mcap_parser_utils.read_session_proto_data(
        "/dummy.mcap", "session_topic"
    )
    self.assertEqual(result, [session_msg])

  @mock.patch.object(mcap_parser_utils, "get_mcap_file_paths")
  @mock.patch.object(mcap_parser_utils, "read_and_parse_mcap_messages")
  def test_read_file_metadata_proto_data(self, mock_parse, mock_paths):
    mock_paths.return_value = ["/dummy.mcap"]
    mock_parse.return_value = []
    with self.assertRaisesRegex(ValueError, "No FileMetadata messages found"):
      mcap_parser_utils.read_file_metadata_proto_data(
          "/dummy.mcap", "meta_topic"
      )

    meta_msg = metadata_pb2.FileMetadata()
    mock_parse.return_value = [meta_msg]
    result = mcap_parser_utils.read_file_metadata_proto_data(
        "/dummy.mcap", "meta_topic"
    )
    self.assertEqual(result, [meta_msg])

  @mock.patch.object(mcap_parser_utils, "get_mcap_file_paths")
  @mock.patch.object(mcap_parser_utils, "read_and_parse_mcap_messages")
  def test_read_proto_data(self, mock_parse, mock_paths):
    mock_paths.return_value = ["/dummy.mcap"]
    ex_ts = example_pb2.Example()
    ex_act = example_pb2.Example()
    ex_pe = example_pb2.Example()
    mock_parse.side_effect = [[ex_ts], [ex_act], [ex_pe]]
    res = mcap_parser_utils.read_proto_data("/dummy.mcap", "ts", "act", "pe")
    self.assertEqual(res.timesteps, [ex_ts])
    self.assertEqual(res.actions, [ex_act])
    self.assertEqual(res.policy_extra, [ex_pe])

  @mock.patch.object(mcap_parser_utils, "_iter_mcap_records")
  def test_read_and_parse_mcap_messages(self, mock_iter):
    mock_msg = mock.MagicMock()
    session = metadata_pb2.Session(task_id="test_task")
    mock_msg.data = session.SerializeToString()
    mock_iter.return_value = [mock_msg]

    msgs = mcap_parser_utils.read_and_parse_mcap_messages(
        ["/dummy.mcap"], "session_topic", metadata_pb2.Session
    )
    self.assertLen(msgs, 1)
    self.assertEqual(msgs[0].task_id, "test_task")

  @mock.patch.object(mcap_parser_utils, "get_mcap_file_paths")
  @mock.patch.object(mcap_parser_utils, "_iter_mcap_records")
  def test_read_raw_mcap_messages(self, mock_iter, mock_paths):
    mock_paths.return_value = ["/dummy.mcap"]
    mock_msg = mock.MagicMock()
    mock_iter.return_value = [mock_msg]

    msgs = mcap_parser_utils.read_raw_mcap_messages("/dummy.mcap", "topic")
    self.assertEqual(msgs, [mock_msg])

  def test_parse_examples_to_dm_env_types(self):
    ts_ex = example_pb2.Example()
    ts_ex.features.feature["step_type"].int64_list.value.append(0)
    ts_ex.features.feature["obs/sensor"].float_list.value.extend([1.0, 2.0])
    ts_ex.features.feature["reward"].float_list.value.append(1.0)
    ts_ex.features.feature["discount"].float_list.value.append(0.9)

    act_ex = example_pb2.Example()
    act_ex.features.feature["act/move"].float_list.value.extend([0.5])

    extra_ex = example_pb2.Example()
    extra_ex.features.feature["extra/info"].float_list.value.extend([42.0])

    ts_spec = gdmr_types.TimeStepSpec(
        step_type=specs.BoundedArray(
            shape=(), dtype=np.uint8, minimum=0, maximum=2
        ),
        reward=specs.Array(shape=(), dtype=np.float32),
        discount=specs.Array(shape=(), dtype=np.float32),
        observation={"sensor": specs.Array(shape=(2,), dtype=np.float32)},
    )
    act_spec = {
        "move": specs.BoundedArray(
            shape=(1,), dtype=np.float32, minimum=-1.0, maximum=1.0
        )
    }
    extra_spec = {
        "info": specs.BoundedArray(
            shape=(1,), dtype=np.float32, minimum=0.0, maximum=100.0
        )
    }

    timesteps, actions, policy_extra = (
        mcap_parser_utils.parse_examples_to_dm_env_types(
            timestep_spec=ts_spec,
            action_spec=act_spec,
            policy_extra_spec=extra_spec,
            timesteps_example=[ts_ex],
            actions_example=[act_ex],
            policy_extra_example=[extra_ex],
            step_type_key="step_type",
            observation_key_prefix="obs",
            reward_key="reward",
            discount_key="discount",
            action_key_prefix="act",
            policy_extra_key_prefix="extra",
        )
    )

    self.assertLen(timesteps, 1)
    self.assertEqual(timesteps[0].step_type, 0)
    self.assertEqual(timesteps[0].reward, 1.0)
    self.assertEqual(timesteps[0].discount, 0.9)
    self.assertIsInstance(timesteps[0].observation, dict)
    np.testing.assert_array_equal(
        timesteps[0].observation["sensor"], [1.0, 2.0]
    )

    self.assertLen(actions, 1)
    self.assertIsInstance(actions[0], dict)
    np.testing.assert_array_equal(actions[0]["move"], [0.5])

    self.assertLen(policy_extra, 1)
    self.assertIsInstance(policy_extra[0], dict)
    np.testing.assert_array_equal(policy_extra[0]["info"], [42.0])


if __name__ == "__main__":
  absltest.main()
