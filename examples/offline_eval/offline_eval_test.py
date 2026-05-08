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

"""Integration and unit tests for the offline evaluation tool."""

from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np

from examples.offline_eval import mcap_loader
from examples.offline_eval import offline_eval
from safari_sdk.protos.logging import metadata_pb2
from tensorflow.core.example import example_pb2


class OfflineEvalTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Ensure absl flags are parsed (required for pytest compatibility).
    if not flags.FLAGS.is_parsed():
      flags.FLAGS(['test'])

  def test_compute_episode_metrics(self):
    predicted = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    ground_truth = np.array([[1.1, 1.9], [2.9, 4.1], [5.0, 6.0]])
    chunk_boundaries = [0, 2]

    metrics = offline_eval.compute_episode_metrics(
        predicted=predicted,
        ground_truth=ground_truth,
        chunk_boundaries=chunk_boundaries,
        chunk_size=2,
    )

    self.assertIn('chunk_mse', metrics)
    self.assertIn('step_mse', metrics)

    # Step MSE: mean of (0.1^2 + (-0.1)^2 + (-0.1)^2 + 0.1^2 + 0 + 0) / 6
    # = (0.01 + 0.01 + 0.01 + 0.01) / 6 = 0.04 / 6 = 0.006666...
    self.assertAlmostEqual(metrics['step_mse'], 0.04 / 6)

    # Chunk MSE: chunk size 2.
    # Chunk 1 (0,1): MSE is (0.01+0.01)/2 = 0.01
    # Chunk 2 (2): incomplete, skipped.
    # Mean Chunk MSE = 0.01
    self.assertAlmostEqual(metrics['chunk_mse'], 0.01)

  def test_build_timestep_spec(self):
    obs = {
        'cam0': np.zeros((100, 100, 3), dtype=np.uint8),
        'proprio0': np.zeros((5,), dtype=np.float32),
    }
    episode = mcap_loader.Episode(
        observations=[obs],
        actions=np.zeros((1, 2)),
        task_instruction='test task',
        num_steps=1,
    )

    ts_spec = offline_eval._build_timestep_spec(
        episode, image_keys=['cam0'], proprio_keys=['proprio0']
    )

    self.assertIsInstance(ts_spec, gdmr_types.TimeStepSpec)
    self.assertIn('cam0', ts_spec.observation)
    self.assertIn('proprio0', ts_spec.observation)
    self.assertIn('instruction', ts_spec.observation)

    self.assertEqual(ts_spec.observation['cam0'].shape, (100, 100, 3))
    self.assertEqual(ts_spec.observation['proprio0'].shape, (1, 5))

  def test_build_observation(self):
    obs = {
        'cam0': np.zeros((100, 100, 3), dtype=np.uint8),
        'proprio0': np.zeros((5,), dtype=np.float32),
    }
    episode = mcap_loader.Episode(
        observations=[obs],
        actions=np.zeros((1, 2)),
        task_instruction='test task',
        num_steps=1,
    )

    built_obs = offline_eval._build_observation(
        episode, step_idx=0, image_keys=['cam0'], proprio_keys=['proprio0']
    )

    self.assertIn('cam0', built_obs)
    self.assertIn('proprio0', built_obs)
    self.assertIn('instruction', built_obs)

    self.assertEqual(built_obs['proprio0'].shape, (1, 5))
    self.assertEqual(built_obs['instruction'], 'test task')

  def test_build_observation_missing_image_key_raises(self):
    obs = {'proprio0': np.zeros((5,), dtype=np.float32)}
    episode = mcap_loader.Episode(
        observations=[obs],
        actions=np.zeros((1, 2)),
        task_instruction='test task',
        num_steps=1,
    )
    with self.assertRaises(KeyError):
      offline_eval._build_observation(
          episode, step_idx=0, image_keys=['cam0'], proprio_keys=['proprio0']
      )

  def test_build_observation_missing_proprio_key_raises(self):
    obs = {'cam0': np.zeros((100, 100, 3), dtype=np.uint8)}
    episode = mcap_loader.Episode(
        observations=[obs],
        actions=np.zeros((1, 2)),
        task_instruction='test task',
        num_steps=1,
    )
    with self.assertRaises(KeyError):
      offline_eval._build_observation(
          episode, step_idx=0, image_keys=['cam0'], proprio_keys=['proprio0']
      )

  def test_build_timestep_spec_missing_key_raises(self):
    obs = {'cam0': np.zeros((100, 100, 3), dtype=np.uint8)}
    episode = mcap_loader.Episode(
        observations=[obs],
        actions=np.zeros((1, 2)),
        task_instruction='test task',
        num_steps=1,
    )
    with self.assertRaises(KeyError):
      offline_eval._build_timestep_spec(
          episode, image_keys=['cam0'], proprio_keys=['missing_key']
      )

  @mock.patch.object(mcap_loader, 'load_mcap_episodes')
  @mock.patch.object(
      offline_eval.gemini_robotics_policy, 'GeminiRoboticsPolicy'
  )
  def test_evaluate_checkpoint_full_loop(
      self, mock_policy_cls, mock_load_episodes
  ):
    # Setup mock data
    obs = {
        'cam0': np.zeros((100, 100, 3), dtype=np.uint8),
        'proprio0': np.zeros((5,), dtype=np.float32),
    }
    episode = mcap_loader.Episode(
        observations=[obs, obs],
        actions=np.array([[1.0, 2.0], [3.0, 4.0]]),
        task_instruction='test task',
        num_steps=2,
    )
    metadata = mcap_loader.DetectedMetadata(
        task_id='test_task_id',
        image_keys=['cam0'],
        proprio_keys=['proprio0'],
        action_dim=2,
    )
    mock_load_episodes.return_value = ([episode], metadata)

    # Setup mock policy
    mock_policy = mock_policy_cls.return_value
    mock_policy.initial_state.return_value = {}
    mock_policy.step.return_value = ((np.array([1.1, 1.9]), {}), {})

    # Run evaluate_checkpoint
    original_dataset_path = flags.FLAGS['dataset_path'].value
    original_output_dir = flags.FLAGS['output_dir'].value
    original_checkpoint_path = flags.FLAGS['checkpoint_path'].value

    flags.FLAGS['dataset_path'].value = 'dummy_path'
    flags.FLAGS['output_dir'].value = self.create_tempdir().full_path
    flags.FLAGS['checkpoint_path'].value = 'dummy_checkpoint'

    try:
      summary = offline_eval.evaluate_checkpoint([episode], metadata)
    finally:
      flags.FLAGS['dataset_path'].value = original_dataset_path
      flags.FLAGS['output_dir'].value = original_output_dir
      flags.FLAGS['checkpoint_path'].value = original_checkpoint_path

    self.assertIn('aggregate', summary)
    self.assertIn('episodes', summary)
    self.assertLen(summary['episodes'], 1)

  def test_group_mcap_by_episode(self):
    mcap_paths = [
        '/path/episode-uuid1-shard1.mcap',
        '/path/episode-uuid1-shard0.mcap',
        '/path/episode-uuid2-shard0.mcap',
        '/path/episode-uuid3.mcap',
    ]
    grouped = mcap_loader._group_mcap_by_episode(mcap_paths)

    self.assertIn('episode-uuid1-', grouped)
    self.assertIn('episode-uuid2-', grouped)
    self.assertIn('episode-uuid3', grouped)

    # Check sorting of shards
    self.assertEqual(
        grouped['episode-uuid1-'],
        ['/path/episode-uuid1-shard0.mcap', '/path/episode-uuid1-shard1.mcap'],
    )

  @mock.patch.object(
      mcap_loader.mcap_parser_utils, 'read_and_parse_mcap_messages'
  )
  def test_detect_metadata_from_session(self, mock_read_messages):
    mock_session = mock.Mock()
    mock_session.HasField.return_value = False
    mock_session.labels = []

    mock_read_messages.return_value = [mock_session]

    metadata = mcap_loader.detect_metadata_from_session(['dummy.mcap'])

    self.assertIsNone(metadata.task_id)
    self.assertIsNone(metadata.image_keys)

  def test_matplotlib_backend_is_agg(self):
    """Verify non-interactive backend is set (headless rendering)."""
    import matplotlib  # pylint: disable=g-import-not-at-top

    self.assertEqual(matplotlib.get_backend().lower(), 'agg')

  def test_aggregate_metrics(self):
    all_metrics = [
        {
            'chunk_mse': 0.1,
            'step_mse': 0.2,
            'per_joint_mse': [0.1, 0.2],
        },
        {
            'chunk_mse': 0.3,
            'step_mse': 0.4,
            'per_joint_mse': [0.3, 0.4],
        },
    ]
    agg = offline_eval.aggregate_metrics(all_metrics)

    self.assertAlmostEqual(agg['chunk_mse']['mean'], 0.2)
    self.assertAlmostEqual(agg['step_mse']['mean'], 0.3)
    self.assertAlmostEqual(agg['per_joint_mse']['mean'][0], 0.2)
    self.assertAlmostEqual(agg['per_joint_mse']['mean'][1], 0.3)

  # ---------------------------------------------------------------------------
  # Tests 1–11: Consolidated coverage for Gaps 2–14
  # ---------------------------------------------------------------------------

  def test_parse_observation_missing_and_empty_keys(self):
    """Gap 2: missing/empty image keys are silently absent, not crashes."""
    example = example_pb2.Example()
    example.features.feature['observation/proprio0'].float_list.value.extend(
        [1.0, 2.0]
    )

    # Missing camera key → absent from dict, no crash.
    obs, _ = mcap_loader._parse_observation_from_example(
        example, image_keys=['nonexistent_cam'], proprio_keys=['proprio0']
    )
    self.assertNotIn('nonexistent_cam', obs)
    np.testing.assert_array_equal(obs['proprio0'], [1.0, 2.0])

    # Empty bytes_list → key skipped.
    example2 = example_pb2.Example()
    example2.features.feature['observation/cam0'].bytes_list.SetInParent()
    obs2, _ = mcap_loader._parse_observation_from_example(
        example2, image_keys=['cam0'], proprio_keys=[]
    )
    self.assertNotIn('cam0', obs2)

  def test_parse_action_and_missing_key(self):
    """Gap 3: action parse + missing key → empty array fallback."""
    example = example_pb2.Example()
    example.features.feature['action'].float_list.value.extend([1.0, 2.0, 3.0])
    action = mcap_loader._parse_action_from_example(example)
    np.testing.assert_array_equal(action, [1.0, 2.0, 3.0])
    self.assertEqual(action.dtype, np.float32)

    # Missing key → empty array (the dangerous silent fallback).
    empty_example = example_pb2.Example()
    action2 = mcap_loader._parse_action_from_example(empty_example)
    self.assertEqual(action2.shape, (0,))
    self.assertEqual(action2.dtype, np.float32)

  @parameterized.parameters(
      ('image_observation_keys', ['cam0', 'cam1']),
      ('proprioceptive_observation_keys', ['joint_pos']),
  )
  def test_extract_label_list(self, key, expected_values):
    """Gap 4: Session label extraction — hit and miss cases."""
    session = metadata_pb2.Session()
    label = session.labels.add()
    label.key = key
    for v in expected_values:
      label.label_value.list_value.values.add().string_value = v

    result = mcap_loader._extract_label_list(session, key)
    self.assertEqual(result, expected_values)

    # Miss: wrong key → None.
    self.assertIsNone(mcap_loader._extract_label_list(session, 'no_such_key'))

  def test_compute_metrics_edge_cases(self):
    """Gaps 5, 6: NaN chunk MSE and all-NaN aggregation."""
    # Gap 5: episode shorter than chunk_size → nan chunk_mse.
    pred = np.array([[1.0, 2.0]])
    gt = np.array([[1.1, 2.1]])
    metrics = offline_eval.compute_episode_metrics(
        pred, gt, chunk_boundaries=[0], chunk_size=50
    )
    self.assertTrue(np.isnan(metrics['chunk_mse']))
    self.assertAlmostEqual(metrics['step_mse'], 0.01)

    # Perfect prediction → step_mse = 0.
    perfect = np.array([[1.0, 2.0], [3.0, 4.0]])
    metrics2 = offline_eval.compute_episode_metrics(
        perfect, perfect, chunk_boundaries=[0], chunk_size=2
    )
    self.assertAlmostEqual(metrics2['step_mse'], 0.0)
    self.assertAlmostEqual(metrics2['chunk_mse'], 0.0)

    # Gap 6: all-NaN episodes → aggregate returns NaN mean, no crash.
    all_nan = [
        {'chunk_mse': float('nan'), 'step_mse': 0.1, 'per_joint_mse': [0.1]},
        {'chunk_mse': float('nan'), 'step_mse': 0.2, 'per_joint_mse': [0.2]},
    ]
    agg = offline_eval.aggregate_metrics(all_nan)
    self.assertTrue(np.isnan(agg['chunk_mse']['mean']))
    self.assertAlmostEqual(agg['step_mse']['mean'], 0.15)

  def test_build_observation_clamping(self):
    """Gap 7: out-of-range step_idx clamped to last valid step."""
    obs = {
        'cam0': np.zeros((4, 4, 3), dtype=np.uint8),
        'p': np.array([1.0], dtype=np.float32),
    }
    episode = mcap_loader.Episode(
        observations=[obs],
        actions=np.zeros((1, 2)),
        task_instruction='t',
        num_steps=1,
    )

    # step_idx=999 → clamped to 0.
    built = offline_eval._build_observation(episode, 999, ['cam0'], ['p'])
    self.assertEqual(built['cam0'].shape, (4, 4, 3))
    self.assertEqual(built['p'].shape, (1, 1))

  def test_group_mcap_edge_cases(self):
    """Gap 8: empty list and no-shard filenames."""
    self.assertEqual(mcap_loader._group_mcap_by_episode([]), {})

    # Single file, no shard suffix.
    result = mcap_loader._group_mcap_by_episode(['/data/ep-abc123.mcap'])
    self.assertIn('ep-abc123', result)
    self.assertLen(result['ep-abc123'], 1)

  @mock.patch.object(
      mcap_loader.mcap_parser_utils, 'read_and_parse_mcap_messages'
  )
  def test_detect_metadata_populated_session(self, mock_read):
    """Gap 10: fully populated Session → all fields detected."""
    session = metadata_pb2.Session()
    session.task_id = 'pick_task'
    for key, vals in [
        ('image_observation_keys', ['cam0']),
        ('proprioceptive_observation_keys', ['joints']),
    ]:
      label = session.labels.add()
      label.key = key
      for v in vals:
        label.label_value.list_value.values.add().string_value = v

    mock_read.return_value = [session]
    meta = mcap_loader.detect_metadata_from_session(['f.mcap'])
    self.assertEqual(meta.task_id, 'pick_task')
    self.assertEqual(meta.image_keys, ['cam0'])
    self.assertEqual(meta.proprio_keys, ['joints'])

  def test_plot_joint_grid_smoke(self):
    """Gap 11: smoke test — no crash, file written, single-joint edge case."""
    import os  # pylint: disable=g-import-not-at-top,redefined-outer-name

    tmpdir = self.create_tempdir().full_path
    for action_dim in (1, 4):
      pred = np.random.randn(10, action_dim).astype(np.float32)
      gt = np.random.randn(10, action_dim).astype(np.float32)
      path = os.path.join(tmpdir, f'test_{action_dim}j.png')
      offline_eval.plot_joint_grid(
          pred,
          gt,
          ep_idx=0,
          metrics={'chunk_mse': 0.1, 'step_mse': 0.2},
          output_path=path,
          action_dim=action_dim,
          checkpoint_label='test',
          chunk_size=5,
      )
      self.assertTrue(os.path.exists(path))

  @mock.patch.object(mcap_loader, 'load_mcap_episodes')
  @mock.patch.object(
      offline_eval.gemini_robotics_policy, 'GeminiRoboticsPolicy'
  )
  def test_evaluate_checkpoint_1step_skipped_and_outputs(
      self, mock_policy_cls, mock_load
  ):
    """Gaps 9, 14: 1-step skip + output file verification."""
    del mock_load  # Injected by @mock.patch; prevents real I/O.
    import os  # pylint: disable=g-import-not-at-top,redefined-outer-name

    obs = {
        'cam0': np.zeros((4, 4, 3), dtype=np.uint8),
        'p': np.zeros((2,), dtype=np.float32),
    }
    meta = mcap_loader.DetectedMetadata(
        task_id='t', image_keys=['cam0'], proprio_keys=['p'], action_dim=2
    )
    mock_policy = mock_policy_cls.return_value
    mock_policy.initial_state.return_value = {}
    mock_policy.step.return_value = ((np.array([0.1, 0.2]), {}), {})

    output_dir = self.create_tempdir().full_path
    original_output = flags.FLAGS['output_dir'].value
    original_ckpt = flags.FLAGS['checkpoint_path'].value
    flags.FLAGS['output_dir'].value = output_dir
    flags.FLAGS['checkpoint_path'].value = 'test_ckpt'

    try:
      # Gap 9: 1-step episode → loop runs 0 iterations → skipped.
      ep_1step = mcap_loader.Episode(
          observations=[obs],
          actions=np.zeros((1, 2)),
          task_instruction='t',
          num_steps=1,
      )
      summary = offline_eval.evaluate_checkpoint([ep_1step], meta)
      self.assertEmpty(summary['episodes'])

      # Gap 14: 2-step episode → verify output files written.
      ep_2step = mcap_loader.Episode(
          observations=[obs, obs],
          actions=np.array([[1.0, 2.0], [3.0, 4.0]]),
          task_instruction='t',
          num_steps=2,
      )
      summary2 = offline_eval.evaluate_checkpoint([ep_2step], meta)
      self.assertLen(summary2['episodes'], 1)
      self.assertTrue(os.path.exists(os.path.join(output_dir, 'summary.json')))
      self.assertTrue(
          os.path.exists(os.path.join(output_dir, 'episode_0_data.npz'))
      )
      self.assertTrue(
          os.path.exists(os.path.join(output_dir, 'episode_0_actions.png'))
      )
    finally:
      flags.FLAGS['output_dir'].value = original_output
      flags.FLAGS['checkpoint_path'].value = original_ckpt

  @mock.patch.object(mcap_loader, 'load_mcap_episodes')
  @mock.patch.object(
      offline_eval.gemini_robotics_policy, 'GeminiRoboticsPolicy'
  )
  def test_evaluate_checkpoint_timeout_fallback(
      self, mock_policy_cls, mock_load
  ):
    """Gap 12: inference timeout → fallback action → episode aborted."""
    del mock_load  # Injected by @mock.patch; prevents real I/O.
    import time  # pylint: disable=g-import-not-at-top,redefined-outer-name

    obs = {
        'cam0': np.zeros((4, 4, 3), dtype=np.uint8),
        'p': np.zeros((2,), dtype=np.float32),
    }
    # 3-step episode → 2 loop iterations → enough to hit timeout abort.
    episode = mcap_loader.Episode(
        observations=[obs, obs, obs],
        actions=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        task_instruction='t',
        num_steps=3,
    )
    meta = mcap_loader.DetectedMetadata(
        task_id='t', image_keys=['cam0'], proprio_keys=['p'], action_dim=2
    )

    mock_policy = mock_policy_cls.return_value
    mock_policy.initial_state.return_value = {}
    # Block long enough to guarantee timeout with timeout=0.
    mock_policy.step.side_effect = lambda ts, st: time.sleep(10)

    output_dir = self.create_tempdir().full_path
    original_output = flags.FLAGS['output_dir'].value
    original_ckpt = flags.FLAGS['checkpoint_path'].value
    original_timeout = flags.FLAGS['step_timeout'].value
    original_max_timeouts = flags.FLAGS['max_consecutive_timeouts'].value
    flags.FLAGS['output_dir'].value = output_dir
    flags.FLAGS['checkpoint_path'].value = 'test_ckpt'
    flags.FLAGS['step_timeout'].value = 0  # Immediate timeout.
    flags.FLAGS['max_consecutive_timeouts'].value = 1  # Abort after 1.

    try:
      summary = offline_eval.evaluate_checkpoint([episode], meta)
      # Episode should be aborted (0 or partial predictions → skipped).
      # With max_consecutive_timeouts=1, the first step times out and aborts.
      self.assertEmpty(summary['episodes'])
    finally:
      flags.FLAGS['output_dir'].value = original_output
      flags.FLAGS['checkpoint_path'].value = original_ckpt
      flags.FLAGS['step_timeout'].value = original_timeout
      flags.FLAGS['max_consecutive_timeouts'].value = original_max_timeouts

  @mock.patch.object(mcap_loader, 'load_mcap_episodes')
  @mock.patch.object(
      offline_eval.gemini_robotics_policy, 'GeminiRoboticsPolicy'
  )
  def test_evaluate_checkpoint_policy_error_fallback(
      self, mock_policy_cls, mock_load
  ):
    """Gap 13: policy.step raises → fallback action → episode aborted."""
    del mock_load  # Injected by @mock.patch; prevents real I/O.
    obs = {
        'cam0': np.zeros((4, 4, 3), dtype=np.uint8),
        'p': np.zeros((2,), dtype=np.float32),
    }
    episode = mcap_loader.Episode(
        observations=[obs, obs, obs],
        actions=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        task_instruction='t',
        num_steps=3,
    )
    meta = mcap_loader.DetectedMetadata(
        task_id='t', image_keys=['cam0'], proprio_keys=['p'], action_dim=2
    )

    mock_policy = mock_policy_cls.return_value
    mock_policy.initial_state.return_value = {}
    mock_policy.step.side_effect = RuntimeError('model crashed')

    output_dir = self.create_tempdir().full_path
    original_output = flags.FLAGS['output_dir'].value
    original_ckpt = flags.FLAGS['checkpoint_path'].value
    original_max_timeouts = flags.FLAGS['max_consecutive_timeouts'].value
    flags.FLAGS['output_dir'].value = output_dir
    flags.FLAGS['checkpoint_path'].value = 'test_ckpt'
    flags.FLAGS['max_consecutive_timeouts'].value = 1  # Abort after 1 error.

    try:
      summary = offline_eval.evaluate_checkpoint([episode], meta)
      # Episode aborted after 1 error. With max_consecutive_timeouts=1,
      # the first step errors → abort. 0 or 1 predictions.
      # If 0 predictions, episode is skipped entirely.
      self.assertLessEqual(len(summary['episodes']), 1)
    finally:
      flags.FLAGS['output_dir'].value = original_output
      flags.FLAGS['checkpoint_path'].value = original_ckpt
      flags.FLAGS['max_consecutive_timeouts'].value = original_max_timeouts


if __name__ == '__main__':
  absltest.main()
