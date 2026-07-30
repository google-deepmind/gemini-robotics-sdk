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

"""Tests for replay_eval module."""

import json

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import dm_env
from dm_env import specs
import numpy as np

from examples.model import replay_eval
from safari_sdk.logging.python import data_validator


def _make_action_spec() -> dict[str, specs.Array]:
  """Creates a simple action spec for testing."""
  return {
      'joint_positions': specs.BoundedArray(
          shape=(6,), dtype=np.float32, minimum=-1.0, maximum=1.0
      ),
  }


def _make_timestep_spec():
  """Creates a simple timestep spec for testing."""
  from gdm_robotics.interfaces import types as gdmr_types  # pylint: disable=g-import-not-at-top

  return gdmr_types.TimeStepSpec(
      step_type=gdmr_types.STEP_TYPE_SPEC,
      reward={},
      discount={},
      observation={
          'joint_positions': specs.Array(shape=(6,), dtype=np.float32),
      },
  )


def _make_logged_actions(n: int) -> list[dict[str, np.ndarray]]:
  """Creates n mock logged actions."""
  return [
      {'joint_positions': np.full((6,), i * 0.1, dtype=np.float32)}
      for i in range(n)
  ]


class MCAPReplayPolicyTest(parameterized.TestCase):
  """Tests for the MCAPReplayPolicy class."""

  def test_step_returns_correct_actions(self):
    """Policy should return logged actions in order."""
    actions = _make_logged_actions(3)
    policy = replay_eval.MCAPReplayPolicy(
        logged_actions=actions,
        timestep_spec=_make_timestep_spec(),
        action_spec=_make_action_spec(),
    )
    state = policy.initial_state()
    dummy_ts = dm_env.transition(reward=0.0, discount=1.0, observation={})

    for i in range(3):
      (action, extra), state = policy.step(dummy_ts, state)
      np.testing.assert_array_almost_equal(
          action['joint_positions'],
          actions[i]['joint_positions'],
      )
      self.assertEmpty(extra)

  def test_exhausted_returns_zeros(self):
    """After all logged actions are used, policy should return zeros."""
    actions = _make_logged_actions(2)
    policy = replay_eval.MCAPReplayPolicy(
        logged_actions=actions,
        timestep_spec=_make_timestep_spec(),
        action_spec=_make_action_spec(),
    )
    state = policy.initial_state()
    dummy_ts = dm_env.transition(reward=0.0, discount=1.0, observation={})

    # Exhaust all actions.
    for _ in range(2):
      _, state = policy.step(dummy_ts, state)

    # Next step should return zeros.
    (action, _), _ = policy.step(dummy_ts, state)
    np.testing.assert_array_equal(
        action['joint_positions'],
        np.zeros(6, dtype=np.float32),
    )

  def test_initial_state_resets(self):
    """initial_state() should reset the policy to replay from the start."""
    actions = _make_logged_actions(3)
    policy = replay_eval.MCAPReplayPolicy(
        logged_actions=actions,
        timestep_spec=_make_timestep_spec(),
        action_spec=_make_action_spec(),
    )
    state = policy.initial_state()
    dummy_ts = dm_env.transition(reward=0.0, discount=1.0, observation={})

    # Step twice.
    _, state = policy.step(dummy_ts, state)
    _, state = policy.step(dummy_ts, state)

    # Reset.
    state = policy.initial_state()

    # Should start from the beginning again.
    (action, _), _ = policy.step(dummy_ts, state)
    np.testing.assert_array_almost_equal(
        action['joint_positions'],
        actions[0]['joint_positions'],
    )

  def test_step_spec_returns_action_spec(self):
    """step_spec() should return the action spec."""
    action_spec = _make_action_spec()
    policy = replay_eval.MCAPReplayPolicy(
        logged_actions=[],
        timestep_spec=_make_timestep_spec(),
        action_spec=action_spec,
    )
    (returned_action_spec, extra_spec), _ = policy.step_spec(
        _make_timestep_spec()
    )
    self.assertEqual(returned_action_spec, action_spec)
    self.assertEmpty(extra_spec)

  def test_zero_action_single_array_spec(self):
    """Policy with a single Array action spec returns a zero array."""
    single_spec = specs.Array(shape=(4,), dtype=np.float32)
    actions = [np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)]
    policy = replay_eval.MCAPReplayPolicy(
        logged_actions=actions,
        timestep_spec=_make_timestep_spec(),
        action_spec=single_spec,
    )
    state = policy.initial_state()
    dummy_ts = dm_env.transition(reward=0.0, discount=1.0, observation={})

    # Exhaust the single action.
    _, state = policy.step(dummy_ts, state)

    # Next step should return zeros.
    (action, _), _ = policy.step(dummy_ts, state)
    np.testing.assert_array_equal(action, np.zeros(4, dtype=np.float32))


class DiscoverMcapFilesTest(absltest.TestCase):
  """Tests for _discover_mcap_files."""

  def test_nonexistent_path_raises(self):
    with self.assertRaises(FileNotFoundError):
      replay_eval._discover_mcap_files('/nonexistent/path')

  def test_non_mcap_file_raises(self):
    # Create a temp file that is not .mcap.
    tmp_dir = self.create_tempdir()
    txt_file = tmp_dir.create_file('not_mcap.txt')
    with self.assertRaises(ValueError):
      replay_eval._discover_mcap_files(txt_file.full_path)

  def test_single_mcap_file(self):
    tmp_dir = self.create_tempdir()
    mcap_file = tmp_dir.create_file('episode.mcap')
    result = replay_eval._discover_mcap_files(mcap_file.full_path)
    self.assertEqual(result, [mcap_file.full_path])

  def test_directory_with_mcap_files(self):
    tmp_dir = self.create_tempdir()
    f1 = tmp_dir.create_file('a.mcap')
    f2 = tmp_dir.create_file('b.mcap')
    tmp_dir.create_file('c.txt')  # Should be ignored.
    result = replay_eval._discover_mcap_files(tmp_dir.full_path)
    self.assertLen(result, 2)
    self.assertIn(f1.full_path, result)
    self.assertIn(f2.full_path, result)

  def test_empty_directory_raises(self):
    tmp_dir = self.create_tempdir()
    with self.assertRaises(ValueError):
      replay_eval._discover_mcap_files(tmp_dir.full_path)


class ClipActionToBoundsTest(parameterized.TestCase):
  """Tests for _clip_action_to_bounds."""

  def test_clip_dict_action(self):
    action = {'joints': np.array([2.0, -2.0, 0.5], dtype=np.float32)}
    action_spec = {
        'joints': specs.BoundedArray(
            shape=(3,), dtype=np.float32, minimum=-1.0, maximum=1.0
        ),
    }
    clipped = replay_eval._clip_action_to_bounds(action, action_spec)
    np.testing.assert_array_equal(
        clipped['joints'], np.array([1.0, -1.0, 0.5], dtype=np.float32)
    )

  def test_clip_single_array_action(self):
    action = np.array([5.0, -5.0], dtype=np.float32)
    action_spec = specs.BoundedArray(
        shape=(2,), dtype=np.float32, minimum=-1.0, maximum=1.0
    )
    clipped = replay_eval._clip_action_to_bounds(action, action_spec)
    np.testing.assert_array_equal(
        clipped, np.array([1.0, -1.0], dtype=np.float32)
    )

  def test_no_clip_for_unbounded_spec(self):
    action = np.array([5.0, -5.0], dtype=np.float32)
    action_spec = specs.Array(shape=(2,), dtype=np.float32)
    clipped = replay_eval._clip_action_to_bounds(action, action_spec)
    np.testing.assert_array_equal(clipped, action)

  def test_clip_dict_action_and_single_bounded_array_spec(self):
    action = {'joints': np.array([2.0, -2.0], dtype=np.float32)}
    action_spec = specs.BoundedArray(
        shape=(2,), dtype=np.float32, minimum=-1.0, maximum=1.0
    )
    clipped = replay_eval._clip_action_to_bounds(action, action_spec)
    self.assertEqual(clipped, action)

  def test_clip_single_array_action_and_dict_spec(self):
    action = np.array([2.0, -2.0], dtype=np.float32)
    action_spec = {
        'joints': specs.BoundedArray(
            shape=(2,), dtype=np.float32, minimum=-1.0, maximum=1.0
        )
    }
    clipped = replay_eval._clip_action_to_bounds(action, action_spec)
    np.testing.assert_array_equal(clipped, action)


class RunValidationAndReportTest(absltest.TestCase):
  """Tests for validation execution, JSON report writing, and CLI flags."""

  def test_write_json_report(self):
    tmp_dir = self.create_tempdir()
    out_path = tmp_dir.create_file('report.json').full_path

    result = data_validator.ValidationResult(
        episode_path='/tmp/test.mcap',
        num_timesteps=10,
        num_actions=9,
        findings=[
            data_validator.ValidationFinding(
                severity=data_validator.Severity.ERROR,
                error_type=data_validator.ErrorType.NAN_VALUES,
                step_idx=1,
                field='obs',
                message='NaN found',
            )
        ],
    )
    replay_eval._write_json_report([result], out_path)

    with open(out_path, 'r') as f:
      data = json.load(f)
    self.assertEqual(data['num_episodes'], 1)
    self.assertEqual(data['num_failed'], 1)
    self.assertEqual(data['episodes'][0]['num_errors'], 1)

  def test_run_hardware_replay_invalid_env(self):
    flags.FLAGS.env_module = 'non_existent_module_foo_bar'
    flags.FLAGS.env_function = 'create_env'
    with self.assertRaises(Exception):
      replay_eval._run_hardware_replay('/tmp/fake.mcap')


if __name__ == '__main__':
  absltest.main()
