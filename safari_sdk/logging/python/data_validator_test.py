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

"""Tests for data_validator module."""

import dm_env
from dm_env import specs
import numpy as np

from absl.testing import absltest
from safari_sdk.logging.python import data_validator


def _make_timestep(step_type, observation=None, reward=0.0, discount=1.0):
  """Helper to create a dm_env.TimeStep."""
  return dm_env.TimeStep(
      step_type=step_type,
      reward=reward,
      discount=discount,
      observation=observation or {},
  )


class ValidateArrayTest(absltest.TestCase):
  """Tests for EpisodeValidator.validate_array()."""

  def setUp(self):
    super().setUp()
    self.validator = data_validator.EpisodeValidator()

  def test_valid_array_no_findings(self):
    """Valid array produces no findings."""
    array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    spec = specs.Array(shape=(3,), dtype=np.float32)
    findings = self.validator.validate_array(
        array, spec, step_idx=0, field="test"
    )
    self.assertEmpty(findings)

  def test_nan_values_detected(self):
    """NaN values produce an ERROR finding."""
    array = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    spec = specs.Array(shape=(3,), dtype=np.float32)
    findings = self.validator.validate_array(
        array, spec, step_idx=0, field="test"
    )
    nan_findings = [
        f
        for f in findings
        if f.error_type == data_validator.ErrorType.NAN_VALUES
    ]
    self.assertLen(nan_findings, 1)
    self.assertEqual(nan_findings[0].severity, data_validator.Severity.ERROR)

  def test_inf_values_detected(self):
    """Inf values produce a WARNING finding."""
    array = np.array([1.0, np.inf, 3.0], dtype=np.float32)
    spec = specs.Array(shape=(3,), dtype=np.float32)
    findings = self.validator.validate_array(
        array, spec, step_idx=0, field="test"
    )
    inf_findings = [
        f
        for f in findings
        if f.error_type == data_validator.ErrorType.INF_VALUES
    ]
    self.assertLen(inf_findings, 1)
    self.assertEqual(inf_findings[0].severity, data_validator.Severity.WARNING)

  def test_shape_mismatch_detected(self):
    """Shape mismatch produces an ERROR finding."""
    array = np.array([1.0, 2.0], dtype=np.float32)
    spec = specs.Array(shape=(3,), dtype=np.float32)
    findings = self.validator.validate_array(
        array, spec, step_idx=0, field="test"
    )
    shape_findings = [
        f
        for f in findings
        if f.error_type == data_validator.ErrorType.SHAPE_MISMATCH
    ]
    self.assertLen(shape_findings, 1)
    self.assertEqual(shape_findings[0].severity, data_validator.Severity.ERROR)

  def test_dtype_mismatch_detected(self):
    """Dtype mismatch produces a WARNING finding."""
    array = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    spec = specs.Array(shape=(3,), dtype=np.float32)
    findings = self.validator.validate_array(
        array, spec, step_idx=0, field="test"
    )
    dtype_findings = [
        f
        for f in findings
        if f.error_type == data_validator.ErrorType.DTYPE_MISMATCH
    ]
    self.assertLen(dtype_findings, 1)
    self.assertEqual(
        dtype_findings[0].severity, data_validator.Severity.WARNING
    )

  def test_bounds_violation_detected(self):
    """Out-of-bounds values produce an ERROR finding."""
    array = np.array([0.5, 2.5, -0.5], dtype=np.float32)
    spec = specs.BoundedArray(
        shape=(3,), dtype=np.float32, minimum=-1.0, maximum=2.0
    )
    findings = self.validator.validate_array(
        array, spec, step_idx=0, field="test"
    )
    bounds_findings = [
        f
        for f in findings
        if f.error_type == data_validator.ErrorType.BOUNDS_VIOLATION
    ]
    self.assertLen(bounds_findings, 1)
    self.assertEqual(bounds_findings[0].severity, data_validator.Severity.ERROR)

  def test_bounds_valid_no_violation(self):
    """In-bounds values produce no bounds finding."""
    array = np.array([0.5, 1.0, -0.5], dtype=np.float32)
    spec = specs.BoundedArray(
        shape=(3,), dtype=np.float32, minimum=-1.0, maximum=2.0
    )
    findings = self.validator.validate_array(
        array, spec, step_idx=0, field="test"
    )
    bounds_findings = [
        f
        for f in findings
        if f.error_type == data_validator.ErrorType.BOUNDS_VIOLATION
    ]
    self.assertEmpty(bounds_findings)

  def test_integer_array_no_nan_check(self):
    """Integer arrays skip NaN/Inf checks (no crash)."""
    array = np.array([1, 2, 3], dtype=np.int32)
    spec = specs.Array(shape=(3,), dtype=np.int32)
    findings = self.validator.validate_array(
        array, spec, step_idx=0, field="test"
    )
    self.assertEmpty(findings)

  def test_string_array_dtype_mismatch(self):
    """String spec with non-string array produces DTYPE_MISMATCH warning."""
    array = np.array([1, 2, 3], dtype=np.int32)
    spec = specs.StringArray(shape=(3,))
    findings = self.validator.validate_array(
        array, spec, step_idx=0, field="string_field"
    )
    dtype_findings = [
        f
        for f in findings
        if f.error_type == data_validator.ErrorType.DTYPE_MISMATCH
    ]
    self.assertLen(dtype_findings, 1)
    self.assertEqual(
        dtype_findings[0].severity, data_validator.Severity.WARNING
    )


class ValidateStepTypesTest(absltest.TestCase):
  """Tests for EpisodeValidator.validate_step_types()."""

  def setUp(self):
    super().setUp()
    self.validator = data_validator.EpisodeValidator()

  def test_valid_progression(self):
    """FIRST -> MID -> LAST produces no findings."""
    timesteps = [
        _make_timestep(step_type=0),
        _make_timestep(step_type=1),
        _make_timestep(step_type=1),
        _make_timestep(step_type=2),
    ]
    findings = self.validator.validate_step_types(timesteps)
    self.assertEmpty(findings)

  def test_wrong_first_step(self):
    """Wrong first step produces an ERROR."""
    timesteps = [
        _make_timestep(step_type=1),
        _make_timestep(step_type=2),
    ]
    findings = self.validator.validate_step_types(timesteps)
    first_findings = [
        f
        for f in findings
        if f.error_type == data_validator.ErrorType.STEP_TYPE_PROGRESSION
        and f.step_idx == 0
    ]
    self.assertLen(first_findings, 1)
    self.assertEqual(first_findings[0].severity, data_validator.Severity.ERROR)

  def test_wrong_last_step(self):
    """Wrong last step produces a WARNING."""
    timesteps = [
        _make_timestep(step_type=0),
        _make_timestep(step_type=1),
        _make_timestep(step_type=1),
    ]
    findings = self.validator.validate_step_types(timesteps)
    last_findings = [
        f
        for f in findings
        if f.error_type == data_validator.ErrorType.STEP_TYPE_PROGRESSION
        and f.step_idx == 2
    ]
    self.assertLen(last_findings, 1)
    self.assertEqual(last_findings[0].severity, data_validator.Severity.WARNING)

  def test_empty_timesteps(self):
    """Empty timesteps produces no findings."""
    findings = self.validator.validate_step_types([])
    self.assertEmpty(findings)

  def test_single_timestep(self):
    """Single FIRST timestep is valid (no last-step check for len=1)."""
    timesteps = [_make_timestep(step_type=0)]
    findings = self.validator.validate_step_types(timesteps)
    self.assertEmpty(findings)


class ValidateEpisodeTest(absltest.TestCase):
  """Tests for EpisodeValidator.validate_episode()."""

  def setUp(self):
    super().setUp()
    self.validator = data_validator.EpisodeValidator()

  def test_empty_episode(self):
    """Empty episode produces an EMPTY_EPISODE error."""
    result = self.validator.validate_episode(
        episode_path="/test/episode.mcap",
        timesteps=[],
        actions=[],
        timestep_spec={"observation": {}},
        action_spec={},
    )
    self.assertFalse(result.passed)
    self.assertEqual(result.num_timesteps, 0)
    self.assertEqual(result.num_actions, 0)
    empty_findings = [
        f
        for f in result.findings
        if f.error_type == data_validator.ErrorType.EMPTY_EPISODE
    ]
    self.assertLen(empty_findings, 1)

  def test_count_mismatch(self):
    """Mismatched timestep/action counts produces a warning."""
    timesteps = [
        _make_timestep(step_type=0),
        _make_timestep(step_type=1),
        _make_timestep(step_type=2),
    ]
    # Expected 2 actions, provide 1.
    actions = [{"joint": np.array([0.0], dtype=np.float32)}]
    result = self.validator.validate_episode(
        episode_path="/test/episode.mcap",
        timesteps=timesteps,
        actions=actions,
        timestep_spec={"observation": {}},
        action_spec={"joint": specs.Array(shape=(1,), dtype=np.float32)},
    )
    count_findings = [
        f
        for f in result.findings
        if f.error_type == data_validator.ErrorType.COUNT_MISMATCH
    ]
    self.assertLen(count_findings, 1)

  def test_valid_episode_passes(self):
    """Clean episode with valid data passes validation."""
    obs_spec = {"joints": specs.Array(shape=(6,), dtype=np.float32)}
    action_spec = {
        "command": specs.BoundedArray(
            shape=(6,), dtype=np.float32, minimum=-1.0, maximum=1.0
        )
    }
    timestep_spec = {"observation": obs_spec, "reward": {}, "discount": {}}

    timesteps = [
        _make_timestep(
            step_type=0,
            observation={"joints": np.zeros(6, dtype=np.float32)},
        ),
        _make_timestep(
            step_type=1,
            observation={"joints": np.ones(6, dtype=np.float32) * 0.5},
        ),
        _make_timestep(
            step_type=2,
            observation={"joints": np.ones(6, dtype=np.float32)},
        ),
    ]
    actions = [
        {"command": np.zeros(6, dtype=np.float32)},
        {"command": np.ones(6, dtype=np.float32) * 0.5},
    ]

    result = self.validator.validate_episode(
        episode_path="/test/episode.mcap",
        timesteps=timesteps,
        actions=actions,
        timestep_spec=timestep_spec,
        action_spec=action_spec,
    )
    self.assertTrue(result.passed)
    self.assertEqual(result.num_timesteps, 3)
    self.assertEqual(result.num_actions, 2)
    self.assertEmpty(result.errors)

  def test_missing_observation_key(self):
    """Missing observation key produces a MISSING_KEY error."""
    obs_spec = {"joints": specs.Array(shape=(6,), dtype=np.float32)}
    timestep_spec = {"observation": obs_spec, "reward": {}, "discount": {}}

    timesteps = [
        _make_timestep(step_type=0, observation={}),  # Missing 'joints'.
        _make_timestep(
            step_type=2, observation={"joints": np.zeros(6, dtype=np.float32)}
        ),
    ]
    actions = [{"command": np.zeros(6, dtype=np.float32)}]

    result = self.validator.validate_episode(
        episode_path="/test/episode.mcap",
        timesteps=timesteps,
        actions=actions,
        timestep_spec=timestep_spec,
        action_spec={"command": specs.Array(shape=(6,), dtype=np.float32)},
    )
    missing_findings = [
        f
        for f in result.findings
        if f.error_type == data_validator.ErrorType.MISSING_KEY
    ]
    self.assertLen(missing_findings, 1)

  def test_reward_and_discount_validation(self):
    """Reward and discount array and dict validation in validate_episode."""
    reward_spec = specs.Array(shape=(1,), dtype=np.float32)
    discount_spec = specs.Array(shape=(1,), dtype=np.float32)
    timestep_spec = {
        "observation": {},
        "reward": reward_spec,
        "discount": discount_spec,
    }

    timesteps = [
        _make_timestep(
            step_type=0,
            reward=np.array([np.nan], dtype=np.float32),
            discount=np.array([np.inf], dtype=np.float32),
        ),
        _make_timestep(
            step_type=2,
            reward=np.array([1.0], dtype=np.float32),
            discount=np.array([1.0], dtype=np.float32),
        ),
    ]
    actions = [np.array([0.0], dtype=np.float32)]
    action_spec = specs.Array(shape=(1,), dtype=np.float32)

    result = self.validator.validate_episode(
        episode_path="/test/episode.mcap",
        timesteps=timesteps,
        actions=actions,
        timestep_spec=timestep_spec,
        action_spec=action_spec,
    )
    self.assertFalse(result.passed)
    nan_findings = [
        f
        for f in result.findings
        if f.error_type == data_validator.ErrorType.NAN_VALUES
    ]
    inf_findings = [
        f
        for f in result.findings
        if f.error_type == data_validator.ErrorType.INF_VALUES
    ]
    self.assertLen(nan_findings, 1)
    self.assertLen(inf_findings, 1)

  def test_reward_and_discount_dict_validation(self):
    """Dict reward and discount validation in validate_episode."""
    timestep_spec = {
        "observation": {},
        "reward": {"r1": specs.Array(shape=(1,), dtype=np.float32)},
        "discount": {"d1": specs.Array(shape=(1,), dtype=np.float32)},
    }
    timesteps = [
        _make_timestep(
            step_type=0,
            reward={"r1": np.array([np.nan], dtype=np.float32)},
            discount={"d1": np.array([1.0], dtype=np.float32)},
        ),
        _make_timestep(
            step_type=2,
            reward={"r1": np.array([0.0], dtype=np.float32)},
            discount={"d1": np.array([1.0], dtype=np.float32)},
        ),
    ]
    actions = [{"cmd": np.array([0.0], dtype=np.float32)}]
    action_spec = {"cmd": specs.Array(shape=(1,), dtype=np.float32)}

    result = self.validator.validate_episode(
        episode_path="/test/episode.mcap",
        timesteps=timesteps,
        actions=actions,
        timestep_spec=timestep_spec,
        action_spec=action_spec,
    )
    self.assertFalse(result.passed)
    self.assertLen(result.errors, 1)

  def test_validation_result_properties(self):
    """Tests ValidationResult.errors, warnings, and passed properties."""
    findings = [
        data_validator.ValidationFinding(
            severity=data_validator.Severity.ERROR,
            error_type=data_validator.ErrorType.NAN_VALUES,
            step_idx=0,
            field="test",
            message="NaN",
        ),
        data_validator.ValidationFinding(
            severity=data_validator.Severity.WARNING,
            error_type=data_validator.ErrorType.DTYPE_MISMATCH,
            step_idx=0,
            field="test",
            message="Dtype",
        ),
    ]
    result = data_validator.ValidationResult(
        episode_path="/test.mcap",
        num_timesteps=1,
        num_actions=0,
        findings=findings,
    )
    self.assertLen(result.errors, 1)
    self.assertLen(result.warnings, 1)
    self.assertFalse(result.passed)

  def test_string_spec_valid_dtypes(self):
    """Tests StringArray and string dtype specs with str, bytes, object arrays."""
    spec = specs.StringArray(shape=(2,))
    for dtype in [np.str_, np.bytes_, np.dtype("O")]:
      arr = np.array(["a", "b"], dtype=dtype)
      findings = self.validator.validate_array(
          arr, spec, step_idx=0, field="str_field"
      )
      self.assertEmpty(findings)

  def test_scalar_spec_shape_mismatch(self):
    """Tests shape mismatch for scalar spec (shape=())."""
    scalar_spec = specs.Array(shape=(), dtype=np.float32)
    non_scalar_arr = np.array([1.0, 2.0], dtype=np.float32)
    findings = self.validator.validate_array(
        non_scalar_arr, scalar_spec, step_idx=0, field="scalar_field"
    )
    self.assertLen(findings, 1)
    self.assertEqual(
        findings[0].error_type, data_validator.ErrorType.SHAPE_MISMATCH
    )

  def test_missing_action_key(self):
    """Tests detection of missing keys in dictionary action specs."""
    action_spec = {"action_a": specs.Array(shape=(1,), dtype=np.float32)}
    action_step = {}  # missing action_a
    timestep_spec = {"observation": {}, "reward": {}, "discount": {}}
    result = self.validator.validate_episode(
        episode_path="/test.mcap",
        timesteps=[_make_timestep(step_type=0)],
        actions=[action_step],
        timestep_spec=timestep_spec,
        action_spec=action_spec,
    )
    self.assertFalse(result.passed)
    self.assertTrue(
        any(
            f.error_type == data_validator.ErrorType.MISSING_KEY
            for f in result.findings
        )
    )


if __name__ == "__main__":
  absltest.main()
