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

"""Episode data validation for Safari SDK MCAP recordings.

Provides offline validation of recorded episode data against session specs.
Checks for common data quality issues including NaN/Inf values, shape/dtype
mismatches, bounds violations, and step-type progression errors.
"""

from collections.abc import Sequence
import dataclasses
import enum
from typing import Any

from dm_env import specs
import numpy as np


class ErrorType(enum.Enum):
  """Types of validation errors."""

  NAN_VALUES = "NAN_VALUES"
  INF_VALUES = "INF_VALUES"
  SHAPE_MISMATCH = "SHAPE_MISMATCH"
  DTYPE_MISMATCH = "DTYPE_MISMATCH"
  BOUNDS_VIOLATION = "BOUNDS_VIOLATION"
  STEP_TYPE_PROGRESSION = "STEP_TYPE_PROGRESSION"
  MISSING_KEY = "MISSING_KEY"
  EMPTY_EPISODE = "EMPTY_EPISODE"
  COUNT_MISMATCH = "COUNT_MISMATCH"


class Severity(enum.Enum):
  """Severity of a validation finding."""

  ERROR = "ERROR"
  WARNING = "WARNING"


@dataclasses.dataclass(frozen=True)
class ValidationFinding:
  """A single validation finding (error or warning)."""

  severity: Severity
  error_type: ErrorType
  step_idx: int | None  # None for episode-level findings
  field: str
  message: str
  expected: Any = None
  actual: Any = None


@dataclasses.dataclass(frozen=True)
class ValidationResult:
  """Result of validating a single episode."""

  episode_path: str
  num_timesteps: int
  num_actions: int
  findings: list[ValidationFinding]

  @property
  def errors(self) -> list[ValidationFinding]:
    return [f for f in self.findings if f.severity == Severity.ERROR]

  @property
  def warnings(self) -> list[ValidationFinding]:
    return [f for f in self.findings if f.severity == Severity.WARNING]

  @property
  def passed(self) -> bool:
    return not self.errors


class EpisodeValidator:
  """Validates MCAP episode data against session specs."""

  def validate_array(
      self,
      array: np.ndarray,
      spec: specs.Array,
      step_idx: int,
      field: str,
  ) -> list[ValidationFinding]:
    """Validates a single array against its spec."""
    findings = []

    # Check shape.
    if spec.shape is not None and array.shape != tuple(spec.shape):
      findings.append(
          ValidationFinding(
              severity=Severity.ERROR,
              error_type=ErrorType.SHAPE_MISMATCH,
              step_idx=step_idx,
              field=field,
              message=(
                  f"Shape mismatch: expected {spec.shape}, got {array.shape}"
              ),
              expected=spec.shape,
              actual=array.shape,
          )
      )
      return findings

    # Check dtype.
    is_string_spec = (
        isinstance(spec, specs.StringArray)
        or np.issubdtype(spec.dtype, np.str_)
        or np.issubdtype(spec.dtype, np.bytes_)
        or spec.dtype == np.dtype("O")
    )
    if is_string_spec:
      if not (
          np.issubdtype(array.dtype, np.str_)
          or np.issubdtype(array.dtype, np.bytes_)
          or array.dtype == np.dtype("O")
      ):
        findings.append(
            ValidationFinding(
                severity=Severity.WARNING,
                error_type=ErrorType.DTYPE_MISMATCH,
                step_idx=step_idx,
                field=field,
                message=f"Dtype mismatch: expected string, got {array.dtype}",
                expected="string",
                actual=str(array.dtype),
            )
        )
    elif array.dtype != spec.dtype:
      findings.append(
          ValidationFinding(
              severity=Severity.WARNING,
              error_type=ErrorType.DTYPE_MISMATCH,
              step_idx=step_idx,
              field=field,
              message=(
                  f"Dtype mismatch: expected {spec.dtype}, got {array.dtype}"
              ),
              expected=str(spec.dtype),
              actual=str(array.dtype),
          )
      )

    # Check NaN (only for float types).
    has_nan = False
    if np.issubdtype(array.dtype, np.floating):
      if np.any(np.isnan(array)):
        has_nan = True
        nan_count = int(np.sum(np.isnan(array)))
        findings.append(
            ValidationFinding(
                severity=Severity.ERROR,
                error_type=ErrorType.NAN_VALUES,
                step_idx=step_idx,
                field=field,
                message=f"Found {nan_count} NaN values",
            )
        )

      # Check Inf (only for float types).
      if np.any(np.isinf(array)):
        inf_count = int(np.sum(np.isinf(array)))
        findings.append(
            ValidationFinding(
                severity=Severity.WARNING,
                error_type=ErrorType.INF_VALUES,
                step_idx=step_idx,
                field=field,
                message=f"Found {inf_count} Inf values",
            )
        )

    # Check bounds (only for BoundedArray).
    if isinstance(spec, specs.BoundedArray) and not has_nan:
      below_min = np.any(array < spec.minimum)
      above_max = np.any(array > spec.maximum)
      if below_min or above_max:
        findings.append(
            ValidationFinding(
                severity=Severity.ERROR,
                error_type=ErrorType.BOUNDS_VIOLATION,
                step_idx=step_idx,
                field=field,
                message=(
                    f"Values out of bounds [{np.min(spec.minimum)}, "
                    f"{np.max(spec.maximum)}]: "
                    f"actual range [{np.min(array)}, {np.max(array)}]"
                ),
            )
        )

    return findings

  def validate_step_types(
      self,
      timesteps: Sequence[Any],
  ) -> list[ValidationFinding]:
    """Validates step-type progression: FIRST -> MID* -> LAST."""
    findings = []
    if not timesteps:
      return findings

    step_types = [ts.step_type for ts in timesteps]

    # First step must be FIRST (0).
    if step_types[0] != 0:
      findings.append(
          ValidationFinding(
              severity=Severity.ERROR,
              error_type=ErrorType.STEP_TYPE_PROGRESSION,
              step_idx=0,
              field="step_type",
              message=f"First step should be FIRST (0), got {step_types[0]}",
              expected=0,
              actual=int(step_types[0]),
          )
      )

    # Last step must be LAST (2).
    if len(step_types) > 1 and step_types[-1] != 2:
      findings.append(
          ValidationFinding(
              severity=Severity.WARNING,
              error_type=ErrorType.STEP_TYPE_PROGRESSION,
              step_idx=len(step_types) - 1,
              field="step_type",
              message=f"Last step should be LAST (2), got {step_types[-1]}",
              expected=2,
              actual=int(step_types[-1]),
          )
      )

    # Middle steps must be MID (1).
    for i in range(1, len(step_types) - 1):
      if step_types[i] != 1:
        findings.append(
            ValidationFinding(
                severity=Severity.WARNING,
                error_type=ErrorType.STEP_TYPE_PROGRESSION,
                step_idx=i,
                field="step_type",
                message=f"Middle step should be MID (1), got {step_types[i]}",
                expected=1,
                actual=int(step_types[i]),
            )
        )
        break  # Only report first violation to avoid noise.

    return findings

  def validate_episode(
      self,
      episode_path: str,
      timesteps: Sequence[Any],
      actions: Sequence[Any],
      timestep_spec: Any,
      action_spec: Any,
  ) -> ValidationResult:
    """Runs all validations on a single episode.

    Args:
      episode_path: Path to the MCAP file.
      timesteps: List of dm_env.TimeStep objects.
      actions: List of action dicts or action arrays.
      timestep_spec: TimeStepSpec object or dict with
        observation/reward/discount specs.
      action_spec: Action spec (dict or single specs.Array).

    Returns:
      A ValidationResult with all findings.
    """
    findings = []

    # Check for empty episode.
    if not timesteps:
      findings.append(
          ValidationFinding(
              severity=Severity.ERROR,
              error_type=ErrorType.EMPTY_EPISODE,
              step_idx=None,
              field="episode",
              message="Episode has no timesteps",
          )
      )
      return ValidationResult(
          episode_path=episode_path,
          num_timesteps=0,
          num_actions=0,
          findings=findings,
      )

    # Check timestep/action count relationship.
    expected_actions = len(timesteps) - 1
    if len(actions) != expected_actions:
      findings.append(
          ValidationFinding(
              severity=Severity.WARNING,
              error_type=ErrorType.COUNT_MISMATCH,
              step_idx=None,
              field="episode",
              message=(
                  f"Expected {expected_actions} actions for {len(timesteps)} "
                  f"timesteps, got {len(actions)}"
              ),
              expected=expected_actions,
              actual=len(actions),
          )
      )

    # Validate step-type progression.
    findings.extend(self.validate_step_types(timesteps))

    # Extract sub-specs supporting both TimeStepSpec object and dict.
    if hasattr(timestep_spec, "observation"):
      obs_spec = timestep_spec.observation
      reward_spec = timestep_spec.reward
      discount_spec = timestep_spec.discount
    elif isinstance(timestep_spec, dict):
      obs_spec = timestep_spec.get("observation", {})
      reward_spec = timestep_spec.get("reward", {})
      discount_spec = timestep_spec.get("discount", {})
    else:
      obs_spec, reward_spec, discount_spec = {}, {}, {}

    # Validate timesteps (observations, rewards, discounts).
    for step_idx, ts in enumerate(timesteps):
      # Validate observations.
      if hasattr(ts, "observation") and isinstance(ts.observation, dict):
        if isinstance(obs_spec, dict):
          for key, spec in obs_spec.items():
            if key not in ts.observation:
              findings.append(
                  ValidationFinding(
                      severity=Severity.ERROR,
                      error_type=ErrorType.MISSING_KEY,
                      step_idx=step_idx,
                      field=f"observation/{key}",
                      message=f"Missing observation key: {key}",
                  )
              )
            elif isinstance(ts.observation[key], np.ndarray):
              findings.extend(
                  self.validate_array(
                      ts.observation[key], spec, step_idx, f"observation/{key}"
                  )
              )

      # Validate rewards.
      if hasattr(ts, "reward") and ts.reward is not None:
        if isinstance(reward_spec, specs.Array) and isinstance(
            ts.reward, np.ndarray
        ):
          findings.extend(
              self.validate_array(ts.reward, reward_spec, step_idx, "reward")
          )
        elif isinstance(reward_spec, dict) and isinstance(ts.reward, dict):
          for key, spec in reward_spec.items():
            if key in ts.reward and isinstance(ts.reward[key], np.ndarray):
              findings.extend(
                  self.validate_array(
                      ts.reward[key], spec, step_idx, f"reward/{key}"
                  )
              )

      # Validate discounts.
      if hasattr(ts, "discount") and ts.discount is not None:
        if isinstance(discount_spec, specs.Array) and isinstance(
            ts.discount, np.ndarray
        ):
          findings.extend(
              self.validate_array(
                  ts.discount, discount_spec, step_idx, "discount"
              )
          )
        elif isinstance(discount_spec, dict) and isinstance(ts.discount, dict):
          for key, spec in discount_spec.items():
            if key in ts.discount and isinstance(ts.discount[key], np.ndarray):
              findings.extend(
                  self.validate_array(
                      ts.discount[key], spec, step_idx, f"discount/{key}"
                  )
              )

    # Validate actions.
    for step_idx, action in enumerate(actions):
      if isinstance(action_spec, specs.Array) and isinstance(
          action, np.ndarray
      ):
        findings.extend(
            self.validate_array(action, action_spec, step_idx, "action")
        )
      elif isinstance(action_spec, dict) and isinstance(action, dict):
        for key, spec in action_spec.items():
          if key not in action:
            findings.append(
                ValidationFinding(
                    severity=Severity.ERROR,
                    error_type=ErrorType.MISSING_KEY,
                    step_idx=step_idx,
                    field=f"action/{key}",
                    message=f"Missing action key: {key}",
                )
            )
          elif isinstance(action[key], np.ndarray):
            findings.extend(
                self.validate_array(
                    action[key], spec, step_idx, f"action/{key}"
                )
            )

    return ValidationResult(
        episode_path=episode_path,
        num_timesteps=len(timesteps),
        num_actions=len(actions),
        findings=findings,
    )
