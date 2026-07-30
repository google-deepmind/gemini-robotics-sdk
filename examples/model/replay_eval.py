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

r"""Replay evaluation tool for Safari SDK MCAP episodes.

Two modes:
  validate: (Default) Offline data quality validation. No robot needed.
  replay:   Replay logged actions through a live robot environment.

Usage:
  # Validate a single episode
  python3 replay_eval.py --mcap_file=/path/to/episode.mcap

  # Validate a directory of episodes
  python3 replay_eval.py --mcap_file=/path/to/episodes_dir

  # Replay through hardware
  python3 replay_eval.py \
    --mode=replay \
    --mcap_file=/path/to/episode.mcap \
    --env_module=safari_sdk.examples.so101.env \
    --env_function=create_so101_environment
"""

from collections.abc import Mapping, Sequence
import dataclasses
import importlib
import json
import os
import sys
import time

from absl import app
from absl import flags
from absl import logging
import dm_env
from dm_env import specs
from gdm_robotics.interfaces import policy as gdmr_policy
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np
from typing_extensions import override

from safari_sdk.logging.python import constants
from safari_sdk.logging.python import data_validator
from safari_sdk.logging.python import mcap_parser_utils
from safari_sdk.logging.python import spec_utils
from safari_sdk.protos.logging import metadata_pb2

_MCAP_FILE = flags.DEFINE_string(
    'mcap_file',
    None,
    'Path to MCAP file or directory.',
)
_MODE = flags.DEFINE_enum(
    'mode',
    'validate',
    ['validate', 'replay'],
    'Operating mode.',
)
_ENV_MODULE = flags.DEFINE_string(
    'env_module',
    None,
    'Environment factory module (required for replay mode).',
)
_ENV_FUNCTION = flags.DEFINE_string(
    'env_function',
    'create_so101_environment',
    'Environment factory function name.',
)
_CONTROL_HZ = flags.DEFINE_float(
    'control_hz',
    None,
    'Control frequency override in Hz. If unset, uses session metadata.',
)
_CHUNK_SIZE = flags.DEFINE_integer(
    'chunk_size',
    50,
    'Steps per interactive chunk in replay mode.',
)
_INTERACTIVE = flags.DEFINE_bool(
    'interactive',
    False,
    'Pause after each chunk in replay mode.',
)
_OUTPUT = flags.DEFINE_string(
    'output',
    None,
    'Path for JSON validation report output.',
)
_STRICT = flags.DEFINE_bool(
    'strict',
    False,
    'Treat warnings as errors (useful for CI gates).',
)


# ============================================================================
# Data container for a loaded episode
# ============================================================================


@dataclasses.dataclass(frozen=True)
class EpisodeData:
  """Container for a loaded MCAP episode."""

  timesteps: list[dm_env.TimeStep]
  actions: list[gdmr_types.ActionType]
  timestep_spec: gdmr_types.TimeStepSpec
  action_spec: specs.Array | Mapping[str, specs.Array]
  session: metadata_pb2.Session


# ============================================================================
# MCAPReplayPolicy
# ============================================================================


class MCAPReplayPolicy(gdmr_policy.Policy[np.ndarray]):
  """Mock policy that replays actions from a recorded MCAP episode.

  This implements the same gdmr_policy.Policy interface as
  GeminiRoboticsPolicy, but returns pre-recorded actions instead of
  calling a model server. Useful for validating the end-to-end data
  pipeline without needing a live model.

  NOTE: This policy does NOT contain interactive prompts (input() calls).
  Interactive chunk-stepping is handled by the replay loop in
  _run_hardware_replay().
  """

  def __init__(
      self,
      logged_actions: list[gdmr_types.ActionType],
      timestep_spec: gdmr_types.TimeStepSpec,
      action_spec: specs.Array | Mapping[str, specs.Array],
  ):
    self._actions = logged_actions
    self._timestep_spec = timestep_spec
    self._action_spec = action_spec
    self._step_idx = 0
    self._dummy_state = np.zeros(())

  @override
  def initial_state(self) -> gdmr_types.StateStructure[np.ndarray]:
    """Resets the policy to the beginning of the logged episode."""
    self._step_idx = 0
    return self._dummy_state

  @override
  def step(
      self,
      timestep: dm_env.TimeStep,
      prev_state: gdmr_types.StateStructure[np.ndarray],
  ) -> tuple[
      tuple[
          gdmr_types.ActionType,
          gdmr_types.ExtraOutputStructure[np.ndarray],
      ],
      gdmr_types.StateStructure[np.ndarray],
  ]:
    """Returns the next logged action.

    If all logged actions have been exhausted, returns a zero action and
    logs a warning.

    Args:
      timestep: Current environment timestep.
      prev_state: Previous policy state.
    """
    del prev_state, timestep  # Unused.

    idx = self._step_idx

    if idx >= len(self._actions):
      logging.warning(
          'Replayed all %d logged actions — returning zeros.',
          len(self._actions),
      )
      action = self._get_zero_action()
      return (action, {}), self._dummy_state

    action = self._actions[idx]
    self._step_idx += 1
    return (action, {}), self._dummy_state

  def _get_zero_action(self) -> gdmr_types.ActionType:
    """Returns a zero-valued action matching the action spec."""
    if isinstance(self._action_spec, Mapping):
      return {
          k: np.zeros(s.shape, dtype=s.dtype)
          for k, s in self._action_spec.items()
      }
    return np.zeros(self._action_spec.shape, dtype=self._action_spec.dtype)

  @override
  def step_spec(self, timestep_spec: gdmr_types.TimeStepSpec) -> tuple[
      tuple[gdmr_types.ActionSpec, gdmr_types.ExtraOutputSpec],
      gdmr_types.StateSpec,
  ]:
    """Returns the action/state spec."""
    return (self._action_spec, {}), specs.Array(shape=(), dtype=np.float32)  # pytype: disable=bad-return-type


# ============================================================================
# Episode Loading
# ============================================================================


def load_episode(mcap_file: str) -> EpisodeData:
  """Loads an MCAP episode and returns parsed timesteps, actions, and specs.

  Uses spec_utils.specs_from_session() to deserialize specs from the
  session metadata — no duplicated spec code.

  Args:
    mcap_file: Path to the MCAP file or directory.

  Returns:
    An EpisodeData containing timesteps, actions, specs, and session.
  """
  logging.info('Reading session metadata from %s...', mcap_file)
  sessions = mcap_parser_utils.read_session_proto_data(
      mcap_root_path=mcap_file,
      session_topic_name=constants.SESSION_TOPIC_NAME,
  )
  if not sessions:
    raise ValueError(
        f'No session metadata proto found in MCAP file: {mcap_file}'
    )

  session = sessions[0]
  if len(sessions) > 1:
    logging.warning(
        'Found %d sessions in %s — using the first one.',
        len(sessions),
        mcap_file,
    )

  logging.info(
      'Found session with %d observation keys.',
      len(session.policy_environment_metadata.feature_specs.observation),
  )

  # Auto-extract specs from the session metadata using shared spec_utils.
  timestep_spec, action_spec, policy_extra_spec = spec_utils.specs_from_session(
      session
  )

  obs_spec = timestep_spec.observation
  assert isinstance(obs_spec, dict)
  logging.info('Observation keys: %s', list(obs_spec.keys()))
  if isinstance(action_spec, Mapping):
    logging.info('Action keys: %s', list(action_spec.keys()))
  else:
    logging.info('Action spec shape: %s', action_spec.shape)

  # Read raw proto data.
  logging.info('Reading proto data from %s...', mcap_file)
  proto_data = mcap_parser_utils.read_proto_data(
      mcap_root_path=mcap_file,
      timestep_topic_name=constants.TIMESTEP_TOPIC_NAME,
      action_topic_name=constants.ACTION_TOPIC_NAME,
      policy_extra_topic_name=constants.POLICY_EXTRA_TOPIC_NAME,
  )
  logging.info(
      'Read %d timesteps, %d actions.',
      len(proto_data.timesteps),
      len(proto_data.actions),
  )

  # Parse protos into dm_env types.
  timesteps, gt_actions, _ = mcap_parser_utils.parse_examples_to_dm_env_types(
      timestep_spec=timestep_spec,
      action_spec=action_spec,
      policy_extra_spec=policy_extra_spec,
      timesteps_example=proto_data.timesteps,
      actions_example=proto_data.actions,
      policy_extra_example=proto_data.policy_extra,
      step_type_key=constants.STEP_TYPE_KEY,
      observation_key_prefix=constants.OBSERVATION_KEY_PREFIX,
      reward_key=constants.REWARD_KEY,
      discount_key=constants.DISCOUNT_KEY,
      action_key_prefix=constants.ACTION_KEY_PREFIX,
      policy_extra_key_prefix=constants.POLICY_EXTRA_PREFIX,
  )

  return EpisodeData(
      timesteps=timesteps,
      actions=gt_actions,
      timestep_spec=timestep_spec,
      action_spec=action_spec,
      session=session,
  )


# ============================================================================
# MCAP File Discovery
# ============================================================================


def _discover_mcap_files(path: str) -> list[str]:
  """Returns MCAP file(s) from a path (single file or directory).

  Args:
    path: Path to a single .mcap file or a directory containing .mcap files.

  Returns:
    Sorted list of .mcap file paths.

  Raises:
    FileNotFoundError: If path does not exist.
    ValueError: If no .mcap files are found.
  """
  if not os.path.exists(path):
    raise FileNotFoundError(f'Path does not exist: {path}')

  if os.path.isfile(path):
    if not path.endswith('.mcap'):
      raise ValueError(f'File is not an MCAP file: {path}')
    return [path]

  # Directory: find all .mcap files recursively.
  mcap_files = []
  for root, _, files in os.walk(path):
    for f in files:
      if f.endswith('.mcap'):
        mcap_files.append(os.path.join(root, f))

  if not mcap_files:
    raise ValueError(f'No .mcap files found in directory: {path}')

  return sorted(mcap_files)


# ============================================================================
# Output Helpers
# ============================================================================


def _print_episode_summary(mcap_file: str, episode: EpisodeData) -> None:
  """Logs a human-readable summary of a loaded episode."""
  session = episode.session
  control_dt = session.policy_environment_metadata.control_timestep
  hz_str = f'{1.0 / control_dt:.0f} Hz' if control_dt > 0 else 'unknown Hz'

  logging.info('--- Episode Summary ---')
  logging.info('  File: %s', mcap_file)
  logging.info('  Timesteps: %d', len(episode.timesteps))
  logging.info('  Actions: %d', len(episode.actions))
  logging.info('  Control rate: %s (dt=%.4fs)', hz_str, control_dt)

  obs_spec = episode.timestep_spec.observation
  if isinstance(obs_spec, dict):
    logging.info('  Observation keys: %s', list(obs_spec.keys()))
  if isinstance(episode.action_spec, Mapping):
    logging.info('  Action keys: %s', list(episode.action_spec.keys()))
  else:
    logging.info('  Action spec shape: %s', episode.action_spec.shape)


def _print_validation_result(result: data_validator.ValidationResult) -> None:
  """Logs a human-readable validation result."""
  status = 'PASS' if result.passed else 'FAIL'
  logging.info(
      '[%s] %s — %d timesteps, %d actions, %d errors, %d warnings',
      status,
      result.episode_path,
      result.num_timesteps,
      result.num_actions,
      len(result.errors),
      len(result.warnings),
  )
  for finding in result.findings:
    level = (
        'ERROR' if finding.severity == data_validator.Severity.ERROR else 'WARN'
    )
    step_str = (
        f'step {finding.step_idx}'
        if finding.step_idx is not None
        else 'episode'
    )
    logging.info(
        '  [%s] %s | %s: %s',
        level,
        step_str,
        finding.field,
        finding.message,
    )


def _write_json_report(
    results: list[data_validator.ValidationResult], output_path: str
) -> None:
  """Writes validation results as a JSON report."""
  report = {
      'num_episodes': len(results),
      'num_passed': sum(1 for r in results if r.passed),
      'num_failed': sum(1 for r in results if not r.passed),
      'episodes': [],
  }

  for result in results:
    episode_report = {
        'path': result.episode_path,
        'passed': result.passed,
        'num_timesteps': result.num_timesteps,
        'num_actions': result.num_actions,
        'num_errors': len(result.errors),
        'num_warnings': len(result.warnings),
        'findings': [],
    }
    for finding in result.findings:
      episode_report['findings'].append({
          'severity': finding.severity.value,
          'error_type': finding.error_type.value,
          'step_idx': finding.step_idx,
          'field': finding.field,
          'message': finding.message,
      })
    report['episodes'].append(episode_report)

  with open(output_path, 'w') as f:
    json.dump(report, f, indent=2)
  logging.info('Wrote JSON report to %s', output_path)


# ============================================================================
# Validate Mode
# ============================================================================


def _run_validation(mcap_files: list[str]) -> None:
  """Runs offline validation on one or more MCAP episode files.

  Loads each episode, runs the data validator, prints results,
  and optionally writes a JSON report.

  Args:
    mcap_files: List of paths to MCAP files to validate.
  """
  validator = data_validator.EpisodeValidator()
  results = []

  for mcap_file in mcap_files:
    logging.info('Validating %s...', mcap_file)
    try:
      episode = load_episode(mcap_file)
    except (
        ValueError,
        FileNotFoundError,
        IndexError,
        TypeError,
        KeyError,
    ) as e:
      logging.error('Failed to load %s: %s', mcap_file, e)
      # Create a failed result for unloadable episodes.
      results.append(
          data_validator.ValidationResult(
              episode_path=mcap_file,
              num_timesteps=0,
              num_actions=0,
              findings=[
                  data_validator.ValidationFinding(
                      severity=data_validator.Severity.ERROR,
                      error_type=data_validator.ErrorType.EMPTY_EPISODE,
                      step_idx=None,
                      field='episode',
                      message=f'Failed to load episode: {e}',
                  )
              ],
          )
      )
      continue

    _print_episode_summary(mcap_file, episode)

    result = validator.validate_episode(
        episode_path=mcap_file,
        timesteps=episode.timesteps,
        actions=episode.actions,
        timestep_spec=episode.timestep_spec,
        action_spec=episode.action_spec,
    )
    results.append(result)
    _print_validation_result(result)

  # Print summary.
  num_passed = sum(1 for r in results if r.passed)
  num_failed = len(results) - num_passed
  logging.info('--- Validation Summary ---')
  logging.info(
      '%d/%d episodes passed, %d failed.',
      num_passed,
      len(results),
      num_failed,
  )

  # Write JSON if requested.
  if _OUTPUT.value:
    _write_json_report(results, _OUTPUT.value)

  # Exit with error code if any failures.
  has_errors = any(r.errors for r in results)
  has_warnings = any(r.warnings for r in results)

  if _STRICT.value and (has_errors or has_warnings):
    sys.exit(1)
  elif has_errors:
    sys.exit(1)


# ============================================================================
# Replay Mode
# ============================================================================


def _run_hardware_replay(mcap_file: str) -> None:
  """Runs hardware replay — sends logged actions to the robot.

  Args:
    mcap_file: Path to the MCAP file to replay.
  """
  episode = load_episode(mcap_file)
  _print_episode_summary(mcap_file, episode)

  # Determine control timestep.
  if _CONTROL_HZ.value is not None:
    control_dt = 1.0 / _CONTROL_HZ.value
    logging.info(
        'Using --control_hz=%.1f (dt=%.4fs)', _CONTROL_HZ.value, control_dt
    )
  else:
    control_dt = episode.session.policy_environment_metadata.control_timestep
    if control_dt <= 0:
      control_dt = 0.02  # Default to 50 Hz.
      logging.warning(
          'No control_timestep in session metadata, defaulting to %.3fs (50'
          ' Hz).',
          control_dt,
      )

  # Dynamically import the environment factory.
  try:
    module = importlib.import_module(_ENV_MODULE.value)
    factory_fn = getattr(module, _ENV_FUNCTION.value)
  except (ModuleNotFoundError, AttributeError) as e:
    raise app.UsageError(
        'Could not load environment factory'
        f' {_ENV_MODULE.value}.{_ENV_FUNCTION.value}: {e}'
    ) from e

  logging.info(
      'Creating robot environment via %s.%s...',
      _ENV_MODULE.value,
      _ENV_FUNCTION.value,
  )
  environment = factory_fn()

  policy = MCAPReplayPolicy(
      logged_actions=episode.actions,
      timestep_spec=episode.timestep_spec,
      action_spec=episode.action_spec,
  )
  state = policy.initial_state()

  try:
    logging.info('Resetting environment...')
    timestep = environment.reset()

    logging.info(
        'Replaying %d actions at %.0f Hz...',
        len(episode.actions),
        1.0 / control_dt,
    )

    step_count = 0
    interactive_mode = _INTERACTIVE.value
    while not timestep.last() and step_count < len(episode.actions):
      step_start = time.monotonic()

      # Interactive chunk-stepping (handled outside the policy).
      if (
          interactive_mode
          and step_count > 0
          and step_count % _CHUNK_SIZE.value == 0
      ):
        logging.info(
            'Completed chunk of %d steps (%d/%d total). '
            'Press Enter to continue, "c" for continuous, "q" to quit.',
            _CHUNK_SIZE.value,
            step_count,
            len(episode.actions),
        )
        try:
          choice = input('> ').strip().lower()
          if choice == 'q':
            logging.info('User requested abort.')
            break
          elif choice == 'c':
            interactive_mode = False
            logging.info('Switching to continuous replay mode.')
        except EOFError:
          logging.warning(
              'EOF encountered on stdin; disabling interactive mode.'
          )
          interactive_mode = False

      # Step the mock policy.
      (action, _), state = policy.step(timestep, state)

      # Clip actions to spec bounds for safety.
      action = _clip_action_to_bounds(action, episode.action_spec)

      timestep = environment.step(action)
      step_count += 1

      elapsed = time.monotonic() - step_start
      sleep_time = max(0.0, control_dt - elapsed)
      if sleep_time > 0:
        time.sleep(sleep_time)

    logging.info('Replay finished after %d steps.', step_count)

  except KeyboardInterrupt:
    logging.info('Interrupted by user.')
  finally:
    logging.info('Closing environment...')
    environment.close()


def _clip_action_to_bounds(
    action: gdmr_types.ActionType,
    action_spec: specs.Array | Mapping[str, specs.Array],
) -> gdmr_types.ActionType:
  """Clips action values to spec bounds for safety.

  Args:
    action: The action to clip.
    action_spec: The spec defining bounds.

  Returns:
    Clipped action.
  """
  if isinstance(action, Mapping) and isinstance(action_spec, Mapping):
    clipped = {}
    for key, val in action.items():
      spec = action_spec.get(key)
      if spec is not None and isinstance(spec, specs.BoundedArray):
        clipped[key] = np.clip(val, spec.minimum, spec.maximum)
      else:
        clipped[key] = val
    return clipped
  elif isinstance(action_spec, specs.BoundedArray) and not isinstance(
      action, Mapping
  ):
    return np.clip(action, action_spec.minimum, action_spec.maximum)
  return action


# ============================================================================
# Main
# ============================================================================


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if not _MCAP_FILE.value:
    raise app.UsageError('--mcap_file is required.')

  mcap_files = _discover_mcap_files(_MCAP_FILE.value)
  logging.info('Discovered %d MCAP file(s).', len(mcap_files))

  if _MODE.value == 'validate':
    _run_validation(mcap_files)
  elif _MODE.value == 'replay':
    if not _ENV_MODULE.value:
      raise ValueError('--env_module is required for replay mode.')
    if len(mcap_files) > 1:
      logging.warning(
          'Replay mode only supports a single MCAP file. '
          'Using the first file: %s',
          mcap_files[0],
      )
    _run_hardware_replay(mcap_files[0])


if __name__ == '__main__':
  app.run(main)
