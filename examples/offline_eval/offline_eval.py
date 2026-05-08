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

r"""Public offline evaluation for Flywheel robotics models.

NOTE: This is an example script for offline evaluation and is not officially supported by the Safari SDK team.

Replays logged MCAP demonstration episodes through a served model checkpoint,
compares predicted actions against ground-truth, and computes aggregate metrics.

Prerequisites:
  1. Start the Docker model server:
     flywheel-cli serve \
       --training_recipe=gemini_robotics_on_device_v1 \
       --model_checkpoint_path=/path/to/checkpoint.chkpt

  2. Run this evaluation:
     python offline_eval.py \
       --dataset_path=/path/to/mcap_episodes/ \
       --checkpoint_path=/path/to/checkpoint.chkpt

All task metadata (image_keys, proprio_keys, task_instruction) is auto-detected
from the MCAP session data. Override with CLI flags if needed.
"""

import json
import os
import queue
import threading
import time
from typing import Any

from absl import app
from absl import flags
from absl import logging
import dm_env
from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np

from examples.offline_eval import mcap_loader
from safari_sdk.model import constants
from safari_sdk.model import gemini_robotics_policy

# --- Required ---
_DATASET_PATH = flags.DEFINE_string(
    'dataset_path',
    None,
    'Path to MCAP episode data (directory or single .mcap file).',
)

# --- Server ---
_SERVE_PORT = flags.DEFINE_integer(
    'offline_eval_port',
    60061,
    'Port where Docker GROD server is running.',
)
_CHECKPOINT_PATH = flags.DEFINE_string(
    'checkpoint_path',
    '',
    'Path to the checkpoint file (for labeling results).',
)

# --- Evaluation ---
_TASK_INSTRUCTION = flags.DEFINE_string(
    'task_instruction',
    None,
    'Override task instruction. Auto-detected from MCAP if not set.',
)
_NUM_EPISODES = flags.DEFINE_integer(
    'num_episodes',
    0,
    'Max episodes to evaluate. 0 = all found.',
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    './eval_results/',
    'Directory for output plots and summary.',
)
_MIN_REPLAN_INTERVAL = flags.DEFINE_integer(
    'min_replan_interval',
    50,
    'Steps between model re-queries (= action chunk size).',
)
_IMAGE_KEYS = flags.DEFINE_list(
    'eval_image_keys',
    None,
    'Override image observation keys (comma-separated).',
)
_PROPRIOCEPTION_KEYS = flags.DEFINE_list(
    'eval_proprioception_keys',
    None,
    'Override proprioceptive observation keys (comma-separated).',
)
_ACTION_DIM = flags.DEFINE_integer(
    'action_dim',
    0,
    'Override action dimensionality. 0 = auto-detect.',
)

_STEP_TIMEOUT = flags.DEFINE_integer(
    'step_timeout',
    30,
    'Per-step inference timeout in seconds.',
)
_MAX_CONSECUTIVE_TIMEOUTS = flags.DEFINE_integer(
    'max_consecutive_timeouts',
    3,
    'Max consecutive timeouts before aborting episode.',
)

_INSTRUCTION_KEY = 'instruction'


# ---------- TimeStepSpec Construction ----------------------------------------
def _build_timestep_spec(
    episode: mcap_loader.Episode,
    image_keys: list[str],
    proprio_keys: list[str],
) -> gdmr_types.TimeStepSpec:
  """Builds a TimeStepSpec from the first observation of an episode.

  Mirrors the pattern in gemini_robotics_policy_example.py — derives all
  shapes directly from the loaded data.

  Args:
    episode: A loaded Episode to derive shapes from.
    image_keys: Camera observation keys.
    proprio_keys: Proprioceptive observation keys.

  Returns:
    A TimeStepSpec for the policy.

  Raises:
    KeyError: If an image or proprio key is not found in episode observations.
  """
  first_obs = episode.observations[0]
  observation_spec = {}

  for cam_key in image_keys:
    if cam_key not in first_obs:
      raise KeyError(
          f"Image key '{cam_key}' not found in episode observations. "
          f'Available keys: {list(first_obs.keys())}'
      )
    h, w, c = first_obs[cam_key].shape
    observation_spec[cam_key] = specs.BoundedArray(
        shape=(h, w, c),
        dtype=np.uint8,
        minimum=0,
        maximum=255,
    )

  for joint_key in proprio_keys:
    if joint_key not in first_obs:
      raise KeyError(
          f"Proprio key '{joint_key}' not found in episode observations. "
          f'Available keys: {list(first_obs.keys())}'
      )
    dim = len(first_obs[joint_key])
    observation_spec[joint_key] = specs.Array(
        shape=(1, dim),
        dtype=np.float32,
    )

  observation_spec[_INSTRUCTION_KEY] = specs.StringArray(shape=())

  return gdmr_types.TimeStepSpec(
      step_type=gdmr_types.STEP_TYPE_SPEC,
      reward={},
      discount={},
      observation=observation_spec,
  )


def _build_observation(
    episode: mcap_loader.Episode,
    step_idx: int,
    image_keys: list[str],
    proprio_keys: list[str],
) -> dict[str, Any]:
  """Build an observation dict from the episode at a given step.

  Args:
    episode: Source episode.
    step_idx: Step index (clamped to valid range).
    image_keys: Camera observation keys.
    proprio_keys: Proprioceptive observation keys.

  Returns:
    Observation dict compatible with GeminiRoboticsPolicy.

  Raises:
    KeyError: If a required image or proprio key is missing from the
      observation at the given step.
  """
  idx = min(step_idx, episode.num_steps - 1)
  raw = episode.observations[idx]
  obs = {}

  for key in image_keys:
    if key not in raw:
      raise KeyError(
          f"Image key '{key}' not found in observation at step {step_idx}. "
          f'Available keys: {list(raw.keys())}'
      )
    obs[key] = raw[key]

  for key in proprio_keys:
    if key not in raw:
      raise KeyError(
          f"Proprio key '{key}' not found in observation at step {step_idx}. "
          f'Available keys: {list(raw.keys())}'
      )
    val = np.array(raw[key], dtype=np.float32)
    obs[key] = val.reshape(1, -1)

  obs[_INSTRUCTION_KEY] = np.array(
      episode.task_instruction,
      dtype=np.object_,
  )
  return obs


# ---------- Metrics -----------------------------------------------------------
def compute_episode_metrics(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    chunk_boundaries: list[int],
    chunk_size: int = 50,
) -> dict[str, Any]:
  """Compute metrics for a single episode.

  Args:
    predicted: Predicted actions, shape (T, action_dim).
    ground_truth: Ground-truth actions, shape (T, action_dim).
    chunk_boundaries: Step indices where model was re-queried.
    chunk_size: Size of each chunk for chunk MSE.

  Returns:
    Dict of metric values.
  """
  # Chunk MSE (standard, unweighted mean over chunk steps)
  chunk_mses = []
  for start in chunk_boundaries:
    end = min(start + chunk_size, len(predicted))
    if end - start < chunk_size:
      continue  # skip incomplete final chunk
    pred_chunk = predicted[start:end]
    gt_chunk = ground_truth[start:end]
    per_step_mse = np.mean((pred_chunk - gt_chunk) ** 2, axis=1)
    chunk_mses.append(float(np.mean(per_step_mse)))

  chunk_mse = float(np.mean(chunk_mses)) if chunk_mses else float('nan')

  # Step MSE — complementary sanity-check; chunk MSE is the primary metric.
  step_mse = float(np.mean((predicted - ground_truth) ** 2))

  # Per-Joint MSE
  per_joint_mse = np.mean(
      (predicted - ground_truth) ** 2,
      axis=0,
  ).tolist()

  return {
      'chunk_mse': chunk_mse,
      'step_mse': step_mse,
      'per_joint_mse': per_joint_mse,
      'num_steps': len(predicted),
      'num_complete_chunks': len(chunk_mses),
  }


def aggregate_metrics(
    all_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
  """Aggregate metrics across episodes.

  Args:
    all_metrics: List of per-episode metric dicts.

  Returns:
    Aggregated metric dict with mean and std.
  """
  chunk_mses = [
      m['chunk_mse'] for m in all_metrics if not np.isnan(m['chunk_mse'])
  ]
  step_mses = [m['step_mse'] for m in all_metrics]
  per_joints = np.array(
      [m['per_joint_mse'] for m in all_metrics],
  )

  return {
      'chunk_mse': {
          'mean': float(np.mean(chunk_mses)) if chunk_mses else float('nan'),
          'std': float(np.std(chunk_mses)) if chunk_mses else float('nan'),
      },
      'step_mse': {
          'mean': float(np.mean(step_mses)) if step_mses else float('nan'),
          'std': float(np.std(step_mses)) if step_mses else float('nan'),
      },
      'per_joint_mse': {
          'mean': per_joints.mean(axis=0).tolist() if per_joints.size else [],
          'std': per_joints.std(axis=0).tolist() if per_joints.size else [],
      },
  }


# ---------- Plotting ----------------------------------------------------------
def plot_joint_grid(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    ep_idx: int,
    metrics: dict[str, Any],
    output_path: str,
    action_dim: int,
    checkpoint_label: str,
    chunk_size: int,
) -> None:
  """Plot predicted vs ground-truth per joint in a grid layout.

  Args:
    predicted: Predicted actions, shape (T, action_dim).
    ground_truth: Ground-truth actions, shape (T, action_dim).
    ep_idx: Episode index.
    metrics: Computed metrics for the episode.
    output_path: Where to save the plot.
    action_dim: Number of action dimensions.
    checkpoint_label: Label for the checkpoint.
    chunk_size: Steps between model re-queries.
  """
  n_rows = max((action_dim + 1) // 2, 1)
  fig, axes = plt.subplots(
      n_rows,
      2,
      figsize=(16, 2.5 * n_rows),
      sharex=True,
  )
  if n_rows == 1:
    axes = axes.reshape(1, 2)

  fig.suptitle(
      f'Episode {ep_idx} — Ckpt: {checkpoint_label}\n'
      f'ChunkMSE={metrics["chunk_mse"]:.6f} · '
      f'StepMSE={metrics["step_mse"]:.6f} · '
      f'Steps={len(predicted)}',
      fontsize=13,
  )

  colors = [
      'red',
      'blue',
      'green',
      'orange',
      'purple',
      'brown',
      'pink',
      'gray',
      'olive',
      'cyan',
  ]
  for i in range(action_dim):
    row, col = i // 2, i % 2
    ax = axes[row, col]

    # Color segments by chunk
    for start in range(0, len(predicted), chunk_size):
      end = min(start + chunk_size, len(predicted))
      color = colors[(start // chunk_size) % len(colors)]
      ax.plot(
          np.arange(start, end),
          predicted[start:end, i],
          color=color,
          marker='o',
          markersize=2,
          linewidth=1.5,
          alpha=0.8,
          label='Predicted' if start == 0 else '',
      )
    ax.plot(
        ground_truth[:, i],
        label='Ground Truth',
        marker='x',
        markersize=2,
        linewidth=1.5,
        alpha=0.8,
        color='black',
    )

    # Vertical lines at chunk boundaries
    for start in range(0, len(predicted), chunk_size):
      ax.axvline(
          x=start,
          color='gray',
          linestyle='--',
          alpha=0.3,
          linewidth=0.5,
      )

    ax.set_ylabel(f'Joint {i}')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

  # Hide unused axes
  for i in range(action_dim, n_rows * 2):
    axes[i // 2, i % 2].set_visible(False)

  axes[-1, 0].set_xlabel('Time Step')
  if action_dim > 1:
    axes[-1, 1].set_xlabel('Time Step')
  plt.tight_layout(rect=[0, 0.03, 1, 0.93])
  plt.savefig(output_path, dpi=300, bbox_inches='tight')
  plt.close(fig)


# ---------- Eval Loop ---------------------------------------------------------
def evaluate_checkpoint(
    episodes: list[mcap_loader.Episode],
    metadata: mcap_loader.DetectedMetadata,
) -> dict[str, Any]:
  """Run offline eval across all episodes.

  Uses the same pattern as gemini_robotics_policy_example.py:
  build a TimeStepSpec, then feed dm_env.transition() timesteps
  directly to policy.step().

  Args:
    episodes: List of loaded Episodes.
    metadata: Auto-detected (or overridden) metadata.

  Returns:
    Summary dict with per-episode and aggregate metrics.
  """
  image_keys = metadata.image_keys
  proprio_keys = metadata.proprio_keys

  # Build timestep spec from first episode
  ts_spec = _build_timestep_spec(episodes[0], image_keys, proprio_keys)

  # Create policy
  logging.info(
      'Creating GeminiRoboticsPolicy on port %d...',
      _SERVE_PORT.value,
  )
  policy = gemini_robotics_policy.GeminiRoboticsPolicy(
      serve_id=f'grpc://localhost:{_SERVE_PORT.value}',
      task_instruction_key=_INSTRUCTION_KEY,
      image_observation_keys=tuple(image_keys),
      proprioceptive_observation_keys=tuple(proprio_keys),
      inference_mode=constants.InferenceMode.SYNCHRONOUS,
      robotics_api_connection=(constants.RoboticsApiConnectionType.LOCAL),
      min_replan_interval=_MIN_REPLAN_INTERVAL.value,
  )
  policy.step_spec(ts_spec)

  os.makedirs(_OUTPUT_DIR.value, exist_ok=True)
  all_episode_metrics = []
  checkpoint_label = os.path.basename(_CHECKPOINT_PATH.value) or 'unknown'

  for ep_idx, episode in enumerate(episodes):
    logging.info(
        'Episode %d/%d (%d steps)',
        ep_idx + 1,
        len(episodes),
        episode.num_steps,
    )
    ep_start = time.time()
    state = policy.initial_state()

    predicted_actions = []
    ground_truth_actions = []
    chunk_boundaries = [0]
    consecutive_timeouts = 0

    for step in range(episode.num_steps - 1):
      if step > 0 and step % _MIN_REPLAN_INTERVAL.value == 0:
        chunk_boundaries.append(step)

      if step % 50 == 0:
        elapsed = time.time() - ep_start
        logging.info(
            '  Step %d/%d (%.1fs)',
            step + 1,
            episode.num_steps,
            elapsed,
        )

      obs = _build_observation(
          episode,
          step,
          image_keys,
          proprio_keys,
      )
      if step == 0:
        timestep = dm_env.restart(observation=obs)
      else:
        timestep = dm_env.transition(
            reward=0.0,
            discount=1.0,
            observation=obs,
        )

      # Inference with timeout
      result_q = queue.Queue()

      def _run(q, ts, st):
        try:
          r = policy.step(ts, st)
          q.put(('ok', r))
        except Exception as e:  # pylint: disable=broad-except
          q.put(('error', e))

      t = threading.Thread(
          target=_run,
          args=(result_q, timestep, state),
          daemon=True,
      )
      t.start()
      t.join(timeout=_STEP_TIMEOUT.value)

      if t.is_alive():
        consecutive_timeouts += 1
        logging.warning(
            'Step %d: timeout (%d consecutive)',
            step,
            consecutive_timeouts,
        )
        if consecutive_timeouts >= _MAX_CONSECUTIVE_TIMEOUTS.value:
          logging.error(
              'Aborting episode: %d consecutive timeouts',
              _MAX_CONSECUTIVE_TIMEOUTS.value,
          )
          break
        action = (
            predicted_actions[-1]
            if predicted_actions
            else np.zeros(metadata.action_dim)
        )
      else:
        status, payload = result_q.get_nowait()
        if status == 'ok':
          (action, _), state = payload
          consecutive_timeouts = 0
        else:
          consecutive_timeouts += 1
          logging.warning('Step %d: error: %s', step, payload)
          if consecutive_timeouts >= _MAX_CONSECUTIVE_TIMEOUTS.value:
            break
          action = (
              predicted_actions[-1]
              if predicted_actions
              else np.zeros(metadata.action_dim)
          )

      predicted_actions.append(np.array(action))
      gt_idx = min(step, episode.num_steps - 1)
      ground_truth_actions.append(episode.actions[gt_idx])

    if not predicted_actions:
      logging.warning(
          'Episode %d produced no predictions, skipping.',
          ep_idx,
      )
      continue

    pred = np.array(predicted_actions)
    gt = np.array(ground_truth_actions)

    ep_metrics = compute_episode_metrics(
        predicted=pred,
        ground_truth=gt,
        chunk_boundaries=chunk_boundaries,
        chunk_size=_MIN_REPLAN_INTERVAL.value,
    )
    ep_metrics['episode_idx'] = ep_idx

    # Save raw data
    np.savez(
        os.path.join(
            _OUTPUT_DIR.value,
            f'episode_{ep_idx}_data.npz',
        ),
        predicted_actions=pred,
        ground_truth_actions=gt,
        chunk_boundaries=np.array(chunk_boundaries),
    )

    # Plot
    plot_joint_grid(
        pred,
        gt,
        ep_idx,
        ep_metrics,
        os.path.join(
            _OUTPUT_DIR.value,
            f'episode_{ep_idx}_actions.png',
        ),
        action_dim=metadata.action_dim,
        checkpoint_label=checkpoint_label,
        chunk_size=_MIN_REPLAN_INTERVAL.value,
    )

    all_episode_metrics.append(ep_metrics)
    elapsed = time.time() - ep_start
    logging.info(
        'Episode %d: ChunkMSE=%.6f StepMSE=%.6f (%.1fs)',
        ep_idx,
        ep_metrics['chunk_mse'],
        ep_metrics['step_mse'],
        elapsed,
    )

  # Aggregate
  agg = aggregate_metrics(all_episode_metrics)
  summary = {
      'checkpoint': _CHECKPOINT_PATH.value or 'unknown',
      'task_id': metadata.task_id,
      'task_instruction': metadata.task_instruction,
      'num_episodes': len(all_episode_metrics),
      'min_replan_interval': _MIN_REPLAN_INTERVAL.value,
      'aggregate': agg,
      'episodes': all_episode_metrics,
  }

  summary_path = os.path.join(_OUTPUT_DIR.value, 'summary.json')
  with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

  # Print results
  print('\n' + '=' * 60)
  print(f'RESULTS — {len(all_episode_metrics)} episodes')
  print('=' * 60)
  for m in all_episode_metrics:
    print(
        f'  Ep {m["episode_idx"]}: '
        f'ChunkMSE={m["chunk_mse"]:.6f} '
        f'StepMSE={m["step_mse"]:.6f}'
    )
  print('-' * 60)
  print(
      '  Chunk MSE:  '
      f'{agg["chunk_mse"]["mean"]:.6f} '
      f'± {agg["chunk_mse"]["std"]:.6f}  ← PRIMARY'
  )
  print(
      '  Step MSE:   '
      f'{agg["step_mse"]["mean"]:.6f} '
      f'± {agg["step_mse"]["std"]:.6f}'
  )

  per_joint_mean = agg['per_joint_mse']['mean']
  if per_joint_mean:
    print(f'  Per-Joint:  {[f"{v:.5f}" for v in per_joint_mean]}')
  else:
    print('  Per-Joint:  (no data)')
  print('=' * 60)
  print(f'Summary: {summary_path}')

  return summary


# ---------- Main --------------------------------------------------------------
def main(argv: list[str]) -> None:
  del argv

  if not _DATASET_PATH.value:
    raise app.UsageError('Flag --dataset_path must be specified.')

  # Load data
  logging.info('Loading MCAP data from: %s', _DATASET_PATH.value)
  episodes, metadata = mcap_loader.load_mcap_episodes(
      dataset_path=_DATASET_PATH.value,
      max_episodes=(_NUM_EPISODES.value if _NUM_EPISODES.value > 0 else None),
      image_keys=_IMAGE_KEYS.value,
      proprio_keys=_PROPRIOCEPTION_KEYS.value,
      task_instruction_override=_TASK_INSTRUCTION.value,
  )

  # Apply action_dim override
  if _ACTION_DIM.value > 0:
    metadata.action_dim = _ACTION_DIM.value

  if not metadata.action_dim:
    raise ValueError(
        'Could not detect action_dim. Provide --action_dim.',
    )
  if metadata.task_instruction is None:
    raise ValueError(
        'Could not detect task instruction. Provide --task_instruction.',
    )

  logging.info('--- Auto-Detected Configuration ---')
  logging.info('  Task ID:       %s', metadata.task_id)
  logging.info('  Instruction:   %s', metadata.task_instruction)
  logging.info('  Image keys:    %s', metadata.image_keys)
  logging.info('  Proprio keys:  %s', metadata.proprio_keys)
  logging.info('  Action dim:    %s', metadata.action_dim)
  logging.info('  Episodes:      %d', len(episodes))
  logging.info('  Chunk size:    %d', _MIN_REPLAN_INTERVAL.value)
  logging.info('  Output dir:    %s', _OUTPUT_DIR.value)

  evaluate_checkpoint(episodes, metadata)


if __name__ == '__main__':
  app.run(main)
