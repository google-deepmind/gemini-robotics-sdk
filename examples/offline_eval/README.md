# Offline Evaluation Tool

Standalone offline evaluation for Flywheel robotics model checkpoints. Replays
MCAP demonstration episodes through a served model and computes
action-prediction metrics.

## Prerequisites

```bash
pip install safari_sdk
pip install -r requirements.txt
```

> **Note:** Run all commands from the repository root directory (the directory
> containing `examples/`).

## Quick Start

### 1. Start the model server

```bash
flywheel-cli serve \
  --training_recipe=gemini_robotics_on_device_v1 \
  --model_checkpoint_path=/path/to/checkpoint.chkpt
```

### 2. Run evaluation

```bash
python examples/offline_eval/offline_eval.py \
  --dataset_path=/path/to/mcap_episodes/ \
  --checkpoint_path=/path/to/checkpoint.chkpt
```

All task metadata (camera names, proprioception keys, task instruction) is
**auto-detected** from the MCAP session data. No manual configuration needed.

## Metrics

| Metric                  | Description                                       |
| ----------------------- | ------------------------------------------------- |
| **Chunk MSE** (primary) | Mean squared error between the predicted chunk of |
:                         : 50 actions and the actual chunk of 50 actions,    :
:                         : averaged over all joints and timesteps.           :
| Single Step MSE         | Mean squared error between the first predicted    |
:                         : action and the actual next action, averaged over  :
:                         : all joints and timesteps.                         :
| Per-Joint MSE           | Step-level MSE broken down by joint index.        |

## Flags

### Evaluation

| Flag                         | Default           | Description               |
| ---------------------------- | ----------------- | ------------------------- |
| `--dataset_path`             | *required*        | Path to MCAP episode data |
:                              :                   : (directory or single      :
:                              :                   : file).                    :
| `--checkpoint_path`          | `""`              | Path to checkpoint (for   |
:                              :                   : labeling results).        :
| `--offline_eval_port`        | `60061`           | Port where Docker GROD    |
:                              :                   : server is running.        :
| `--task_instruction`         | auto              | Override task instruction |
:                              :                   : string.                   :
| `--num_episodes`             | `0` (all)         | Max episodes to evaluate. |
| `--output_dir`               | `./eval_results/` | Output directory.         |
| `--min_replan_interval`      | `50`              | Action chunk size (steps  |
:                              :                   : between model             :
:                              :                   : re-queries).              :
| `--eval_image_keys`          | auto              | Override camera keys.     |
| `--eval_proprioception_keys` | auto              | Override proprioception   |
:                              :                   : keys.                     :
| `--action_dim`               | auto              | Override action           |
:                              :                   : dimensionality.           :

### Robustness

| Flag                         | Default | Description                     |
| ---------------------------- | ------- | ------------------------------- |
| `--step_timeout`             | `30`    | Per-step inference timeout in   |
:                              :         : seconds.                        :
| `--max_consecutive_timeouts` | `3`     | Max consecutive timeouts before |
:                              :         : aborting an episode.            :

## Outputs

-   `summary.json` — Per-episode + aggregate metrics.
-   `episode_N_actions.png` — Joint-grid plot (predicted vs ground-truth).
-   `episode_N_data.npz` — Raw predicted/ground-truth action arrays.
