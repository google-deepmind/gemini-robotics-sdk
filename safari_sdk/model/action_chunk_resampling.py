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

"""Functions for resampling action chunks."""

import numpy as np


def apply_linear_resampling(
    action_chunk: np.ndarray, resample_factor: float
) -> np.ndarray:
  """Resamples an action chunk using piecewise linear interpolation.

  This resampling maps the original trajectory to a new set of steps by
  uniformly sampling between the first and last steps of the original
  trajectory (or returning only the first step if the resampled length is 1).

  If chunk_length <= 1, the original chunk is returned early without changes.

  Args:
    action_chunk: 2D array of shape (chunk_length, action_dim).
    resample_factor: The factor by which to resample the action chunk; the
      output length will be ceil(chunk_length / resample_factor).

  Returns:
    2D array of shape (resampled_length, action_dim) where
    resampled_length = ceil(chunk_length / resample_factor).
  """
  if action_chunk.ndim != 2:
    raise ValueError(
        f"action_chunk must be 2D, but got shape {action_chunk.shape}"
    )
  source_chunk_length = action_chunk.shape[0]
  if source_chunk_length <= 1:
    return action_chunk
  resampled_length = int(np.ceil(source_chunk_length / resample_factor))
  dtype = (
      action_chunk.dtype
      if np.issubdtype(action_chunk.dtype, np.floating)
      else np.float64
  )
  source_steps = np.arange(source_chunk_length, dtype=dtype)
  resampled_steps = np.linspace(
      0, source_chunk_length - 1, resampled_length, dtype=dtype
  )
  return np.column_stack([
      np.interp(resampled_steps, source_steps, action_chunk[:, j])
      for j in range(action_chunk.shape[1])
  ]).astype(dtype, copy=False)


def resampled_to_min_source_length(
    resampled_length: int, resample_factor: float
) -> int:
  """Minimum number of source actions needed for a target resampled length.

  Given a desired number of resampled actions and a resampling factor, returns
  the minimum number of source actions needed to produce the desired resampled
  length keeping the first and last actions from the source fixed.

  Example: resampled_length = 3, resample_factor = 2.0 -> 5 actions are needed.
  Example: resampled_length = 5, resample_factor = 1.5 -> 7 actions are needed.

  Args:
    resampled_length: The number of actions to produce after resampling.
    resample_factor: The factor by which to resample the action chunk.

  Returns:
    Minimum number of actions needed to produce the desired resampled length.
  """
  if resampled_length <= 0:
    return 0
  return int(np.floor((resampled_length - 1) * resample_factor)) + 1
