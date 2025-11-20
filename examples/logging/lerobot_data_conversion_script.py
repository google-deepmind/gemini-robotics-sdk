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

"""A script to convert LeRobot datasets to MCAP format."""

from collections.abc import Sequence
import datetime
import os
import re
import shutil

from absl import app
from absl import flags
from absl import logging
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from safari_sdk.logging.python import mcap_lerobot_logger

_DATASET_NAME = flags.DEFINE_string(
    'lerobot_dataset_name',
    default=None,
    help=(
        'Name of the LeRobot dataset to load. e.g. '
        'lerobot/aloha_static_cups_open'
    ),
    required=True,
)
_TASK_ID = flags.DEFINE_string(
    'task_id',
    'lerobot_test_task',
    'Task ID for the logger, used to identify data for later finetuning.',
)

_EPISODE_START_TIME_NS = flags.DEFINE_multi_string(
    'episode_start_time_ns',
    default=None,
    help=(
        'Start time of the episode, in nanoseconds since UNIX epoch,'
        ' in the format '
        '<lerobot_episode_id>:<timestamp in nanos since unix epoch>. It '
        'can be specified multiple times.'
    ),
    required=True,
)

_OUTPUT_DIRECTORY = flags.DEFINE_string(
    'output_directory',
    '/tmp/converted_lerobot_log',
    'Directory to save MCAP files.',
)

_NUM_EPISODES = flags.DEFINE_integer(
    'num_episodes',
    0,
    'Number of episodes to process. Default value 0 means all episodes.',
)


_PROPRIO_KEY = flags.DEFINE_string(
    'proprio_key',
    'state',
    'The key of the proprio data in the observation.',
)

_MAX_WORKERS = flags.DEFINE_integer(
    'max_workers',
    200,
    'Maximum number of threads for parallel processing and logging.',
)


def validate_episode_start_time_ns_format(values):
  seconds_in_a_year = datetime.timedelta(days=365).total_seconds()
  nanoseconds_per_second = 1e9
  earliest_expected_time = (
      (2000 - 1970) * seconds_in_a_year * nanoseconds_per_second
  )

  def get_time(text):
    return int(text.split(':')[1])

  return all(re.fullmatch(r'[0-9]+:[0-9]+', value) for value in values) and all(
      get_time(value) > earliest_expected_time for value in values
  )


flags.register_validator(
    'episode_start_time_ns',
    validate_episode_start_time_ns_format,
    message=(
        'episode_start_time_ns must be specified in '
        'the format <episode id>:<epoch time in ns>, and '
        'the time needs to be in _nanoseconds_.'
    ),
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  dataset_folder_name = _DATASET_NAME.value.split('/')[-1]
  output_directory = os.path.join(_OUTPUT_DIRECTORY.value, dataset_folder_name)

  if os.path.exists(output_directory):
    shutil.rmtree(output_directory)
  os.makedirs(output_directory, exist_ok=True)

  logging.info('Logs will be written to: %s', output_directory)

  logging.info('--- Loading and processing  from "%s" ---', _DATASET_NAME.value)
  dataset = LeRobotDataset(_DATASET_NAME.value)

  episode_start_time_ns = dict(
      [map(int, item.split(':')) for item in _EPISODE_START_TIME_NS.value]
  )

  mcap_lerobot_logger.convert_lerobot_data_to_mcap(
      dataset=dataset,
      task_id=_TASK_ID.value,
      output_directory=output_directory,
      proprioceptive_observation_keys=[_PROPRIO_KEY.value],
      episodes_limit=_NUM_EPISODES.value,
      max_workers=_MAX_WORKERS.value,
      episode_start_timestamps_ns=episode_start_time_ns,
  )

  logging.info('\n--- Script finished successfully!')


if __name__ == '__main__':
  app.run(main)
