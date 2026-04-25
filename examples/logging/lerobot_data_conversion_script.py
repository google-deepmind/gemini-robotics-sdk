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
import lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from packaging import version

from safari_sdk.logging.python import mcap_lerobot_logger

# Check lerobot version is >= 0.4.0
_MIN_LEROBOT_VERSION = '0.4.0'
if version.parse(lerobot.__version__) < version.parse(_MIN_LEROBOT_VERSION):
  raise ImportError(
      f'This script requires lerobot version >= {_MIN_LEROBOT_VERSION}, '
      f'but found version {lerobot.__version__}. Please upgrade lerobot: '
      f'pip install --upgrade lerobot>={_MIN_LEROBOT_VERSION}'
  )


_ONE_MINUTE_NS = 60 * 1_000_000_000
_LEROBOT_TIMESTAMP_DELTA_KEY = 'timestamp'

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

_DATASET_START_TIME = flags.DEFINE_string(
    'dataset_start_time',
    default=None,
    help=(
        'Start time of the dataset in ISO 8601 format (e.g.'
        ' 2020-01-23T13:22:34.123). If not provided, datetime.now() will be'
        ' used.'
    ),
    required=False,
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
    required=False,
)


def _check_dataset_time_and_episode_start_time_ns_are_mutually_exclusive(
    flags_dict,
):
  return not (
      flags_dict['dataset_start_time'] is not None
      and flags_dict['episode_start_time_ns'] is not None
  )


flags.register_multi_flags_validator(
    ['dataset_start_time', 'episode_start_time_ns'],
    _check_dataset_time_and_episode_start_time_ns_are_mutually_exclusive,
    message=(
        '--dataset_start_time and --episode_start_time_ns cannot both be'
        ' provided.'
    ),
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


_MAX_WORKERS = flags.DEFINE_integer(
    'max_workers',
    1,
    'Maximum number of threads for parallel processing and logging. '
    'Parallelization is not currently supported.',
)


def _validate_max_workers(values: int) -> bool:
  return values == 1


flags.register_validator(
    'max_workers',
    _validate_max_workers,
    message=(
        'max_workers must be 1 (parallelization is not currently supported).'
    ),
)


def validate_episode_start_time_ns_format(values):
  """Validates the format of the episode_start_time_ns flag.

  The flag should be a list of strings, where each string is in the format
  "<episode id>:<timestamp in nanos since unix epoch>". The timestamp must
  be a valid nanosecond timestamp.

  Args:
    values: The value of the episode_start_time_ns flag.

  Returns:
    True if the format is valid or the flag is not provided, False otherwise.
  """
  # episode_start_time_ns is optionally provided, so we can return True if it's
  # not provided.
  if not values:
    return True

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

  if _EPISODE_START_TIME_NS.value is not None:
    logging.warning(
        '--episode_start_time_ns is deprecated and will be removed in a future'
        ' version. Please use --dataset_start_time instead.'
    )
    episode_start_time_ns_by_episode_id = dict(
        [map(int, item.split(':')) for item in _EPISODE_START_TIME_NS.value]
    )
  else:
    episode_start_time_ns_by_episode_id = {}
    if _DATASET_START_TIME.value is None:
      start_dt = datetime.datetime.now()
    else:
      start_dt = datetime.datetime.fromisoformat(_DATASET_START_TIME.value)
    current_start_ns = int(start_dt.timestamp() * 1e9)

    # Compute a start time of each episode in the dataset as one minute
    # after the previous episode's end time.
    for i, episode_id in enumerate(range(dataset.num_episodes)):
      episode_start_time_ns_by_episode_id[episode_id] = current_start_ns
      # Calculate duration
      last_index = int(dataset.meta.episodes[i]['dataset_to_index']) - 1
      last_item = dataset[last_index]
      duration_s = float(last_item[_LEROBOT_TIMESTAMP_DELTA_KEY])
      duration_ns = int(duration_s * 1e9)
      current_start_ns += duration_ns + _ONE_MINUTE_NS

  mcap_lerobot_logger.convert_lerobot_data_to_mcap(
      dataset=dataset,
      task_id=_TASK_ID.value,
      output_directory=output_directory,
      episodes_limit=_NUM_EPISODES.value,
      max_workers=_MAX_WORKERS.value,
      episode_start_timestamps_ns=episode_start_time_ns_by_episode_id,
  )

  logging.info('\n--- Script finished successfully!')


if __name__ == '__main__':
  app.run(main)
