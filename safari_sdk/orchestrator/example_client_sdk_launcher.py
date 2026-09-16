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

r"""Example of launcher for binaries on the robot using Orchestrator SDK.

Launches binaries on the robot by fetching artifacts and launch commands from
the active robot job / work unit in Orchestrator.
"""

from collections.abc import Sequence
import logging
import sys

from absl import app
from absl import flags

from safari_sdk.orchestrator.helpers import orchestrator_launch_helper

_JOB_TYPE_CODE = flags.DEFINE_string(
    "job_type_code",
    None,
    "The job type code name for the robot job.",
    required=True,
)
_ROBOT_ID = flags.DEFINE_string(
    "robot_id",
    None,
    "The ID of the robot to connect to Orchestrator.",
)
_HOSTNAME = flags.DEFINE_string(
    "hostname",
    None,
    "Optional hostname of the robot.",
)
_DOWNLOAD_DIR = flags.DEFINE_string(
    "download_dir",
    "/tmp/orchestrator_launcher_artifacts",
    "Local directory where downloaded artifacts should be saved.",
)
_POLL_INTERVAL_SEC = flags.DEFINE_float(
    "poll_interval_sec",
    1.0,
    "Interval in seconds between process health monitoring polls.",
)

flags.register_multi_flags_validator(
    ["robot_id", "hostname"],
    lambda f: bool(f["robot_id"] or f["hostname"]),
    message="At least one of --robot_id or --hostname must be provided.",
)


def main(argv: Sequence[str]) -> None:
  del argv  # Unused.

  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s [%(levelname)s] %(message)s",
  )

  launcher = orchestrator_launch_helper.OrchestratorLaunchHelper(
      job_type_codes=[_JOB_TYPE_CODE.value],
      robot_id=_ROBOT_ID.value,
      hostname=_HOSTNAME.value,
      download_dir=_DOWNLOAD_DIR.value,
  )

  exit_code = launcher.run(poll_interval=_POLL_INTERVAL_SEC.value)
  sys.exit(exit_code)


if __name__ == "__main__":
  app.run(main)
