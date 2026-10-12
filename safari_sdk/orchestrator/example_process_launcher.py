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

r"""Example of standalone process launcher using ProcessLaunchHelper.

Launches a list of commands sequentially in separate terminal windows and
manages their process lifecycles using only the native Python 3 standard
library (no absl or external dependencies).
"""

import argparse
from collections.abc import Sequence
import logging
import sys

from safari_sdk.orchestrator.helpers import process_launch_helper


def _parse_commands(argv: Sequence[str] | None = None) -> list[str]:
  """Parses the list of commands to execute in order from CLI arguments."""
  parser = argparse.ArgumentParser(
      description=(
          "Launches a list of commands sequentially in separate terminal"
          " windows using ProcessLaunchHelper."
      )
  )
  parser.add_argument(
      "--commands",
      "-c",
      nargs="+",
      action="extend",
      default=[],
      help="List of commands to execute in the order passed in.",
  )
  parser.add_argument(
      "positional_commands",
      nargs="*",
      metavar="COMMAND",
      help="List of commands to execute in the order passed in.",
  )
  args = parser.parse_args(argv)
  commands = list(args.commands) + list(args.positional_commands)
  if not commands:
    parser.error("At least one command to execute must be provided.")
  return commands


def main(argv: Sequence[str] | None = None) -> int:
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s [%(levelname)s] %(message)s",
  )

  commands = _parse_commands(argv)
  launcher = process_launch_helper.ProcessLaunchHelper()
  return launcher.run(commands)


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
