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

"""Example of using Orchestrator client SDK for robot and operator info.

For more details on how to use the Orchestrator client SDK, please refer to the
docstring of the main integration example at:
  orchestrator/example_client_sdk_integration.py

For more details on each of the Orchestrator client SDK API methods, please
refer to the docstring of the helper file itself:
  orchestrator/helpers/orchestrator_helper.py.
"""

from collections.abc import Sequence
import time

from absl import app
from absl import flags

from safari_sdk.orchestrator.helpers import orchestrator_helper

# Required flags.
_ROBOT_ID = flags.DEFINE_string(
    name="robot_id",
    default=None,
    help="This robot's ID.",
    required=True,
)

_JOB_TYPE = flags.DEFINE_enum_class(
    name="job_type",
    default=orchestrator_helper.JOB_TYPE.ALL,
    enum_class=orchestrator_helper.JOB_TYPE,
    help="Type of job to run.",
)

# The flags below are optional.
_RAISE_ERROR = flags.DEFINE_bool(
    "raise_error",
    default=False,
    help=(
        "Whether to raise the error as an exception or just show it as a"
        " messsage. Default = False."
    ),
)


def _print_robot_info_response(response: orchestrator_helper.RESPONSE) -> None:
  """Prints out details of the current robot information."""
  print("\n - Current robot information -")

  print(" ----------------------------------------------------------------\n")
  print(f" Robot ID: {response.robot_id}")
  print(f" Is operational: {response.is_operational}\n")
  print(f" Operator ID: {response.operator_id}\n")
  print(f" Robot job ID: {response.robot_job_id}")
  print(f" Work unit ID: {response.work_unit_id}")
  print(f" Work unit stage: {response.work_unit_stage}")
  print(" ----------------------------------------------------------------\n")


def run_example(
    orchestrator_client: orchestrator_helper.OrchestratorHelper,
) -> None:
  """Runs mock eval loop."""

  print(" - Getting current robot info -\n")
  response = orchestrator_client.get_current_robot_info()
  if not response.success:
    print(f"\n - ERROR: {response.error_message} -\n")
    return

  _print_robot_info_response(response=response)
  time.sleep(1)

  print(" - Setting operator ID to: 'test_operator_id' -\n")
  response = orchestrator_client.set_current_robot_operator_id(
      operator_id="test_operator_id"
  )
  if not response.success:
    print(f"\n - ERROR: {response.error_message} -\n")
    return

  print(" - Getting current robot info again to verify operator ID -\n")
  response = orchestrator_client.get_current_robot_info()
  if not response.success:
    print(f"\n - ERROR: {response.error_message} -\n")
    return

  _print_robot_info_response(response=response)
  time.sleep(1)

  print(" - Clearing operator ID field in robot information -\n")
  response = orchestrator_client.set_current_robot_operator_id(operator_id="")
  if not response.success:
    print(f"\n - ERROR: {response.error_message} -\n")
    return

  print(" - Getting current robot info again to verify no operator ID -\n")
  response = orchestrator_client.get_current_robot_info()
  if not response.success:
    print(f"\n - ERROR: {response.error_message} -\n")
    return

  _print_robot_info_response(response=response)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  print(" - Initializing and connecting to orchestrator -\n")
  orchestrator_client = orchestrator_helper.OrchestratorHelper(
      robot_id=_ROBOT_ID.value,
      job_type=_JOB_TYPE.value,
      raise_error=_RAISE_ERROR.value,
  )
  response = orchestrator_client.connect()
  if not response.success:
    print(f"\n - ERROR: {response.error_message} -\n")
    return

  print(" - Running example of getting robot info and setting operator ID -\n")
  run_example(orchestrator_client=orchestrator_client)

  print(" - Disconnecting from orchestrator -\n")
  orchestrator_client.disconnect()

  print(" - Example run completed -\n")


if __name__ == "__main__":
  app.run(main)
