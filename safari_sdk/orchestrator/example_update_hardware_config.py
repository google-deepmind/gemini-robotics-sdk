# Copyright 2026 Google LLC
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

"""Example of using Orchestrator client SDK to update robot hardware config."""

from collections.abc import Sequence

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


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  print(" - Initializing and connecting to orchestrator -\n")
  helper = orchestrator_helper.OrchestratorHelper(
      robot_id=_ROBOT_ID.value,
      job_type=orchestrator_helper.JOB_TYPE.ALL,
  )
  response = helper.connect()
  if not response.success:
    print(f"\n - ERROR: {response.error_message} -\n")
    return
  print(" - Successfully connected to orchestrator -\n")

  print(" - Testing empty components list check -\n")
  response = helper.update_robot_hardware_config(components=[])
  if not response.success:
    print(f" - Expected failure for empty list: {response.error_message} -\n")
  else:
    print(
        "\n - ERROR: update_robot_hardware_config unexpectedly succeeded for"
        " empty list! -\n"
    )

  print(" - Constructing fake hardware info -\n")
  components = [
      orchestrator_helper.ROBOT_HARDWARE_COMPONENT(
          component_name="left_gripper",
          serial_number="SN12345",
          model="Inspire Hand",
          firmware_number="1.0.0",
      ),
      orchestrator_helper.ROBOT_HARDWARE_COMPONENT(
          component_name="right_gripper",
          serial_number="SN67890",
          model="Inspire Hand",
          firmware_number="1.0.1",
      ),
  ]

  print(f" - Updating ORCA with {len(components)} components -\n")
  response = helper.update_robot_hardware_config(components=components)
  if response.success:
    print(" - Successfully updated ORCA with hardware components -\n")
  else:
    print(f"\n - ERROR: Failed to update ORCA: {response.error_message} -\n")

  print(" - Disconnecting from orchestrator -\n")
  helper.disconnect()
  print(" - Example completed -\n")


if __name__ == "__main__":
  app.run(main)
