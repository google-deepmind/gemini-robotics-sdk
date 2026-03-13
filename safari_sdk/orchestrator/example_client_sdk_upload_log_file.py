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

"""Example of uploading a log file to Orchestrator using the SDK.

This is an fully working example of how a binary would need to upload a text
log file to Orchestrator.

For more details on each of the Orchestrator client SDK API methods, please
refer to the docstring of the helper file itself:
  orchestrator/helpers/orchestrator_helper.py.
"""

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

_JOB_TYPE = flags.DEFINE_enum_class(
    name="job_type",
    default=orchestrator_helper.JOB_TYPE.ALL,
    enum_class=orchestrator_helper.JOB_TYPE,
    help="Type of job to run.",
)

# The flags below are optional.
_RAISE_ERROR = flags.DEFINE_bool(
    name="raise_error",
    default=False,
    help=(
        "Whether to raise the error as an exception or just show it as a"
        " messsage. Default = False."
    ),
)


def _print_orchestrator_work_unit_info(
    work_unit: orchestrator_helper.WORK_UNIT,
) -> None:
  """Prints out details of the given work unit dataclass."""
  print(" ----------------------------------------------------------------\n")
  print(f" Robot Job ID: {work_unit.robotJobId}")
  print(f" Work Unit ID: {work_unit.workUnitId}")
  print(f" Work Unit stage: {work_unit.stage}")
  print(f" Work Unit outcome: {work_unit.outcome}")
  print(f" Work Unit note: {work_unit.note}\n")
  print(" ----------------------------------------------------------------\n")


def run_mock_eval_loop(
    orchestrator_client: orchestrator_helper.OrchestratorHelper,
) -> bool:
  """Runs mock eval loop."""
  print(" - Requesting work unit... -\n")

  response = orchestrator_client.request_work_unit()
  if not response.success:
    print(f"\n - ERROR: {response.error_message} -\n")
    return False
  if response.no_more_robot_job:
    print(" - No robot job available -\n")
    return False
  if response.no_more_work_unit:
    print(" - No work unit available -\n")
    return False
  print(" - Sucessfully requested work unit -\n")

  if response.launch_command:
    print(
        "\n - Robot Job contains launch command information -\n"
        f"{response.launch_command}\n"
    )

  response = orchestrator_client.get_current_work_unit()
  if not response.success:
    print(f"\n - ERROR: {response.error_message} -\n")
    return False

  print(" - Current work unit information -\n")
  work_unit = response.work_unit
  assert work_unit is not None
  _print_orchestrator_work_unit_info(work_unit=work_unit)

  return True


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

  print(" - Running mock eval -\n")
  upload_log = run_mock_eval_loop(orchestrator_client=orchestrator_client)

  if upload_log:
    print("   Assuming we have a mock log file we want to upload called:")
    print("       example_client_sdk_upload_log_file.log\n")
    print("   To upload this log file, the user needs to extract the content")
    print("   of the log file as bytes, so the 'upload_text_log_artifact'")
    print("   Orchestrator Helper API can be used.\n")

    print(" - Mocking the opening and reading the log file... -\n")

    print(" - Mocking the context of the log file as... -\n\n")
    mock_log_file_content = """
This is a mock log file for testing purposes by example_client_sdk_upload_log_file.py

There is no actual data in this log file, so please ignore the content.

Again, this is just a mock log file for testing purposes.
  """
    print(mock_log_file_content)
    print("")

    print(" - Uploading the mock log file content... -\n")
    response = orchestrator_client.upload_text_log_artifact(
        source_file_name="example_client_sdk_upload_log_file.log",
        text_file_bytes=mock_log_file_content.encode("utf-8"),
    )
    if not response.success:
      print(f"\n - ERROR: {response.error_message} -\n")
    else:
      print(
          " - Log file uploaded successfully as artifact ID:"
          f" {response.artifact_id} -\n"
      )

  print(" - Disconnecting from orchestrator -\n")
  orchestrator_client.disconnect()

  print(" - Mock eval completed -\n")


if __name__ == "__main__":
  app.run(main)
