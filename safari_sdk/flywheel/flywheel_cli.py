# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The main file for the Robotics SDK training CLI."""

import copy
import datetime
import json

from absl import app
from absl import flags

from safari_sdk import __version__  # pylint: disable=g-importing-member
from safari_sdk import auth
from safari_sdk.flywheel import upload_data


_COMMANDS_LIST = [
    "train",
    "list",
    "download",
    "data_stats",
    "list_serve",
    "serve",
    "help",
    "upload_data",
    "version",
]

# Mapping from recipe name to training type.
_RECIPE_TO_TYPE_MAP = {
    "narrow": "TRAINING_TYPE_NARROW",
    "gemini_robotics_v1": "TRAINING_TYPE_GEMINI_ROBOTICS_V1",
    "gemini_robotics_on_device_v1": (
        "TRAINING_TYPE_GEMINI_ROBOTICS_ON_DEVICE_V1"
    ),
}

_TRAINING_JOB_ID = flags.DEFINE_string(
    name="training_job_id", default=None, help="The training job id to use."
)

_MODEL_CHECKPOINT_NUMBER = flags.DEFINE_integer(
    name="model_checkpoint_number",
    default=None,
    help="The model checkpoint number to use.",
)

_TRAINING_RECIPE = flags.DEFINE_enum(
    name="training_recipe",
    default="narrow",
    enum_values=list(_RECIPE_TO_TYPE_MAP.keys()),
    help="The training recipe to use.",
)

_TASK_ID = flags.DEFINE_list(
    name="task_id", default=None, help="The task id to use."
)

_START_DATE = flags.DEFINE_string(
    name="start_date",
    default=None,
    help="The start date to use. Format: YYYYMMDD.",
)

_END_DATE = flags.DEFINE_string(
    name="end_date", default=None, help="The end date to use. Format: YYYYMMDD."
)

_JSON_OUTPUT = flags.DEFINE_bool(
    name="json_output",
    default=False,
    help="Whether to output the response in json format.",
)

_UPLOAD_DATA_API_ENDPOINT = flags.DEFINE_string(
    "upload_data_api_endpoint",
    "https://roboticsdeveloper.googleapis.com/upload/v1/dataIngestion:uploadData",
    "Data ingestion service endpoint.",
)

_UPLOAD_DATA_ROBOT_ID = flags.DEFINE_string(
    "upload_data_robot_id",
    None,
    "Typically the identifier of the robot or human collector. Alphanumeric "
    "and fewer than 60 characters.",
)

_UPLOAD_DATA_DIRECTORY = flags.DEFINE_string(
    "upload_data_directory",
    None,
    "Directory where the data files are stored.",
)

_ARTIFACT_ID = flags.DEFINE_string(
    "artifact_id",
    None,
    "Artifact id to download. This comes from the 'train' and 'list' commands.",
)

_HELP_STRING = f"""Usage: flywheel-cli command --api_key=api_key <additional flags>

Commands:
  train: Train a model, need flags:
    --task_id: The task id to use.
    --start_date: The start date to use. Format: YYYYMMDD.
    --end_date: The end date to use. Format: YYYYMMDD.
    --training_recipe: The training recipe to use, one of [{", ".join(_RECIPE_TO_TYPE_MAP.keys())}]

  data_stats: Show data stats currently available for training.

  list: List available models.

  list_serve: List available serving jobs.

  serve: Serve a model, need flags:
    --training_job_id: The training job id to use.
    --model_checkpoint_number: The model checkpoint number to use.

  download: Download artifacts from a training job, need flags:
    --training_job_id: Download the artifacts from this training job.
    or --artifact_id: Download the artifacts from this artifact.
      This comes from the 'train' and 'list' commands.

  upload_data: Upload data to the data ingestion service.
    --upload_data_robot_id: The robot id to use.
    --upload_data_directory: The directory where the data files are stored.

  help: Show this help message.

  version: Show the version of the SDK.

Note: The API key can be specified with the --api_key flag or in a file named
"api_key.json" in one of the paths specified in the auth module."""


class FlywheelCli:
  """The training CLI."""

  def __init__(self):
    self._service = auth.get_service()
    self._base_request_body = {}

  def handle_train(self) -> None:
    """Handles the train commands.

    Trains a model.

    Needs task_id, start_date, end_date flags.
    """

    body = copy.deepcopy(self._base_request_body)
    body |= {
        "training_data_filters": {
            "task_id": _TASK_ID.value,
            "start_date": _START_DATE.value,
            "end_date": _END_DATE.value,
        },
        "training_type": _RECIPE_TO_TYPE_MAP[_TRAINING_RECIPE.value],
    }
    response = self._service.orchestrator().startTraining(body=body).execute()

    print(json.dumps(response, indent=4))

  def handle_data_stats(self) -> None:
    """Handles the data stats commands."""
    body = copy.deepcopy(self._base_request_body)
    response = (
        self._service.orchestrator().trainingDataDetails(body=body).execute()
    )

    def _print_row(col1, col2, col3, col4, col5):
      print(f"{col1:40s}{col2:40s}{col3:20s}{col4:20s}{col5:20s}")

    if _JSON_OUTPUT.value:
      print(json.dumps(response, indent=4))
    elif response.get("taskDates"):
      _print_row(
          "Robot id",
          "Task id",
          "Date",
          "Count",
          "Success count",
      )
      for task_date in response.get("taskDates"):
        robot_id = task_date.get("robotId")
        task_id = task_date.get("taskId")
        dates = task_date.get("dates")
        daily_counts = task_date.get("dailyCounts")
        success_counts = task_date.get("successCounts")
        for date, daily_count, success_count in zip(
            dates, daily_counts, success_counts
        ):
          _print_row(
              str(robot_id),
              str(task_id),
              str(date),
              str(daily_count),
              str(success_count),
          )
    else:
      print("No data stats found.")

  def handle_list_training_jobs(self) -> None:
    """Handles the list commands.

    List all training jobs.
    """
    body = copy.deepcopy(self._base_request_body)
    response = self._service.orchestrator().trainingJobs(body=body).execute()

    def _print_row(col1, col2, col3, col4, col5, col6, col7):
      print(
          f"{col1:40s}{col2:40s}{col3:40s}{col4:30s}{col5:20s}{col6:20s}{col7:20s}"
      )

    if _JSON_OUTPUT.value:
      print(json.dumps(response, indent=4))
    elif response.get("trainingJobs"):
      _print_row(
          "Training jobs id",
          "Status",
          "Training type",
          "Task id",
          "robot id",
          "Start date",
          "End date",
      )
      for training_job in response.get("trainingJobs"):
        training_data_filters = training_job.get("trainingDataFilters")
        if training_data_filters:
          robot_id = training_data_filters.get("robotId")
          task_id = training_data_filters.get("taskId")
          start_date = training_data_filters.get("startDate")
          end_date = training_data_filters.get("endDate")
        else:
          robot_id = task_id = start_date = end_date = None
        _print_row(
            str(training_job.get("trainingJobId")),
            str(training_job.get("stage")),
            str(training_job.get("trainingType")),
            str(task_id),
            str(robot_id),
            str(start_date),
            str(end_date),
        )
    else:
      print("No training jobs found.")

  def handle_list_serving_jobs(self) -> None:
    """Handles the serving jobs commands.

    List all serving jobs.
    """
    body = copy.deepcopy(self._base_request_body)
    response = self._service.orchestrator().servingJobs(body=body).execute()

    def _print_row(col1, col2, col3, col4, col5, col6, col7, col8):
      print(
          f"{col1:40s}{col2:40s}{col3:40s}{col4:40s}{col5:30s}{col6:20s}{col7:20s}{col8:40s}"
      )

    if _JSON_OUTPUT.value:
      print(json.dumps(response, indent=4))
    elif response.get("servingJobs"):
      _print_row(
          "Serving jobs id",
          "Training job id",
          "Model chkpt #",
          "Status",
          "Task id",
          "robot id",
          "Start date",
          "End date",
      )
      for serving_job in response.get("servingJobs"):
        training_job_id = serving_job.get("trainingJobId")
        training_data_filters = serving_job.get("trainingDataFilters")
        if training_data_filters:
          robot_id = training_data_filters.get("robotId")
          task_id = training_data_filters.get("taskId")
          start_date = training_data_filters.get("startDate")
          end_date = training_data_filters.get("endDate")
        else:
          robot_id = task_id = start_date = end_date = None
        _print_row(
            str(serving_job.get("servingJobId")),
            str(training_job_id),
            str(serving_job.get("modelCheckpointNumber")),
            str(serving_job.get("stage")),
            str(task_id),
            str(robot_id),
            str(start_date),
            str(end_date),
        )
    else:
      print("No serving jobs found.")

  def handle_serve(self) -> None:
    """Handles the serve commands.

    Serve a model.
    """
    body = copy.deepcopy(self._base_request_body)
    body |= {
        "training_job_id": _TRAINING_JOB_ID.value,
        "model_checkpoint_number": _MODEL_CHECKPOINT_NUMBER.value,
    }
    response = self._service.orchestrator().serveModel(body=body).execute()

    print(json.dumps(response, indent=4))

  def handle_upload_data(self) -> None:
    """Handles the upload data commands."""
    upload_data.upload_data_directory(
        api_endpoint=_UPLOAD_DATA_API_ENDPOINT.value,
        data_directory=_UPLOAD_DATA_DIRECTORY.value,
        robot_id=_UPLOAD_DATA_ROBOT_ID.value,
    )

  def handle_download_training_artifacts(self) -> None:
    """Handles the download commands.

    Download artifacts from a training job.
    """
    body = copy.deepcopy(self._base_request_body)
    body |= {
        "training_job_id": _TRAINING_JOB_ID.value,
    }
    response = (
        self._service.orchestrator().trainingArtifact(body=body).execute()
    )

    def _print_row(col1, col2):
      print(f"{col1:40s}{col2:20s}")

    if _JSON_OUTPUT.value:
      print(json.dumps(response, indent=4))
    else:
      _print_row("#", "Artifact URL")
      for i, uri in enumerate(response.get("uris")):
        _print_row(str(i), uri)

  def handle_download_artifact_id(self) -> None:
    """Handles the download commands."""
    body = copy.deepcopy(self._base_request_body)
    body |= {
        "artifact_id": _ARTIFACT_ID.value,
    }
    response = self._service.orchestrator().loadArtifact(body=body).execute()
    print(json.dumps(response, indent=4))

  # TODO: Do not require an api key for version or help.
  def handle_version(self) -> None:
    """Handles the version commands."""
    print(f"Version: {__version__}")

  def parse_flag(self, command: str) -> None:
    """Parses command flags."""
    if not auth.get_api_key():
      raise ValueError("API key is required.")

    match command:
      case "train":
        if not _TASK_ID.value:
          raise ValueError("Task is is required.")
        if not _START_DATE.value:
          raise ValueError("Start date is required.")
        if not _is_valid_date(_START_DATE.value):
          raise ValueError(
              "Start date is not in the correct format YYYYMMDD. Got:"
              f" {_START_DATE.value}"
          )
        if not _END_DATE.value:
          raise ValueError("End date is required.")
        if not _is_valid_date(_END_DATE.value):
          raise ValueError(
              "End date is not in the correct format YYYYMMDD. Got:"
              f" {_END_DATE.value}"
          )
        if not _is_valid_start_end_date_pair(
            _START_DATE.value, _END_DATE.value
        ):
          raise ValueError(
              "Start date must be before or equal to end date. Start date:"
              f" {_START_DATE.value} End date: {_END_DATE.value}"
          )
        self.handle_train()
      case "serve":
        if not _TRAINING_JOB_ID.value:
          raise ValueError("Training job id is required.")
        if not _MODEL_CHECKPOINT_NUMBER.value:
          raise ValueError("Model checkpoint number is required.")
        if _MODEL_CHECKPOINT_NUMBER.value < 0:
          raise ValueError(
              "Model checkpoint number must be positive non-zero number. Got:"
              f" {_MODEL_CHECKPOINT_NUMBER.value}"
          )
        self.handle_serve()
      case "list":
        self.handle_list_training_jobs()
      case "list_serve":
        self.handle_list_serving_jobs()
      case "data_stats":
        self.handle_data_stats()
      case "download":
        if _TRAINING_JOB_ID.value:
          self.handle_download_training_artifacts()
        elif _ARTIFACT_ID.value:
          self.handle_download_artifact_id()
        else:
          raise ValueError(
              "Download command requires either training_job_id or artifact_id."
          )
      case "upload_data":
        if not _UPLOAD_DATA_ROBOT_ID.value:
          raise ValueError("Upload data robot id is required.")
        if not _UPLOAD_DATA_DIRECTORY.value:
          raise ValueError("Upload data directory is required.")
        self.handle_upload_data()
      case "version":
        self.handle_version()
      case _:
        show_help()


def _is_valid_date(date: str) -> bool:
  """Checks if the date is in the format YYYYMMDD."""
  if len(date) != 8:
    return False
  try:
    datetime.datetime.strptime(date, "%Y%m%d")
    return True
  except ValueError:
    return False


def _is_valid_start_end_date_pair(start_date: str, end_date: str) -> bool:
  """Checks if the start and end date are in the correct order."""
  start = datetime.datetime.strptime(start_date, "%Y%m%d")
  end = datetime.datetime.strptime(end_date, "%Y%m%d")
  return start <= end


def show_help() -> None:
  """Shows the help message."""
  print(_HELP_STRING)


def cli_main() -> None:
  """The main function for the CLI."""
  app.run(main)


def main(argv: list[str]) -> None:
  if len(argv) != 2 or argv[1] == "help" or argv[1] not in _COMMANDS_LIST:
    show_help()
    return

  command = argv[1]
  flywheel_cli = FlywheelCli()
  flywheel_cli.parse_flag(command)


if __name__ == "__main__":
  app.run(main)
