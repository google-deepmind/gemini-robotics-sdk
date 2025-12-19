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
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Sequence
import urllib

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

_CHECKPOINT_TYPE_MAP = {
    "aloha": "CHECKPOINT_TYPE_ALOHA",
    "franka_duo": "CHECKPOINT_TYPE_FRANKA_DUO",
}

_TRAINING_JOB_ID = flags.DEFINE_string(
    name="training_job_id", default=None, help="The training job id to use."
)

_MODEL_CHECKPOINT_NUMBER = flags.DEFINE_integer(
    name="model_checkpoint_number",
    default=None,
    help="The model checkpoint number to use.",
)

_MODEL_CHECKPOINT_PATH = flags.DEFINE_string(
    name="model_checkpoint_path",
    default=None,
    help="The local gemini_robotics_on_device model checkpoint path to use.",
)

_SERVE_PORT = flags.DEFINE_integer(
    name="serve_port",
    default=60061,
    help="The port to use for serving.",
)

_GPU_MEM_FRACTION = flags.DEFINE_float(
    name="gpu_mem_fraction",
    default=0.8,
    help="The GPU memory fraction to use for serving.",
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

_ROBOT_ID = flags.DEFINE_list(
    name="robot_id", default=None, help="The robot id to use."
)

_START_DATE = flags.DEFINE_string(
    name="start_date",
    default=None,
    help="The start date to use. Format: YYYYMMDD.",
)

_END_DATE = flags.DEFINE_string(
    name="end_date", default=None, help="The end date to use. Format: YYYYMMDD."
)

_MAX_TRAINING_STEPS = flags.DEFINE_integer(
    name="max_training_steps",
    default=10000,
    help="The maximum number of training steps to use.",
)

_CHECKPOINT_EVERY_N_STEPS = flags.DEFINE_integer(
    name="checkpoint_every_n_steps",
    default=None,
    help=(
        "The number of steps to checkpoint. If not set, the default is"
        " max_training_steps / 5."
    ),
)

_CHECKPOINT_TYPE = flags.DEFINE_enum(
    name="checkpoint_type",
    default="aloha",
    enum_values=list(_CHECKPOINT_TYPE_MAP.keys()),
    help="The checkpoint type to use.",
)

_IMAGE_KEYS = flags.DEFINE_list(
    name="image_keys",
    default=[],
    help=(
        "The image keys to use for training. They should be a subset of the"
        " available image_observation_keys logged by EpisodicLogger."
    ),
)

_PROPRIOCEPTION_KEYS = flags.DEFINE_list(
    name="proprioception_keys",
    default=[],
    help=(
        "The proprioception keys to use for training. They should be a subset"
        " of the available proprioceptive_observation_keys logged by"
        " EpisodicLogger."
    ),
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

_USE_CPU = flags.DEFINE_bool(
    "use_cpu",
    False,
    "Whether to use CPU for serving. If False, GPU will be used.",
)

_HELP_STRING = f"""Usage: flywheel-cli command --api_key=api_key <additional flags>

Commands:
  train: Train a model, need flags:
    --robot_id: The robot id to use. (Optional)
    --task_id: The task id to use.
    --start_date: The start date to use. Format: YYYYMMDD.
    --end_date: The end date to use. Format: YYYYMMDD.
    --training_recipe: The training recipe to use, one of [{', '.join(_RECIPE_TO_TYPE_MAP.keys())}]

  data_stats: Show data stats currently available for training.

  list: List available models.

  list_serve: List available serving jobs.

  serve: Serve a model.
    For gemini_robotics_v1 recipe:
      --training_job_id: The training job id to use.
      --model_checkpoint_number: The model checkpoint number to use.
    For gemini_robotics_on_device_v1 recipe (requires Docker):
      --training_job_id: The training job id to use. (Optional, defaults to base model)
      --model_checkpoint_path: The local gemini_robotics_on_device model
        checkpoint path to use if you have downloaded the checkpoint and saved
        it locally. (Optional)

      This will download the model and start a serving docker container.

  download: Download artifacts from a training job. This is an interactive command.
    --training_job_id: Lists artifacts from this training job to download.
    or --artifact_id: Download a specific artifact. If it is a docker image,
      you will be prompted to load it.
      Artifacts are saved to a temporary directory by default.

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
            "robot_id": _ROBOT_ID.value,
            "task_id": _TASK_ID.value,
            "start_date": _START_DATE.value,
            "end_date": _END_DATE.value,
        },
        "training_type": _RECIPE_TO_TYPE_MAP[_TRAINING_RECIPE.value],
        "tracer": time.time_ns(),
    }
    if _TRAINING_RECIPE.value == "gemini_robotics_on_device_v1":
      body |= {
          "training_config": {
              "max_training_steps": _MAX_TRAINING_STEPS.value,
              "checkpoint_every_n_steps": _CHECKPOINT_EVERY_N_STEPS.value,
              "checkpoint_type": _CHECKPOINT_TYPE_MAP[_CHECKPOINT_TYPE.value],
              "image_keys": _IMAGE_KEYS.value,
              "proprioception_keys": _PROPRIOCEPTION_KEYS.value,
          }
      }
    response = self._service.orchestrator().startTraining(body=body).execute()

    print(json.dumps(response, indent=4))

  def handle_data_stats(self) -> None:
    """Handles the data stats commands."""
    body = copy.deepcopy(self._base_request_body)
    body |= {"tracer": time.time_ns()}
    response = (
        self._service.orchestrator().trainingDataDetails(body=body).execute()
    )

    if _JSON_OUTPUT.value:
      print(json.dumps(response, indent=4))
    elif response.get("taskDates"):
      headers = ["Robot id", "Task id", "Date", "Count", "Success count"]
      rows = []
      for task_date in response.get("taskDates"):
        robot_id = task_date.get("robotId")
        task_id = task_date.get("taskId")
        dates = task_date.get("dates")
        daily_counts = task_date.get("dailyCounts")
        success_counts = task_date.get("successCounts")
        for date, daily_count, success_count in zip(
            dates, daily_counts, success_counts
        ):
          rows.append([
              str(robot_id),
              str(task_id),
              str(date),
              str(daily_count),
              str(success_count),
          ])
      _print_responsive_table(headers, rows)
    else:
      print("No data stats found.")

  def handle_list_training_jobs(self) -> None:
    """Handles the list commands.

    List all training jobs.
    """
    body = copy.deepcopy(self._base_request_body)
    body |= {"tracer": time.time_ns()}
    response = self._service.orchestrator().trainingJobs(body=body).execute()

    if _JSON_OUTPUT.value:
      print(json.dumps(response, indent=4))
    elif response.get("trainingJobs"):
      headers = [
          "Training jobs id",
          "Status",
          "Training type",
          "Task id",
          "Robot id",
          "Start date",
          "End date",
      ]
      # Relative weights for columns.
      rows = []
      for training_job in response.get("trainingJobs"):
        training_data_filters = training_job.get("trainingDataFilters")
        if training_data_filters:
          robot_id = training_data_filters.get("robotId")
          task_id = training_data_filters.get("taskId")
          start_date = training_data_filters.get("startDate")
          end_date = training_data_filters.get("endDate")
        else:
          robot_id = task_id = start_date = end_date = None
        rows.append([
            str(training_job.get("trainingJobId")),
            str(training_job.get("stage")),
            str(training_job.get("trainingType")),
            str(task_id),
            str(robot_id),
            str(start_date),
            str(end_date),
        ])
      _print_responsive_table(headers, rows)
    else:
      print("No training jobs found.")

  def handle_list_serving_jobs(self) -> None:
    """Handles the serving jobs commands.

    List all serving jobs.
    """
    body = copy.deepcopy(self._base_request_body)
    body |= {"tracer": time.time_ns()}
    response = self._service.orchestrator().servingJobs(body=body).execute()

    if _JSON_OUTPUT.value:
      print(json.dumps(response, indent=4))
    elif response.get("servingJobs"):
      headers = [
          "Serving jobs id",
          "Training job id",
          "Model chkpt #",
          "Status",
          "Task id",
          "Robot id",
          "Start date",
          "End date",
      ]
      # Relative weights for columns.
      rows = []
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
        rows.append([
            str(serving_job.get("servingJobId")),
            str(training_job_id),
            str(serving_job.get("modelCheckpointNumber")),
            str(serving_job.get("stage")),
            str(task_id),
            str(robot_id),
            str(start_date),
            str(end_date),
        ])
      _print_responsive_table(headers, rows)
    else:
      print("No serving jobs found.")

  def _handle_serve_gemini_robotics_v1(self) -> None:
    """Handles the serve commands for gemini_robotics_v1."""
    body = copy.deepcopy(self._base_request_body)
    body |= {
        "training_job_id": _TRAINING_JOB_ID.value,
        "model_checkpoint_number": _MODEL_CHECKPOINT_NUMBER.value,
        "tracer": time.time_ns(),
    }
    response = self._service.orchestrator().serveModel(body=body).execute()

    print(json.dumps(response, indent=4))

  def _handle_serve_gemini_robotics_on_device_v1(self) -> None:
    """Handles the serve commands for gemini_robotics_on_device_v1."""
    if _MODEL_CHECKPOINT_PATH.value is None:
      if _TRAINING_JOB_ID.value is None:
        print("No training job id provided. Using the base model checkpoint...")
        flags.FLAGS.set_default("training_job_id", "grod_chkpt_aloha_v1")

      checkpoint_path = self.handle_download_training_artifacts()
    else:
      checkpoint_path = _MODEL_CHECKPOINT_PATH.value
    file_dir = os.path.dirname(checkpoint_path)
    file_name = os.path.basename(checkpoint_path)
    try:
      print(f"\nStarting serving docker with checkpoint: {checkpoint_path} ...")
      commands = ["docker", "run", "-it"]

      if not _USE_CPU.value:
        commands.extend([
            "--gpus",
            "device=0",
            "-e",
            f"XLA_PYTHON_CLIENT_MEM_FRACTION={_GPU_MEM_FRACTION.value}",
        ])

      commands.extend([
          "-p",
          f"{_SERVE_PORT.value}:60061",
          "-v",
          f"{file_dir}:/checkpoint",
          "google-deepmind/gemini_robotics_on_device",
          f"--checkpoint_path=/checkpoint/{file_name}",
      ])

      if _IMAGE_KEYS.value:
        commands.append(f"--image_keys={','.join(_IMAGE_KEYS.value)}")
      if _PROPRIOCEPTION_KEYS.value:
        commands.append(
            f"--proprio_keys={','.join(_PROPRIOCEPTION_KEYS.value)}"
        )

      print(f"Running commands: {' '.join(commands)}")
      subprocess.run(
          commands,
          check=True,
          text=True,
      )
    except subprocess.CalledProcessError as e:
      print(
          f"\n[ERROR] Failed to run serving docker (exit code: {e.returncode})."
      )
      print(
          "\nHint: Did you forget to load the docker image? Try `flywheel-cli"
          " download --artifact_id=grod_model_server_docker`."
      )

  def handle_serve(self) -> None:
    """Handles the serve commands.

    Serve a model.
    """
    if _TRAINING_RECIPE.value == "gemini_robotics_v1":
      self._handle_serve_gemini_robotics_v1()
    elif _TRAINING_RECIPE.value == "gemini_robotics_on_device_v1":
      self._handle_serve_gemini_robotics_on_device_v1()
    # else conditions are checked in parse_flag() and should not be reached.

  def handle_upload_data(self) -> None:
    """Handles the upload data commands."""
    upload_data.upload_data_directory(
        api_endpoint=_UPLOAD_DATA_API_ENDPOINT.value,
        data_directory=_UPLOAD_DATA_DIRECTORY.value,
        robot_id=_UPLOAD_DATA_ROBOT_ID.value,
    )

  def handle_download_training_artifacts(self) -> str | None:
    """Handles the download commands.

    Download artifacts from a training job.

    Returns:
      The name of the downloaded file or None if no file was downloaded.
    """
    body = copy.deepcopy(self._base_request_body)
    body |= {
        "training_job_id": _TRAINING_JOB_ID.value,
        "tracer": time.time_ns(),
    }
    response = (
        self._service.orchestrator().trainingArtifact(body=body).execute()
    )

    if _JSON_OUTPUT.value:
      print(json.dumps(response, indent=4))
    else:
      uris = response.get("uris")
      if not uris:
        print("No artifacts found.")
        return

      print("\nAvailable artifacts to download:")
      artifact_names = []
      uri_from_name = {}
      for uri in uris:
        # Try to find a descriptive name like 'checkpoint_...'
        match = re.search(r"(checkpoint_[\w.-]+)", uri)
        if match:
          name = match.group(1)
        else:
          # Fallback to the last part of the URL path
          parsed_uri = urllib.parse.urlparse(uri)
          name = os.path.basename(parsed_uri.path)
        artifact_names.append(name)
        uri_from_name[name] = uri
      # Sort the artifact names by number to be more intuitive.
      artifact_names = sorted(
          artifact_names,
          key=lambda s: int("".join(re.findall(r"\d+", s)))
          if re.findall(r"\d+", s)
          else -1,
      )

      for i, name in enumerate(artifact_names):
        print(f"  [{i}] {name}")

      try:
        artifact_number_str = input(
            "\n> Enter artifact # to download (or press Enter to skip): "
        )
        if not artifact_number_str:
          return
        artifact_number = int(artifact_number_str)
        if not 0 <= artifact_number < len(uris):
          print("Invalid artifact number.")
          return
        selected_name = artifact_names[artifact_number]
        default_file_name = os.path.join(
            tempfile.gettempdir(),
            "grod",
            f"{_TRAINING_JOB_ID.value}_{selected_name}.chkpt",
        ).replace(".chkpt.chkpt", ".chkpt")
        file_name = input(
            f"> Save artifact as (default: {default_file_name}): "
        )
        if not file_name:
          file_name = default_file_name
        if os.path.exists(file_name):
          overwrite = input(f"> File '{file_name}' exists. Overwrite? (y/n): ")
          if overwrite.lower() != "y":
            print("Download cancelled.")
            return file_name
        _download_url_to_file(uri_from_name[selected_name], file_name)
        return file_name
      except ValueError:
        print("Invalid input. Please enter a number.")
        return
      except KeyboardInterrupt:
        print("\nDownload cancelled.")
        return

  def handle_download_artifact_id(self) -> None:
    """Handles the download commands."""
    body = copy.deepcopy(self._base_request_body)
    body |= {
        "artifact_id": _ARTIFACT_ID.value,
        "tracer": time.time_ns(),
    }
    response = self._service.orchestrator().loadArtifact(body=body).execute()
    artifact = response.get("artifact")
    if not artifact:
      print("No artifact found.")
      return
    uri = artifact.get("uri")
    if not uri:
      print(f"URI is not specified for artifact: {_ARTIFACT_ID.value}.")
      return

    container_dir = os.path.join(tempfile.gettempdir(), "grod")
    default_filename = os.path.join(container_dir, f"{_ARTIFACT_ID.value}.tar")
    filename = input(f"\n> Save artifact as (default: {default_filename}): ")
    print(f"Filename: {filename}")
    if not filename:
      filename = default_filename
    _download_url_to_file(uri, filename)

    if "docker" in _ARTIFACT_ID.value:
      load_docker_image = input(
          "\n> This artifact appears to be a docker image. Load it? (y/n): "
      )
      if load_docker_image.lower() == "y":
        try:
          print("\nLoading docker image...")
          result = subprocess.run(
              ["docker", "load", "-i", filename],
              capture_output=True,
              check=True,
              text=True,
          )
          print(result.stdout)
        except subprocess.CalledProcessError as e:
          print(f"\n[ERROR] Failed to load docker image: {e.stderr}")

  def parse_flag(self, command: str) -> None:
    """Parses command flags."""
    if not auth.get_api_key():
      raise ValueError("API key is required.")

    match command:
      case "train":
        if not _TASK_ID.value:
          raise ValueError("Task id is required.")
        if not _ROBOT_ID.value:
          print("No Robot id is specified. Using data from all robots...")
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
        support_training_recipes = [
            "gemini_robotics_v1",
            "gemini_robotics_on_device_v1",
        ]
        if _TRAINING_RECIPE.value not in support_training_recipes:
          raise ValueError(
              "Serving is only supported for training recipe:"
              f" {support_training_recipes}. Got: {_TRAINING_RECIPE.value}"
          )
        if _TRAINING_RECIPE.value == "gemini_robotics_v1":
          if not _TRAINING_JOB_ID.value:
            raise ValueError("Training job id is required.")
          if not _MODEL_CHECKPOINT_NUMBER.value:
            raise ValueError("Model checkpoint number is required.")
          if _MODEL_CHECKPOINT_NUMBER.value < 0:
            raise ValueError(
                "Model checkpoint number must be positive non-zero number. Got:"
                f" {_MODEL_CHECKPOINT_NUMBER.value}"
            )
        elif _TRAINING_RECIPE.value == "gemini_robotics_on_device_v1":
          if _MODEL_CHECKPOINT_NUMBER.value:
            raise ValueError(
                "Model checkpoint number is not supported for training recipe:"
                f" {_TRAINING_RECIPE.value}"
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
      case _:
        show_help()


def _print_responsive_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> None:
  """Prints a table with column widths that adapt to content."""
  if not rows and not headers:
    return
  num_columns = len(headers)
  col_widths = [len(h) for h in headers]
  for row in rows:
    for i, cell in enumerate(row):
      col_widths[i] = max(col_widths[i], len(str(cell)))

  separator = "  "

  def _print_line(items, widths):
    print(
        separator.join(
            [f"{str(item):<{widths[i]}}" for i, item in enumerate(items)]
        )
    )

  def _print_hr(widths):
    total_width = sum(widths) + len(separator) * (num_columns - 1)
    print("-" * total_width)

  _print_hr(col_widths)
  _print_line(headers, col_widths)
  _print_hr(col_widths)
  for row in rows:
    _print_line(row, col_widths)


def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
  """A reporthook for urlretrieve to print a progress bar."""
  downloaded = block_num * block_size
  if total_size > 0:
    percent = min(100, 100 * downloaded / total_size)
    bar = "█" * int(percent / 2)
    sys.stdout.write(f"\r|{bar:<50}| {percent:.1f}%")
    sys.stdout.flush()
  else:
    # Total size not known.
    sys.stdout.write(f"\rDownloaded {downloaded / (1024*1024):.2f} MB")
    sys.stdout.flush()


def _download_url_to_file(url: str, filename: str) -> None:
  """Downloads content from a URL and saves it to a file."""
  try:
    dirname = os.path.dirname(filename)
    if dirname:
      os.makedirs(dirname, exist_ok=True)
    print(f"\nDownloading artifact to {filename} ...")
    urllib.request.urlretrieve(url, filename, reporthook=_reporthook)
    print()  # New line after progress bar.
    print("Download complete!")
  except urllib.error.URLError as e:
    print(f"\n[ERROR] Error downloading artifact {url}: {e}")


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


def main(argv: Sequence[str]) -> None:
  if len(argv) != 2 or argv[1] == "help" or argv[1] not in _COMMANDS_LIST:
    show_help()
    return

  if argv[1] == "version":
    print(f"Version: {__version__}")
    return

  command = argv[1]
  flywheel_cli = FlywheelCli()
  flywheel_cli.parse_flag(command)


if __name__ == "__main__":
  app.run(main)
