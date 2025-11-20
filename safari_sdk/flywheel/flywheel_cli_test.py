# Copyright 2025 DeepMind Technologies Limited
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

"""Unit tests for the flywheel cli."""

import io
import json
import os
import tempfile
from unittest import mock
import urllib.error

from absl.testing import flagsaver
from absl.testing import parameterized

from absl.testing import absltest
from safari_sdk import __version__  # pylint: disable=g-importing-member
from safari_sdk.flywheel import flywheel_cli


_DATA_STATS_TEST_DATA = {
    "taskDates": [
        {
            "robotId": "test_robot_id",
            "taskId": "test_task_id",
            "dates": ["2024-12-01", "2024-12-02"],
            "dailyCounts": [100, 200],
            "successCounts": [50, 150],
        },
    ],
}
_TRAINING_JOB_WITH_FILTERS = {
    "trainingJobs": [
        {
            "trainingJobId": "test_training_job_id",
            "stage": "COMPLETED",
            "trainingDataFilters": {
                "robotId": "test_robot_id",
                "taskId": "test_task_id",
                "startDate": "2024-12-01",
                "endDate": "2024-12-02",
            },
            "trainingType": "TRAINING_TYPE_ACT",
        },
    ],
}
_TRAINING_JOB_NO_FILTERS = {
    "trainingJobs": [{
        "trainingJobId": "test_training_job_id",
        "stage": "COMPLETED",
        "trainingType": "TRAINING_TYPE_ACT",
    }],
}

_SERVING_JOB_WITH_FILTERS = {
    "servingJobs": [
        {
            "servingJobId": "test_serving_job_id",
            "stage": "COMPLETED",
            "trainingJobId": "test_training_job_id",
            "modelCheckpointNumber": 1,
            "trainingDataFilters": {
                "robotId": "test_robot_id",
                "taskId": "test_task_id",
                "startDate": "2024-12-01",
                "endDate": "2024-12-02",
            },
        },
    ],
}
_SERVING_JOB_NO_FILTERS = {
    "servingJobs": [
        {
            "servingJobId": "test_serving_job_id",
            "stage": "COMPLETED",
            "trainingJobId": "test_training_job_id",
            "modelCheckpointNumber": 1,
        },
    ],
}
_SERVE_MODEL_RETURN_VALUE = {"serving_job_id": "test_serving_job_id"}
_URI_JSON_OUTPUT = """{
    "uris": [
        "test_uri_1",
        "test_uri_2"
    ]
}
"""


class FlywheelCliTest(parameterized.TestCase):
  # Text output is not tested because the output prone to change, and the
  # test will be flaky.

  def setUp(self):
    super().setUp()
    flagsaver.flagsaver(
        api_key="test_api_key",
        json_output=True,
    ).__enter__()
    with mock.patch("googleapiclient.discovery.build") as mock_build:
      self.service_mock = mock.Mock()
      self._orchestrator = mock_build.return_value.orchestrator
      mock_build.return_value.orchestrator.return_value = self.service_mock

      self._cli = flywheel_cli.FlywheelCli()

      mock_build.assert_called_once()

  @parameterized.named_parameters(
      ("narrow", None, "narrow", "TRAINING_TYPE_NARROW"),
      (
          "gemini_robotics_v1",
          None,
          "gemini_robotics_v1",
          "TRAINING_TYPE_GEMINI_ROBOTICS_V1",
      ),
      (
          "narrow_with_robot_id",
          "test_robot_id",
          "narrow",
          "TRAINING_TYPE_NARROW",
      ),
      (
          "gemini_robotics_on_device_v1",
          "test_robot_id",
          "gemini_robotics_on_device_v1",
          "TRAINING_TYPE_GEMINI_ROBOTICS_ON_DEVICE_V1",
      ),
  )
  def test_train(self, robot_id, recipe, training_type):
    self.service_mock.startTraining.return_value.execute.return_value = {
        "training_job_id": "test_training_job_id"
    }
    req_flags = {
        "task_id": "test_task_id",
        "start_date": "20240101",
        "end_date": "20240102",
        "training_recipe": recipe,
        "max_training_steps": 12345,
        "checkpoint_every_n_steps": 123,
        "checkpoint_type": "aloha",
        "image_keys": ["image1", "image2"],
        "proprioception_keys": ["prop1", "prop2"],
    }
    if robot_id:
      req_flags["robot_id"] = robot_id

    expected_body = {
        "training_data_filters": {
            "robot_id": robot_id if robot_id else None,
            "task_id": "test_task_id",
            "start_date": "20240101",
            "end_date": "20240102",
        },
        "training_type": training_type,
    }
    if recipe == "gemini_robotics_on_device_v1":
      expected_body["training_config"] = {
          "max_training_steps": 12345,
          "checkpoint_every_n_steps": 123,
          "checkpoint_type": "CHECKPOINT_TYPE_ALOHA",
          "image_keys": ["image1", "image2"],
          "proprioception_keys": ["prop1", "prop2"],
      }

    with flagsaver.flagsaver(**req_flags):
      self._cli.handle_train()
      self.service_mock.startTraining.assert_called_once_with(
          body=expected_body
      )
      self.service_mock.startTraining.return_value.execute.assert_called_once_with()

  @parameterized.named_parameters(
      ("text_output_no_data", False, {}, "No data stats found.\n"),
      (
          "json_output_with_data",
          True,
          _DATA_STATS_TEST_DATA,
          json.dumps(_DATA_STATS_TEST_DATA, indent=4) + "\n",
      ),
      ("json_output_no_data", True, {}, "{}\n"),
  )
  def test_data_stats(self, json_output, return_value, expected_output):
    mock_stdout = io.StringIO()
    self.service_mock.trainingDataDetails.return_value.execute.return_value = (
        return_value
    )
    with flagsaver.flagsaver(json_output=json_output):
      with mock.patch("sys.stdout", mock_stdout):
        self._cli.handle_data_stats()
        self.service_mock.trainingDataDetails.assert_called_once_with(body={})
        self.service_mock.trainingDataDetails.return_value.execute.assert_called_once()
        self.assertEqual(mock_stdout.getvalue(), expected_output)

  @parameterized.named_parameters(
      ("text_output_no_jobs", False, {}, "No training jobs found.\n"),
      (
          "json_output_with_filters",
          True,
          _TRAINING_JOB_WITH_FILTERS,
          json.dumps(_TRAINING_JOB_WITH_FILTERS, indent=4) + "\n",
      ),
      (
          "json_output_no_filters",
          True,
          _TRAINING_JOB_NO_FILTERS,
          json.dumps(_TRAINING_JOB_NO_FILTERS, indent=4) + "\n",
      ),
      ("json_output_no_jobs", True, {}, "{}\n"),
  )
  def test_list(self, json_output, return_value, expected_output):
    mock_stdout = io.StringIO()
    self.service_mock.trainingJobs.return_value.execute.return_value = (
        return_value
    )
    with mock.patch("sys.stdout", mock_stdout):
      with flagsaver.flagsaver(json_output=json_output):
        self._cli.handle_list_training_jobs()

      self.service_mock.trainingJobs.assert_called_once_with(body={})
      self.service_mock.trainingJobs.return_value.execute.assert_called_once()
      self.assertEqual(mock_stdout.getvalue(), expected_output)

  @parameterized.named_parameters(
      ("text_output_no_jobs", False, {}, "No serving jobs found.\n"),
      (
          "json_output_with_filters",
          True,
          _SERVING_JOB_WITH_FILTERS,
          json.dumps(_SERVING_JOB_WITH_FILTERS, indent=4) + "\n",
      ),
      (
          "json_output_no_filters",
          True,
          _SERVING_JOB_NO_FILTERS,
          json.dumps(_SERVING_JOB_NO_FILTERS, indent=4) + "\n",
      ),
      ("json_output_no_jobs", True, {}, "{}\n"),
  )
  def test_list_serve(self, json_output, return_value, expected_output):
    mock_stdout = io.StringIO()
    self.service_mock.servingJobs.return_value.execute.return_value = (
        return_value
    )
    with mock.patch("sys.stdout", mock_stdout):
      with flagsaver.flagsaver(json_output=json_output):
        self._cli.handle_list_serving_jobs()

      self.service_mock.servingJobs.assert_called_once_with(body={})
      self.service_mock.servingJobs.return_value.execute.assert_called_once()
      self.assertEqual(mock_stdout.getvalue(), expected_output)

  def test_serve_gemini_robotics_v1(self):
    self.service_mock.serveModel.return_value.execute.return_value = {
        "serving_job_id": "test_serving_job_id"
    }
    with flagsaver.flagsaver(
        training_recipe="gemini_robotics_v1",
        training_job_id="test_training_job_id",
        model_checkpoint_number=1,
    ):
      self._cli.handle_serve()
      self.service_mock.serveModel.assert_called_once_with(
          body={
              "training_job_id": "test_training_job_id",
              "model_checkpoint_number": 1,
          }
      )
      self.service_mock.serveModel.return_value.execute.assert_called_once_with()

  @parameterized.named_parameters(
      (
          "with_download_defaults",
          {"training_job_id": "test_training_job_id"},
          True,
          60061,
          0.8,
          "/tmp/grod/test_training_job_id_0.chkpt",
      ),
      (
          "with_path_custom_flags",
          {
              "model_checkpoint_path": "/test/path/model.chkpt",
              "serve_port": 12345,
              "gpu_mem_fraction": 0.5,
          },
          False,
          12345,
          0.5,
          "/test/path/model.chkpt",
      ),
  )
  @mock.patch("subprocess.run")
  def test_serve_gemini_robotics_on_device_v1(
      self,
      flags_dict,
      expect_download,
      port,
      mem_fraction,
      checkpoint_path,
      mock_subprocess_run,
  ):
    with flagsaver.flagsaver(
        training_recipe="gemini_robotics_on_device_v1", **flags_dict
    ):
      with mock.patch.object(
          self._cli,
          "handle_download_training_artifacts",
          return_value=checkpoint_path,
      ) as mock_download:
        self._cli.handle_serve()
        if expect_download:
          mock_download.assert_called_once()
        else:
          mock_download.assert_not_called()

        file_dir = os.path.dirname(checkpoint_path)
        file_name = os.path.basename(checkpoint_path)
        expected_docker_command = [
            "docker",
            "run",
            "-it",
            "--gpus",
            "device=0",
            "-p",
            f"{port}:60061",
            "-v",
            f"{file_dir}:/checkpoint",
            "-e",
            f"XLA_PYTHON_CLIENT_MEM_FRACTION={mem_fraction}",
            "google-deepmind/gemini_robotics_on_device",
            f"--checkpoint_path=/checkpoint/{file_name}",
        ]
        mock_subprocess_run.assert_called_once_with(
            expected_docker_command, check=True, text=True
        )

  def test_download_training_artifacts(self):
    with flagsaver.flagsaver(training_job_id="test_training_job_id"):
      mock_stdout = io.StringIO()
      self.service_mock.trainingArtifact.return_value.execute.return_value = {
          "uris": ["test_uri_1", "test_uri_2"]
      }

      with mock.patch("sys.stdout", mock_stdout):
        self._cli.handle_download_training_artifacts()

        self.service_mock.trainingArtifact.assert_called_once_with(
            body={
                "training_job_id": "test_training_job_id",
            }
        )
        self.service_mock.trainingArtifact.return_value.execute.assert_called_once()
        self.assertEqual(
            mock_stdout.getvalue(),
            _URI_JSON_OUTPUT,
        )

  @mock.patch(
      "safari_sdk.flywheel.flywheel_cli._download_url_to_file"
  )
  @mock.patch("os.path.exists", return_value=False)
  @mock.patch("builtins.input", side_effect=["0", ""])
  def test_download_training_artifacts_interactive(
      self, mock_input, mock_exists, mock_download
  ):
    """Mocks "flywheel-cli download" command, interactive mode."""
    with flagsaver.flagsaver(
        training_job_id="test_training_job_id", json_output=False
    ):
      self.service_mock.trainingArtifact.return_value.execute.return_value = {
          "uris": [
              "https://storage.googleapis.com/foo/checkpoint_10",
              "https://storage.googleapis.com/foo/checkpoint_2",
          ]
      }

      returned_filename = self._cli.handle_download_training_artifacts()

      self.service_mock.trainingArtifact.assert_called_once_with(
          body={
              "training_job_id": "test_training_job_id",
          }
      )
      self.service_mock.trainingArtifact.return_value.execute.assert_called_once()
      self.assertEqual(mock_input.call_count, 2)
      mock_exists.assert_called_once()

      expected_filename = os.path.join(
          tempfile.gettempdir(),
          "grod",
          "test_training_job_id_checkpoint_2.chkpt",
      )
      mock_download.assert_called_once_with(
          "https://storage.googleapis.com/foo/checkpoint_2",
          expected_filename,
      )
      self.assertEqual(returned_filename, expected_filename)

  @mock.patch(
      "safari_sdk.flywheel.flywheel_cli._download_url_to_file"
  )
  @mock.patch("builtins.input", return_value="")
  def test_download_artifact_id(self, mock_input, mock_download):
    with flagsaver.flagsaver(artifact_id="test_artifact_id"):
      self.service_mock.loadArtifact.return_value.execute.return_value = {
          "artifact": {
              "uri": "test_uri_1"
          }
      }

      self._cli.handle_download_artifact_id()

      self.service_mock.loadArtifact.assert_called_once_with(
          body={"artifact_id": "test_artifact_id"}
      )
      self.service_mock.loadArtifact.return_value.execute.assert_called_once()
      mock_input.assert_called_once()
      mock_download.assert_called_once()

  @mock.patch(
      "safari_sdk.flywheel.flywheel_cli._download_url_to_file"
  )
  @mock.patch("builtins.input", return_value="")
  def test_download_artifact_id_with_empty_response(
      self, mock_input, mock_download
  ):
    with flagsaver.flagsaver(artifact_id="test_artifact_id"):
      self.service_mock.loadArtifact.return_value.execute.return_value = {}

      self._cli.handle_download_artifact_id()

      self.service_mock.loadArtifact.assert_called_once_with(
          body={"artifact_id": "test_artifact_id"}
      )
      self.service_mock.loadArtifact.return_value.execute.assert_called_once()
      mock_input.assert_not_called()
      mock_download.assert_not_called()

  @mock.patch(
      "safari_sdk.flywheel.flywheel_cli._download_url_to_file"
  )
  @mock.patch("builtins.input", return_value="")
  def test_download_artifact_id_with_empty_artifact(
      self, mock_input, mock_download
  ):
    with flagsaver.flagsaver(artifact_id="test_artifact_id"):
      self.service_mock.loadArtifact.return_value.execute.return_value = {
          "artifact": {}
      }

      self._cli.handle_download_artifact_id()

      self.service_mock.loadArtifact.assert_called_once_with(
          body={"artifact_id": "test_artifact_id"}
      )
      self.service_mock.loadArtifact.return_value.execute.assert_called_once()
      mock_input.assert_not_called()
      mock_download.assert_not_called()

  def test_show_help(self):
    mock_stdout = io.StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      flywheel_cli.show_help()
      self.assertEqual(
          mock_stdout.getvalue().rstrip(),
          flywheel_cli._HELP_STRING,
      )

  @parameterized.named_parameters(
      (
          "train_missing_api_key",
          "train",
          {},
          ValueError,
      ),
      (
          "train_missing_project_id",
          "train",
          {"api_key": "test_api_key"},
          ValueError,
      ),
      (
          "train_missing_task_id",
          "train",
          {
              "api_key": "test_api_key",
              "start_date": "20240101",
              "end_date": "20240102",
              "training_recipe": "narrow",
          },
          ValueError,
      ),
      (
          "train_missing_start_date",
          "train",
          {
              "api_key": "test_api_key",
              "task_id": "test_task_id",
              "end_date": "20240102",
              "training_recipe": "narrow",
          },
          ValueError,
      ),
      (
          "train_bad_start_date_1",
          "train",
          {
              "api_key": "test_api_key",
              "task_id": "test_task_id",
              "start_date": "202401012",
              "end_date": "20240102",
          },
          ValueError,
      ),
      (
          "train_bad_start_date_2",
          "train",
          {
              "api_key": "test_api_key",
              "task_id": "test_task_id",
              "start_date": "240101",
              "end_date": "20240102",
          },
          ValueError,
      ),
      (
          "train_bad_start_date_3",
          "train",
          {
              "api_key": "test_api_key",
              "task_id": "test_task_id",
              "start_date": "2024-01-01",
              "end_date": "20240102",
          },
          ValueError,
      ),
      (
          "train_bad_start_date_4",
          "train",
          {
              "api_key": "test_api_key",
              "task_id": "test_task_id",
              "start_date": "20241301",
              "end_date": "20240102",
          },
          ValueError,
      ),
      (
          "train_missing_end_date",
          "train",
          {
              "api_key": "test_api_key",
              "task_id": "test_task_id",
              "start_date": "20240101",
              "training_recipe": "narrow",
          },
          ValueError,
      ),
      (
          "train_bad_end_date",
          "train",
          {
              "api_key": "test_api_key",
              "task_id": "test_task_id",
              "start_date": "20240101",
              "end_date": "202401022",
          },
          ValueError,
      ),
      (
          "train_bad_end_date_1",
          "train",
          {
              "api_key": "test_api_key",
              "task_id": "test_task_id",
              "start_date": "20240101",
              "end_date": "202401012",
          },
          ValueError,
      ),
      (
          "train_bad_end_date_2",
          "train",
          {
              "api_key": "test_api_key",
              "task_id": "test_task_id",
              "start_date": "20240101",
              "end_date": "240101",
          },
          ValueError,
      ),
      (
          "train_bad_end_date_3",
          "train",
          {
              "api_key": "test_api_key",
              "task_id": "test_task_id",
              "start_date": "20240101",
              "end_date": "2024-01-01",
          },
          ValueError,
      ),
      (
          "train_bad_end_date_4",
          "train",
          {
              "api_key": "test_api_key",
              "task_id": "test_task_id",
              "start_date": "20240101",
              "end_date": "20241301",
          },
          ValueError,
      ),
      (
          "train_start_date_after_end_date",
          "train",
          {
              "api_key": "test_api_key",
              "task_id": "test_task_id",
              "start_date": "20240102",
              "end_date": "20240101",
          },
          ValueError,
      ),
      (
          "serve_unsupported_recipe",
          "serve",
          {
              "api_key": "test_api_key",
              "training_recipe": "narrow",
          },
          ValueError,
      ),
      (
          "serve_missing_training_job_id",
          "serve",
          {
              "api_key": "test_api_key",
              "training_recipe": "gemini_robotics_v1",
              "model_checkpoint_number": 1,
          },
          ValueError,
      ),
      (
          "serve_missing_model_checkpoint_number",
          "serve",
          {
              "api_key": "test_api_key",
              "training_recipe": "gemini_robotics_v1",
              "training_job_id": "test_training_job_id",
          },
          ValueError,
      ),
      (
          "serve_bad_model_checkpoint_number_as_zero",
          "serve",
          {
              "api_key": "test_api_key",
              "training_recipe": "gemini_robotics_v1",
              "training_job_id": "test_training_job_id",
              "model_checkpoint_number": 0,
          },
          ValueError,
      ),
      (
          "serve_bad_model_checkpoint_number_as_negative",
          "serve",
          {
              "api_key": "test_api_key",
              "training_recipe": "gemini_robotics_v1",
              "training_job_id": "test_training_job_id",
              "model_checkpoint_number": -1,
          },
          ValueError,
      ),
      (
          "serve_on_device_with_checkpoint_number",
          "serve",
          {
              "api_key": "test_api_key",
              "training_recipe": "gemini_robotics_on_device_v1",
              "model_checkpoint_number": 1,
          },
          ValueError,
      ),
      (
          "download_missing_training_job_id",
          "download",
          {
              "api_key": "test_api_key",
          },
          ValueError,
      ),
  )
  def test_parse_flags_errors(self, command, params, expected_exception):
    with flagsaver.flagsaver(**params):
      with self.assertRaises(expected_exception):
        self._cli.parse_flag(command)

  @parameterized.named_parameters(
      (
          "train",
          "train",
          {
              "api_key": "test_api_key",
              "task_id": "test_task_id",
              "start_date": "20240101",
              "end_date": "20240102",
              "training_recipe": "narrow",
          },
      ),
      (
          "serve_gemini_robotics_v1",
          "serve",
          {
              "api_key": "test_api_key",
              "training_recipe": "gemini_robotics_v1",
              "training_job_id": "test_training_job_id",
              "model_checkpoint_number": 1,
          },
      ),
      (
          "serve_gemini_robotics_on_device_v1",
          "serve",
          {
              "api_key": "test_api_key",
              "training_recipe": "gemini_robotics_on_device_v1",
              "training_job_id": "test_training_job_id",
          },
      ),
      (
          "list",
          "list",
          {
              "api_key": "test_api_key",
          },
      ),
      (
          "list_serve",
          "list_serve",
          {
              "api_key": "test_api_key",
          },
      ),
      (
          "data_stats",
          "data_stats",
          {
              "api_key": "test_api_key",
          },
      ),
      (
          "download",
          "download",
          {
              "api_key": "test_api_key",
              "training_job_id": "test_training_job_id",
          },
      ),
      (
          "version",
          "version",
          {
              "api_key": "test_api_key",
          },
      ),
      (
          "help",
          "help",
          {
              "api_key": "test_api_key",
          },
      ),
  )
  @mock.patch("subprocess.run")
  def test_parse_flags_success(self, command, params, mock_subprocess_run):
    self.service_mock.startTraining.return_value.execute.return_value = {
        "training_job_id": "test_training_job_id"
    }
    self.service_mock.trainingDataDetails.return_value.execute.return_value = (
        _DATA_STATS_TEST_DATA
    )
    self.service_mock.trainingJobs.return_value.execute.return_value = (
        _TRAINING_JOB_WITH_FILTERS
    )
    self.service_mock.servingJobs.return_value.execute.return_value = (
        _SERVING_JOB_WITH_FILTERS
    )
    self.service_mock.serveModel.return_value.execute.return_value = (
        _SERVE_MODEL_RETURN_VALUE
    )
    self.service_mock.trainingArtifact.return_value.execute.return_value = {
        "uris": ["test_uri_1", "test_uri_2"]
    }
    with flagsaver.flagsaver(**params):
      with mock.patch.object(
          self._cli,
          "handle_download_training_artifacts",
          return_value="/fake/path.chkpt",
      ), mock.patch.object(self._cli, "handle_download_artifact_id"):
        self._cli.parse_flag(command)

    if (
        command == "serve"
        and params.get("training_recipe") == "gemini_robotics_on_device_v1"
    ):
      mock_subprocess_run.assert_called_once()
    else:
      mock_subprocess_run.assert_not_called()

  @parameterized.named_parameters(
      ("with_directory", "/tmp/test_dir/test_file.txt", True),
      ("without_directory", "test_file.txt", False),
  )
  @mock.patch("urllib.request.urlretrieve")
  @mock.patch("os.makedirs")
  def test_download_url_to_file_success(
      self, filename, should_call_makedirs, mock_makedirs, mock_urlretrieve
  ):
    mock_stdout = io.StringIO()
    url = "http://example.com/file"
    with mock.patch("sys.stdout", mock_stdout):
      flywheel_cli._download_url_to_file(url, filename)

    if should_call_makedirs:
      mock_makedirs.assert_called_once_with(
          os.path.dirname(filename), exist_ok=True
      )
    else:
      mock_makedirs.assert_not_called()

    mock_urlretrieve.assert_called_once_with(
        url, filename, reporthook=flywheel_cli._reporthook
    )
    output = mock_stdout.getvalue()
    self.assertIn(f"Downloading artifact to {filename} ...", output)
    self.assertIn("Download complete!", output)

  @mock.patch(
      "urllib.request.urlretrieve",
      side_effect=urllib.error.URLError("test error"),
  )
  @mock.patch("os.makedirs")
  def test_download_url_to_file_failure(self, mock_makedirs, mock_urlretrieve):
    mock_stdout = io.StringIO()
    filename = "/tmp/test_file.txt"
    url = "http://example.com/file"
    with mock.patch("sys.stdout", mock_stdout):
      flywheel_cli._download_url_to_file(url, filename)

    mock_makedirs.assert_called_once_with(
        os.path.dirname(filename), exist_ok=True
    )
    mock_urlretrieve.assert_called_once_with(
        url, filename, reporthook=flywheel_cli._reporthook
    )
    self.assertIn(
        f"\n[ERROR] Error downloading artifact {url}: <urlopen error test"
        " error>",
        mock_stdout.getvalue(),
    )


if __name__ == "__main__":
  absltest.main()
