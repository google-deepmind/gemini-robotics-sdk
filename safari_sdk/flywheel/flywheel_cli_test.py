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
import pathlib
import subprocess
import tempfile
from unittest import mock
import urllib.error

from absl import flags
from absl.testing import flagsaver
from absl.testing import parameterized

from absl.testing import absltest
from safari_sdk.flywheel import flywheel_cli


_DATA_STATS_TEST_DATA = {
    "taskDates": [
        {
            "robotId": "test_robot_id",
            "taskId": "test_task_id",
            "dates": ["20241201", "20241202"],
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
      (
          "narrow",
          None,
          "narrow",
          "TRAINING_TYPE_NARROW",
          None,
      ),
      (
          "gemini_robotics_v1",
          None,
          "gemini_robotics_v1",
          "TRAINING_TYPE_GEMINI_ROBOTICS_V1",
          None,
      ),
      (
          "narrow_with_robot_id",
          "test_robot_id",
          "narrow",
          "TRAINING_TYPE_NARROW",
          None,
      ),
      (
          "gemini_robotics_on_device_v1",
          "test_robot_id",
          "gemini_robotics_on_device_v1",
          "TRAINING_TYPE_GEMINI_ROBOTICS_ON_DEVICE_V1",
          None,
      ),
      (
          "narrow_with_max_episodes",
          None,
          "narrow",
          "TRAINING_TYPE_NARROW",
          50,
      ),
  )
  def test_train(self, robot_id, recipe, training_type, max_episodes):
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
    if max_episodes is not None:
      req_flags["max_episodes"] = max_episodes

    expected_body = {
        "training_data_filters": {
            "robot_id": robot_id if robot_id else None,
            "task_id": "test_task_id",
            "start_date": "20240101",
            "end_date": "20240102",
        },
        "training_type": training_type,
        "tracer": mock.ANY,
    }
    if max_episodes is not None:
      expected_body["training_data_filters"]["max_episode_count"] = max_episodes
      if recipe == "gemini_robotics_on_device_v1":
        expected_body["training_data_filters"]["seed"] = mock.ANY
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

  def test_train_small_run_auto_default_checkpoint(self):
    """Verifies auto-default checkpoint is 0 when max_steps < 100."""
    self.service_mock.startTraining.return_value.execute.return_value = {
        "training_job_id": "test_training_job_id"
    }
    req_flags = {
        "task_id": "test_task_id",
        "start_date": "20240101",
        "end_date": "20240102",
        "training_recipe": "gemini_robotics_on_device_v1",
        "max_training_steps": 50,
        # checkpoint_every_n_steps intentionally not set (None default).
    }

    with flagsaver.flagsaver(**req_flags):
      self._cli.handle_train()
      call_body = self.service_mock.startTraining.call_args[1]["body"]
      self.assertEqual(
          call_body["training_config"]["checkpoint_every_n_steps"], 0
      )

  def test_train_with_explicit_seed(self):
    self.service_mock.startTraining.return_value.execute.return_value = {
        "training_job_id": "test_training_job_id"
    }
    req_flags = {
        "task_id": "test_task_id",
        "start_date": "20240101",
        "end_date": "20240102",
        "training_recipe": "gemini_robotics_on_device_v1",
        "max_episodes": 10,
        "seed": 42,
    }

    with flagsaver.flagsaver(**req_flags):
      self._cli.handle_train()
      call_body = self.service_mock.startTraining.call_args[1]["body"]
      self.assertEqual(call_body["training_data_filters"]["seed"], 42)

  def test_train_generates_random_seed(self):
    self.service_mock.startTraining.return_value.execute.return_value = {
        "training_job_id": "test_training_job_id"
    }
    req_flags = {
        "task_id": "test_task_id",
        "start_date": "20240101",
        "end_date": "20240102",
        "training_recipe": "gemini_robotics_on_device_v1",
        "max_episodes": 10,
    }

    mock_stdout = io.StringIO()
    with flagsaver.flagsaver(**req_flags):
      with mock.patch("sys.stdout", mock_stdout):
        self._cli.handle_train()

    call_body = self.service_mock.startTraining.call_args[1]["body"]
    self.assertIn("seed", call_body["training_data_filters"])
    generated_seed = call_body["training_data_filters"]["seed"]
    self.assertIsInstance(generated_seed, int)
    # Verify seed is within JSON-safe integer range
    self.assertLessEqual(generated_seed, 2**53 - 1)
    self.assertGreaterEqual(generated_seed, 0)

    output = mock_stdout.getvalue()
    self.assertIn(f"Using generated random seed: {generated_seed}", output)

  def test_train_seed_without_max_episodes_raises_error(self):
    """Verifies that --seed without --max_episodes raises ValueError."""
    self.service_mock.startTraining.return_value.execute.return_value = {
        "training_job_id": "test_training_job_id"
    }
    req_flags = {
        "task_id": "test_task_id",
        "start_date": "20240101",
        "end_date": "20240102",
        "training_recipe": "gemini_robotics_on_device_v1",
        "seed": 42,
    }

    with flagsaver.flagsaver(**req_flags):
      with self.assertRaisesRegex(ValueError, "--seed requires --max_episodes"):
        self._cli.handle_train()

  @parameterized.named_parameters(
      (
          "text_output_no_data",
          False,
          {},
          "No data stats found.\n",
      ),
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
        self.service_mock.trainingDataDetails.assert_called_once_with(
            body={"tracer": mock.ANY}
        )
        self.service_mock.trainingDataDetails.return_value.execute.assert_called_once()
        self.assertEqual(mock_stdout.getvalue(), expected_output)

  @parameterized.named_parameters(
      (
          "text_output_no_jobs",
          False,
          {},
          "No training jobs found.\n",
      ),
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

      self.service_mock.trainingJobs.assert_called_once_with(
          body={"tracer": mock.ANY}
      )
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

      self.service_mock.servingJobs.assert_called_once_with(
          body={"tracer": mock.ANY}
      )
      self.service_mock.servingJobs.return_value.execute.assert_called_once()
      self.assertEqual(mock_stdout.getvalue(), expected_output)

  @parameterized.named_parameters(
      (
          "json_output_with_data",
          True,
          {"status": "RUNNING"},
          '{\n    "status": "RUNNING"\n}\n',
      ),
      ("json_output_no_data", True, {}, "{}\n"),
  )
  def test_status(self, json_output, return_value, expected_output):
    mock_stdout = io.StringIO()
    self.service_mock.trainingJobStatus.return_value.execute.return_value = (
        return_value
    )
    with flagsaver.flagsaver(
        json_output=json_output, training_job_id="test_job_id"
    ):
      with mock.patch("sys.stdout", mock_stdout):
        self._cli.handle_status()
        self.service_mock.trainingJobStatus.assert_called_once_with(
            body={"trainingJobId": "test_job_id", "tracer": mock.ANY}
        )
        self.service_mock.trainingJobStatus.return_value.execute.assert_called_once()
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
              "tracer": mock.ANY,
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
          False,
          [],
          [],
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
          False,
          [],
          [],
      ),
      (
          "with_cpu_keys_flags",
          {
              "model_checkpoint_path": "/test/path/model.chkpt",
              "use_cpu": True,
              "image_keys": ["image1", "image2"],
              "proprioception_keys": ["prop1", "prop2"],
          },
          False,
          60061,
          0.8,
          "/test/path/model.chkpt",
          True,
          ["image1", "image2"],
          ["prop1", "prop2"],
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
      use_cpu,
      image_keys,
      proprio_keys,
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
        with mock.patch("pathlib.Path.exists", return_value=True), mock.patch(
            "pathlib.Path.is_file", return_value=True
        ):
          self._cli.handle_serve()
        if expect_download:
          mock_download.assert_called_once()
        else:
          mock_download.assert_not_called()

        file_dir = os.path.dirname(checkpoint_path)
        file_name = os.path.basename(checkpoint_path)
        expected_docker_command = ["docker", "run", "-it", "--rm"]
        if not use_cpu:
          expected_docker_command.extend([
              "--gpus",
              "device=0",
              "-e",
              f"XLA_PYTHON_CLIENT_MEM_FRACTION={mem_fraction}",
          ])
        expected_docker_command.extend([
            "-p",
            f"{port}:60061",
            "-v",
            f"{file_dir}:/checkpoint",
            "google-deepmind/gemini_robotics_on_device:latest",
            f"--checkpoint_path=/checkpoint/{file_name}",
        ])
        if image_keys:
          expected_docker_command.append(f"--image_keys={','.join(image_keys)}")
        if proprio_keys:
          expected_docker_command.append(
              f"--proprio_keys={','.join(proprio_keys)}"
          )
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
                "tracer": mock.ANY,
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
              "tracer": mock.ANY,
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
          "artifact": {"uri": "test_uri_1"}
      }

      self._cli.handle_download_artifact_id()

      self.service_mock.loadArtifact.assert_called_once_with(
          body={"artifact_id": "test_artifact_id", "tracer": mock.ANY}
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
          body={"artifact_id": "test_artifact_id", "tracer": mock.ANY}
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
          body={"artifact_id": "test_artifact_id", "tracer": mock.ANY}
      )
      self.service_mock.loadArtifact.return_value.execute.assert_called_once()
      mock_input.assert_not_called()
      mock_download.assert_not_called()

  @mock.patch(
      "shutil.get_terminal_size", return_value=os.terminal_size((200, 24))
  )
  def test_print_responsive_table(self, _):
    mock_stdout = io.StringIO()
    headers = ["H1", "Header2"]
    rows = [["d1", "data2"], ["data1-long", "d2"]]
    with mock.patch("sys.stdout", mock_stdout):
      flywheel_cli._print_responsive_table(headers, rows)

    expected = (
        "-------------------\n"
        "H1          Header2\n"
        "-------------------\n"
        "d1          data2\n"
        "data1-long  d2\n"
    )
    self.assertEqual(mock_stdout.getvalue(), expected)

  @mock.patch(
      "shutil.get_terminal_size", return_value=os.terminal_size((30, 24))
  )
  def test_print_responsive_table_truncation(self, _):
    mock_stdout = io.StringIO()
    headers = ["Col1", "Col2"]
    rows = [
        ["short", "this_is_a_very_long_cell_value_that_should_be_truncated"]
    ]
    with mock.patch("sys.stdout", mock_stdout):
      flywheel_cli._print_responsive_table(headers, rows)

    expected = (
        "------------------------------\n"
        "Col1   Col2\n"
        "------------------------------\n"
        "short  this_is_a_very_long_...\n"
    )
    self.assertEqual(mock_stdout.getvalue(), expected)

  def test_show_help(self):
    mock_stdout = io.StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      flywheel_cli.show_help()
      output = mock_stdout.getvalue()
      self.assertIn(flywheel_cli._HELP_STRING, output)
      # Verify show_help includes argparse usage info (consistent with --help).
      self.assertIn("usage:", output)

  def test_cli_main_help(self):
    mock_stdout = io.StringIO()
    with mock.patch("sys.argv", ["flywheel-cli", "--help"]):
      with mock.patch("sys.stdout", mock_stdout):
        with self.assertRaises(SystemExit):
          flywheel_cli.cli_main()
        self.assertIn(
            flywheel_cli._HELP_STRING,
            mock_stdout.getvalue(),
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
          "train_missing_training_recipe",
          "train",
          {
              "api_key": "test_api_key",
              "task_id": "test_task_id",
              "start_date": "20240101",
              "end_date": "20240102",
          },
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
          "train_max_episodes_zero",
          "train",
          {
              "api_key": "test_api_key",
              "task_id": "test_task_id",
              "start_date": "20240101",
              "end_date": "20240102",
              "max_episodes": 0,
          },
          ValueError,
      ),
      (
          "train_max_episodes_negative",
          "train",
          {
              "api_key": "test_api_key",
              "task_id": "test_task_id",
              "start_date": "20240101",
              "end_date": "20240102",
              "max_episodes": -5,
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
      (
          "status_missing_training_job_id",
          "status",
          {
              "api_key": "test_api_key",
          },
          ValueError,
      ),
      (
          "download_bad_artifact_id_with_period",
          "download",
          {
              "api_key": "test_api_key",
              "artifact_id": "test.pb",
          },
          ValueError,
      ),
      (
          "download_bad_artifact_id_starts_with_hyphen",
          "download",
          {
              "api_key": "test_api_key",
              "artifact_id": "-invalid_id",
          },
          ValueError,
      ),
      (
          "download_bad_artifact_id_too_short",
          "download",
          {
              "api_key": "test_api_key",
              "artifact_id": "a",
          },
          ValueError,
      ),
      (
          "download_bad_artifact_id_too_long",
          "download",
          {
              "api_key": "test_api_key",
              "artifact_id": "a" * 37,
          },
          ValueError,
      ),
      (
          "upload_data_bad_robot_id_too_long",
          "upload_data",
          {
              "api_key": "test_api_key",
              "upload_data_robot_id": "a" * 64,
              "upload_data_directory": "/tmp/data",
          },
          flags.IllegalFlagValueError,
      ),
      (
          "upload_data_bad_robot_id_bad_chars",
          "upload_data",
          {
              "api_key": "test_api_key",
              "upload_data_robot_id": "bad/id",
              "upload_data_directory": "/tmp/data",
          },
          flags.IllegalFlagValueError,
      ),
      (
          "train_missing_proprioception_keys",
          "train",
          {
              "api_key": "test_api_key",
              "task_id": "test_task_id",
              "start_date": "20240101",
              "end_date": "20240102",
              "training_recipe": "gemini_robotics_on_device_v1",
          },
          ValueError,
      ),
      (
          "train_bad_max_training_steps",
          "train",
          {
              "api_key": "test_api_key",
              "task_id": "test_task_id",
              "start_date": "20240101",
              "end_date": "20240102",
              "max_training_steps": 100000,
          },
          flags.IllegalFlagValueError,
      ),
      (
          "train_bad_checkpoint_every_n_steps",
          "train",
          {
              "api_key": "test_api_key",
              "task_id": "test_task_id",
              "start_date": "20240101",
              "end_date": "20240102",
              "checkpoint_every_n_steps": 1,
          },
          flags.IllegalFlagValueError,
      ),
      (
          "train_negative_max_training_steps",
          "train",
          {
              "api_key": "test_api_key",
              "task_id": "test_task_id",
              "start_date": "20240101",
              "end_date": "20240102",
              "max_training_steps": -1,
          },
          flags.IllegalFlagValueError,
      ),
      (
          "train_negative_checkpoint_every_n_steps",
          "train",
          {
              "api_key": "test_api_key",
              "task_id": "test_task_id",
              "start_date": "20240101",
              "end_date": "20240102",
              "checkpoint_every_n_steps": -1,
          },
          flags.IllegalFlagValueError,
      ),
      (
          "train_small_run_with_custom_checkpoint",
          "train",
          {
              "api_key": "test_api_key",
              "task_id": "test_task_id",
              "start_date": "20240101",
              "end_date": "20240102",
              "max_training_steps": 50,
              "checkpoint_every_n_steps": 100,
          },
          flags.IllegalFlagValueError,
      ),
  )

  def test_parse_flags_errors(self, command, params, expected_exception):
    with self.assertRaises(expected_exception):
      with flagsaver.flagsaver(**params):
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
          "train_with_max_episodes",
          "train",
          {
              "api_key": "test_api_key",
              "task_id": "test_task_id",
              "start_date": "20240101",
              "end_date": "20240102",
              "training_recipe": "narrow",
              "max_episodes": 50,
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
          "download_valid_artifact_id",
          "download",
          {
              "api_key": "test_api_key",
              "artifact_id": "valid_artifact_id",
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
      (
          "upload_data",
          "upload_data",
          {
              "api_key": "test_api_key",
              "upload_data_robot_id": "valid_id",
              "upload_data_directory": "/tmp/data",
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
        with mock.patch("pathlib.Path.exists", return_value=True), mock.patch(
            "pathlib.Path.is_file", return_value=True
        ):
          self._cli.parse_flag(command)

    if (
        command == "serve"
        and params.get("training_recipe") == "gemini_robotics_on_device_v1"
    ):
      mock_subprocess_run.assert_called_once()
    else:
      mock_subprocess_run.assert_not_called()

  def test_strip_whitespace_from_flags_strips_task_id(self):
    """Verifies task_id whitespace is stripped and a warning is printed."""
    self.service_mock.startTraining.return_value.execute.return_value = {
        "training_job_id": "test_training_job_id"
    }
    mock_stdout = io.StringIO()
    with flagsaver.flagsaver(
        task_id=[" test_task_id "],
        start_date="20240101",
        end_date="20240102",
        training_recipe="narrow",
    ):
      with mock.patch("sys.stdout", mock_stdout):
        self._cli.parse_flag("train")

      # Verify the request body has the stripped task_id.
      call_args = self.service_mock.startTraining.call_args
      actual_task_id = call_args[1]["body"]["training_data_filters"]["task_id"]
      self.assertEqual(actual_task_id, ["test_task_id"])

    self.assertIn("WARNING", mock_stdout.getvalue())
    self.assertIn("task_id", mock_stdout.getvalue())

  def test_strip_whitespace_from_flags_strips_training_job_id(self):
    """Verifies training_job_id whitespace is stripped."""
    mock_stdout = io.StringIO()
    with flagsaver.flagsaver(
        training_job_id=" test_job_id ",
    ):
      with mock.patch("sys.stdout", mock_stdout):
        flywheel_cli._strip_whitespace_from_flags()

      self.assertEqual(flags.FLAGS["training_job_id"].value, "test_job_id")
    self.assertIn("WARNING", mock_stdout.getvalue())
    self.assertIn("training_job_id", mock_stdout.getvalue())

  def test_strip_whitespace_from_flags_no_warning_for_clean_flags(self):
    """Verifies no warning is printed when flags have no whitespace."""
    mock_stdout = io.StringIO()
    with flagsaver.flagsaver(
        task_id=["clean_task_id"],
        robot_id=["clean_robot_id"],
    ):
      with mock.patch("sys.stdout", mock_stdout):
        flywheel_cli._strip_whitespace_from_flags()

    self.assertEqual(mock_stdout.getvalue(), "")

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

  @mock.patch("subprocess.run")
  def test_download_artifact_id_docker_load_success(self, mock_subprocess_run):
    with flagsaver.flagsaver(artifact_id="test_docker_artifact"):
      self.service_mock.loadArtifact.return_value.execute.return_value = {
          "artifact": {"uri": "test_uri_1"}
      }
      mock_stdout = io.StringIO()
      with mock.patch("builtins.input", return_value="y") as mock_input:
        with mock.patch("sys.stdout", mock_stdout):
          with mock.patch.object(flywheel_cli, "_download_url_to_file"):
            self._cli.handle_download_artifact_id()

        self.assertEqual(mock_input.call_count, 2)
      self.service_mock.loadArtifact.assert_called_once()
      mock_subprocess_run.assert_called_once_with(
          ["docker", "load", "-i", mock.ANY],
          capture_output=True,
          check=True,
          text=True,
      )

  @mock.patch("subprocess.run")
  def test_download_artifact_id_docker_load_failure(self, mock_subprocess_run):
    with flagsaver.flagsaver(artifact_id="test_docker_artifact"):
      self.service_mock.loadArtifact.return_value.execute.return_value = {
          "artifact": {"uri": "test_uri_1"}
      }
      mock_subprocess_run.side_effect = subprocess.CalledProcessError(
          1, "docker load", stderr="error loading image"
      )
      mock_stdout = io.StringIO()
      with mock.patch("builtins.input", return_value="y"):
        with mock.patch("sys.stdout", mock_stdout):
          with mock.patch.object(flywheel_cli, "_download_url_to_file"):
            self._cli.handle_download_artifact_id()

      self.assertIn(
          "[ERROR] Failed to load docker image",
          mock_stdout.getvalue(),
      )

  def test_train_with_only_successful_episodes(self):
    with flagsaver.flagsaver(
        training_recipe="narrow",
        task_id=["task1"],
        start_date="20240101",
        end_date="20240102",
        only_successful_episodes=True,
        api_key="test_key",
    ):
      self.service_mock.startTraining.return_value.execute.return_value = {
          "job_id": "job_1"
      }
      self._cli.handle_train()

      self.service_mock.startTraining.assert_called_once()
      call_args = self.service_mock.startTraining.call_args[1]["body"]
      self.assertTrue(
          call_args["training_data_filters"]["only_successful_episodes"]
      )

  def test_train_auto_checkpoint_calculation(self):
    with flagsaver.flagsaver(
        training_recipe="gemini_robotics_on_device_v1",
        task_id=["task1"],
        start_date="20240101",
        end_date="20240102",
        max_training_steps=5000,
        checkpoint_every_n_steps=0,
        api_key="test_key",
        proprioception_keys=["proprio1"],
    ):
      self.service_mock.startTraining.return_value.execute.return_value = {
          "job_id": "job_1"
      }
      self._cli.handle_train()

      self.service_mock.startTraining.assert_called_once()
      call_args = self.service_mock.startTraining.call_args[1]["body"]
      self.assertEqual(
          call_args["training_config"]["checkpoint_every_n_steps"],
          200,
      )

  def test_download_training_artifacts_file_exists_no_overwrite(self):
    with flagsaver.flagsaver(training_job_id="job_1", json_output=False):
      self.service_mock.trainingArtifact.return_value.execute.return_value = {
          "uris": ["http://example.com/checkpoint_1.ckpt"]
      }
      mock_stdout = io.StringIO()
      with mock.patch("builtins.input", side_effect=["0", "", "n"]):
        with mock.patch("os.path.exists", return_value=True):
          with mock.patch("sys.stdout", mock_stdout):
            self._cli.handle_download_training_artifacts()

      self.assertIn("Download cancelled.", mock_stdout.getvalue())

  def test_download_training_artifacts_invalid_input(self):
    with flagsaver.flagsaver(training_job_id="job_1", json_output=False):
      self.service_mock.trainingArtifact.return_value.execute.return_value = {
          "uris": ["http://example.com/checkpoint_1.ckpt"]
      }
      mock_stdout = io.StringIO()
      with mock.patch("builtins.input", return_value="invalid"):
        with mock.patch("sys.stdout", mock_stdout):
          self._cli.handle_download_training_artifacts()

      self.assertIn(
          "Invalid input. Please enter a number.",
          mock_stdout.getvalue(),
      )

  @mock.patch("subprocess.run")
  def test_serve_on_device_docker_failure_latest_hint(
      self, mock_subprocess_run
  ):
    with flagsaver.flagsaver(
        training_recipe="gemini_robotics_on_device_v1",
        training_job_id="job_1",
        docker_tag="latest",
        json_output=False,
        model_checkpoint_path="/tmp/fake.chkpt",
    ):
      self.service_mock.trainingArtifact.return_value.execute.return_value = {
          "uris": ["http://example.com/checkpoint_1.ckpt"]
      }
      mock_subprocess_run.side_effect = subprocess.CalledProcessError(
          1, "docker run", stderr="image not found"
      )
      mock_stdout = io.StringIO()
      with mock.patch.object(flywheel_cli, "_download_url_to_file"):
        with mock.patch("pathlib.Path.exists", return_value=True):
          with mock.patch("sys.stdout", mock_stdout):
            self._cli.handle_serve()

      self.assertIn(
          "Did you forget to load the docker image?",
          mock_stdout.getvalue(),
      )

  @mock.patch("subprocess.run")
  def test_serve_on_device_docker_failure_versioned_hint(
      self, mock_subprocess_run
  ):
    with flagsaver.flagsaver(
        training_recipe="gemini_robotics_on_device_v1",
        training_job_id="job_1",
        docker_tag="1.0.0",
        json_output=False,
        model_checkpoint_path="/tmp/fake.chkpt",
    ):
      self.service_mock.trainingArtifact.return_value.execute.return_value = {
          "uris": ["http://example.com/checkpoint_1.ckpt"]
      }
      mock_subprocess_run.side_effect = subprocess.CalledProcessError(
          1, "docker run", stderr="image not found"
      )
      mock_stdout = io.StringIO()
      with mock.patch.object(flywheel_cli, "_download_url_to_file"):
        with mock.patch("pathlib.Path.exists", return_value=True):
          with mock.patch("sys.stdout", mock_stdout):
            self._cli.handle_serve()

      self.assertIn(
          "To use a versioned docker image, be sure to load it first.",
          mock_stdout.getvalue(),
      )

  def test_strip_whitespace_from_flags_strips_robot_id(self):
    with flagsaver.flagsaver(robot_id=[" robot1 ", "robot2 "]):
      mock_stdout = io.StringIO()
      with mock.patch("sys.stdout", mock_stdout):
        flywheel_cli._strip_whitespace_from_flags()

      self.assertEqual(flywheel_cli._ROBOT_ID.value, ["robot1", "robot2"])

  def test_strip_whitespace_from_flags_strips_artifact_id(self):
    with flagsaver.flagsaver(artifact_id=" art1 "):
      mock_stdout = io.StringIO()
      with mock.patch("sys.stdout", mock_stdout):
        flywheel_cli._strip_whitespace_from_flags()

      self.assertEqual(flywheel_cli._ARTIFACT_ID.value, "art1")

  def test_data_stats_table_output(self):
    """Covers the table-printing branch in handle_data_stats."""
    mock_stdout = io.StringIO()
    self.service_mock.trainingDataDetails.return_value.execute.return_value = (
        _DATA_STATS_TEST_DATA
    )
    with flagsaver.flagsaver(json_output=False):
      with mock.patch("sys.stdout", mock_stdout):
        self._cli.handle_data_stats()

    output = mock_stdout.getvalue()
    self.assertIn("test_robot_id", output)
    self.assertIn("test_task_id", output)
    self.assertIn("2024-12-01", output)
    self.assertIn("2024-12-02", output)
    self.assertIn("100", output)
    self.assertIn("200", output)
    self.assertNotIn("Success count", output)
    self.assertNotIn("50", output)
    self.assertNotIn("150", output)

  @parameterized.named_parameters(
      (
          "with_filters",
          _TRAINING_JOB_WITH_FILTERS,
      ),
      (
          "no_filters",
          _TRAINING_JOB_NO_FILTERS,
      ),
  )
  def test_list_training_jobs_table_output(self, return_value):
    """Covers the table-printing branch in handle_list_training_jobs."""
    mock_stdout = io.StringIO()
    self.service_mock.trainingJobs.return_value.execute.return_value = (
        return_value
    )
    with flagsaver.flagsaver(json_output=False):
      with mock.patch("sys.stdout", mock_stdout):
        self._cli.handle_list_training_jobs()

    output = mock_stdout.getvalue()
    self.assertIn("test_training_job_id", output)
    self.assertIn("COMPLETED", output)


class ResolveDownloadPathTest(parameterized.TestCase):
  """Tests for the _resolve_download_path helper function."""

  @mock.patch("pathlib.Path.expanduser")
  @mock.patch("pathlib.Path.is_dir")
  def test_tilde_expansion_to_existing_directory(
      self, mock_is_dir, mock_expanduser
  ):
    """Tests tilde expansion when the resolved path is an existing directory."""
    # Mock expanduser to simulate ~ -> /home/testuser
    mock_expanduser.return_value = pathlib.Path("/home/testuser/Downloads")
    mock_is_dir.return_value = True

    result = flywheel_cli._resolve_download_path(
        "~/Downloads", "/default/path/flywheel_checkpoint.ckpt"
    )

    self.assertEqual(
        result, "/home/testuser/Downloads/flywheel_checkpoint.ckpt"
    )

  @mock.patch("pathlib.Path.expanduser")
  @mock.patch("pathlib.Path.is_dir")
  def test_tilde_expansion_to_nonexistent_path(
      self, mock_is_dir, mock_expanduser
  ):
    """Tests tilde expansion when the resolved path does not exist."""
    mock_expanduser.return_value = pathlib.Path("/home/testuser/new_file.ckpt")
    mock_is_dir.return_value = False

    result = flywheel_cli._resolve_download_path(
        "~/new_file.ckpt", "/default/path/flywheel_checkpoint.ckpt"
    )

    self.assertEqual(result, "/home/testuser/new_file.ckpt")

  @mock.patch("pathlib.Path.is_dir")
  def test_existing_directory_input(self, mock_is_dir):
    """Tests that an existing directory gets the default filename appended."""
    mock_is_dir.return_value = True

    result = flywheel_cli._resolve_download_path(
        "/tmp/models", "/default/path/flywheel_checkpoint.ckpt"
    )

    self.assertEqual(result, "/tmp/models/flywheel_checkpoint.ckpt")

  @mock.patch("pathlib.Path.is_dir")
  def test_existing_file_input(self, mock_is_dir):
    """Tests that an existing file path remains unchanged."""
    mock_is_dir.return_value = False

    result = flywheel_cli._resolve_download_path(
        "/tmp/existing_file.ckpt", "/default/path/flywheel_checkpoint.ckpt"
    )

    self.assertEqual(result, "/tmp/existing_file.ckpt")

  @mock.patch("pathlib.Path.is_dir")
  def test_nonexistent_path_input(self, mock_is_dir):
    """Tests that a non-existent path remains unchanged."""
    mock_is_dir.return_value = False

    result = flywheel_cli._resolve_download_path(
        "/tmp/new_dest", "/default/path/flywheel_checkpoint.ckpt"
    )

    self.assertEqual(result, "/tmp/new_dest")

  def test_empty_input(self):
    """Tests that empty input returns the default full path."""
    result = flywheel_cli._resolve_download_path(
        None, "/default/path/flywheel_checkpoint.ckpt"
    )

    self.assertEqual(result, "/default/path/flywheel_checkpoint.ckpt")

  def test_empty_string_input(self):
    """Tests that empty string input returns the default full path."""
    result = flywheel_cli._resolve_download_path(
        "", "/default/path/flywheel_checkpoint.ckpt"
    )

    self.assertEqual(result, "/default/path/flywheel_checkpoint.ckpt")


class FormatDateTest(absltest.TestCase):

  def test_yyyymmdd_format(self):
    self.assertEqual(flywheel_cli._format_date("20241201"), "2024-12-01")

  def test_yyyy_mm_dd_passthrough(self):
    self.assertEqual(flywheel_cli._format_date("2024-12-01"), "2024-12-01")

  def test_none_input(self):
    self.assertEqual(flywheel_cli._format_date(None), "None")

  def test_empty_string(self):
    self.assertEqual(flywheel_cli._format_date(""), "")

  def test_malformed_string(self):
    self.assertEqual(flywheel_cli._format_date("not-a-date"), "not-a-date")


if __name__ == "__main__":
  absltest.main()
