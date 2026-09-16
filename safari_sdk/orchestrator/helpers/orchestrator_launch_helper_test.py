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

"""Unit tests for orchestrator_launch_helper.py."""

import logging
import os
import signal
import tempfile
from unittest import mock

from absl.testing import absltest

from safari_sdk.orchestrator.client.dataclass import api_response
from safari_sdk.orchestrator.client.dataclass import work_unit as work_unit_data
from safari_sdk.orchestrator.helpers import orchestrator_helper
from safari_sdk.orchestrator.helpers import orchestrator_launch_helper


class OrchestratorLaunchHelperTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_helper = mock.create_autospec(
        orchestrator_helper.OrchestratorHelper
    )
    self.test_download_dir = tempfile.mkdtemp()
    self.launcher = orchestrator_launch_helper.OrchestratorLaunchHelper(
        job_type_codes=["agentic"],
        robot_id="test_robot_1",
        download_dir=self.test_download_dir,
        orchestrator_helper_instance=self.mock_helper,
    )

  def tearDown(self):
    super().tearDown()
    self.launcher._processes.clear()

  def test_connect_success(self):
    self.mock_helper.connect.return_value = (
        api_response.OrchestratorAPIResponse(
            success=True, robot_id="test_robot_1"
        )
    )
    result = self.launcher.connect()
    self.assertTrue(result)
    self.mock_helper.connect.assert_called_once()

  def test_connect_failure(self):
    self.mock_helper.connect.return_value = (
        api_response.OrchestratorAPIResponse(
            success=False, error_message="Connection refused"
        )
    )
    result = self.launcher.connect()
    self.assertFalse(result)

  def test_disconnect(self):
    self.mock_helper.connect.return_value = (
        api_response.OrchestratorAPIResponse(success=True)
    )
    self.launcher.connect()
    self.launcher.disconnect()
    self.mock_helper.disconnect.assert_called_once()

  def test_fetch_launch_info_from_orchestrator(self):
    wu_context = work_unit_data.WorkUnitContext(
        launcherArtifacts=[
            work_unit_data.LauncherArtifact(
                artifactId="art_launcher_1",
            ),
            work_unit_data.LauncherArtifact(
                artifactId="art_launcher_2",
            ),
            work_unit_data.LauncherArtifact(
                artifactId="art_launcher_custom",
                launchCommand="python3 -m custom_service --port=8080",
            ),
        ],
        policies=[
            work_unit_data.PolicyDetails(
                name="harness_spec",
                usage="harness",
                artifactIds=["art_harness_1"],
                parameters=[
                    work_unit_data.KvMsg(
                        key="launch_command",
                        value=work_unit_data.KvMsgValue(
                            stringValue="python3 -m harness_server"
                        ),
                        type=work_unit_data.KvMsgValueType.KV_MSG_VALUE_TYPE_STRING,
                    )
                ],
            ),
            work_unit_data.PolicyDetails(
                name="er_spec",
                usage="er",
                artifactIds=["art_er_1"],
            ),
            work_unit_data.PolicyDetails(
                name="vla_spec",
                usage="vla",
                artifactIds=["art_vla_1"],
            ),
        ],
    )
    wu = work_unit_data.WorkUnit(
        workUnitId="wu_123",
        robotJobId="job_456",
        context=wu_context,
    )
    self.mock_helper.get_current_work_unit.return_value = (
        api_response.OrchestratorAPIResponse(
            success=True,
            work_unit=wu,
            launch_command="echo 'Starting orchestrator'",
        )
    )

    artifact_ids, launch_commands = (
        self.launcher.fetch_launch_info_from_orchestrator()
    )

    self.assertIn("art_launcher_1", artifact_ids)
    self.assertIn("art_launcher_2", artifact_ids)
    self.assertIn("art_launcher_custom", artifact_ids)
    self.assertIn("art_harness_1", artifact_ids)
    self.assertIn("art_er_1", artifact_ids)
    self.assertIn("art_vla_1", artifact_ids)
    self.assertIn("python3 -m custom_service --port=8080", launch_commands)
    self.assertIn("python3 -m harness_server", launch_commands)
    self.assertIn("echo 'Starting orchestrator'", launch_commands)

  def test_fetch_launch_info_with_launch_order(self):
    wu_context = work_unit_data.WorkUnitContext(
        launcherArtifacts=[
            work_unit_data.LauncherArtifact(
                artifactId="art_secondary",
                launchCommand="cmd_secondary",
                launchOrder=2,
            ),
            work_unit_data.LauncherArtifact(
                artifactId="art_primary",
                launchCommand="cmd_primary",
                launchOrder=1,
            ),
        ],
    )
    wu = work_unit_data.WorkUnit(
        workUnitId="wu_123",
        robotJobId="job_456",
        context=wu_context,
    )
    self.mock_helper.get_current_work_unit.return_value = (
        api_response.OrchestratorAPIResponse(
            success=True,
            work_unit=wu,
        )
    )

    artifact_ids, launch_commands = (
        self.launcher.fetch_launch_info_from_orchestrator()
    )

    self.assertEqual(artifact_ids, ["art_primary", "art_secondary"])
    self.assertEqual(launch_commands, ["cmd_primary", "cmd_secondary"])

  @mock.patch("urllib.request.urlretrieve")
  def test_download_artifacts(self, mock_urlretrieve):
    def fake_retrieve(url, path):
      del url  # Unused.
      with open(path, "w") as f:
        f.write("fake_binary_content")

    mock_urlretrieve.side_effect = fake_retrieve
    self.mock_helper.get_artifact.side_effect = [
        api_response.OrchestratorAPIResponse(
            success=True,
            artifact=mock.MagicMock(
                uri="http://storage.googleapis.com/test/bin_agentic",
                version="v1.0",
                commitTime="2026-08-20T00:00:00Z",
            ),
        ),
        api_response.OrchestratorAPIResponse(
            success=True,
            artifact=mock.MagicMock(
                uri="http://storage.googleapis.com/test/bin_policy",
                version="v1.0",
                commitTime="2026-08-20T00:00:00Z",
            ),
        ),
    ]

    paths = self.launcher.download_artifacts(["art_1", "art_2"])

    self.assertLen(paths, 2)
    self.assertIn("art_1", paths)
    self.assertIn("art_2", paths)
    self.assertEqual(mock_urlretrieve.call_count, 2)
    self.assertTrue(self.launcher.is_artifact_cached("art_1"))
    self.assertTrue(self.launcher.is_artifact_cached("art_2"))

  @mock.patch("urllib.request.urlretrieve")
  def test_download_artifacts_cached_skips_download(self, mock_urlretrieve):
    def fake_retrieve(url, path):
      del url  # Unused.
      with open(path, "w") as f:
        f.write("fake_binary_content")

    mock_urlretrieve.side_effect = fake_retrieve
    self.mock_helper.get_artifact.return_value = (
        api_response.OrchestratorAPIResponse(
            success=True,
            artifact=mock.MagicMock(
                uri="http://storage.googleapis.com/test/bin_agentic",
                version="v1.0",
                commitTime="2026-08-20T00:00:00Z",
            ),
        )
    )

    # First download
    paths1 = self.launcher.download_artifacts(["art_1"])
    self.assertEqual(mock_urlretrieve.call_count, 1)

    # Second download against same artifact / version -> should be skipped!
    paths2 = self.launcher.download_artifacts(["art_1"])
    self.assertEqual(mock_urlretrieve.call_count, 1)
    self.assertEqual(paths1["art_1"], paths2["art_1"])

  @mock.patch("urllib.request.urlretrieve")
  def test_download_artifacts_version_mismatch_warns_and_redownloads(
      self, mock_urlretrieve
  ):
    def fake_retrieve(url, path):
      del url  # Unused.
      with open(path, "w") as f:
        f.write("fake_binary_content")

    mock_urlretrieve.side_effect = fake_retrieve

    # Initial version on server
    self.mock_helper.get_artifact.return_value = (
        api_response.OrchestratorAPIResponse(
            success=True,
            artifact=mock.MagicMock(
                uri="http://storage.googleapis.com/test/bin_agentic",
                version="v1.0",
                commitTime="2026-08-20T00:00:00Z",
            ),
        )
    )
    self.launcher.download_artifacts(["art_1"])
    self.assertEqual(mock_urlretrieve.call_count, 1)

    # Server version changes to v2.0
    self.mock_helper.get_artifact.return_value = (
        api_response.OrchestratorAPIResponse(
            success=True,
            artifact=mock.MagicMock(
                uri="http://storage.googleapis.com/test/bin_agentic",
                version="v2.0",
                commitTime="2026-08-20T01:00:00Z",
            ),
        )
    )

    is_same, local_v, server_v = self.launcher.check_artifact_version("art_1")
    self.assertFalse(is_same)
    self.assertIsNotNone(local_v)
    self.assertIsNotNone(server_v)
    assert local_v is not None and server_v is not None
    self.assertIn("v:v1.0", local_v)
    self.assertIn("v:v2.0", server_v)

    # Redownload should detect mismatch, clear old cache, and redownload
    with self.assertLogs(level="WARNING") as log:
      self.launcher.download_artifacts(["art_1"])
      self.assertEqual(mock_urlretrieve.call_count, 2)
      self.assertTrue(
          any("does not match server version" in line for line in log.output)
      )

    # After update, version should match
    is_same_now, _, _ = self.launcher.check_artifact_version("art_1")
    self.assertTrue(is_same_now)

  def test_download_artifacts_missing_url_raises_runtime_error(self):
    self.mock_helper.get_artifact.return_value = (
        api_response.OrchestratorAPIResponse(
            success=False,
            error_message="Artifact not found",
        )
    )
    self.mock_helper.get_artifact_uri.return_value = (
        api_response.OrchestratorAPIResponse(
            success=False,
            error_message="URI not found",
        )
    )

    with self.assertRaisesRegex(
        RuntimeError,
        "Failed to get download URI for artifact \\[art_missing\\]",
    ):
      self.launcher.download_artifacts(["art_missing"])

  @mock.patch("urllib.request.urlretrieve")
  def test_download_artifacts_network_error_raises_runtime_error(
      self, mock_urlretrieve
  ):
    self.mock_helper.get_artifact.return_value = (
        api_response.OrchestratorAPIResponse(
            success=True,
            artifact=mock.MagicMock(
                uri="http://storage.googleapis.com/test/bin_fail",
                version="v1.0",
                commitTime="2026-08-20T00:00:00Z",
            ),
        )
    )
    mock_urlretrieve.side_effect = ConnectionResetError(
        "Connection reset by peer"
    )

    with self.assertRaisesRegex(
        RuntimeError, "Failed downloading artifact \\[art_fail\\]"
    ):
      self.launcher.download_artifacts(["art_fail"])

  @mock.patch("urllib.request.urlretrieve")
  def test_download_artifacts_missing_file_on_disk_raises_runtime_error(
      self, mock_urlretrieve
  ):
    del mock_urlretrieve  # urlretrieve does not create the file
    self.mock_helper.get_artifact.return_value = (
        api_response.OrchestratorAPIResponse(
            success=True,
            artifact=mock.MagicMock(
                uri="http://storage.googleapis.com/test/bin_missing_file",
                version="v1.0",
                commitTime="2026-08-20T00:00:00Z",
            ),
        )
    )

    with self.assertRaisesRegex(RuntimeError, "does not exist at"):
      self.launcher.download_artifacts(["art_missing_file"])

  def test_run_aborts_on_artifact_download_failure(self):
    self.mock_helper.connect.return_value = (
        api_response.OrchestratorAPIResponse(success=True)
    )
    wu = work_unit_data.WorkUnit(
        workUnitId="wu_123",
        robotJobId="job_456",
        context=work_unit_data.WorkUnitContext(
            launcherArtifacts=[
                work_unit_data.LauncherArtifact(artifactId="art_bad"),
            ]
        ),
    )
    self.mock_helper.get_current_work_unit.return_value = (
        api_response.OrchestratorAPIResponse(
            success=True,
            work_unit=wu,
        )
    )
    # Artifact lookup fails
    self.mock_helper.get_artifact.return_value = (
        api_response.OrchestratorAPIResponse(success=False)
    )
    self.mock_helper.get_artifact_uri.return_value = (
        api_response.OrchestratorAPIResponse(success=False)
    )

    exit_code = self.launcher.run()

    self.assertEqual(exit_code, 1)
    self.mock_helper.disconnect.assert_called_once()

  def test_clear_artifact_cache(self):
    test_file = os.path.join(self.test_download_dir, "test_file.bin")
    with open(test_file, "w") as f:
      f.write("data")

    self.launcher._save_cache_manifest({
        "art_test": {
            "artifact_id": "art_test",
            "server_fingerprint": "v:1.0",
            "local_file_path": test_file,
            "target_path": test_file,
        }
    })
    self.assertTrue(self.launcher.is_artifact_cached("art_test"))

    self.launcher.clear_artifact_cache("art_test")
    self.assertFalse(self.launcher.is_artifact_cached("art_test"))
    self.assertFalse(os.path.exists(test_file))

  @mock.patch("subprocess.Popen")
  def test_launch_commands_in_sequence(self, mock_popen):
    mock_process1 = mock.MagicMock()
    mock_process1.pid = 1001

    mock_process2 = mock.MagicMock()
    mock_process2.pid = 1002

    mock_popen.side_effect = [mock_process1, mock_process2]

    commands = ["echo 'Step 1'", "echo 'Step 2'"]
    launched = self.launcher.launch_commands_in_sequence(
        commands, delay_between_commands_sec=0.0
    )

    self.assertLen(launched, 2)
    self.assertEqual(mock_popen.call_count, 2)

    # Verify each call was given independent I/O (not piped back to launcher)
    for call in mock_popen.call_args_list:
      _, kwargs = call
      self.assertIsNone(kwargs.get("stdin"))
      self.assertIsNone(kwargs.get("stdout"))
      self.assertIsNone(kwargs.get("stderr"))
      self.assertTrue(kwargs.get("start_new_session"))

  def test_monitor_processes_all_running(self):
    mock_proc1 = mock.MagicMock()
    mock_proc1.poll.return_value = None
    mock_proc2 = mock.MagicMock()
    mock_proc2.poll.return_value = None

    self.launcher._processes = [mock_proc1, mock_proc2]
    self.assertTrue(self.launcher.monitor_processes())

  def test_monitor_processes_one_exited(self):
    mock_proc1 = mock.MagicMock()
    mock_proc1.poll.return_value = None
    mock_proc2 = mock.MagicMock()
    mock_proc2.poll.return_value = 1
    mock_proc2.pid = 9999

    self.launcher._processes = [mock_proc1, mock_proc2]
    self.assertFalse(self.launcher.monitor_processes())

  @mock.patch("os.killpg")
  @mock.patch("os.getpgid", return_value=12345)
  def test_terminate_all_processes(self, mock_getpgid, mock_killpg):
    mock_proc = mock.MagicMock()
    mock_proc.pid = mock_getpgid.return_value
    mock_proc.poll.side_effect = [None, 0]

    self.launcher._processes = [mock_proc]
    self.launcher.terminate_all_processes(timeout_sec=0.1)

    mock_killpg.assert_called_with(12345, signal.SIGTERM)
    self.assertEmpty(self.launcher.running_processes)

  def test_format_and_log_work_unit_info(self):
    wu_context = work_unit_data.WorkUnitContext(
        scenePresetId="preset_123",
        sceneEpisodeIndex=3,
        orchestratorTaskId="task_789",
        scenePresetDetails=work_unit_data.ScenePresetDetails(
            setupInstructions="Place object on table",
            grouping=["group_a"],
            parameters=[
                work_unit_data.KvMsg(
                    key="param1",
                    value=work_unit_data.KvMsgValue(stringValue="val1"),
                    type=work_unit_data.KvMsgValueType.KV_MSG_VALUE_TYPE_STRING,
                )
            ],
            referenceImages=[
                work_unit_data.SceneReferenceImage(
                    artifactId="ref_art_1",
                    sourceTopic="/camera/image",
                    rawImageWidth=640,
                    rawImageHeight=480,
                    renderedCanvasWidth=320,
                    renderedCanvasHeight=240,
                )
            ],
            sceneObjects=[
                work_unit_data.SceneObject(
                    objectId="obj_1",
                    overlayTextLabels=work_unit_data.OverlayTextLabel(
                        labels=[work_unit_data.OverlayText(text="Target Cup")]
                    ),
                    evaluationLocation=work_unit_data.FixedLocation(
                        overlayIcon=work_unit_data.OverlayObjectIcon.OVERLAY_OBJECT_ICON_CIRCLE,
                        layerOrder=1,
                        rgbHexColorValue="#FF0000",
                        location=work_unit_data.PixelVector(
                            coordinate=work_unit_data.PixelLocation(
                                x=100, y=200
                            ),
                            direction=work_unit_data.PixelDirection(rad=1.57),
                        ),
                        containerArea=work_unit_data.ContainerArea(
                            circle=work_unit_data.ShapeCircle(
                                center=work_unit_data.PixelLocation(
                                    x=100, y=200
                                ),
                                radius=50,
                            ),
                            box=work_unit_data.ShapeBox(
                                x=80, y=180, w=40, h=40
                            ),
                        ),
                    ),
                    sceneReferenceImageArtifactId="ref_art_1",
                )
            ],
        ),
        policies=[
            work_unit_data.PolicyDetails(
                name="test_policy",
                description="test description",
                usage="vla",
                artifactIds=["art_vla"],
                parameters=[
                    work_unit_data.KvMsg(
                        key="model_name",
                        value=work_unit_data.KvMsgValue(
                            stringValue="gemini_robotics"
                        ),
                        type=work_unit_data.KvMsgValueType.KV_MSG_VALUE_TYPE_STRING,
                    )
                ],
            )
        ],
        launcherArtifacts=[
            work_unit_data.LauncherArtifact(
                artifactId="launcher_bin_1",
                launchCommand="run.sh",
                launchOrder=1,
            )
        ],
        successScores=[
            work_unit_data.SuccessScore(definition="grasp", score=1.0)
        ],
        questions=[
            work_unit_data.Question(
                question="Was the grasp successful?",
                questionnaireId="q_1",
                whenToAsk=[
                    work_unit_data.QuestionCondition.QUESTION_CONDITION_ALWAYS
                ],
                answerFormat=work_unit_data.AnswerType.ANSWER_TYPE_YES_NO,
                allowedAnswers=["Yes", "No"],
            )
        ],
    )
    wu = work_unit_data.WorkUnit(
        workUnitId="wu_999",
        robotJobId="job_888",
        obfuscatedJobTypeCode="agentic_eval",
        stage=work_unit_data.WorkUnitStage.WORK_UNIT_STAGE_CREATED,
        outcome=work_unit_data.WorkUnitOutcome.WORK_UNIT_OUTCOME_UNSPECIFIED,
        note="Initial state",
        context=wu_context,
    )

    formatted_str = (
        orchestrator_launch_helper.format_orchestrator_work_unit_info(wu)
    )
    self.assertIn("Work Unit ID: wu_999", formatted_str)
    self.assertIn("Robot Job ID: job_888", formatted_str)
    self.assertIn("Robot Job type: agentic_eval", formatted_str)
    self.assertIn("Scene Preset ID: preset_123", formatted_str)
    self.assertIn("Setup Instructions: Place object on table", formatted_str)
    self.assertIn("param1: val1", formatted_str)
    self.assertIn("Artifact ID: ref_art_1", formatted_str)
    self.assertIn("Object ID: obj_1", formatted_str)
    self.assertIn("Overlay Text Label: Target Cup", formatted_str)
    self.assertIn("Policy Name: test_policy", formatted_str)
    self.assertIn("model_name: gemini_robotics", formatted_str)
    self.assertIn("Artifact ID: launcher_bin_1", formatted_str)
    self.assertIn("Launch Command: run.sh", formatted_str)
    self.assertIn("Was the grasp successful?", formatted_str)

    with mock.patch.object(logging, "info") as mock_logging_info:
      orchestrator_launch_helper.log_orchestrator_work_unit_info(wu)
      mock_logging_info.assert_called_once()
      call_args = mock_logging_info.call_args[0]
      self.assertIn(formatted_str, call_args[1])

  @mock.patch.object(orchestrator_launch_helper, "log_work_unit_info")
  def test_fetch_launch_info_logs_work_unit_info_on_work_unit(
      self, mock_log_wu
  ):
    wu = work_unit_data.WorkUnit(
        workUnitId="wu_123",
        robotJobId="job_456",
        context=work_unit_data.WorkUnitContext(),
    )
    self.mock_helper.get_current_work_unit.return_value = (
        api_response.OrchestratorAPIResponse(
            success=True,
            work_unit=wu,
        )
    )

    self.launcher.fetch_launch_info_from_orchestrator()
    mock_log_wu.assert_called_once_with(wu)

  def test_init_missing_robot_id_and_hostname_raises_value_error(self):
    with self.assertRaises(ValueError):
      orchestrator_launch_helper.OrchestratorLaunchHelper(
          job_type_codes=["agentic"],
          robot_id=None,
          hostname=None,
          orchestrator_helper_instance=self.mock_helper,
      )

  def test_init_with_hostname_success(self):
    launcher = orchestrator_launch_helper.OrchestratorLaunchHelper(
        job_type_codes=["agentic"],
        hostname="robot_hostname_1",
        orchestrator_helper_instance=self.mock_helper,
    )
    self.assertIsNotNone(launcher)

  def test_build_terminal_command_with_gnome_terminal(self):
    with mock.patch("shutil.which", return_value="/usr/bin/gnome-terminal"):
      cmd = self.launcher._build_terminal_command(
          "./fake_binary", "proc-1", cwd="/tmp/my_artifacts"
      )
      self.assertEqual(cmd[0], "gnome-terminal")
      self.assertIn("--wait", cmd)
      self.assertIn("--working-directory=/tmp/my_artifacts", cmd)
      # Check that cd '/tmp/my_artifacts' is in the shell script
      shell_script = cmd[-1]
      self.assertIn("cd '/tmp/my_artifacts';", shell_script)
      self.assertIn("./fake_binary", shell_script)

  def test_build_terminal_command_with_xfce4_terminal(self):
    def fake_which(name):
      return "/usr/bin/xfce4-terminal" if name == "xfce4-terminal" else None

    with mock.patch("shutil.which", side_effect=fake_which):
      cmd = self.launcher._build_terminal_command(
          "./fake_binary", "proc-1", cwd="/tmp/my_artifacts"
      )
      self.assertEqual(cmd[0], "xfce4-terminal")
      self.assertIn("--disable-server", cmd)
      self.assertIn("--working-directory=/tmp/my_artifacts", cmd)

  def test_build_terminal_command_with_konsole(self):
    def fake_which(name):
      return "/usr/bin/konsole" if name == "konsole" else None

    with mock.patch("shutil.which", side_effect=fake_which):
      cmd = self.launcher._build_terminal_command(
          "./fake_binary", "proc-1", cwd="/tmp/my_artifacts"
      )
      self.assertEqual(cmd[0], "konsole")
      self.assertIn("--nofork", cmd)
      self.assertIn("--workdir", cmd)
      self.assertIn("/tmp/my_artifacts", cmd)

  def test_build_terminal_command_fallback_without_gui(self):
    with mock.patch("shutil.which", return_value=None):
      cmd = self.launcher._build_terminal_command(
          "./fake_binary", "proc-1", cwd="/tmp/my_artifacts"
      )
      self.assertEqual(cmd[0], "bash")
      self.assertEqual(cmd[1], "-ic")
      self.assertIn("cd '/tmp/my_artifacts';", cmd[2])
      self.assertIn("./fake_binary", cmd[2])


if __name__ == "__main__":
  absltest.main()
