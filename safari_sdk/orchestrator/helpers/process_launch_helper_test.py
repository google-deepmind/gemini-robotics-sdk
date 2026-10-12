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

"""Unit tests for process_launch_helper.py."""

import os
import signal
import tempfile
from unittest import mock

from absl.testing import absltest

from safari_sdk.orchestrator.helpers import process_launch_helper


class ProcessLaunchHelperTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_download_dir = tempfile.mkdtemp()
    self.launcher = process_launch_helper.ProcessLaunchHelper(
        download_dir=self.test_download_dir,
        robot_id="test_robot_1",
    )

  def tearDown(self):
    super().tearDown()
    self.launcher.processes.clear()

  @mock.patch("subprocess.Popen")
  def test_launch_commands_in_sequence(self, mock_popen):
    mock_process1 = mock.MagicMock()
    mock_process1.pid = 1001

    mock_process2 = mock.MagicMock()
    mock_process2.pid = 1002

    mock_popen.side_effect = [mock_process1, mock_process2]

    commands = [
        "echo 'Step 1 on {robot_id}'",
        "{download_dir}/bin --dir={artifact_dir}",
    ]
    launched = self.launcher.launch_commands_in_sequence(
        commands, delay_between_commands_sec=0.0
    )

    self.assertLen(launched, 2)
    self.assertEqual(mock_popen.call_count, 2)
    self.assertLen(self.launcher.log_files, 2)

    for idx, log_file_path in enumerate(self.launcher.log_files):
      self.assertEqual(os.path.dirname(log_file_path), "/tmp")
      self.assertTrue(os.path.exists(log_file_path))
      log_filename = os.path.basename(log_file_path)
      self.assertRegex(
          log_filename,
          rf"^proc-{idx + 1}_\d{{4}}_\d{{2}}_\d{{2}}_\d{{2}}_\d{{2}}_\d{{2}}\.log$",
      )

    for idx, call in enumerate(mock_popen.call_args_list):
      args, kwargs = call
      cmd_args = args[0]
      shell_script = cmd_args[-1]
      self.assertIn("2>&1 | tee -a", shell_script)
      self.assertIn(self.launcher.log_files[idx], shell_script)
      self.assertIsNone(kwargs.get("stdin"))
      self.assertIsNone(kwargs.get("stdout"))
      self.assertIsNone(kwargs.get("stderr"))
      self.assertTrue(kwargs.get("start_new_session"))
      self.assertEqual(
          kwargs.get("env", {}).get("ORCA_ARTIFACT_DIR"),
          self.test_download_dir,
      )

  def test_monitor_processes_all_running(self):
    mock_proc1 = mock.MagicMock()
    mock_proc1.poll.return_value = None
    mock_proc2 = mock.MagicMock()
    mock_proc2.poll.return_value = None

    self.launcher.processes = [mock_proc1, mock_proc2]
    self.assertTrue(self.launcher.monitor_processes())

  def test_monitor_processes_one_exited(self):
    mock_proc1 = mock.MagicMock()
    mock_proc1.poll.return_value = None
    mock_proc2 = mock.MagicMock()
    mock_proc2.poll.return_value = 1
    mock_proc2.pid = 9999

    self.launcher.processes = [mock_proc1, mock_proc2]
    self.assertFalse(self.launcher.monitor_processes())

  @mock.patch("os.killpg")
  @mock.patch("os.getpgid", return_value=12345)
  def test_terminate_all_processes(self, mock_getpgid, mock_killpg):
    mock_proc = mock.MagicMock()
    mock_proc.pid = mock_getpgid.return_value
    mock_proc.poll.side_effect = [None, 0]

    self.launcher.processes = [mock_proc]
    self.launcher.terminate_all_processes(timeout_sec=0.1)

    mock_killpg.assert_called_with(12345, signal.SIGTERM)
    self.assertEmpty(self.launcher.running_processes)

  def test_build_terminal_command_with_gnome_terminal(self):
    with mock.patch("shutil.which", return_value="/usr/bin/gnome-terminal"):
      cmd = self.launcher.build_terminal_command(
          "./fake_binary", "proc-1", cwd="/tmp/my_artifacts"
      )
      self.assertEqual(cmd[0], "gnome-terminal")
      self.assertIn("--wait", cmd)
      self.assertIn("--working-directory=/tmp/my_artifacts", cmd)
      shell_script = cmd[-1]
      self.assertIn("cd '/tmp/my_artifacts';", shell_script)
      self.assertIn("./fake_binary", shell_script)

  def test_build_terminal_command_with_xfce4_terminal(self):
    def fake_which(name):
      return "/usr/bin/xfce4-terminal" if name == "xfce4-terminal" else None

    with mock.patch("shutil.which", side_effect=fake_which):
      cmd = self.launcher.build_terminal_command(
          "./fake_binary", "proc-1", cwd="/tmp/my_artifacts"
      )
      self.assertEqual(cmd[0], "xfce4-terminal")
      self.assertIn("--disable-server", cmd)
      self.assertIn("--working-directory=/tmp/my_artifacts", cmd)

  def test_build_terminal_command_with_konsole(self):
    def fake_which(name):
      return "/usr/bin/konsole" if name == "konsole" else None

    with mock.patch("shutil.which", side_effect=fake_which):
      cmd = self.launcher.build_terminal_command(
          "./fake_binary", "proc-1", cwd="/tmp/my_artifacts"
      )
      self.assertEqual(cmd[0], "konsole")
      self.assertIn("--nofork", cmd)
      self.assertIn("--workdir", cmd)
      self.assertIn("/tmp/my_artifacts", cmd)

  def test_build_terminal_command_fallback_without_gui(self):
    with mock.patch("shutil.which", return_value=None):
      cmd = self.launcher.build_terminal_command(
          "./fake_binary", "proc-1", cwd="/tmp/my_artifacts"
      )
      self.assertEqual(cmd[0], "bash")
      self.assertEqual(cmd[1], "-ic")
      self.assertIn("cd '/tmp/my_artifacts';", cmd[2])
      self.assertIn("./fake_binary", cmd[2])


if __name__ == "__main__":
  absltest.main()
