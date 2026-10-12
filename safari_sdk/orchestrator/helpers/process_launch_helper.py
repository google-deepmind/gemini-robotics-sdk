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

"""Helper library for sequential multi-window process launching and lifecycle management.

This module relies strictly on the Python standard library (no third-party
dependencies such as absl) so that it can be executed directly in bare host
Python environments as well as inside container environments.
"""

from collections.abc import Sequence
import datetime
import logging
import os
import shutil
import signal
import subprocess
import threading
import time
from typing import Any


class ProcessLaunchHelper:
  """Helper class to manage sequential multi-window process launching and lifecycle."""

  def __init__(
      self,
      *,
      download_dir: str = "/tmp/orchestrator_launcher_artifacts",
      robot_id: str | None = None,
      log_dir: str = "/tmp",
  ):
    """Initializes the ProcessLaunchHelper.

    Args:
      download_dir: Local directory where artifacts/binaries are stored.
      robot_id: Optional ID of the robot for placeholder substitution.
      log_dir: Directory where per-process log files are saved. Defaults to
        '/tmp'.
    """
    self._download_dir = download_dir
    self._robot_id = robot_id
    self._log_dir = log_dir
    self._processes: list[subprocess.Popen[Any]] = []
    self._log_files: list[str] = []
    self._shutdown_event = threading.Event()
    self._signal_received = False

  @property
  def download_dir(self) -> str:
    """Returns the artifact download directory path."""
    return self._download_dir

  @download_dir.setter
  def download_dir(self, value: str) -> None:
    """Sets the artifact download directory path."""
    self._download_dir = value

  @property
  def robot_id(self) -> str | None:
    """Returns the robot ID."""
    return self._robot_id

  @robot_id.setter
  def robot_id(self, value: str | None) -> None:
    """Sets the robot ID."""
    self._robot_id = value

  @property
  def log_dir(self) -> str:
    """Returns the configured directory for process log files."""
    return self._log_dir

  @log_dir.setter
  def log_dir(self, value: str) -> None:
    """Sets the directory for process log files."""
    self._log_dir = value

  @property
  def log_files(self) -> list[str]:
    """Returns a shallow copy of the log file paths for launched processes."""
    return list(self._log_files)

  @property
  def processes(self) -> list[subprocess.Popen[Any]]:
    """Returns the underlying list of managed subprocesses."""
    return self._processes

  @processes.setter
  def processes(self, value: list[subprocess.Popen[Any]]) -> None:
    """Sets the underlying list of managed subprocesses."""
    self._processes = value

  @property
  def running_processes(self) -> list[subprocess.Popen[Any]]:
    """Returns a shallow copy of currently managed processes."""
    return list(self._processes)

  @property
  def shutdown_event(self) -> threading.Event:
    """Returns the shutdown threading.Event."""
    return self._shutdown_event

  @shutdown_event.setter
  def shutdown_event(self, value: threading.Event) -> None:
    """Sets the shutdown threading.Event."""
    self._shutdown_event = value

  @property
  def signal_received(self) -> bool:
    """Returns whether a termination signal has been received."""
    return self._signal_received

  @signal_received.setter
  def signal_received(self, value: bool) -> None:
    """Sets whether a termination signal has been received."""
    self._signal_received = value

  def find_terminal_emulator(self) -> str | None:
    """Finds an available terminal emulator binary on the system."""
    for term in [
        "gnome-terminal",
        "x-terminal-emulator",
        "xterm",
        "xfce4-terminal",
        "konsole",
    ]:
      if shutil.which(term):
        return term
    return None

  def _find_terminal_emulator(self) -> str | None:
    """Finds an available terminal emulator binary on the system."""
    return self.find_terminal_emulator()

  def _generate_log_file_path(
      self,
      title: str,
      start_time: datetime.datetime | None = None,
  ) -> str:
    """Generates a timestamped log file path in log_dir ('/tmp' by default)."""
    timestamp_str = (start_time or datetime.datetime.now()).strftime(
        "%Y_%m_%d_%H_%M_%S"
    )
    safe_label = "".join(
        c if c.isalnum() or c in ("-", "_") else "_" for c in title
    )
    target_dir = self._log_dir or "/tmp"
    filename = f"{safe_label}_{timestamp_str}.log"
    return os.path.join(target_dir, filename)

  def build_terminal_command(
      self,
      command: str,
      title: str,
      cwd: str | None = None,
      log_file: str | None = None,
  ) -> list[str]:
    """Builds the argument list to launch a command inside its own terminal window.

    All stdout and stderr output from the command is displayed live on the
    terminal screen and simultaneously saved to `log_file` via `tee`.

    Args:
      command: The shell command to run.
      title: Title/label for the terminal window and log file.
      cwd: Optional working directory for the command.
      log_file: Optional path to the dedicated log file. If None, a timestamped
        log file path is generated automatically.

    Returns:
      Command argument list suitable for subprocess.Popen.
    """
    terminal_emulator = self._find_terminal_emulator()
    resolved_log_file = log_file or self._generate_log_file_path(title)
    cd_command = f"cd '{cwd}'; " if cwd else ""
    safe_title = title.replace("'", "'\\''")
    safe_command_str = command.replace("'", "'\\''")
    safe_log_file = resolved_log_file.replace("'", "'\\''")
    shell_script = (
        f"{cd_command}"
        "set -o pipefail; "
        "{ "
        "echo '========================================'; "
        f"echo '[{safe_title}] {safe_command_str}'; "
        f"echo '[{safe_title}] Log file: {safe_log_file}'; "
        "echo '========================================'; "
        f"{command}; "
        f"}} 2>&1 | tee -a '{safe_log_file}'"
    )

    if terminal_emulator == "gnome-terminal":
      display = os.environ.get("DISPLAY", ":0")
      cmd = [
          "gnome-terminal",
          "--wait",
          f"--display={display}",
          f"--title={title}",
      ]
      if cwd:
        cmd.append(f"--working-directory={cwd}")
      cmd.extend([
          "-q",
          "--",
          "bash",
          "-ic",
          shell_script,
      ])
      return cmd
    elif terminal_emulator in (
        "xterm",
        "x-terminal-emulator",
        "xfce4-terminal",
    ):
      cmd = [terminal_emulator]
      if terminal_emulator == "xfce4-terminal":
        cmd.append("--disable-server")
        if cwd:
          cmd.append(f"--working-directory={cwd}")
      cmd.extend([
          "-T",
          title,
          "-e",
          "bash",
          "-ic",
          shell_script,
      ])
      return cmd
    elif terminal_emulator == "konsole":
      cmd = [
          "konsole",
          "--nofork",
          "--title",
          title,
      ]
      if cwd:
        cmd.extend(["--workdir", cwd])
      cmd.extend([
          "-e",
          "bash",
          "-ic",
          shell_script,
      ])
      return cmd
    else:
      # Fallback to direct bash if no GUI terminal emulator is available
      return ["bash", "-ic", shell_script]

  def _build_terminal_command(
      self,
      command: str,
      title: str,
      cwd: str | None = None,
      log_file: str | None = None,
  ) -> list[str]:
    """Builds the argument list to launch a command inside its own terminal window."""
    return self.build_terminal_command(
        command, title, cwd=cwd, log_file=log_file
    )

  def launch_commands_in_sequence(
      self,
      launch_commands: Sequence[str],
      working_dir: str | None = None,
      env_vars: dict[str, str] | None = None,
      delay_between_commands_sec: float = 0.5,
  ) -> list[subprocess.Popen[Any]]:
    """Launches the given commands in sequence, each in its own terminal window.

    Each command is executed sequentially one after another in its own terminal
    window. All output is displayed on the terminal screen and simultaneously
    saved to a dedicated log file whose filename reflects the command start
    time.

    Args:
      launch_commands: Commands to execute sequentially.
      working_dir: Working directory for the commands.
      env_vars: Environment variables dictionary.
      delay_between_commands_sec: Delay between consecutive launches.

    Returns:
      List of launched subprocess.Popen objects.
    """
    cwd = working_dir or self._download_dir
    os.makedirs(cwd, exist_ok=True)
    target_log_dir = self._log_dir or cwd
    os.makedirs(target_log_dir, exist_ok=True)

    env = os.environ.copy()
    if env_vars:
      env.update(env_vars)
    # Add download_dir to PATH so binaries can be located easily
    env["PATH"] = f"{self._download_dir}:{env.get('PATH', '')}"
    env["ORCA_ARTIFACT_DIR"] = self._download_dir

    for idx, raw_command in enumerate(launch_commands):
      if self._shutdown_event.is_set() or self._signal_received:
        logging.warning("Launch aborted due to shutdown event.")
        break

      command = raw_command.strip()
      if not command:
        continue

      # Substitute any placeholders
      command = command.format(
          download_dir=self._download_dir,
          artifact_dir=self._download_dir,
          robot_id=self._robot_id,
      )

      command_label = f"proc-{idx + 1}"
      start_time = datetime.datetime.now()
      log_file_path = self._generate_log_file_path(
          command_label, start_time=start_time
      )
      # Ensure the dedicated log file is created on disk at command start time.
      with open(log_file_path, "a", encoding="utf-8"):
        pass

      logging.info(
          "Launching process (%d/%d) in dedicated terminal window: [%s]"
          " (CWD: %s, Log: %s)",
          idx + 1,
          len(launch_commands),
          command,
          cwd,
          log_file_path,
      )

      cmd_args = self._build_terminal_command(
          command, command_label, cwd=cwd, log_file=log_file_path
      )

      logging.info("Running command: %s", cmd_args)

      try:
        # Launch each process in its own terminal window.
        process: subprocess.Popen[Any] = subprocess.Popen(
            cmd_args,
            start_new_session=True,
            stdin=None,
            stdout=None,
            stderr=None,
            cwd=cwd,
            env=env,
        )
        self._processes.append(process)
        self._log_files.append(log_file_path)

        logging.info(
            "Launched process [%s] (PID %d) in its own terminal window. Output"
            " saved to [%s].",
            command_label,
            process.pid,
            log_file_path,
        )

        if delay_between_commands_sec > 0:
          time.sleep(delay_between_commands_sec)

      except Exception as e:
        logging.error("Failed to launch command [%s]: %s", command, e)
        # Terminate previously launched processes on launch failure
        self.terminate_all_processes()
        raise

    return self._processes

  def monitor_processes(self) -> bool:
    """Monitors the health and status of all launched processes.

    Returns:
      True if all processes are running healthy, False if any process exited.
    """
    for idx, process in enumerate(self._processes):
      return_code = process.poll()
      if return_code is not None:
        if return_code == 0:
          logging.info(
              "Process (index %d, PID %d) completed normally.", idx, process.pid
          )
        else:
          logging.warning(
              "Process (index %d, PID %d) exited unexpectedly with code %d.",
              idx,
              process.pid,
              return_code,
          )
        return False

    return True

  def wait_for_processes(self, poll_interval: float = 1.0) -> None:
    """Monitors running processes in a loop until one exits or shutdown is requested.

    Args:
      poll_interval: Process monitoring poll interval in seconds.
    """
    logging.info("Entering process monitoring loop. Press Ctrl+C to stop.")
    while not self._shutdown_event.is_set():
      all_alive = self.monitor_processes()
      if not all_alive:
        logging.info("One or more processes exited.")
        break
      time.sleep(poll_interval)

  def terminate_all_processes(self, timeout_sec: float = 10.0) -> None:
    """Terminates all launched processes and their child process trees.

    Args:
      timeout_sec: Timeout in seconds to wait for SIGTERM before sending
        SIGKILL.
    """
    logging.info(
        "Terminating all %d launched processes...", len(self._processes)
    )
    self._shutdown_event.set()
    current_pid = os.getpid()
    current_pgid = os.getpgrp()

    # Send SIGTERM to all process groups
    for process in self._processes:
      if (
          process.poll() is None
          and process.pid
          and process.pid > 1
          and process.pid != current_pid
      ):
        try:
          pgid = os.getpgid(process.pid)
          if pgid > 1 and pgid != current_pgid:
            logging.info(
                "Sending SIGTERM to process group %d (PID %d)...",
                pgid,
                process.pid,
            )
            os.killpg(pgid, signal.SIGTERM)
          else:
            process.terminate()
        except ProcessLookupError:
          pass
        except Exception as e:  # pylint: disable=broad-exception-caught
          logging.debug("Error sending SIGTERM to PID %d: %s", process.pid, e)

    # Wait for graceful termination
    start_time = time.time()
    for process in self._processes:
      remaining_timeout = max(0.1, timeout_sec - (time.time() - start_time))
      try:
        process.wait(timeout=remaining_timeout)
      except subprocess.TimeoutExpired:
        pass

    # Send SIGKILL to any remaining processes
    for process in self._processes:
      if (
          process.poll() is None
          and process.pid
          and process.pid > 1
          and process.pid != current_pid
      ):
        try:
          pgid = os.getpgid(process.pid)
          if pgid > 1 and pgid != current_pgid:
            logging.warning(
                "Process group %d (PID %d) did not terminate in time. Sending"
                " SIGKILL...",
                pgid,
                process.pid,
            )
            os.killpg(pgid, signal.SIGKILL)
          else:
            process.kill()
        except ProcessLookupError:
          pass
        except Exception as e:  # pylint: disable=broad-exception-caught
          logging.debug("Error sending SIGKILL to PID %d: %s", process.pid, e)

    self._processes.clear()
    logging.info("All launched processes have been terminated.")

  def setup_signal_handlers(self) -> None:
    """Registers signal handlers for graceful shutdown on SIGINT, SIGTERM, SIGHUP."""

    def _handle_signal(signum: int, frame: Any) -> None:
      """Handles termination signals."""
      del frame  # Unused.
      signal_name = signal.Signals(signum).name
      logging.info(
          "Received termination signal: %s (%d). Initiating shutdown...",
          signal_name,
          signum,
      )
      self._signal_received = True
      self._shutdown_event.set()
      self.terminate_all_processes()

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
      try:
        signal.signal(sig, _handle_signal)
      except (ValueError, OSError):
        pass

  def run(
      self,
      launch_commands: Sequence[str],
      working_dir: str | None = None,
      env_vars: dict[str, str] | None = None,
      delay_between_commands_sec: float = 0.5,
      poll_interval: float = 1.0,
  ) -> int:
    """Launches commands sequentially and monitors them until completion or interrupt.

    Args:
      launch_commands: Commands to execute sequentially.
      working_dir: Optional working directory for the commands.
      env_vars: Optional environment variables dictionary.
      delay_between_commands_sec: Delay between consecutive launches.
      poll_interval: Process monitoring poll interval in seconds.

    Returns:
      Exit code: 0 on success, non-zero on error.
    """
    self.setup_signal_handlers()
    try:
      if launch_commands:
        self.launch_commands_in_sequence(
            launch_commands,
            working_dir=working_dir,
            env_vars=env_vars,
            delay_between_commands_sec=delay_between_commands_sec,
        )
        self.wait_for_processes(poll_interval=poll_interval)
      return 0
    except KeyboardInterrupt:
      logging.info("Process launcher interrupted by user.")
      return 0
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.error("Process launcher execution failed: %s", e)
      return 1
    finally:
      self.terminate_all_processes()
