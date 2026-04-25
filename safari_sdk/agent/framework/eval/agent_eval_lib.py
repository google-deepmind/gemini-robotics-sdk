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

r"""Agent Evals with ORCA, agent backend, and robot backend."""

import json
import os
import select
import subprocess
import sys
import termios
import threading
import time
from typing import Any

from absl import logging
import httpx

from safari_sdk.agent.framework import types
from safari_sdk.agent.framework.eval import agent_eval_util
from safari_sdk.orchestrator.helpers import orchestrator_helper


_AGENT_BINARY_COMMAND_KEY = "agent_binary_command"
_ROBOT_BINARY_COMMAND_KEY = "robot_binary_command"

_ROBOT_BACKEND_URL = "http://localhost:8888"
_AGENT_BACKEND_URL = "http://localhost:8887"

_DEFAULT_BACKEND_START_TIMEOUT_SECONDS = 300.0
_DEFAULT_AGENT_BACKEND_START_TIMEOUT_SECONDS = 60.0


class AgentEvaluator:
  """Manages agent evaluations with ORCA, agent backend, and robot backend."""

  # ===========================================================================
  # Initialization
  # ===========================================================================

  def __init__(
      self,
      orchestrator_client: orchestrator_helper.OrchestratorHelper,
      robot_id: str,
      launch_subprocess: bool = True,
      show_subprocess_logs: bool = False,
      save_subprocess_logs: bool = True,
      subprocess_log_dir: str | None = None,
  ):
    self._orchestrator_client = orchestrator_client
    self._robot_id = robot_id
    self._launch_subprocess = launch_subprocess
    self._show_subprocess_logs = show_subprocess_logs
    self._save_subprocess_logs = save_subprocess_logs
    self._subprocess_log_dir = subprocess_log_dir

    if self._subprocess_log_dir and not os.path.exists(
        self._subprocess_log_dir
    ):
      os.makedirs(self._subprocess_log_dir)

    self._fns = agent_eval_util.SetUserIoConnectionAsConsole()
    self._robot_backend_process: subprocess.Popen[Any] | None = None
    self._agent_backend_process: subprocess.Popen[Any] | None = None

    self._operator_abort_event = threading.Event()
    self._stop_monitoring_abort_event = threading.Event()
    self._stop_streaming_event = threading.Event()
    self._agent_session_id: str | None = None

  # ===========================================================================
  # Backend lifecycle management
  # ===========================================================================

  def start_robot_backend(self, policy_details):
    """Starts the robot backend and waits for it to be ready."""
    self._fns.print("\nStarting robot backend...")
    if self._launch_subprocess:
      if (
          self._robot_backend_process is not None
          and self._robot_backend_process.poll() is None
      ):
        self._fns.print("Robot backend is already running.")
      else:
        runner_path = agent_eval_util.get_runner_path_from_policy_details(
            policy_details, _ROBOT_BINARY_COMMAND_KEY
        )
        policy_details_json_b64 = agent_eval_util.serialize_policy_details(
            policy_details
        )
        self._robot_backend_process = agent_eval_util.start_backend_subprocess(
            runner_path,
            policy_details_json_b64,
            "Robot backend",
            save_logs=self._save_subprocess_logs,
            log_dir=self._subprocess_log_dir,
            show_logs=self._show_subprocess_logs,
        )
    self._fns.print("Waiting for the robot backend to be ready...")

    backend_start_timeout = policy_details.get_parameter_value(
        "backend_start_timeout", _DEFAULT_BACKEND_START_TIMEOUT_SECONDS
    )
    robot_backend_ready = self._check_if_server_is_responsive(
        server_url=f"{_ROBOT_BACKEND_URL}/get_state/",
        timeout=backend_start_timeout,
    )
    if not robot_backend_ready:
      raise ValueError(
          "Robot backend failed to become ready after"
          f" {backend_start_timeout} seconds."
      )
    self._fns.print("\nRobot backend is ready.\n")

  def start_agent_backend(self, policy_details):
    """Starts the agent backend and waits for it to be ready."""
    self._terminate_agent_backend()
    self._fns.print("\nStarting agent backend...")
    if self._launch_subprocess:
      runner_path = agent_eval_util.get_runner_path_from_policy_details(
          policy_details, _AGENT_BINARY_COMMAND_KEY
      )
      policy_details_json_b64 = agent_eval_util.serialize_policy_details(
          policy_details
      )

      self._agent_backend_process = agent_eval_util.start_backend_subprocess(
          runner_path,
          policy_details_json_b64,
          "Agent backend",
          save_logs=self._save_subprocess_logs,
          log_dir=self._subprocess_log_dir,
          show_logs=self._show_subprocess_logs,
      )
    self._fns.print("Waiting for the agent backend to be ready...")
    agent_backend_ready = self._check_if_server_is_responsive(
        server_url=f"{_AGENT_BACKEND_URL}/get_framework_status/",
        timeout=_DEFAULT_AGENT_BACKEND_START_TIMEOUT_SECONDS,
    )
    if not agent_backend_ready:
      raise ValueError(
          "Agent backend failed to become ready after"
          f" {_DEFAULT_AGENT_BACKEND_START_TIMEOUT_SECONDS} seconds."
      )
    self._fns.print("\nAgent backend is ready.\n")

  def _terminate_agent_backend(self) -> dict[str, str] | None:
    """Terminates the agent backend process."""
    if self._agent_backend_process is None:
      return
    info = {}
    try:
      response = httpx.get(f"{_AGENT_BACKEND_URL}/terminate/")
      response.raise_for_status()
      self._fns.print(
          f"Agent backend terminate status Code: {response.status_code}")
      self._fns.print(
          f"Agent backend terminate Response JSON: {response.json()}\n")
      time.sleep(5)  # Wait for the agent backend to terminate.
      info.update(response.json())
      info["status_code"] = response.status_code
    except httpx.HTTPError as e:
      logging.warning("Failed to terminate agent backend via endpoint.")
      info["error_message"] = str(e)

    agent_eval_util.terminate_process(
        self._agent_backend_process, "Agent backend"
    )
    self._agent_backend_process = None
    return info

  def _terminate_robot_backend(self):
    """Shuts down the robot backend process."""
    if self._robot_backend_process is not None:
      # Process was launched as subprocess, terminate it.
      is_responsive = self._check_if_server_is_responsive(
          server_url=f"{_ROBOT_BACKEND_URL}/get_state/",
          timeout=1.0,
      )
      if is_responsive:
        logging.info("Robot backend is responsive. Sending /sleep/ request.")
        try:
          response = httpx.get(f"{_ROBOT_BACKEND_URL}/sleep/")
          response.raise_for_status()
        except httpx.RequestError:
          logging.exception(
              "/sleep/ endpoint on the robot backend not reachable."
          )
        time.sleep(5)
      agent_eval_util.terminate_process(
          self._robot_backend_process, "Robot backend"
      )
      self._robot_backend_process = None

  # ===========================================================================
  # Backend communication / status
  # ===========================================================================

  def _check_if_server_is_responsive(
      self,
      server_url: str,
      retry_interval=5.0,
      timeout=60.0,
  ) -> bool:
    """Checks if the server is ready."""
    self._fns.print(f"\nChecking if server is responsive: {server_url}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
      try:
        response = httpx.get(server_url)
        response.raise_for_status()
        return True
      except httpx.HTTPError:
        self._fns.print(
            f"The server is not ready yet. Retrying in {retry_interval}"
            " seconds..."
        )
        time.sleep(retry_interval)
    self._fns.print(f"\nServer is not ready after {timeout} seconds.")
    return False

  def _get_agent_backend_status(self) -> dict[str, Any] | None:
    """Gets the agent backend status."""
    try:
      response = httpx.get(f"{_AGENT_BACKEND_URL}/get_framework_status/")
      response.raise_for_status()
      return response.json()
    except httpx.RequestError:
      logging.exception("Failed to get agent backend status.")
      return None

  def _get_robot_backend_status(self) -> bool:
    """Checks if the robot backend is healthy."""
    return self._check_if_server_is_responsive(
        server_url=f"{_ROBOT_BACKEND_URL}/get_state/",
        timeout=1.0,
    )

  def _request_with_retries(
      self,
      method: str,
      url: str,
      timeout: float = 60.0,
      max_retries: int = 3,
      retry_interval: float = 2.0,
      **kwargs,
  ) -> httpx.Response:
    """Sends a request with retries."""
    for attempt in range(max_retries):
      try:
        response = httpx.request(method, url, timeout=timeout, **kwargs)
        response.raise_for_status()  # pytype: disable=wrong-arg-types
        return response
      except (httpx.HTTPError, httpx.RequestError):
        logging.warning(
            "Request %s to %s failed. Attempt %d of %d.",
            method,
            url,
            attempt + 1,
            max_retries,
            exc_info=True,
        )
        if attempt < max_retries - 1:
          time.sleep(retry_interval)
    # If all retries fail, try one last time to propagate the error.
    response = httpx.request(method, url, timeout=timeout, **kwargs)
    response.raise_for_status()
    return response

  # ===========================================================================
  # Agent backend API calls
  # ===========================================================================

  def _send_work_unit_details_to_agent_backend(
      self, work_unit_details: dict[str, Any]
  ):
    """Sends the work unit details to the agent backend via HTTP GET request."""
    url = f"{_AGENT_BACKEND_URL}/publish_event/"
    self._fns.print(f"\nWork unit details: {work_unit_details}")
    data = {
        "source": str(types.EventSource.EXTERNAL_CONTROLLER.value),
        "type": str(types.EventType.LOG_SESSION_METADATA.value),
        "metadata": work_unit_details,
    }
    self._fns.print(
        f"\nSending work unit details to agent backend: {data}"
    )
    try:
      response = httpx.post(url, json=data)
      response.raise_for_status()
      self._fns.print(f"Status Code: {response.status_code}")
      self._fns.print(f"Response JSON: {response.json()}\n")
    except httpx.RequestError:
      logging.exception("Failed to send work unit details to agent backend.")

  def _send_instruction_to_agent_backend(self, agent_instruction: str) -> None:
    """Sends the LH task to the agent backend via HTTP GET request."""
    url = f"{_AGENT_BACKEND_URL}/execute_lh_task/"
    params = {"lh_task": agent_instruction}
    self._fns.print(
        f"\nSending instruction to agent backend: {agent_instruction}")
    try:
      response = httpx.get(url, params=params)
      response.raise_for_status()
      self._fns.print(f"Status Code: {response.status_code}")
      self._fns.print(f"Response JSON: {response.json()}\n")
    except httpx.RequestError:
      logging.exception("Failed to send LH task to agent backend.")

  def _get_agent_session_id_from_agent_backend(self) -> str | None:
    """Gets the agent session id from the agent backend via HTTP GET request."""
    url = f"{_AGENT_BACKEND_URL}/get_agent_session_id/"
    self._fns.print("\nGetting agent session id from agent backend...")
    try:
      response = httpx.get(url)
      response.raise_for_status()
      self._fns.print(f"Status Code: {response.status_code}")
      response_json = response.json()
      self._fns.print(f"Response JSON: {response_json}\n")
      return response_json.get("agent_session_id")
    except httpx.RequestError:
      logging.exception("Failed to get agent session id from agent backend.")
      return None

  def _set_agent_session_id_on_robot_backend(self, agent_session_id: str):
    """Sets the agent session ID on the robot backend."""
    self._fns.print("\nSetting agent session ID on the robot backend...")
    try:
      response = httpx.get(
          f"{_ROBOT_BACKEND_URL}/set_agent_session_id/",
          params={"agent_session_id": agent_session_id},
      )
      response.raise_for_status()
      self._fns.print(f"Status Code: {response.status_code}")
      self._fns.print(f"Response JSON: {response.json()}\n")
    except httpx.RequestError:
      logging.exception("Failed to set agent session ID on the robot backend.")

  # ===========================================================================
  # Robot control
  # ===========================================================================

  def _reset_robot(self):
    """Resets the robot (open grippers and then reset)."""
    # Send Open grippers command to the robot backend.
    self._fns.print("\nOpening grippers.")
    try:
      response = self._request_with_retries(
          "GET", f"{_ROBOT_BACKEND_URL}/open_grippers/"
      )
      self._fns.print(f"Status Code: {response.status_code}")
      self._fns.print(f"Response JSON: {response.json()}\n")
    except httpx.HTTPError:
      logging.exception("Failed to open grippers.")
    time.sleep(1)
    self._fns.print("\nSending reset command to robot backend.")
    try:
      response = self._request_with_retries(
          "GET", f"{_ROBOT_BACKEND_URL}/reset/"
      )
      self._fns.print(f"Status Code: {response.status_code}")
      self._fns.print(f"Response JSON: {response.json()}\n")
    except httpx.HTTPError:
      logging.exception("Failed to send reset command to the robot backend.")

  # ===========================================================================
  # Operator interaction (bidirectional dialog)
  # ===========================================================================

  def _stream_agent_responses(self):
    """Streams and displays agent responses from the backend."""
    url = f"{_AGENT_BACKEND_URL}/stream_terminal_output/"
    try:
      with httpx.stream("GET", url, timeout=None) as response:
        for line in response.iter_lines():
          if self._stop_streaming_event.is_set():
            break
          if line.startswith("data: "):
            try:
              data = json.loads(line[6:])
              self._display_agent_event(data)
            except json.JSONDecodeError:
              logging.warning("Failed to parse agent event: %s", line)
    except httpx.RequestError:
      logging.warning("Agent response stream disconnected.")

  def _display_agent_event(self, event_data: dict[str, Any]):
    """Formats and displays an agent event to the operator."""
    event_type = event_data.get("type", "")
    data = event_data.get("data", "")

    if "MODEL_TURN" in event_type:
      self._fns.print(f"\n[Agent]: {data}")
    elif "TOOL_CALL" in event_type:
      self._fns.print(f"\n[Tool Call]: {data}")
    elif "TOOL_RESULT" in event_type:
      self._fns.print(f"\n[Tool Result]: {data}")
    elif "OUTPUT_TRANSCRIPT" in event_type:
      self._fns.print(f"\n[Agent Speech Transcript]: {data}")

  def _send_follow_up_interaction(self, interaction: str) -> None:
    """Sends a follow-up interaction to the agent backend."""
    url = f"{_AGENT_BACKEND_URL}/execute_interaction/"
    params = {"interaction": interaction}
    self._fns.print(f"\n>>> Operator: {interaction}")
    try:
      response = httpx.get(url, params=params, timeout=5.0)
      response.raise_for_status()
    except httpx.RequestError:
      logging.warning("Failed to send follow-up interaction to agent.")

  def _watch_for_operator_input(self):
    """Watches for operator input to send to agent or abort episode.

    - Empty line (just Enter): Aborts the episode
    - Non-empty line: Sends the text as a follow-up interaction to the agent
    """
    timeout = 1.0
    while not self._stop_monitoring_abort_event.is_set():
      rlist, _, _ = select.select([sys.stdin], [], [], timeout)
      if rlist:
        user_input = sys.stdin.readline()
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
        user_input = user_input.strip()
        if not user_input:
          self._operator_abort_event.set()
          self._stop_monitoring_abort_event.set()
        else:
          self._send_follow_up_interaction(user_input)

  def _get_operator_rating(self, work_unit):
    """Gets the post-episode rating from the operator."""
    # Operator post-episode rating.
    if work_unit.context.successScores:
      ss_score = []
      ss_definition = []
      for ss in work_unit.context.successScores:
        ss_score.append(ss.score if ss.score else 0.0)
        ss_definition.append(ss.definition)

      use_checklist_format = (
          work_unit.context.scenePresetDetails.get_parameter_value(
              key="score_with_checklist", default_value=False
          )
      )  # pytype: disable=wrong-arg-types
      if use_checklist_format:
        selected_ss_idx = []
        remaining_ss_definition = ss_definition[:]

        while True:
          user_input = self._fns.input(
              prompt="\nSelect success score options as a checklist.",
              choices=remaining_ss_definition,
              is_cancelable=True,
          )
          if user_input is None:
            break

          idx = ss_definition.index(user_input)
          selected_ss_idx.append(idx)

          idx = remaining_ss_definition.index(user_input)
          remaining_ss_definition.pop(idx)

          if not remaining_ss_definition:
            break

        success_score_raw = 0.0
        success_score_definition = ""
        for idx in selected_ss_idx:
          success_score_raw += ss_score[idx]
          if success_score_definition:
            success_score_definition += f" | {ss_definition[idx]}"
          else:
            success_score_definition = ss_definition[idx]

        success_score = float(success_score_raw) / sum(ss_score)

        if remaining_ss_definition:
          is_success = False
          outcome = (
              orchestrator_helper.WORK_UNIT_OUTCOME.WORK_UNIT_OUTCOME_FAILURE
          )
        else:
          is_success = True
          outcome = (
              orchestrator_helper.WORK_UNIT_OUTCOME.WORK_UNIT_OUTCOME_SUCCESS
          )
      else:
        # Ask operator to select a success score from the list of available
        # scores.
        user_input = self._fns.input(
            prompt="\nSelect a success score.",
            choices=ss_definition,
        )
        idx = ss_definition.index(user_input)
        success_score = ss_score[idx]
        success_score_definition = user_input
        is_success = success_score == 1.0
        if is_success:
          outcome = (
              orchestrator_helper.WORK_UNIT_OUTCOME.WORK_UNIT_OUTCOME_SUCCESS
          )
        else:
          outcome = (
              orchestrator_helper.WORK_UNIT_OUTCOME.WORK_UNIT_OUTCOME_FAILURE
          )
    else:
      success_score = None
      success_score_definition = None
      # Ask operator to confirm if episode is successful or not.
      user_input = self._fns.input(
          prompt="\nConfirm episode result.",
          choices=["Success", "Failure", "Invalid"],
      )
      match user_input:
        case "Success":
          is_success = True
          outcome = (
              orchestrator_helper.WORK_UNIT_OUTCOME.WORK_UNIT_OUTCOME_SUCCESS
          )
        case "Failure":
          is_success = False
          outcome = (
              orchestrator_helper.WORK_UNIT_OUTCOME.WORK_UNIT_OUTCOME_FAILURE
          )
        case "Invalid":
          is_success = False
          outcome = (
              orchestrator_helper.WORK_UNIT_OUTCOME.WORK_UNIT_OUTCOME_INVALID
          )
        case _:
          is_success = False
          outcome = (
              orchestrator_helper.WORK_UNIT_OUTCOME.WORK_UNIT_OUTCOME_INVALID
          )

    episode_note = self._fns.input(
        prompt="\nPlease enter any additional information about this episode: "
    )
    return (
        is_success,
        outcome,
        success_score,
        success_score_definition,
        episode_note,
    )

  # ===========================================================================
  # ORCA workflow
  # ===========================================================================

  def _connect_to_orca(self):
    """Connects to Orca."""
    response = self._orchestrator_client.connect()
    if not response.success:
      raise ConnectionError(
          f"Failed to connect to Orca: {response.error_message}"
      )

  def _get_work_unit(self) -> orchestrator_helper.WORK_UNIT | None:
    """Gets the work unit from ORCA for this episode."""
    response = self._orchestrator_client.request_work_unit()
    if response.success:
      if response.no_more_robot_job:
        self._fns.print("\nNo robot job available.")
        return None
      if response.no_more_work_unit:
        self._fns.print("\nNo work unit available.")
        return None
      logging.info(" - Sucessfully requested work unit -\n")
    else:
      logging.error("\n - ERROR: %s -\n", response.error_message)
      self._fns.print(f"\n - ERROR: {response.error_message} -\n")
      return None
    return response.work_unit

  def _prep_work_unit(self):
    """Preps the work unit."""
    response = self._orchestrator_client.start_work_unit_software_asset_prep()
    if not response.success:
      logging.error("\n - ERROR: %s -\n", response.error_message)
      return False
    response = self._orchestrator_client.start_work_unit_scene_prep()
    if not response.success:
      logging.error("\n - ERROR: %s -\n", response.error_message)
      return False
    return True

  def _complete_work_unit(
      self,
      work_unit: orchestrator_helper.WORK_UNIT,
      agent_backend_termination_info: dict[str, str] | None,
  ):
    """Completes the work unit with ORCA.

    Args:
      work_unit: The work unit to complete.
      agent_backend_termination_info: The termination info of the agent backend.
    """

    def _get_agent_log_values() -> tuple[int, int, str] | tuple[None, ...]:
      if agent_backend_termination_info is None:
        return None, None, None
      start_ns = agent_backend_termination_info.get("logger_start_nsec")
      end_ns = agent_backend_termination_info.get("logger_end_nsec")
      session_log_type = agent_backend_termination_info.get("session_log_type")
      if start_ns is None or end_ns is None or session_log_type is None:
        return None, None, None
      return int(start_ns), int(end_ns), session_log_type

    (
        _,  # is_success
        outcome,
        success_score,
        success_score_definition,
        episode_note,
    ) = self._get_operator_rating(work_unit)
    # TODO: Add support for questionnaire.
    session_start_time_ns, session_end_time_ns, session_log_type = (
        _get_agent_log_values()
    )
    self._orchestrator_client.complete_work_unit(
        outcome=outcome,
        success_score=success_score,
        success_score_definition=success_score_definition,
        note=f"Agent session ID: {self._agent_session_id}\n{episode_note}",
        session_start_time_ns=session_start_time_ns,
        session_end_time_ns=session_end_time_ns,
        session_log_type=session_log_type
    )

  def _get_work_unit_details(
      self,
      work_unit,
      agent_instruction: str,
      operator_instruction: str,
      policy_name: str | None = None
  ) -> dict[str, str]:
    """Gets the work unit details."""
    work_unit_details = {
        "orchestrator_robot_job_id": work_unit.robotJobId,
        "orchestrator_work_unit_id": work_unit.workUnitId,
        "orchestrator_task_id": work_unit.context.orchestratorTaskId,
        "orchestrator_scene_preset_id": work_unit.context.scenePresetId,
        "orchestrator_work_unit_agent_instruction": agent_instruction,
        "orchestrator_work_unit_operator_instruction": operator_instruction,
    }
    if policy_name:
      work_unit_details["orchestrator_work_unit_policy_name"] = policy_name
    return work_unit_details

  # ===========================================================================
  # Main execution
  # ===========================================================================

  def execute_work_unit(self, work_unit):
    """Executes a work unit end-to-end.

    Extracts work unit info, runs the episode, collects operator rating,
    and saves episode data.

    Args:
      work_unit: The work unit to execute.

    Returns:
      A tuple of (termination_reason, outcome, success_score,
      success_score_definition, episode_note).
    """
    scene_details = work_unit.context.scenePresetDetails
    agent_instruction = scene_details.get_parameter_value(
        "agent_instruction", None
    )
    operator_instruction = scene_details.get_parameter_value(
        "operator_instruction", None
    )
    if agent_instruction is None and operator_instruction is None:
      raise ValueError(
          "Either `agent_instruction` or `operator_instruction` must be "
          "specified in the work unit."
      )
    self._fns.print("====== Episode task ======")

    scene_id = work_unit.context.scenePresetId
    self._fns.print(f"Scene id: {scene_id}")
    time_limit_seconds = scene_details.get_parameter_value("time_limit", 360)
    policy_details = work_unit.context.policyDetails  # pytype: disable=attribute-error
    self._fns.print(f"Policy name: {policy_details.name}")
    self._fns.print(f"Agent instruction: {agent_instruction}")
    self._fns.print(f"Operator instruction: {operator_instruction}")

    policy_name = policy_details.name if policy_details else None
    work_unit_details = self._get_work_unit_details(
        work_unit, agent_instruction, operator_instruction, policy_name)

    self._fns.print(f"Work unit details: {work_unit_details}")

    user_input = self._fns.input(
        prompt="Start running episode or quit evaluation?",
        choices=["Start", "Quit"],
    )
    if user_input == "Quit":
      self._fns.print("\nOperator requested to quit evaluation.\n")
      agent_backend_termination_info = self._terminate_agent_backend()
      return (agent_eval_util.TerminationReason.OPERATOR_ABORT,
              agent_backend_termination_info)

    self.start_robot_backend(policy_details)
    self.start_agent_backend(policy_details)

    self._orchestrator_client.start_work_unit_execution()
    self._agent_session_id = self._get_agent_session_id_from_agent_backend()
    self._set_agent_session_id_on_robot_backend(self._agent_session_id)
    if agent_instruction:
      self._send_instruction_to_agent_backend(agent_instruction)
    if operator_instruction:
      self._fns.print(
          "OPERATOR INSTRUCTION: Interaction to execute:"
          f" {operator_instruction}."
      )
    # The work unit details will be sent to the external controller which will
    # publish them to the event bus wherein they will be logged by the stream
    # logger. This is needed to correctly associate SSOT sessions with ORCA
    # work units.
    self._send_work_unit_details_to_agent_backend(work_unit_details)

    episode_start_time = time.time()
    episode_limit_time = episode_start_time + time_limit_seconds
    self._operator_abort_event.clear()
    self._stop_monitoring_abort_event.clear()
    self._stop_streaming_event.clear()

    self._fns.print("\n--- Dialog Mode ---")
    self._fns.print(
        "Type a message to talk to the agent. Press Enter on empty line to"
        " abort.\n"
    )

    stream_agent_responses_thread = threading.Thread(
        target=self._stream_agent_responses,
        daemon=True,
    )
    stream_agent_responses_thread.start()

    monitor_operator_input_thread = threading.Thread(
        target=self._watch_for_operator_input,
        daemon=True,
    )
    monitor_operator_input_thread.start()

    last_status_print_time = 0.0
    while True:
      agent_backend_status = self._get_agent_backend_status()
      if agent_backend_status is not None:
        last_status_print_time = agent_eval_util.print_every_x_seconds(
            agent_backend_status, last_status_print_time, self._fns.print
        )
        match agent_backend_status["framework_status"]:
          case "READY" | "RUNNING":
            pass  # Agent is still working, continue polling
          case "FINISHED":
            self._fns.print(
                "\nAgent finished the task. Elapsed time:"
                f" {time.time() - episode_start_time:.2f}"
            )
            termination_reason = (
                agent_eval_util.TerminationReason.AGENT_TERMINATION_SIGNAL
            )
            break
          case _:
            raise ValueError(
                "Agent backend is in an unexpected state:"
                f" {agent_backend_status}"
            )

      if not self._get_robot_backend_status():
        termination_reason = agent_eval_util.TerminationReason.BACKEND_UNHEALTHY
        break

      if time.time() > episode_limit_time:
        self._fns.print(
            f"\nEpisode timed out after {time_limit_seconds} seconds."
        )
        termination_reason = (
            agent_eval_util.TerminationReason.TIME_LIMIT_REACHED
        )
        break

      if self._operator_abort_event.is_set():
        self._fns.print("\nOperator requested to abort the episode.")
        termination_reason = agent_eval_util.TerminationReason.OPERATOR_ABORT
        break

      time.sleep(1)

    self._stop_monitoring_abort_event.set()
    self._stop_streaming_event.set()
    if (
        stream_agent_responses_thread is not None
        and stream_agent_responses_thread.is_alive()
    ):
      stream_agent_responses_thread.join(timeout=1.0)
    if (
        monitor_operator_input_thread is not None
        and monitor_operator_input_thread.is_alive()
    ):
      monitor_operator_input_thread.join(timeout=1.0)
    self._fns.print(f"\nEpisode terminated because {termination_reason}.")

    agent_backend_termination_info = self._terminate_agent_backend()
    self._reset_robot()
    return termination_reason, agent_backend_termination_info

  def run_eval(self):
    """Runs the agentic eval loop."""
    self._connect_to_orca()
    work_unit_idx = 0
    while True:
      work_unit_idx += 1
      self._fns.print(f"====== Starting Episode {work_unit_idx} ======")
      work_unit = self._get_work_unit()
      if work_unit is None:
        break
      if not self._prep_work_unit():
        break
      termination_reason, agent_backend_termination_info = (
          self.execute_work_unit(work_unit)
      )
      if termination_reason == agent_eval_util.TerminationReason.OPERATOR_ABORT:
        logging.info("Operator requested to abort the episode.")
        break
      self._complete_work_unit(work_unit, agent_backend_termination_info)
      self._fns.print(f"====== Completed Episode {work_unit_idx} ======")

    # All experiments are done. Shutdown.
    self._fns.print("\nShutting down subprocesses.")
    self._terminate_agent_backend()
    self._terminate_robot_backend()
    self._fns.print("\nShutdown complete. Have a nice day!")
