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

"""Orchestrator launch helper library.

Provides capabilities to:
1. Connect to Orchestrator Server via OrchestratorHelper.
2. Fetch current robot job / work unit launch information.
3. Download and prepare required artifacts.
4. Launch downloaded binaries/commands in explicitly specified sequence.
5. Monitor process health and output streams.
6. Ensure all launched processes and child process trees are gracefully
terminated
   upon receiving termination signals.
"""

from collections.abc import Sequence
import json
import logging
import os
import shutil
import signal
import subprocess
import threading
import time
from typing import Any
import urllib.request

from safari_sdk.orchestrator.helpers import orchestrator_helper

_MANIFEST_FILENAME = ".artifact_cache_manifest.json"


def format_orchestrator_work_unit_info(
    work_unit: orchestrator_helper.WORK_UNIT,
) -> str:
  """Formats all details of the given work unit dataclass into a string."""
  lines: list[str] = []
  lines.append(f" Work Unit dataclass: {work_unit}\n")
  lines.append(
      " ----------------------------------------------------------------\n"
  )
  lines.append(f" Robot Job type: {work_unit.obfuscatedJobTypeCode}")
  lines.append(f" Robot Job ID: {work_unit.robotJobId}")
  lines.append(f" Work Unit ID: {work_unit.workUnitId}")
  lines.append(f" Work Unit stage: {work_unit.stage}")
  lines.append(f" Work Unit outcome: {work_unit.outcome}")
  lines.append(f" Work Unit note: {work_unit.note}\n")

  work_unit_context = work_unit.context
  if work_unit_context is not None:
    lines.append(f" Scene Preset ID: {work_unit_context.scenePresetId}")
    lines.append(f" Scene episode index: {work_unit_context.sceneEpisodeIndex}")
    lines.append(
        f" Orchestrator Task ID: {work_unit_context.orchestratorTaskId}\n"
    )

    success_scores = work_unit_context.successScores
    if success_scores is not None:
      for s_score in success_scores:
        lines.append(" Success Scores:")
        lines.append(f"   Definition: {s_score.definition}")
        lines.append(f"   Score: {s_score.score}\n")

    scene_details = work_unit_context.scenePresetDetails
    if scene_details is not None:
      lines.append(f" Setup Instructions: {scene_details.setupInstructions}")
      scene_params = (
          scene_details.get_all_parameters()
          if hasattr(scene_details, "get_all_parameters")
          else {}
      )
      if scene_params:
        lines.append(" Parameters:")
        for s_key, s_value in scene_params.items():
          lines.append(f"   {s_key}: {s_value}")
      lines.append(f" Grouping: {scene_details.grouping}")

      if scene_details.referenceImages:
        for ref_img in scene_details.referenceImages:
          lines.append(" Reference Image:")
          lines.append(f"   Artifact ID: {ref_img.artifactId}")
          lines.append(f"   Source Topic: {ref_img.sourceTopic}")
          lines.append(f"   Image width: {ref_img.rawImageWidth}")
          lines.append(f"   Image height: {ref_img.rawImageHeight}")
          lines.append(f"   UI width: {ref_img.renderedCanvasWidth}")
          lines.append(f"   UI height: {ref_img.renderedCanvasHeight}\n")

      if scene_details.sceneObjects:
        for s_obj in scene_details.sceneObjects:
          lines.append(" Scene Object:")
          lines.append(f"   Object ID: {s_obj.objectId}")
          if s_obj.overlayTextLabels and s_obj.overlayTextLabels.labels:
            for t_label in s_obj.overlayTextLabels.labels:
              lines.append(f"   Overlay Text Label: {t_label.text}")
          if s_obj.evaluationLocation:
            lines.append(f"   Icon: {s_obj.evaluationLocation.overlayIcon}")
            lines.append(
                f"   Layer Order: {s_obj.evaluationLocation.layerOrder}"
            )
            lines.append(
                "   RGB Hex Color Value:"
                f" {s_obj.evaluationLocation.rgbHexColorValue}"
            )

            if s_obj.evaluationLocation.location:
              if s_obj.evaluationLocation.location.coordinate:
                lines.append("   Coordinate: (UI frame)")
                lines.append(
                    f"     x: {s_obj.evaluationLocation.location.coordinate.x}"
                )
                lines.append(
                    f"     y: {s_obj.evaluationLocation.location.coordinate.y}"
                )
              if s_obj.evaluationLocation.location.direction:
                lines.append("   Direction:")
                lines.append(
                    "     radian:"
                    f" {s_obj.evaluationLocation.location.direction.rad}"
                )

            if s_obj.evaluationLocation.containerArea:
              if s_obj.evaluationLocation.containerArea.circle:
                if s_obj.evaluationLocation.containerArea.circle.center:
                  lines.append("   Coordinate: (UI frame)")
                  lines.append(
                      "     x:"
                      f" {s_obj.evaluationLocation.containerArea.circle.center.x}"
                  )
                  lines.append(
                      "     y:"
                      f" {s_obj.evaluationLocation.containerArea.circle.center.y}"
                  )
                lines.append(
                    "   Radius:"
                    f" {s_obj.evaluationLocation.containerArea.circle.radius}"
                )
              if s_obj.evaluationLocation.containerArea.box:
                lines.append("   Coordinate: (UI frame)")
                lines.append(
                    f"     x: {s_obj.evaluationLocation.containerArea.box.x}"
                )
                lines.append(
                    f"     y: {s_obj.evaluationLocation.containerArea.box.y}"
                )
                lines.append(
                    f"   Width: {s_obj.evaluationLocation.containerArea.box.w}"
                )
                lines.append(
                    f"   Height: {s_obj.evaluationLocation.containerArea.box.h}"
                )
          lines.append(
              "   Reference Image Artifact ID:"
              f" {s_obj.sceneReferenceImageArtifactId}\n"
          )

    if work_unit_context.launcherArtifacts:
      for launcher_art in work_unit_context.launcherArtifacts:
        lines.append(" Launcher Artifact:")
        lines.append(f"   Artifact ID: {launcher_art.artifactId}")
        if launcher_art.launchCommand:
          lines.append(f"   Launch Command: {launcher_art.launchCommand}")
        if launcher_art.launchOrder is not None:
          lines.append(f"   Launch Order: {launcher_art.launchOrder}")
        lines.append("")

    if work_unit_context.policies:
      for p in work_unit_context.policies:
        lines.append(" Policy Details:")
        lines.append(f"   Policy Name: {p.name}")
        lines.append(f"   Policy Description: {p.description}")
        if p.usage:
          lines.append(f"   Policy Usage: {p.usage}")
        if p.artifactIds:
          lines.append(f"   Policy Artifact IDs: {p.artifactIds}")
        p_params = (
            p.get_all_parameters() if hasattr(p, "get_all_parameters") else {}
        )
        if p_params:
          lines.append("   Parameters:")
          for p_key, p_value in p_params.items():
            lines.append(f"     {p_key}: {p_value}")
        lines.append("")
    elif work_unit_context.policyDetails is not None:
      policy_details = work_unit_context.policyDetails
      lines.append(f" Policy Name: {policy_details.name}")
      lines.append(f" Policy Description: {policy_details.description}")
      if policy_details.usage:
        lines.append(f" Policy Usage: {policy_details.usage}")
      if policy_details.artifactIds:
        lines.append(f" Policy Artifact IDs: {policy_details.artifactIds}")
      policy_params = (
          policy_details.get_all_parameters()
          if hasattr(policy_details, "get_all_parameters")
          else {}
      )
      if policy_params:
        lines.append(" Parameters:")
        for p_key, p_value in policy_params.items():
          lines.append(f"   {p_key}: {p_value}")
      lines.append("")

    if work_unit_context.robotJobAssets:
      for r_asset in work_unit_context.robotJobAssets:
        lines.append(" Robot Job Asset:")
        lines.append(f"   Asset Type: {r_asset.assetType}")
        lines.append(f"   Download URI: {r_asset.downloadUri}")
        lines.append("")

    questions = work_unit_context.questions
    if questions is not None:
      for q in questions:
        lines.append(" Questionnaire:")
        lines.append(f"   Questionnaire ID: {q.questionnaireId}")
        lines.append(f"   Question: {q.question}")
        lines.append(f"   When to ask: {q.whenToAsk}")
        lines.append(f"   Answer format: {q.answerFormat}")
        lines.append(f"   Allowed answers: {q.allowedAnswers}")
      lines.append("")

  lines.append(
      " ----------------------------------------------------------------\n"
  )
  return "\n".join(lines)


def log_orchestrator_work_unit_info(
    work_unit: orchestrator_helper.WORK_UNIT,
) -> None:
  """Logs all details of the given work unit dataclass in a single logging.info call."""
  formatted_info = format_orchestrator_work_unit_info(work_unit)
  logging.info("\n%s", formatted_info)


def format_work_unit_info(
    work_unit: orchestrator_helper.WORK_UNIT,
) -> str:
  """Formats all details of the given work unit dataclass into a string."""
  return format_orchestrator_work_unit_info(work_unit)


def log_work_unit_info(
    work_unit: orchestrator_helper.WORK_UNIT,
) -> None:
  """Logs all details of the given work unit dataclass in a single logging.info call."""
  log_orchestrator_work_unit_info(work_unit)


class OrchestratorLaunchHelper:
  """Helper class to manage artifact downloads, sequential launching, and process lifecycle."""

  def __init__(
      self,
      *,
      job_type_codes: list[str],
      robot_id: str | None = None,
      hostname: str | None = None,
      download_dir: str = "/tmp/orchestrator_launcher_artifacts",
      orchestrator_helper_instance: (
          orchestrator_helper.OrchestratorHelper | None
      ) = None,
  ):
    """Initializes the OrchestratorLaunchHelper.

    Args:
      job_type_codes: Optional list of job type codes (e.g. ['agentic']).
      robot_id: Optional ID of the robot.
      hostname: Optional robot hostname.
      download_dir: Local directory to store downloaded artifacts.
      orchestrator_helper_instance: Optional existing OrchestratorHelper
        instance.

    Raises:
      ValueError: If neither robot_id nor hostname is provided.
    """
    self._robot_id = robot_id
    self._hostname = hostname
    if not self._robot_id and not self._hostname:
      raise ValueError(
          "At least one of --robot_id or --hostname must be provided."
      )

    self._job_type_codes = job_type_codes
    self._download_dir = download_dir
    self._helper = (
        orchestrator_helper_instance
        or orchestrator_helper.OrchestratorHelper(
            robot_id=self._robot_id or "",
            job_type_codes=self._job_type_codes,
            hostname=self._hostname,
        )
    )
    self._processes: list[subprocess.Popen[Any]] = []
    self._shutdown_event = threading.Event()
    self._downloaded_artifact_paths: dict[str, str] = {}
    self._is_connected = False
    self._signal_received = False

  @property
  def download_dir(self) -> str:
    """Returns the artifact download directory path."""
    return self._download_dir

  @property
  def running_processes(self) -> list[subprocess.Popen[Any]]:
    """Returns list of currently managed processes."""
    return list(self._processes)

  def log_work_unit_info(
      self, work_unit: orchestrator_helper.WORK_UNIT
  ) -> None:
    """Logs all details of the given work unit dataclass in a single logging.info call."""
    log_orchestrator_work_unit_info(work_unit)

  def format_work_unit_info(
      self, work_unit: orchestrator_helper.WORK_UNIT
  ) -> str:
    """Formats all details of the given work unit dataclass into a string."""
    return format_orchestrator_work_unit_info(work_unit)

  def connect(self) -> bool:
    """Connects to the Orchestrator server.

    Returns:
      True if connection succeeded, False otherwise.
    """
    logging.info("Connecting to Orchestrator server...")
    response = self._helper.connect()
    if not response.success:
      logging.error(
          "Failed to connect to Orchestrator: %s", response.error_message
      )
      self._is_connected = False
      return False

    self._is_connected = True
    logging.info("Successfully connected to Orchestrator.")
    return True

  def disconnect(self) -> None:
    """Disconnects from the Orchestrator server."""
    if self._is_connected:
      logging.info("Disconnecting from Orchestrator server...")
      self._helper.disconnect()
      self._is_connected = False

  def _get_active_work_unit_response(
      self,
  ) -> orchestrator_helper.RESPONSE:
    """Queries the active work unit or requests a new one if none is active."""
    response = self._helper.get_current_work_unit()
    if not response.success or response.work_unit is None:
      logging.info("No active work unit found, requesting work unit...")
      response = self._helper.request_work_unit()
    return response

  def _extract_from_launcher_artifacts(
      self,
      launcher_artifacts: Sequence[Any] | None,
  ) -> tuple[list[str], list[str]]:
    """Extracts artifact IDs and launch commands from launcher artifacts.

    Args:
      launcher_artifacts: Sequence of LauncherArtifact objects, or None.

    Returns:
      A tuple of (artifact_ids, launch_commands).
    """
    if not launcher_artifacts:
      return [], []

    artifact_ids: list[str] = []
    launch_commands: list[str] = []
    sorted_artifacts = sorted(
        launcher_artifacts,
        key=lambda x: x.launchOrder if x.launchOrder is not None else 999999,
    )
    for launcher_art in sorted_artifacts:
      if launcher_art.artifactId:
        artifact_ids.append(launcher_art.artifactId)
      if launcher_art.launchCommand:
        launch_commands.append(launcher_art.launchCommand)
    return artifact_ids, launch_commands

  def _extract_from_policies(
      self,
      context: orchestrator_helper.WORK_UNIT_CONTEXT,
  ) -> tuple[list[str], list[str]]:
    """Extracts artifact IDs and launch commands from policy details.

    Args:
      context: The work unit context containing policy information.

    Returns:
      A tuple of (artifact_ids, launch_commands).
    """
    artifact_ids: list[str] = []
    launch_commands: list[str] = []

    policies: list[orchestrator_helper.POLICY_DETAILS] = []
    if context.policies:
      policies.extend(context.policies)
    if context.policyDetails and context.policyDetails not in policies:
      policies.append(context.policyDetails)

    for details in policies:
      if not details:
        continue
      if details.artifactIds:
        artifact_ids.extend(details.artifactIds)
      params = (
          details.get_all_parameters()
          if hasattr(details, "get_all_parameters")
          else {}
      )
      if "launch_command" in params and isinstance(
          params["launch_command"], str
      ):
        launch_commands.append(params["launch_command"])
      if "launch_commands" in params and isinstance(
          params["launch_commands"], list
      ):
        launch_commands.extend(params["launch_commands"])
    return artifact_ids, launch_commands

  def _extract_response_launch_commands(
      self,
      response: orchestrator_helper.RESPONSE,
  ) -> list[str]:
    """Extracts top-level response and robot job launch commands.

    Args:
      response: The API response from the Orchestrator.

    Returns:
      A list of launch commands found on the response or robot job.
    """
    commands: list[str] = []
    if response.launch_command:
      commands.append(response.launch_command)
    if response.robot_job and response.robot_job.launchCommand:
      commands.append(response.robot_job.launchCommand)
    return commands

  def fetch_launch_info_from_orchestrator(
      self,
  ) -> tuple[list[str], list[str]]:
    """Fetches launcher artifact IDs and launch commands from current job/work unit.

    Returns:
      A tuple of (artifact_ids, launch_commands).
    """
    logging.info("Fetching current work unit launch info from Orchestrator...")
    artifact_ids: list[str] = []
    launch_commands: list[str] = []

    response = self._get_active_work_unit_response()
    if response.success and response.work_unit:
      work_unit = response.work_unit
      log_work_unit_info(work_unit)
      if work_unit.context:
        art_ids, cmds = self._extract_from_launcher_artifacts(
            work_unit.context.launcherArtifacts
        )
        artifact_ids.extend(art_ids)
        launch_commands.extend(cmds)

        policy_art_ids, policy_cmds = self._extract_from_policies(
            work_unit.context
        )
        artifact_ids.extend(policy_art_ids)
        launch_commands.extend(policy_cmds)

      launch_commands.extend(self._extract_response_launch_commands(response))

    # De-duplicate while preserving order
    unique_artifacts = list(dict.fromkeys(artifact_ids))
    unique_commands = list(dict.fromkeys(launch_commands))

    logging.info(
        "Found %d artifact IDs: %s", len(unique_artifacts), unique_artifacts
    )
    logging.info(
        "Found %d launch commands: %s", len(unique_commands), unique_commands
    )
    return unique_artifacts, unique_commands

  def _get_manifest_path(self) -> str:
    """Returns the absolute path to the local artifact cache manifest file."""
    return os.path.join(self._download_dir, _MANIFEST_FILENAME)

  def _load_cache_manifest(self) -> dict[str, dict[str, Any]]:
    """Loads the local artifact cache manifest."""
    manifest_path = self._get_manifest_path()
    if not os.path.exists(manifest_path):
      return {}
    try:
      with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning(
          "Failed to load artifact cache manifest [%s]: %s", manifest_path, e
      )
      return {}

  def _save_cache_manifest(self, manifest: dict[str, dict[str, Any]]) -> None:
    """Saves the local artifact cache manifest."""
    manifest_path = self._get_manifest_path()
    try:
      os.makedirs(self._download_dir, exist_ok=True)
      tmp_manifest_path = f"{manifest_path}.tmp"
      with open(tmp_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
      os.replace(tmp_manifest_path, manifest_path)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning(
          "Failed to save artifact cache manifest [%s]: %s", manifest_path, e
      )

  def _get_server_artifact_info(
      self, artifact_id: str
  ) -> tuple[str | None, str | None, str | None, str | None]:
    """Retrieves server artifact information including download URL, version, commit time, and fingerprint.

    Args:
      artifact_id: The ID of the artifact to inspect on the server.

    Returns:
      A tuple of (download_url, version, commit_time, server_fingerprint).
      If the artifact cannot be found or resolved, returns (None, None, None,
      None).
    """
    download_url: str | None = None
    version: str | None = None
    commit_time: str | None = None

    # First try get_artifact to get full metadata
    try:
      artifact_response = self._helper.get_artifact(artifact_id)
      if artifact_response.success and artifact_response.artifact:
        art = artifact_response.artifact
        download_url = art.uri
        version = art.version
        commit_time = art.commitTime
    except Exception:  # pylint: disable=broad-exception-caught
      pass

    # If download_url wasn't obtained, fallback Runningto get_artifact_uri
    if not download_url:
      try:
        uri_response = self._helper.get_artifact_uri(artifact_id)
        if uri_response.success and uri_response.artifact_uri:
          download_url = uri_response.artifact_uri
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error(
            "Failed to query artifact URI for [%s]: %s", artifact_id, e
        )

    if not download_url:
      return None, None, None, None

    # Construct server fingerprint from version, commit_time, and/or url
    fingerprint_parts = []
    if version:
      fingerprint_parts.append(f"v:{version}")
    if commit_time:
      fingerprint_parts.append(f"ct:{commit_time}")
    if download_url:
      base_url = download_url.split("?")[0]
      fingerprint_parts.append(f"url:{base_url}")

    server_fingerprint = (
        "|".join(fingerprint_parts) if fingerprint_parts else download_url
    )
    return download_url, version, commit_time, server_fingerprint

  def check_artifact_version(
      self, artifact_id: str
  ) -> tuple[bool, str | None, str | None]:
    """Checks if the local cached version matches the server version for an artifact.

    Args:
      artifact_id: The ID of the artifact to check.

    Returns:
      A tuple of (is_same, local_version_fingerprint,
      server_version_fingerprint).
      If the artifact is not cached locally or the target file does not exist,
      returns (False, None, server_version_fingerprint).
    """
    manifest = self._load_cache_manifest()
    cached_entry = manifest.get(artifact_id)

    _, _, _, server_fingerprint = self._get_server_artifact_info(artifact_id)

    if not cached_entry:
      return False, None, server_fingerprint

    target_path = cached_entry.get("target_path")
    if not isinstance(target_path, str) or not os.path.exists(target_path):
      logging.info(
          "Cached path [%s] for artifact [%s] does not exist on disk.",
          target_path,
          artifact_id,
      )
      return False, None, server_fingerprint

    local_fingerprint = cached_entry.get("server_fingerprint")
    if not isinstance(local_fingerprint, str) or not local_fingerprint:
      return False, None, server_fingerprint

    if server_fingerprint and local_fingerprint == server_fingerprint:
      return True, local_fingerprint, server_fingerprint

    return False, local_fingerprint, server_fingerprint

  def is_artifact_cached(self, artifact_id: str) -> bool:
    """Returns True if the artifact is cached locally and its target path exists."""
    manifest = self._load_cache_manifest()
    entry = manifest.get(artifact_id)
    if not entry:
      return False
    target_path = entry.get("target_path")
    return bool(
        isinstance(target_path, str)
        and target_path
        and os.path.exists(target_path)
    )

  def get_cached_artifact_path(self, artifact_id: str) -> str | None:
    """Returns the local path to the cached artifact if it exists, else None."""
    manifest = self._load_cache_manifest()
    entry = manifest.get(artifact_id)
    if not entry:
      return None
    target_path = entry.get("target_path")
    if isinstance(target_path, str) and os.path.exists(target_path):
      return target_path
    return None

  def clear_artifact_cache(self, artifact_id: str | None = None) -> None:
    """Clears the local cached files and manifest entry for an artifact or all artifacts.

    Args:
      artifact_id: The ID of the artifact to clear, or None to clear all cached
        artifacts.
    """
    manifest = self._load_cache_manifest()

    if artifact_id is not None:
      entry = manifest.pop(artifact_id, None)
      if entry:
        local_file_path = entry.get("local_file_path")
        target_path = entry.get("target_path")
        for path_to_clean in {local_file_path, target_path}:
          if isinstance(path_to_clean, str) and os.path.exists(path_to_clean):
            try:
              if os.path.isdir(path_to_clean):
                shutil.rmtree(path_to_clean)
              else:
                os.remove(path_to_clean)
              logging.info(
                  "Removed cached file/directory [%s] for artifact [%s].",
                  path_to_clean,
                  artifact_id,
              )
            except OSError as e:
              logging.warning(
                  "Failed to remove cached path [%s]: %s", path_to_clean, e
              )
      self._downloaded_artifact_paths.pop(artifact_id, None)
      self._save_cache_manifest(manifest)
      logging.info("Cleared cache for artifact [%s].", artifact_id)
    else:
      # Clear all artifacts
      for _, entry in list(manifest.items()):
        local_file_path = entry.get("local_file_path")
        target_path = entry.get("target_path")
        for path_to_clean in {local_file_path, target_path}:
          if isinstance(path_to_clean, str) and os.path.exists(path_to_clean):
            try:
              if os.path.isdir(path_to_clean):
                shutil.rmtree(path_to_clean)
              else:
                os.remove(path_to_clean)
            except OSError as e:
              logging.warning(
                  "Failed to remove cached path [%s]: %s", path_to_clean, e
              )
      self._downloaded_artifact_paths.clear()
      self._save_cache_manifest({})
      logging.info("Cleared all cached artifacts in [%s].", self._download_dir)

  def _get_valid_cached_artifact_path(
      self,
      artifact_id: str,
      server_fingerprint: str | None,
  ) -> str | None:
    """Returns valid cached target path if up-to-date, else invalidates cache.

    Args:
      artifact_id: The ID of the artifact to check.
      server_fingerprint: The server fingerprint to validate against.

    Returns:
      The cached target path if cached and up-to-date, otherwise None.
    """
    manifest = self._load_cache_manifest()
    cached_entry = manifest.get(artifact_id)
    if not cached_entry:
      return None

    cached_target_path = cached_entry.get("target_path")
    local_fingerprint = cached_entry.get("server_fingerprint")
    target_exists = isinstance(cached_target_path, str) and os.path.exists(
        cached_target_path
    )

    if (
        target_exists
        and isinstance(cached_target_path, str)
        and server_fingerprint
        and local_fingerprint == server_fingerprint
    ):
      logging.info(
          "Artifact [%s] is already cached and up-to-date at [%s]"
          " (version: %s). Skipping re-download.",
          artifact_id,
          cached_target_path,
          server_fingerprint,
      )
      return cached_target_path

    if (
        target_exists
        and server_fingerprint
        and local_fingerprint != server_fingerprint
    ):
      logging.warning(
          "Local cached version of artifact [%s] (local: %s) does not match"
          " server version (server: %s). Invalidating local cache and"
          " redownloading.",
          artifact_id,
          local_fingerprint,
          server_fingerprint,
      )
    elif not target_exists:
      logging.info(
          "Cached file for artifact [%s] is missing from disk. Redownloading.",
          artifact_id,
      )

    self.clear_artifact_cache(artifact_id)
    return None

  def _download_file(self, artifact_id: str, download_url: str) -> str:
    """Downloads an artifact from a URL and returns the local file path.

    Args:
      artifact_id: The ID of the artifact.
      download_url: The URL to download from.

    Returns:
      The local file path where the artifact was saved.

    Raises:
      RuntimeError: If downloading the artifact fails or the downloaded file
        does not exist on disk.
    """
    file_name = (
        download_url.split("?")[0].split("/")[-1] or f"artifact_{artifact_id}"
    )
    local_file_path = os.path.join(self._download_dir, file_name)

    logging.info(
        "Downloading artifact [%s] to [%s]...", artifact_id, local_file_path
    )
    try:
      urllib.request.urlretrieve(download_url, local_file_path)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.error("Failed downloading artifact [%s]: %s", artifact_id, e)
      raise RuntimeError(
          f"Failed downloading artifact [{artifact_id}] from [{download_url}]"
          f" to [{local_file_path}]: {e}"
      ) from e

    if not os.path.exists(local_file_path):
      logging.error(
          "Downloaded file for artifact [%s] does not exist at [%s].",
          artifact_id,
          local_file_path,
      )
      raise RuntimeError(
          f"Downloaded file for artifact [{artifact_id}] does not exist at"
          f" [{local_file_path}]."
      )

    try:
      os.chmod(local_file_path, 0o755)
    except OSError:
      pass

    return local_file_path

  def _update_cache_manifest_entry(
      self,
      *,
      artifact_id: str,
      server_version: str | None,
      server_commit_time: str | None,
      server_fingerprint: str | None,
      download_url: str,
      target_path: str,
  ) -> None:
    """Updates and saves the cache manifest with a new artifact entry."""
    manifest = self._load_cache_manifest()
    manifest[artifact_id] = {
        "artifact_id": artifact_id,
        "server_version": server_version,
        "server_commit_time": server_commit_time,
        "server_fingerprint": server_fingerprint,
        "download_url": download_url,
        "local_file_path": target_path,
        "target_path": target_path,
        "download_timestamp": time.time(),
    }
    self._save_cache_manifest(manifest)

  def _download_or_get_cached_artifact(self, artifact_id: str) -> str:
    """Downloads an artifact or retrieves it from cache if up-to-date.

    Args:
      artifact_id: The ID of the artifact.

    Returns:
      The local path to the artifact file.

    Raises:
      RuntimeError: If retrieving the download URI fails, downloading the
        artifact fails, or the downloaded file does not exist on disk.
    """
    logging.info("Checking artifact [%s]...", artifact_id)
    download_url, version, commit_time, server_fingerprint = (
        self._get_server_artifact_info(artifact_id)
    )

    if not download_url:
      logging.error(
          "Failed to get download URI for artifact [%s].", artifact_id
      )
      raise RuntimeError(
          f"Failed to get download URI for artifact [{artifact_id}]."
      )

    cached_path = self._get_valid_cached_artifact_path(
        artifact_id, server_fingerprint
    )
    if cached_path:
      return cached_path

    target_path = self._download_file(artifact_id, download_url)
    self._update_cache_manifest_entry(
        artifact_id=artifact_id,
        server_version=version,
        server_commit_time=commit_time,
        server_fingerprint=server_fingerprint,
        download_url=download_url,
        target_path=target_path,
    )
    return target_path

  def download_artifacts(self, artifact_ids: Sequence[str]) -> dict[str, str]:
    """Downloads or retrieves from local cache all specified artifacts.

    If an artifact is already cached and the local version matches the server
    version, the cached version is reused to avoid unnecessary network
    downloads.
    If the local version does not match the server version, a warning is logged,
    the stale local cache is invalidated and deleted, and the artifact is
    redownloaded.

    Args:
      artifact_ids: Sequence of artifact IDs to download.

    Returns:
      Dictionary mapping artifact_id to local downloaded/cached file or
      directory path.

    Raises:
      RuntimeError: If retrieving the download URI fails, downloading the
        artifact fails, or the downloaded file does not exist on disk.
    """
    os.makedirs(self._download_dir, exist_ok=True)
    downloaded_paths: dict[str, str] = {}

    for artifact_id in artifact_ids:
      if not artifact_id:
        continue
      target_path = self._download_or_get_cached_artifact(artifact_id)
      downloaded_paths[artifact_id] = target_path
      self._downloaded_artifact_paths[artifact_id] = target_path

    logging.info("Completed downloading %d artifacts.", len(downloaded_paths))
    return downloaded_paths

  def _find_terminal_emulator(self) -> str | None:
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

  def _build_terminal_command(
      self, command: str, title: str, cwd: str | None = None
  ) -> list[str]:
    """Builds the argument list to launch a command inside its own terminal window."""
    terminal_emulator = self._find_terminal_emulator()
    cd_command = f"cd '{cwd}'; " if cwd else ""
    safe_title = title.replace("'", "'\\''")
    safe_command_str = command.replace("'", "'\\''")
    shell_script = (
        f"{cd_command}"
        "echo '========================================'; "
        f"echo '[{safe_title}] {safe_command_str}'; "
        "echo '========================================'; "
        f"{command}"
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

  def launch_commands_in_sequence(
      self,
      launch_commands: Sequence[str],
      working_dir: str | None = None,
      env_vars: dict[str, str] | None = None,
      delay_between_commands_sec: float = 0.5,
  ) -> list[subprocess.Popen[Any]]:
    """Launches the given commands in sequence, each in its own terminal window.

    Each command is executed sequentially one after another in its own terminal
    window.
    The stdin, stdout, and stderr streams belong directly to the opened terminal
    window
    and are not redirected back to this launcher process.

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
      logging.info(
          "Launching process (%d/%d) in dedicated terminal window: [%s]"
          " (CWD: %s)",
          idx + 1,
          len(launch_commands),
          command,
          cwd,
      )

      cmd_args = self._build_terminal_command(command, command_label, cwd=cwd)

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

        logging.info(
            "Launched process [%s] (PID %d) in its own terminal window.",
            command_label,
            process.pid,
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

  def terminate_all_processes(self, timeout_sec: float = 5.0) -> None:
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

  def run(self, poll_interval: float = 1.0) -> int:
    """Runs the full launcher workflow end-to-end.

    Args:
      poll_interval: Process monitoring poll interval in seconds.

    Returns:
      Exit code: 0 on success, non-zero on error.
    """
    self.setup_signal_handlers()

    if not self.connect():
      return 1

    try:
      artifact_ids, launch_commands = self.fetch_launch_info_from_orchestrator()
      if not launch_commands and not artifact_ids:
        logging.warning(
            "No launch commands or artifact IDs found from Orchestrator."
        )

      if artifact_ids:
        self.download_artifacts(artifact_ids)

      if launch_commands:
        self.launch_commands_in_sequence(launch_commands)

        logging.info("Entering process monitoring loop. Press Ctrl+C to stop.")
        while not self._shutdown_event.is_set():
          all_alive = self.monitor_processes()
          if not all_alive:
            logging.info("One or more processes exited.")
            break
          time.sleep(poll_interval)

      return 0

    except KeyboardInterrupt:
      logging.info("Launcher interrupted by user.")
      return 0
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.error("Launcher execution failed: %s", e)
      return 1
    finally:
      self.terminate_all_processes()
      self.disconnect()
