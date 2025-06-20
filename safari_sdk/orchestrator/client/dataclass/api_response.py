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

"""Orchestrator API response format."""

import dataclasses

from googleapiclient import discovery
import numpy as np
from PIL import Image

from safari_sdk.orchestrator.client.dataclass import robot_job as robot_job_data
from safari_sdk.orchestrator.client.dataclass import work_unit as work_unit_data


@dataclasses.dataclass(frozen=True, kw_only=True)
class OrchestratorAPIResponse:
  """Orchestrator API response.

  All orchestrator client API calls will return their response with this
  dataclass. The only exception is for the disconnect() API call that is not
  expected to return any data. This gives an explicit information on if the API
  call was successful or not, as well as the requested data if any.

  Attributes:
    success: Whether the API call was successful.
    error_message: Error message if the API call was not successful.
    no_more_robot_job: If true, there are no active robot jobs available to this
      robot.
    no_more_work_unit: If true, there are no active work units available to this
      robot.
    is_visual_overlay_found: If true, there are visual overlay information
      specified within the current work unit.
    project_id: Project ID for the current robot job and work unit.
    robot_id: Robot ID for the current robot job and work unit.
    robot_job: Actual RobotJob dataclass object, containing all the values
      of the current robot job.
    robot_job_id: Robot job ID for the current robot job and work unit.
    work_unit_id: Work unit ID for the current work unit.
    work_unit: Actual WorkUnit dataclass object, containing all the values
      of the current work unit.
    work_unit_stage: Current stage of the work unit, if any.
    operator_id: Current operator ID for the robot, if any.
    is_operational: Whether the robot is operational.
    server_connection: Server connection to the orchestrator server.
    visual_overlay_renderer_keys: List of keys to access specific visual overlay
      renderers.
    visual_overlay_image: Image with visual overlay drawn on it.
  """

  success: bool = False
  error_message: str = ""
  no_more_robot_job: bool = False
  no_more_work_unit: bool = False
  is_visual_overlay_found: bool = False
  project_id: str | None = None
  robot_id: str | None = None
  robot_job: robot_job_data.RobotJob | None = None
  robot_job_id: str | None = None
  work_unit_id: str | None = None
  work_unit: work_unit_data.WorkUnit | None = None
  work_unit_stage: work_unit_data.WorkUnitStage | None = None
  operator_id: str | None = None
  is_operational: bool | None = None
  server_connection: discovery.Resource | None = None
  visual_overlay_renderer_keys: list[str] | None = None
  visual_overlay_image: Image.Image | np.ndarray | bytes | None = None
