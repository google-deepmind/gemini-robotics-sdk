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

"""Current robot information in Orchestrator."""

import dataclasses
import dataclasses_json
from safari_sdk.orchestrator.client.dataclass import work_unit

# pylint: disable=invalid-name


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class CurrentRobotInfoResponse:
  """Current information about the robot."""

  robotId: str
  isOperational: bool | None = False
  operatorId: str | None = None
  robotJobId: str | None = None
  workUnitId: str | None = None
  stage: work_unit.WorkUnitStage | None = None

  def __post_init__(self):
    if self.stage is None:
      self.stage = work_unit.WorkUnitStage.WORK_UNIT_STAGE_UNSPECIFIED
    elif isinstance(self.stage, str):
      self.stage = work_unit.WorkUnitStage(self.stage)
