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

"""Orchestrator work unit information."""

import dataclasses
import enum
from typing import Any
import dataclasses_json

# pylint: disable=invalid-name


class WorkUnitStage(enum.Enum):
  WORK_UNIT_STAGE_UNSPECIFIED = "WORK_UNIT_STAGE_UNSPECIFIED"
  WORK_UNIT_STAGE_CREATED = "WORK_UNIT_STAGE_CREATED"
  WORK_UNIT_STAGE_QUEUED_TO_ROBOT = "WORK_UNIT_STAGE_QUEUED_TO_ROBOT"
  WORK_UNIT_STAGE_ROBOT_SOFTWARE_ASSETS_PREP = (
      "WORK_UNIT_STAGE_ROBOT_SOFTWARE_ASSETS_PREP"
  )
  WORK_UNIT_STAGE_ROBOT_SCENE_PREP = "WORK_UNIT_STAGE_ROBOT_SCENE_PREP"
  WORK_UNIT_STAGE_ROBOT_EXECUTION = "WORK_UNIT_STAGE_ROBOT_EXECUTION"
  WORK_UNIT_STAGE_COMPLETED = "WORK_UNIT_STAGE_COMPLETED"
  WORK_UNIT_STAGE_CANCELLED = "WORK_UNIT_STAGE_CANCELLED"


class WorkUnitOutcome(enum.Enum):
  WORK_UNIT_OUTCOME_UNSPECIFIED = "WORK_UNIT_OUTCOME_UNSPECIFIED"
  WORK_UNIT_OUTCOME_SUCCESS = "WORK_UNIT_OUTCOME_SUCCESS"
  WORK_UNIT_OUTCOME_FAILURE = "WORK_UNIT_OUTCOME_FAILURE"
  WORK_UNIT_OUTCOME_INVALID = "WORK_UNIT_OUTCOME_INVALID"

  def num_value(self) -> int:
    match self:
      case WorkUnitOutcome.WORK_UNIT_OUTCOME_SUCCESS:
        return 1
      case WorkUnitOutcome.WORK_UNIT_OUTCOME_FAILURE:
        return 2
      case WorkUnitOutcome.WORK_UNIT_OUTCOME_INVALID:
        return 3
      case _:
        return 0


class KvMsgValueType(enum.Enum):
  KV_MSG_VALUE_TYPE_UNSPECIFIED = "KV_MSG_VALUE_TYPE_UNSPECIFIED"
  KV_MSG_VALUE_TYPE_STRING = "KV_MSG_VALUE_TYPE_STRING"
  KV_MSG_VALUE_TYPE_STRING_LIST = "KV_MSG_VALUE_TYPE_STRING_LIST"
  KV_MSG_VALUE_TYPE_INT = "KV_MSG_VALUE_TYPE_INT"
  KV_MSG_VALUE_TYPE_INT_LIST = "KV_MSG_VALUE_TYPE_INT_LIST"
  KV_MSG_VALUE_TYPE_FLOAT = "KV_MSG_VALUE_TYPE_FLOAT"
  KV_MSG_VALUE_TYPE_FLOAT_LIST = "KV_MSG_VALUE_TYPE_FLOAT_LIST"
  KV_MSG_VALUE_TYPE_BOOL = "KV_MSG_VALUE_TYPE_BOOL"
  KV_MSG_VALUE_TYPE_BOOL_LIST = "KV_MSG_VALUE_TYPE_BOOL_LIST"
  KV_MSG_VALUE_TYPE_JSON = "KV_MSG_VALUE_TYPE_JSON"


class OverlayObjectIcon(enum.Enum):
  OVERLAY_OBJECT_ICON_UNSPECIFIED = "OVERLAY_OBJECT_ICON_UNSPECIFIED"
  OVERLAY_OBJECT_ICON_CIRCLE = "OVERLAY_OBJECT_ICON_CIRCLE"
  OVERLAY_OBJECT_ICON_ARROW = "OVERLAY_OBJECT_ICON_ARROW"
  OVERLAY_OBJECT_ICON_SQUARE = "OVERLAY_OBJECT_ICON_SQUARE"
  OVERLAY_OBJECT_ICON_TRIANGLE = "OVERLAY_OBJECT_ICON_TRIANGLE"
  OVERLAY_OBJECT_ICON_CONTAINER = "OVERLAY_OBJECT_ICON_CONTAINER"


class RobotJobAssetType(enum.Enum):
  ROBOT_JOB_ASSET_TYPE_UNSPECIFIED = "ROBOT_JOB_ASSET_TYPE_UNSPECIFIED"
  ROBOT_JOB_ASSET_TYPE_DOCKER_IMAGE = "ROBOT_JOB_ASSET_TYPE_DOCKER_IMAGE"
  ROBOT_JOB_ASSET_TYPE_PAR = "ROBOT_JOB_ASSET_TYPE_PAR"
  ROBOT_JOB_ASSET_TYPE_CHECKPOINT = "ROBOT_JOB_ASSET_TYPE_CHECKPOINT"


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class KvMsgValue:
  """Actual values for Key-Value pair message.

  Although all fields in this class can be populated, the actual value that is
  returned when called via get_value() is determined by the KvMsgValueType
  in the parent KvMsg class.

  Attributes:
    stringValue: Singular string object.
    stringListValue: List of string objects.
    intValue: Singular integer object.
    intListValue: List of integer objects.
    floatValue: Singular float object.
    floatListValue: List of float objects.
    boolValue: Singular boolean object.
    boolListValue: List of boolean objects.
    jsonValue: JSON object in string format.
  """

  stringValue: str | None = None
  stringListValue: list[str] | None = None
  intValue: int | None = None
  intListValue: list[int] | None = None
  floatValue: float | None = None
  floatListValue: list[float] | None = None
  boolValue: bool | None = None
  boolListValue: list[bool] | None = None
  jsonValue: str | None = None


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class KvMsg:
  """Key-Value pair message.

  Attributes:
    key: Key of the key-value pair message.
    type: Enum to indicate which value field to return within KvMsgValue.
    value: Value of the key-value pair message.
  """

  key: str | None = None
  type: KvMsgValueType | None = None
  value: KvMsgValue | None = None

  def get_value(self) -> Any:
    if self.value is None:
      return None
    match self.type:
      case KvMsgValueType.KV_MSG_VALUE_TYPE_STRING:
        return self.value.stringValue
      case KvMsgValueType.KV_MSG_VALUE_TYPE_STRING_LIST:
        return self.value.stringListValue
      case KvMsgValueType.KV_MSG_VALUE_TYPE_INT:
        return self.value.intValue
      case KvMsgValueType.KV_MSG_VALUE_TYPE_INT_LIST:
        return self.value.intListValue
      case KvMsgValueType.KV_MSG_VALUE_TYPE_FLOAT:
        return self.value.floatValue
      case KvMsgValueType.KV_MSG_VALUE_TYPE_FLOAT_LIST:
        return self.value.floatListValue
      case KvMsgValueType.KV_MSG_VALUE_TYPE_BOOL:
        return self.value.boolValue
      case KvMsgValueType.KV_MSG_VALUE_TYPE_BOOL_LIST:
        return self.value.boolListValue
      case KvMsgValueType.KV_MSG_VALUE_TYPE_JSON:
        return self.value.jsonValue
      case _:
        return None


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class SceneReferenceImage:
  """Reference image information use to create visual overlay renderer.

  Attributes:
    artifactId: The artifact id of the reference image in GCS.
    renderedCanvasWidth: The width of the rendered canvas in Orchestrator
      website UI.
    renderedCanvasHeight: The height of the rendered canvas in Orchestrator
      website UI.
    sourceTopic: The source topic of the reference image as provided by the user
      into the Orchestrator website UI.
    rawImageWidth: The width of the raw image as provided by the user into the
      Orchestrator website UI.
    rawImageHeight: The height of the raw image as provided by the user into
      the Orchestrator website UI.
  """

  artifactId: str | None = None
  renderedCanvasWidth: int | None = None
  renderedCanvasHeight: int | None = None
  sourceTopic: str | None = None
  rawImageWidth: int | None = None
  rawImageHeight: int | None = None


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class PixelLocation:
  """Pixel location in the image frame.

  Upper left corner of the image frame is (0, 0).
  """

  x: int | None = None
  y: int | None = None


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class PixelDirection:
  """Direction in radians in the image frame.

  Radian of 0 is right, pi/2 is up, pi or -pi is left, and -pi/2 is down.
  """

  rad: float | None = None


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class PixelVector:
  """Required information to render icons by the visual overlay renderer.

  Direction is only needed for arrow icons.
  """

  coordinate: PixelLocation | None = None
  direction: PixelDirection | None = None


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class ShapeCircle:
  """Required information to render a circle by the visual overlay renderer."""

  center: PixelLocation | None = None
  radius: int | None = None


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class ShapeBox:
  """Required information to render a rectangle by the visual overlay renderer.

  X and Y are the top left corner of the rectangle. W and H are the width and
  height of the rectangle.
  """

  x: int | None = None
  y: int | None = None
  w: int | None = None
  h: int | None = None


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class ContainerArea:
  """Required information to render a container by the visual overlay renderer.

  The container can be a rectangle or a circle, which is determined by which
  shape field is used.
  """

  circle: ShapeCircle | None = None
  box: ShapeBox | None = None


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class FixedLocation:
  """Required infomation to render an object by the visual overlay renderer."""

  overlayIcon: OverlayObjectIcon | None = None
  layerOrder: int | None = None
  rgbHexColorValue: str | None = None
  location: PixelVector | None = None
  containerArea: ContainerArea | None = None


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class OverlayText:
  """Actual text to be rendered by the visual overlay renderer."""

  text: str | None = None


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class OverlayTextLabel:
  """List of text labels to be rendered by the visual overlay renderer.

  Currently, only the first label in the list will be rendered.
  """

  labels: list[OverlayText] | None = None


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class SceneObject:
  """Individual scene object information for visual overlay rendering."""

  objectId: str | None = None
  overlayTextLabels: OverlayTextLabel | None = None
  evaluationLocation: FixedLocation | None = None
  sceneReferenceImageArtifactId: str | None = None


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class ScenePresetDetails:
  """Scene preset details.

  Attributes:
    setupInstructions: Instructions for setting up the scene.
    parameters: List of key-value pair parameters for the scene.
    grouping: List of grouping names for the scene.
    referenceImages: List of reference images for the scene.
    sceneObjects: List of scene objects for the scene.
  """

  setupInstructions: str | None = None
  parameters: list[KvMsg] | None = None
  grouping: list[str] | None = None
  referenceImages: list[SceneReferenceImage] | None = None
  sceneObjects: list[SceneObject] | None = None

  def get_all_parameters(self) -> dict[str, Any]:
    if self.parameters is None:
      return {}
    return {kv.key: kv.get_value() for kv in self.parameters}

  def get_parameter_value(self, key: str, default_value: Any = None) -> Any:
    if self.parameters is None:
      return default_value
    for parameter in self.parameters:
      if parameter.key == key:
        return parameter.get_value()
    return default_value


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class PolicyDetails:
  """Policy details.

  Attributes:
    name: Name of the policy.
    description: Description of the policy.
    parameters: List of key-value pair parameters for the policy.
  """

  name: str | None = None
  description: str | None = None
  parameters: list[KvMsg] | None = None

  def get_all_parameters(self) -> dict[str, Any]:
    if self.parameters is None:
      return {}
    return {kv.key: kv.get_value() for kv in self.parameters}

  def get_parameter_value(self, key: str, default_value: Any = None) -> Any:
    if self.parameters is None:
      return default_value
    for parameter in self.parameters:
      if parameter.key == key:
        return parameter.get_value()
    return default_value


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class RobotJobAsset:
  """Robot job asset information."""

  downloadUri: str | None = None
  assetType: RobotJobAssetType | None = None


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class WorkUnitContext:
  """Work unit context information."""

  scenePresetId: str | None = None
  sceneEpisodeIndex: int | None = None
  scenePresetDetails: ScenePresetDetails | None = None
  orchestratorTaskId: str | None = None
  policyDetails: PolicyDetails | None = None
  robotJobAssets: list[RobotJobAsset] | None = None


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class WorkUnit:
  """Orchestrator work unit information."""

  projectId: str | None = None
  robotJobId: str | None = None
  workUnitId: str | None = None
  context: WorkUnitContext | None = None
  stage: WorkUnitStage | None = None
  outcome: WorkUnitOutcome | None = None
  note: str | None = None

  def __post_init__(self):
    if self.stage is None:
      self.stage = WorkUnitStage.WORK_UNIT_STAGE_UNSPECIFIED
    elif isinstance(self.stage, str):
      self.stage = WorkUnitStage(self.stage)

    if self.outcome is None:
      self.outcome = WorkUnitOutcome.WORK_UNIT_OUTCOME_UNSPECIFIED
    elif isinstance(self.outcome, str):
      self.outcome = WorkUnitOutcome(self.outcome)


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class WorkUnitResponse:
  """Orchestrator work unit information."""

  workUnit: WorkUnit
