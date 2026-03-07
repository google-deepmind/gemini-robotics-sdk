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

"""Example of integrating and using Orchestrator client SDK in observer mode.

This is an fully working example of how an user would integrate and use the
Orchestrator client SDK in observer mode. It shows how the user would call the
Orchestrator client SDK API methods.

For more details on each of the Orchestrator client SDK API methods, please
refer to the docstring of the helper file itself:
  orchestrator/helpers/orchestrator_helper.py.
"""

from collections.abc import Sequence
import time

from absl import app
from absl import flags

from safari_sdk.orchestrator.helpers import orchestrator_helper

# Required flags.
_ROBOT_ID = flags.DEFINE_string(
    name="robot_id",
    default=None,
    help="This robot's ID.",
    required=True,
)

_JOB_TYPE = flags.DEFINE_enum_class(
    name="job_type",
    default=orchestrator_helper.JOB_TYPE.ALL,
    enum_class=orchestrator_helper.JOB_TYPE,
    help="Type of job to run.",
)

# The flags below are optional.
_RAISE_ERROR = flags.DEFINE_bool(
    name="raise_error",
    default=False,
    help=(
        "Whether to raise the error as an exception or just show it as a"
        " messsage. Default = False."
    ),
)


def _print_orchestrator_work_unit_info(
    work_unit: orchestrator_helper.WORK_UNIT,
) -> None:
  """Prints out details of the given work unit dataclass."""
  print(f" Work Unit dataclass: {work_unit}\n")
  print(" ----------------------------------------------------------------\n")
  print(f" Robot Job ID: {work_unit.robotJobId}")
  print(f" Work Unit ID: {work_unit.workUnitId}")
  print(f" Work Unit stage: {work_unit.stage}")
  print(f" Work Unit outcome: {work_unit.outcome}")
  print(f" Work Unit note: {work_unit.note}\n")

  work_unit_context = work_unit.context
  if work_unit_context is not None:
    print(f" Scene Preset ID: {work_unit_context.scenePresetId}")
    print(f" Scene episode index: {work_unit_context.sceneEpisodeIndex}")
    print(f" Orchestrator Task ID: {work_unit_context.orchestratorTaskId}\n")

    success_scores = work_unit_context.successScores
    if success_scores is not None:
      for s_score in success_scores:
        print(" Success Scores:")
        print(f"   Definition: {s_score.definition}")
        print(f"   Score: {s_score.score}\n")

    scene_details = work_unit_context.scenePresetDetails
    if scene_details is not None:
      print(f" Setup Instructions: {scene_details.setupInstructions}")
      scene_params = scene_details.get_all_parameters()
      if scene_params:
        print(" Parameters:")
        for s_key, s_value in scene_params.items():
          print(f"   {s_key}: {s_value}")
      print(f" Grouping: {scene_details.grouping}")

      if scene_details.referenceImages:
        for ref_img in scene_details.referenceImages:
          print(" Reference Image:")
          print(f"   Artifact ID: {ref_img.artifactId}")
          print(f"   Source Topic: {ref_img.sourceTopic}")
          print(f"   Image width: {ref_img.rawImageWidth}")
          print(f"   Image height: {ref_img.rawImageHeight}")
          print(f"   UI width: {ref_img.renderedCanvasWidth}")
          print(f"   UI height: {ref_img.renderedCanvasHeight}\n")

      if scene_details.sceneObjects:
        for s_obj in scene_details.sceneObjects:
          print(" Scene Object:")
          print(f"   Object ID: {s_obj.objectId}")
          for t_label in s_obj.overlayTextLabels.labels:
            print(f"   Overlay Text Label: {t_label.text}")
          print(f"   Icon: {s_obj.evaluationLocation.overlayIcon}")
          print(f"   Layer Order: {s_obj.evaluationLocation.layerOrder}")
          print(
              "   RGB Hex Color Value:"
              f" {s_obj.evaluationLocation.rgbHexColorValue}"
          )

          if s_obj.evaluationLocation.location:
            print("   Coordinate: (UI frame)")
            print(f"     x: {s_obj.evaluationLocation.location.coordinate.x}")
            print(f"     y: {s_obj.evaluationLocation.location.coordinate.y}")
            if s_obj.evaluationLocation.location.direction:
              print("   Direction:")
              print(
                  "     radian:"
                  f" {s_obj.evaluationLocation.location.direction.rad}"
              )

          if s_obj.evaluationLocation.containerArea:
            if s_obj.evaluationLocation.containerArea.circle:
              print("   Coordinate: (UI frame)")
              print(
                  "     x:"
                  f" {s_obj.evaluationLocation.containerArea.circle.center.x}"
              )
              print(
                  "     y:"
                  f" {s_obj.evaluationLocation.containerArea.circle.center.y}"
              )
              print(
                  "   Radius:"
                  f" {s_obj.evaluationLocation.containerArea.circle.radius}"
              )
            if s_obj.evaluationLocation.containerArea.box:
              print("   Coordinate: (UI frame)")
              print(f"     x: {s_obj.evaluationLocation.containerArea.box.x}")
              print(f"     y: {s_obj.evaluationLocation.containerArea.box.y}")
              print(f"   Width: {s_obj.evaluationLocation.containerArea.box.w}")
              print(
                  f"   Height: {s_obj.evaluationLocation.containerArea.box.h}"
              )
          print(
              "   Reference Image Artifact ID:"
              f" {s_obj.sceneReferenceImageArtifactId}\n"
          )

    policy_details = work_unit_context.policyDetails
    if policy_details is not None:
      print(f" Policy Name: {policy_details.name}")
      print(f" Policy Description: {policy_details.description}")
      policy_params = policy_details.get_all_parameters()
      if policy_params:
        print(" Parameters:")
        for p_key, p_value in policy_params.items():
          print(f"   {p_key}: {p_value}")
      print("")

    questions = work_unit_context.questions
    if questions is not None:
      for q in questions:
        print(" Questionnaire:")
        print(f"   Question: {q.question}")
        print(f"   When to ask: {q.whenToAsk}")
        print(f"   Answer format: {q.answerFormat}")
        print(f"   Allowed answers: {q.allowedAnswers}")
      print("")

  print(" ----------------------------------------------------------------\n")


def run_mock_observer_mode_loop(
    orchestrator_client: orchestrator_helper.OrchestratorHelper,
) -> None:
  """Runs mock eval loop."""
  print(" - Asking to observe latest work unit -\n")

  response = orchestrator_client.observe_latest_work_unit()
  if not response.success:
    print(f"\n - ERROR: {response.error_message} -\n")
    return
  if response.no_more_work_unit:
    print(" - No work unit available to observe -\n")
    return
  print(" - Sucessfully got latest work unit for observation -\n")

  print(" - Current work unit information -\n")
  work_unit = response.work_unit
  assert work_unit is not None
  _print_orchestrator_work_unit_info(work_unit=work_unit)

  print("[Sleeping for 5 seconds to simulate observer program execution]\n")
  time.sleep(5)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  print(" - Initializing and connecting to orchestrator -\n")
  orchestrator_client = orchestrator_helper.OrchestratorHelper(
      robot_id=_ROBOT_ID.value,
      job_type=_JOB_TYPE.value,
      observer_mode=True,
      raise_error=_RAISE_ERROR.value,
  )
  response = orchestrator_client.connect()
  if not response.success:
    print(f"\n - ERROR: {response.error_message} -\n")
    return

  if response.latest_robot_release_configs:
    print(" - Latest robot release configs -\n")
    for config in response.latest_robot_release_configs:
      print(f"   {config.key}: {config.get_value()}")
    print("")

  print(" - Running mock observer program -\n")
  run_mock_observer_mode_loop(
      orchestrator_client=orchestrator_client,
  )

  print(" - Disconnecting from orchestrator -\n")
  orchestrator_client.disconnect()

  print(" - Mock eval completed -\n")


if __name__ == "__main__":
  app.run(main)
