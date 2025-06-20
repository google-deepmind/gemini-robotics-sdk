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

"""Example of integrating and using Orchestrator client SDK into a mock binary.

This is an fully working example of how an user would integrate and use the
Orchestrator client SDK in a mock eval binary. It shows how, where and when the
user would call each of the Orchestrator client SDK API methods.

For more details on each of the Orchestrator client SDK API methods, please
refer to the docstring of the helper file itself:
  orchestrator/helpers/orchestrator_helper.py.
"""

from collections.abc import Sequence
import dataclasses
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
_NUM_WORK_UNITS_TO_RUN = flags.DEFINE_integer(
    name="num_work_units_to_run",
    default=None,
    help="Number of work units to run.  Default = run all assigned work units.",
)

_RAISE_ERROR = flags.DEFINE_bool(
    name="raise_error",
    default=False,
    help=(
        "Whether to raise the error as an exception or just show it as a"
        " messsage. Default = False."
    ),
)


@dataclasses.dataclass
class EvalPolicyParams:
  orchestrator_client: orchestrator_helper.OrchestratorHelper
  num_work_units_to_run: int | None


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
  print(" ----------------------------------------------------------------\n")


def run_mock_eval_loop(params: EvalPolicyParams) -> None:
  """Runs mock eval loop."""
  current_work_unit_idx = 1

  while True:
    current_work_unit_msg = f" - Requesting work unit #{current_work_unit_idx}"
    if params.num_work_units_to_run is None:
      current_work_unit_msg += " -\n"
    else:
      current_work_unit_msg += f" / {params.num_work_units_to_run} -\n"
    print(current_work_unit_msg)

    response = params.orchestrator_client.request_work_unit()
    if response.success:
      if response.no_more_robot_job:
        print(" - No robot job available -\n")
        break
      if response.no_more_work_unit:
        print(" - No work unit available -\n")
        break
      print(" - Sucessfully requested work unit -\n")
    else:
      print(f"\n - ERROR: {response.error_message} -\n")
      break

    response = params.orchestrator_client.get_current_work_unit()
    if not response.success:
      print(f"\n - ERROR: {response.error_message} -\n")
      break

    print(" - Current work unit information -\n")
    work_unit = response.work_unit
    assert work_unit is not None
    _print_orchestrator_work_unit_info(work_unit=work_unit)

    print(" - Starting software asset prep for current work unit -\n")
    response = params.orchestrator_client.start_work_unit_software_asset_prep()
    if not response.success:
      print(f"\n - ERROR: {response.error_message} -\n")
      break

    print("[Sleeping for 1 seconds to simulate software asset prep]\n")
    time.sleep(1)

    print(" - Starting scene prep for current work unit -\n")
    response = params.orchestrator_client.start_work_unit_scene_prep()
    if not response.success:
      print(f"\n - ERROR: {response.error_message} -\n")
      break

    print(" - Checking if current work unit have visual overlay info -\n")
    response = (
        params.orchestrator_client.is_visual_overlay_in_current_work_unit()
    )
    if not response.success:
      print(f"\n - ERROR: {response.error_message} -\n")
      break
    generate_visual_overlays = response.is_visual_overlay_found

    if not generate_visual_overlays:
      print(" - No visual overlay information found -\n")
    else:
      print(" - Creating visual overlays for current work unit -\n")
      response = (
          params.orchestrator_client.create_visual_overlays_for_current_work_unit()
      )
      if not response.success:
        print(f"\n - ERROR: {response.error_message} -\n")
        break

      print(" - Getting list of visual overlay renderer key names -\n")
      response = (
          params.orchestrator_client.list_visual_overlay_renderer_keys()
      )
      if not response.success:
        print(f"\n - ERROR: {response.error_message} -\n")
        break

      renderer_keys = response.visual_overlay_renderer_keys
      if renderer_keys:
        renderer_idx_display = 1
        print(" - Found following visual overlay renderer -")
        for r_key in renderer_keys:
          print(f"   #{renderer_idx_display}: {r_key}")
          renderer_idx_display += 1

        print("\n - Rendering overlay with current camera image for... -")
        renderer_idx_display = 1
        for r_key in renderer_keys:
          print(f"   #{renderer_idx_display}: {r_key}")
          response = params.orchestrator_client.render_visual_overlay(
              renderer_key=r_key,
              # Provide the latest image for the new_image argument here.
              new_image=None,
          )
          if not response.success:
            print(f"\n - ERROR: {response.error_message} -\n")
            break
          renderer_idx_display += 1

        # You only need to extract the overlay images in your desired format.
        # The following code shows how to extract the overlay images as PIL
        # image.
        print("\n - Extracting overlay images as PIL image... -")
        renderer_idx_display = 1
        for r_key in renderer_keys:
          print(f"   #{renderer_idx_display}: {r_key}")
          response = (
              params.orchestrator_client.get_visual_overlay_image_as_pil_image(
                  renderer_key=r_key
              )
          )

          # To extract as numpy array, use the following code instead:
          # response = (
          #     params.orchestrator_client.get_visual_overlay_image_as_np_array(
          #         renderer_key=r_key
          #     )
          # )

          # To extract as bytes image, use the following code instead:
          # response = (
          #     params.orchestrator_client.get_visual_overlay_image_as_bytes(
          #         renderer_key=r_key
          #     )
          # )

          if not response.success:
            print(f"\n - ERROR: {response.error_message} -\n")
            break
          renderer_idx_display += 1

        print("\n - Mock feeding your UI with the latest overlay images -\n")
      else:
        print(" - No visual overlay renderer keys found -\n")

    print("[Sleeping for 1 seconds to simulate scene prep]\n")
    time.sleep(1)

    print(" - Starting episode execution for current work unit -\n")
    response = params.orchestrator_client.start_work_unit_execution()
    if not response.success:
      print(f"\n - ERROR: {response.error_message} -\n")
      break

    print("[Sleeping for 3 seconds to simulate episode execution]\n")
    time.sleep(3)

    print(" - Marking current work unit as completed -\n")
    response = params.orchestrator_client.complete_work_unit(
        outcome=orchestrator_helper.WORK_UNIT_OUTCOME.WORK_UNIT_OUTCOME_SUCCESS,
        note="This is a mock episode from SDK example code.",
    )
    if not response.success:
      print(f"\n - ERROR: {response.error_message} -\n")
      break

    current_work_unit_idx += 1
    if (
        params.num_work_units_to_run is not None
        and current_work_unit_idx > params.num_work_units_to_run
    ):
      break

    print("[Sleeping for 3 seconds to simulate reset between episodes]\n")
    time.sleep(3)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if (
      _NUM_WORK_UNITS_TO_RUN.value is not None
      and _NUM_WORK_UNITS_TO_RUN.value <= 0
  ):
    raise app.UsageError(
        "Max number of work units to run must be greater than zero."
    )

  print(" - Initializing and connecting to orchestrator -\n")
  orchestrator_client = orchestrator_helper.OrchestratorHelper(
      robot_id=_ROBOT_ID.value,
      job_type=_JOB_TYPE.value,
      raise_error=_RAISE_ERROR.value,
  )
  response = orchestrator_client.connect()
  if not response.success:
    print(f"\n - ERROR: {response.error_message} -\n")
    return

  params = EvalPolicyParams(
      orchestrator_client=orchestrator_client,
      num_work_units_to_run=_NUM_WORK_UNITS_TO_RUN.value,
  )

  print(" - Running mock eval -\n")
  run_mock_eval_loop(params=params)

  print(" - Disconnecting from orchestrator -\n")
  orchestrator_client.disconnect()

  print(" - Mock eval completed -\n")


if __name__ == "__main__":
  app.run(main)
