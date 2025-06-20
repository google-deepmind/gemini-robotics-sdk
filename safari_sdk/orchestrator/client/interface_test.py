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

"""Unit tests for interface.py."""

from unittest import mock

from absl import flags
from absl.testing import absltest

from safari_sdk.orchestrator.client import interface

FLAGS = flags.FLAGS
FLAGS.mark_as_parsed()


class InterfaceTest(absltest.TestCase):

  def test_connect_good(self):

    FLAGS.api_key = "mock_test_key"
    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )

    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_build.return_value = mock.Mock(
          spec=interface.auth.discovery.Resource
      )
      response = interface_lib.connect()
      self.assertTrue(response.success)

  def test_connect_bad_connect(self):

    FLAGS.api_key = None
    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.COLLECTION,
    )

    response = interface_lib.connect()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, "Auth: No API key provided by flag or file."
    )

  def test_get_current_connection_good(self, *_):

    FLAGS.api_key = "mock_test_key"
    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.EVALUATION,
    )

    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_build.return_value = mock.Mock(
          spec=interface.auth.discovery.Resource
      )
      response = interface_lib.connect()
      self.assertTrue(response.success)

    response = interface_lib.get_current_connection()
    self.assertTrue(response.success)
    self.assertIsInstance(
        response.server_connection, interface.auth.discovery.Resource
    )
    self.assertEqual(response.robot_id, "test_robot_id")

  def test_get_current_connection_bad_active_connection(self):
    FLAGS.api_key = None
    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )
    interface_lib._connection = None

    response = interface_lib.get_current_connection()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_ACTIVE_CONNECTION
    )

  @mock.patch(
      "safari_sdk.orchestrator.client.libs.current_robot.OrchestratorCurrentRobotInfo.get_current_robot_info",
      return_value=interface.current_robot._RESPONSE(
          success=True,
          robot_id="test_robot_id",
          operator_id="test_operator_id",
          is_operational=True,
      ),
  )
  def test_get_current_robot_info_good(self, *_):
    FLAGS.api_key = "mock_test_key"
    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )

    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_build.return_value = mock.Mock(
          spec=interface.auth.discovery.Resource
      )
      response = interface_lib.connect()
      self.assertTrue(response.success)

    response = interface_lib.get_current_robot_info()
    self.assertTrue(response.success)
    self.assertEqual(response.robot_id, "test_robot_id")
    self.assertTrue(response.is_operational)
    self.assertEqual(response.operator_id, "test_operator_id")

  def test_get_current_robot_info_bad_active_connection(self):

    FLAGS.api_key = None
    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )
    interface_lib._connection = None
    interface_lib._current_robot_lib = mock.MagicMock(
        spec=interface.current_robot.OrchestratorCurrentRobotInfo
    )

    response = interface_lib.get_current_robot_info()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_ACTIVE_CONNECTION
    )

    interface_lib._connection = mock.Mock(
        spec=interface.auth.discovery.Resource
    )
    interface_lib._current_robot_lib = None

    response = interface_lib.get_current_robot_info()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_ACTIVE_CONNECTION
    )

  @mock.patch(
      "safari_sdk.orchestrator.client.libs.current_robot.OrchestratorCurrentRobotInfo.set_current_robot_operator_id",
      return_value=interface.current_robot._RESPONSE(
          success=True,
          robot_id="test_robot_id",
          operator_id="test_operator_id",
          is_operational=True,
      ),
  )
  def test_set_current_robot_operator_id_good(self, *_):

    FLAGS.api_key = "mock_test_key"
    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )

    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_build.return_value = mock.Mock(
          spec=interface.auth.discovery.Resource
      )
      response = interface_lib.connect()
      self.assertTrue(response.success)

    response = interface_lib.set_current_robot_operator_id(
        operator_id="test_operator_id"
    )
    self.assertTrue(response.success)
    self.assertEqual(response.robot_id, "test_robot_id")
    self.assertTrue(response.is_operational)
    self.assertEqual(response.operator_id, "test_operator_id")

  def test_set_current_robot_operator_id_bad_active_connection(self):

    FLAGS.api_key = None
    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )
    interface_lib._connection = None
    interface_lib._current_robot_lib = mock.MagicMock(
        spec=interface.current_robot.OrchestratorCurrentRobotInfo
    )

    response = interface_lib.set_current_robot_operator_id(
        operator_id="test_operator_id"
    )
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_ACTIVE_CONNECTION
    )

    interface_lib._connection = mock.Mock(
        spec=interface.auth.discovery.Resource
    )
    interface_lib._current_robot_lib = None

    response = interface_lib.set_current_robot_operator_id(
        operator_id="test_operator_id"
    )
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_ACTIVE_CONNECTION
    )

  @mock.patch(
      "safari_sdk.orchestrator.client.libs.robot_job.OrchestratorRobotJob.request_robot_job",
      return_value=interface.robot_job_work_unit._RESPONSE(
          success=True,
          robot_id="test_robot_id",
          robot_job_id="test_robot_job_id",
      ),
  )
  @mock.patch(
      "safari_sdk.orchestrator.client.libs.robot_job_work_unit.OrchestratorRobotJobWorkUnit.request_work_unit",
      return_value=interface.robot_job_work_unit._RESPONSE(
          success=True,
          robot_id="test_robot_id",
          robot_job_id="test_robot_job_id",
          work_unit_id="test_work_unit_id",
          work_unit=interface.robot_job_work_unit.work_unit.WorkUnit(
              robotJobId="test_robot_job_id",
              workUnitId="test_work_unit_id",
          ),
      ),
  )
  def test_request_robot_job_work_unit_good(self, *_):

    FLAGS.api_key = "mock_test_key"
    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )

    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_build.return_value = mock.Mock(
          spec=interface.auth.discovery.Resource
      )
      response = interface_lib.connect()
      self.assertTrue(response.success)

    response = interface_lib.request_robot_job_work_unit()
    self.assertTrue(response.success)
    self.assertEqual(response.robot_id, "test_robot_id")
    self.assertEqual(response.robot_job_id, "test_robot_job_id")
    self.assertEqual(response.work_unit_id, "test_work_unit_id")
    self.assertIsInstance(
        response.work_unit, interface.robot_job_work_unit.work_unit.WorkUnit
    )

  def test_request_robot_job_work_unit_bad_active_connection(self):

    FLAGS.api_key = None
    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )
    interface_lib._connection = None
    interface_lib._robot_job_lib = mock.MagicMock(
        spec=interface.robot_job.OrchestratorRobotJob
    )
    interface_lib._robot_job_work_unit_lib = mock.MagicMock(
        spec=interface.robot_job_work_unit.OrchestratorRobotJobWorkUnit
    )

    response = interface_lib.request_robot_job_work_unit()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_ACTIVE_CONNECTION
    )

    interface_lib._connection = mock.Mock(
        spec=interface.auth.discovery.Resource
    )
    interface_lib._robot_job_lib = None
    interface_lib._robot_job_work_unit_lib = mock.MagicMock(
        spec=interface.robot_job_work_unit.OrchestratorRobotJobWorkUnit
    )

    response = interface_lib.request_robot_job_work_unit()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_ACTIVE_CONNECTION
    )

    interface_lib._connection = mock.Mock(
        spec=interface.auth.discovery.Resource
    )
    interface_lib._robot_job_lib = mock.MagicMock(
        spec=interface.robot_job.OrchestratorRobotJob
    )
    interface_lib._robot_job_work_unit_lib = None

    response = interface_lib.request_robot_job_work_unit()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_ACTIVE_CONNECTION
    )

  @mock.patch(
      "safari_sdk.orchestrator.client.libs.robot_job.OrchestratorRobotJob.request_robot_job",
      return_value=interface.robot_job_work_unit._RESPONSE(
          success=False, error_message="Error from request robot job."
      ),
  )
  def test_request_robot_job_work_unit_bad_request_robot_job(self, *_):

    FLAGS.api_key = "mock_test_key"
    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )

    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_build.return_value = mock.Mock(
          spec=interface.auth.discovery.Resource
      )
      response = interface_lib.connect()
      self.assertTrue(response.success)

    response = interface_lib.request_robot_job_work_unit()
    self.assertFalse(response.success)
    self.assertEqual(response.error_message, "Error from request robot job.")

  @mock.patch(
      "safari_sdk.orchestrator.client.libs.robot_job.OrchestratorRobotJob.request_robot_job",
      return_value=interface.robot_job_work_unit._RESPONSE(
          success=True,
          no_more_robot_job=True,
          error_message="No more robot job.",
      ),
  )
  def test_request_robot_job_work_unit_no_more_robot_job(self, *_):

    FLAGS.api_key = "mock_test_key"
    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )

    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_build.return_value = mock.Mock(
          spec=interface.auth.discovery.Resource
      )
      response = interface_lib.connect()
      self.assertTrue(response.success)

    response = interface_lib.request_robot_job_work_unit()
    self.assertTrue(response.success)
    self.assertTrue(response.no_more_robot_job)
    self.assertEqual(response.error_message, "No more robot job.")

  @mock.patch(
      "safari_sdk.orchestrator.client.libs.robot_job.OrchestratorRobotJob.request_robot_job",
      return_value=interface.robot_job_work_unit._RESPONSE(
          success=True,
          robot_id="test_robot_id",
          robot_job_id="test_robot_job_id",
      ),
  )
  @mock.patch(
      "safari_sdk.orchestrator.client.libs.robot_job_work_unit.OrchestratorRobotJobWorkUnit.get_current_work_unit",
      return_value=interface.robot_job_work_unit._RESPONSE(
          success=True,
          robot_id="test_robot_id",
          robot_job_id="test_robot_job_id",
          work_unit_id="test_work_unit_id",
          work_unit=interface.robot_job_work_unit.work_unit.WorkUnit(
              robotJobId="test_robot_job_id",
              workUnitId="test_work_unit_id",
          ),
      ),
  )
  def test_get_current_robot_job_work_unit_good(self, *_):

    FLAGS.api_key = "mock_test_key"
    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )

    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_build.return_value = mock.Mock(
          spec=interface.auth.discovery.Resource
      )
      response = interface_lib.connect()
      self.assertTrue(response.success)

    response = interface_lib.get_current_robot_job_work_unit()
    self.assertTrue(response.success)
    self.assertEqual(response.robot_id, "test_robot_id")
    self.assertEqual(response.robot_job_id, "test_robot_job_id")
    self.assertEqual(response.work_unit_id, "test_work_unit_id")
    self.assertIsInstance(
        response.work_unit, interface.robot_job_work_unit.work_unit.WorkUnit
    )

  def test_get_current_robot_job_work_unit_bad_active_connection(self):

    FLAGS.api_key = None
    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )
    interface_lib._connection = None
    interface_lib._robot_job_lib = mock.MagicMock(
        spec=interface.robot_job.OrchestratorRobotJob
    )
    interface_lib._robot_job_work_unit_lib = mock.MagicMock(
        spec=interface.robot_job_work_unit.OrchestratorRobotJobWorkUnit
    )

    response = interface_lib.get_current_robot_job_work_unit()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_ACTIVE_CONNECTION
    )

    interface_lib._connection = mock.Mock(
        spec=interface.auth.discovery.Resource
    )
    interface_lib._robot_job_lib = None
    interface_lib._robot_job_work_unit_lib = mock.MagicMock(
        spec=interface.robot_job_work_unit.OrchestratorRobotJobWorkUnit
    )

    response = interface_lib.get_current_robot_job_work_unit()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_ACTIVE_CONNECTION
    )

    interface_lib._connection = mock.Mock(
        spec=interface.auth.discovery.Resource
    )
    interface_lib._robot_job_lib = mock.MagicMock(
        spec=interface.robot_job.OrchestratorRobotJob
    )
    interface_lib._robot_job_work_unit_lib = None

    response = interface_lib.get_current_robot_job_work_unit()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_ACTIVE_CONNECTION
    )

  @mock.patch(
      "safari_sdk.orchestrator.client.libs.robot_job_work_unit.OrchestratorRobotJobWorkUnit.get_current_work_unit",
      return_value=interface.robot_job_work_unit._RESPONSE(
          success=True,
          robot_id="test_robot_id",
          robot_job_id="test_robot_job_id",
          work_unit_id="test_work_unit_id",
          work_unit=interface.robot_job_work_unit.work_unit.WorkUnit(
              robotJobId="test_robot_job_id",
              workUnitId="test_work_unit_id",
              context=interface.robot_job_work_unit.work_unit.WorkUnitContext(
                  scenePresetId="test_scene_preset_id",
                  sceneEpisodeIndex=1,
                  scenePresetDetails=interface.robot_job_work_unit.work_unit.ScenePresetDetails(
                      sceneObjects=[
                          interface.robot_job_work_unit.work_unit.SceneObject(
                              objectId="test_object_id",
                              overlayTextLabels=interface.robot_job_work_unit.work_unit.OverlayTextLabel(
                                  labels=[
                                      interface.robot_job_work_unit.work_unit.OverlayText(
                                          text="test_overlay_text_label",
                                      )
                                  ],
                              ),
                              evaluationLocation=interface.robot_job_work_unit.work_unit.FixedLocation(
                                  overlayIcon=interface.robot_job_work_unit.work_unit.OverlayObjectIcon.OVERLAY_OBJECT_ICON_CIRCLE,
                                  layerOrder=1,
                                  rgbHexColorValue="FF0000",
                                  location=interface.robot_job_work_unit.work_unit.PixelVector(
                                      coordinate=interface.robot_job_work_unit.work_unit.PixelLocation(
                                          x=10,
                                          y=10,
                                      )
                                  ),
                              ),
                              sceneReferenceImageArtifactId="test_artifact_id",
                          ),
                      ],
                      referenceImages=[
                          interface.robot_job_work_unit.work_unit.SceneReferenceImage(
                              artifactId="test_artifact_id",
                              sourceTopic="test_source_topic",
                              rawImageWidth=100,
                              rawImageHeight=100,
                              renderedCanvasWidth=100,
                              renderedCanvasHeight=100,
                          ),
                      ],
                  ),
              ),
          ),
      ),
  )
  def test_is_visual_overlay_in_current_work_unit_good(self, *_):

    FLAGS.api_key = "mock_test_key"
    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )

    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_build.return_value = mock.Mock(
          spec=interface.auth.discovery.Resource
      )
      response = interface_lib.connect()
      self.assertTrue(response.success)

    response = interface_lib.is_visual_overlay_in_current_work_unit()
    self.assertTrue(response.success)
    self.assertTrue(response.is_visual_overlay_found)

  @mock.patch(
      "safari_sdk.orchestrator.client.libs.robot_job_work_unit.OrchestratorRobotJobWorkUnit.get_current_work_unit",
      return_value=interface.robot_job_work_unit._RESPONSE(
          success=True,
          robot_id="test_robot_id",
          robot_job_id="test_robot_job_id",
          work_unit_id="test_work_unit_id",
          work_unit=interface.robot_job_work_unit.work_unit.WorkUnit(
              robotJobId="test_robot_job_id",
              workUnitId="test_work_unit_id",
              context=interface.robot_job_work_unit.work_unit.WorkUnitContext(
                  scenePresetId="test_scene_preset_id",
                  sceneEpisodeIndex=1,
                  scenePresetDetails=interface.robot_job_work_unit.work_unit.ScenePresetDetails(
                      referenceImages=[
                          interface.robot_job_work_unit.work_unit.SceneReferenceImage(
                              artifactId="test_artifact_id",
                              sourceTopic="test_source_topic",
                              rawImageWidth=100,
                              rawImageHeight=100,
                              renderedCanvasWidth=100,
                              renderedCanvasHeight=100,
                          ),
                      ],
                  ),
              ),
          ),
      ),
  )
  def test_is_visual_overlay_in_current_work_unit_good_with_no_scene_objects(
      self, *_
  ):

    FLAGS.api_key = "mock_test_key"
    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )

    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_build.return_value = mock.Mock(
          spec=interface.auth.discovery.Resource
      )
      response = interface_lib.connect()
      self.assertTrue(response.success)

    response = interface_lib.is_visual_overlay_in_current_work_unit()
    self.assertTrue(response.success)
    self.assertTrue(response.is_visual_overlay_found)

  def test_is_visual_overlay_in_current_work_unit_bad_active_connection(
      self
  ):

    FLAGS.api_key = None
    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )
    interface_lib._connection = None
    interface_lib._robot_job_work_unit_lib = mock.MagicMock(
        spec=interface.robot_job_work_unit.OrchestratorRobotJobWorkUnit
    )

    response = interface_lib.is_visual_overlay_in_current_work_unit()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_ACTIVE_CONNECTION
    )

    interface_lib._connection = mock.Mock(
        spec=interface.auth.discovery.Resource
    )
    interface_lib._robot_job_work_unit_lib = None

    response = interface_lib.is_visual_overlay_in_current_work_unit()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_ACTIVE_CONNECTION
    )

  @mock.patch(
      "safari_sdk.orchestrator.client.libs.robot_job_work_unit.OrchestratorRobotJobWorkUnit.get_current_work_unit",
      return_value=interface.robot_job_work_unit._RESPONSE(
          success=True,
          robot_id="test_robot_id",
          robot_job_id="test_robot_job_id",
          work_unit_id="test_work_unit_id",
          work_unit=interface.robot_job_work_unit.work_unit.WorkUnit(
              robotJobId="test_robot_job_id",
              workUnitId="test_work_unit_id",
          ),
      ),
  )
  def test_is_visual_overlay_in_current_work_unit_with_no_context(
      self, *_
  ):

    FLAGS.api_key = "mock_test_key"
    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )

    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_build.return_value = mock.Mock(
          spec=interface.auth.discovery.Resource
      )
      response = interface_lib.connect()
      self.assertTrue(response.success)

    response = interface_lib.is_visual_overlay_in_current_work_unit()
    self.assertTrue(response.success)
    self.assertFalse(response.is_visual_overlay_found)

  @mock.patch(
      "safari_sdk.orchestrator.client.libs.robot_job_work_unit.OrchestratorRobotJobWorkUnit.get_current_work_unit",
      return_value=interface.robot_job_work_unit._RESPONSE(
          success=True,
          robot_id="test_robot_id",
          robot_job_id="test_robot_job_id",
          work_unit_id="test_work_unit_id",
          work_unit=interface.robot_job_work_unit.work_unit.WorkUnit(
              robotJobId="test_robot_job_id",
              workUnitId="test_work_unit_id",
              context=interface.robot_job_work_unit.work_unit.WorkUnitContext(
                  scenePresetId="test_scene_preset_id",
                  sceneEpisodeIndex=1,
              ),
          ),
      ),
  )
  def test_is_visual_overlay_in_current_work_unit_with_no_scene_preset_details(
      self, *_
  ):

    FLAGS.api_key = "mock_test_key"
    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )

    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_build.return_value = mock.Mock(
          spec=interface.auth.discovery.Resource
      )
      response = interface_lib.connect()
      self.assertTrue(response.success)

    response = interface_lib.is_visual_overlay_in_current_work_unit()
    self.assertTrue(response.success)
    self.assertFalse(response.is_visual_overlay_found)

  @mock.patch(
      "safari_sdk.orchestrator.client.libs.robot_job_work_unit.OrchestratorRobotJobWorkUnit.get_current_work_unit",
      return_value=interface.robot_job_work_unit._RESPONSE(
          success=True,
          robot_id="test_robot_id",
          robot_job_id="test_robot_job_id",
          work_unit_id="test_work_unit_id",
          work_unit=interface.robot_job_work_unit.work_unit.WorkUnit(
              robotJobId="test_robot_job_id",
              workUnitId="test_work_unit_id",
              context=interface.robot_job_work_unit.work_unit.WorkUnitContext(
                  scenePresetId="test_scene_preset_id",
                  sceneEpisodeIndex=1,
                  scenePresetDetails=interface.robot_job_work_unit.work_unit.ScenePresetDetails(
                      sceneObjects=[
                          interface.robot_job_work_unit.work_unit.SceneObject(
                              objectId="test_object_id",
                              overlayTextLabels=interface.robot_job_work_unit.work_unit.OverlayTextLabel(
                                  labels=[
                                      interface.robot_job_work_unit.work_unit.OverlayText(
                                          text="test_overlay_text_label",
                                      )
                                  ],
                              ),
                              evaluationLocation=interface.robot_job_work_unit.work_unit.FixedLocation(
                                  overlayIcon=interface.robot_job_work_unit.work_unit.OverlayObjectIcon.OVERLAY_OBJECT_ICON_CIRCLE,
                                  layerOrder=1,
                                  rgbHexColorValue="FF0000",
                                  location=interface.robot_job_work_unit.work_unit.PixelVector(
                                      coordinate=interface.robot_job_work_unit.work_unit.PixelLocation(
                                          x=10,
                                          y=10,
                                      )
                                  ),
                              ),
                              sceneReferenceImageArtifactId="test_artifact_id",
                          ),
                      ],
                  ),
              ),
          ),
      ),
  )
  def test_is_visual_overlay_in_current_work_unit_with_no_reference_images(
      self, *_
  ):

    FLAGS.api_key = "mock_test_key"
    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )

    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_build.return_value = mock.Mock(
          spec=interface.auth.discovery.Resource
      )
      response = interface_lib.connect()
      self.assertTrue(response.success)

    response = interface_lib.is_visual_overlay_in_current_work_unit()
    self.assertTrue(response.success)
    self.assertFalse(response.is_visual_overlay_found)

  @mock.patch(
      "safari_sdk.orchestrator.client.libs.robot_job_work_unit.OrchestratorRobotJobWorkUnit.get_current_work_unit",
      return_value=interface.robot_job_work_unit._RESPONSE(
          success=True,
          robot_id="test_robot_id",
          robot_job_id="test_robot_job_id",
          work_unit_id="test_work_unit_id",
          work_unit=interface.robot_job_work_unit.work_unit.WorkUnit(
              robotJobId="test_robot_job_id",
              workUnitId="test_work_unit_id",
              context=interface.robot_job_work_unit.work_unit.WorkUnitContext(
                  scenePresetId="test_scene_preset_id",
                  sceneEpisodeIndex=1,
                  scenePresetDetails=interface.robot_job_work_unit.work_unit.ScenePresetDetails(
                      sceneObjects=[
                          interface.robot_job_work_unit.work_unit.SceneObject(
                              objectId="test_object_id",
                              overlayTextLabels=interface.robot_job_work_unit.work_unit.OverlayTextLabel(
                                  labels=[
                                      interface.robot_job_work_unit.work_unit.OverlayText(
                                          text="test_overlay_text_label",
                                      )
                                  ],
                              ),
                              evaluationLocation=interface.robot_job_work_unit.work_unit.FixedLocation(
                                  overlayIcon=interface.robot_job_work_unit.work_unit.OverlayObjectIcon.OVERLAY_OBJECT_ICON_CIRCLE,
                                  layerOrder=1,
                                  rgbHexColorValue="FF0000",
                                  location=interface.robot_job_work_unit.work_unit.PixelVector(
                                      coordinate=interface.robot_job_work_unit.work_unit.PixelLocation(
                                          x=10,
                                          y=10,
                                      )
                                  ),
                              ),
                              sceneReferenceImageArtifactId="test_artifact_id",
                          ),
                      ],
                      referenceImages=[
                          interface.robot_job_work_unit.work_unit.SceneReferenceImage(
                              artifactId="test_artifact_id",
                              sourceTopic="test_source_topic",
                              rawImageWidth=100,
                              rawImageHeight=100,
                              renderedCanvasWidth=100,
                              renderedCanvasHeight=100,
                          ),
                      ],
                  ),
              ),
          ),
      ),
  )
  def test_create_visual_overlays_for_current_work_unit_good(self, *_):

    FLAGS.api_key = "mock_test_key"
    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )

    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_build.return_value = mock.Mock(
          spec=interface.auth.discovery.Resource
      )
      response = interface_lib.connect()
      self.assertTrue(response.success)

    response = interface_lib.create_visual_overlays_for_current_work_unit()
    self.assertTrue(response.success)
    self.assertLen(interface_lib._visual_overlay, 1)
    self.assertIn("test_source_topic", interface_lib._visual_overlay)
    self.assertIsInstance(
        interface_lib._visual_overlay["test_source_topic"],
        interface.visual_overlay.OrchestratorRenderer,
    )

  @mock.patch(
      "safari_sdk.orchestrator.client.libs.robot_job_work_unit.OrchestratorRobotJobWorkUnit.get_current_work_unit",
      return_value=interface.robot_job_work_unit._RESPONSE(
          success=True,
          robot_id="test_robot_id",
          robot_job_id="test_robot_job_id",
          work_unit_id="test_work_unit_id",
          work_unit=interface.robot_job_work_unit.work_unit.WorkUnit(
              robotJobId="test_robot_job_id",
              workUnitId="test_work_unit_id",
              context=interface.robot_job_work_unit.work_unit.WorkUnitContext(
                  scenePresetId="test_scene_preset_id",
                  sceneEpisodeIndex=1,
                  scenePresetDetails=interface.robot_job_work_unit.work_unit.ScenePresetDetails(
                      referenceImages=[
                          interface.robot_job_work_unit.work_unit.SceneReferenceImage(
                              artifactId="test_artifact_id",
                              sourceTopic="test_source_topic",
                              rawImageWidth=100,
                              rawImageHeight=100,
                              renderedCanvasWidth=100,
                              renderedCanvasHeight=100,
                          ),
                      ],
                  ),
              ),
          ),
      ),
  )
  def test_create_visual_overlays_for_current_work_unit_good_with_no_scene_objects(
      self, *_
  ):

    FLAGS.api_key = "mock_test_key"
    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )

    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_build.return_value = mock.Mock(
          spec=interface.auth.discovery.Resource
      )
      response = interface_lib.connect()
      self.assertTrue(response.success)

    response = interface_lib.create_visual_overlays_for_current_work_unit()
    self.assertTrue(response.success)
    self.assertLen(interface_lib._visual_overlay, 1)
    self.assertIn("test_source_topic", interface_lib._visual_overlay)
    self.assertIsInstance(
        interface_lib._visual_overlay["test_source_topic"],
        interface.visual_overlay.OrchestratorRenderer,
    )

  def test_create_visual_overlays_for_current_work_unit_bad_active_connection(
      self
  ):

    FLAGS.api_key = None
    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )
    interface_lib._connection = None
    interface_lib._robot_job_work_unit_lib = mock.MagicMock(
        spec=interface.robot_job_work_unit.OrchestratorRobotJobWorkUnit
    )

    response = interface_lib.create_visual_overlays_for_current_work_unit()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_ACTIVE_CONNECTION
    )

    interface_lib._connection = mock.Mock(
        spec=interface.auth.discovery.Resource
    )
    interface_lib._robot_job_work_unit_lib = None

    response = interface_lib.create_visual_overlays_for_current_work_unit()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_ACTIVE_CONNECTION
    )

  @mock.patch(
      "safari_sdk.orchestrator.client.libs.robot_job_work_unit.OrchestratorRobotJobWorkUnit.get_current_work_unit",
      return_value=interface.robot_job_work_unit._RESPONSE(
          success=True,
          robot_id="test_robot_id",
          robot_job_id="test_robot_job_id",
          work_unit_id="test_work_unit_id",
          work_unit=interface.robot_job_work_unit.work_unit.WorkUnit(
              robotJobId="test_robot_job_id",
              workUnitId="test_work_unit_id",
          ),
      ),
  )
  def test_create_visual_overlays_for_current_work_unit_with_no_context(
      self, *_
  ):

    FLAGS.api_key = "mock_test_key"
    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )

    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_build.return_value = mock.Mock(
          spec=interface.auth.discovery.Resource
      )
      response = interface_lib.connect()
      self.assertTrue(response.success)

    response = interface_lib.create_visual_overlays_for_current_work_unit()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_WORK_UNIT_CONTEXT
    )

  @mock.patch(
      "safari_sdk.orchestrator.client.libs.robot_job_work_unit.OrchestratorRobotJobWorkUnit.get_current_work_unit",
      return_value=interface.robot_job_work_unit._RESPONSE(
          success=True,
          robot_id="test_robot_id",
          robot_job_id="test_robot_job_id",
          work_unit_id="test_work_unit_id",
          work_unit=interface.robot_job_work_unit.work_unit.WorkUnit(
              robotJobId="test_robot_job_id",
              workUnitId="test_work_unit_id",
              context=interface.robot_job_work_unit.work_unit.WorkUnitContext(
                  scenePresetId="test_scene_preset_id",
                  sceneEpisodeIndex=1,
              ),
          ),
      ),
  )
  def test_create_visual_overlays_for_current_work_unit_with_no_scene_preset_details(
      self, *_
  ):

    FLAGS.api_key = "mock_test_key"
    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )

    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_build.return_value = mock.Mock(
          spec=interface.auth.discovery.Resource
      )
      response = interface_lib.connect()
      self.assertTrue(response.success)

    response = interface_lib.create_visual_overlays_for_current_work_unit()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_SCENE_PRESET_DETAILS
    )

  @mock.patch(
      "safari_sdk.orchestrator.client.libs.robot_job_work_unit.OrchestratorRobotJobWorkUnit.get_current_work_unit",
      return_value=interface.robot_job_work_unit._RESPONSE(
          success=True,
          robot_id="test_robot_id",
          robot_job_id="test_robot_job_id",
          work_unit_id="test_work_unit_id",
          work_unit=interface.robot_job_work_unit.work_unit.WorkUnit(
              robotJobId="test_robot_job_id",
              workUnitId="test_work_unit_id",
              context=interface.robot_job_work_unit.work_unit.WorkUnitContext(
                  scenePresetId="test_scene_preset_id",
                  sceneEpisodeIndex=1,
                  scenePresetDetails=interface.robot_job_work_unit.work_unit.ScenePresetDetails(
                      sceneObjects=[
                          interface.robot_job_work_unit.work_unit.SceneObject(
                              objectId="test_object_id",
                              overlayTextLabels=interface.robot_job_work_unit.work_unit.OverlayTextLabel(
                                  labels=[
                                      interface.robot_job_work_unit.work_unit.OverlayText(
                                          text="test_overlay_text_label",
                                      )
                                  ],
                              ),
                              evaluationLocation=interface.robot_job_work_unit.work_unit.FixedLocation(
                                  overlayIcon=interface.robot_job_work_unit.work_unit.OverlayObjectIcon.OVERLAY_OBJECT_ICON_CIRCLE,
                                  layerOrder=1,
                                  rgbHexColorValue="FF0000",
                                  location=interface.robot_job_work_unit.work_unit.PixelVector(
                                      coordinate=interface.robot_job_work_unit.work_unit.PixelLocation(
                                          x=10,
                                          y=10,
                                      )
                                  ),
                              ),
                              sceneReferenceImageArtifactId="test_artifact_id",
                          ),
                      ],
                  ),
              ),
          ),
      ),
  )
  def test_create_visual_overlays_for_current_work_unit_with_no_reference_images(
      self, *_
  ):

    FLAGS.api_key = "mock_test_key"
    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )

    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_build.return_value = mock.Mock(
          spec=interface.auth.discovery.Resource
      )
      response = interface_lib.connect()
      self.assertTrue(response.success)

    response = interface_lib.create_visual_overlays_for_current_work_unit()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_REFERENCE_IMAGES
    )

  def test_list_visual_overlay_renderer_keys_good(self):

    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )
    interface_lib._visual_overlay = {
        "test_renderer_key_1": interface.visual_overlay.OrchestratorRenderer(
            scene_reference_image_data=(
                interface.robot_job_work_unit.work_unit.SceneReferenceImage(
                    artifactId="test_artifact_id_1",
                    sourceTopic="test_renderer_key_1",
                    rawImageWidth=10,
                    rawImageHeight=10,
                    renderedCanvasWidth=10,
                    renderedCanvasHeight=10,
                )
            )
        ),
        "test_renderer_key_2": interface.visual_overlay.OrchestratorRenderer(
            scene_reference_image_data=(
                interface.robot_job_work_unit.work_unit.SceneReferenceImage(
                    artifactId="test_artifact_id_2",
                    sourceTopic="test_renderer_key_2",
                    rawImageWidth=20,
                    rawImageHeight=20,
                    renderedCanvasWidth=20,
                    renderedCanvasHeight=20,
                )
            )
        ),
        "test_renderer_key_3": interface.visual_overlay.OrchestratorRenderer(
            scene_reference_image_data=(
                interface.robot_job_work_unit.work_unit.SceneReferenceImage(
                    artifactId="test_artifact_id_3",
                    sourceTopic="test_renderer_key_3",
                    rawImageWidth=30,
                    rawImageHeight=30,
                    renderedCanvasWidth=30,
                    renderedCanvasHeight=30,
                )
            )
        ),
        "test_renderer_key_4": interface.visual_overlay.OrchestratorRenderer(
            scene_reference_image_data=(
                interface.robot_job_work_unit.work_unit.SceneReferenceImage(
                    artifactId="test_artifact_id_4",
                    sourceTopic="test_renderer_key_4",
                    rawImageWidth=40,
                    rawImageHeight=40,
                    renderedCanvasWidth=40,
                    renderedCanvasHeight=40,
                )
            )
        ),
    }

    response = interface_lib.list_visual_overlay_renderer_keys()
    self.assertTrue(response.success)
    self.assertLen(response.visual_overlay_renderer_keys, 4)
    self.assertSameElements(
        response.visual_overlay_renderer_keys,
        [
            "test_renderer_key_1",
            "test_renderer_key_2",
            "test_renderer_key_3",
            "test_renderer_key_4",
        ],
    )

  def test_list_visual_overlay_renderer_keys_no_visual_overlay_found(self):

    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )

    response = interface_lib.list_visual_overlay_renderer_keys()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_RENDERER_FOUND
    )

  def test_get_visual_overlay_image_as_pil_image_good(self):

    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )
    interface_lib._visual_overlay = {
        "test_renderer_key_1": interface.visual_overlay.OrchestratorRenderer(
            scene_reference_image_data=(
                interface.robot_job_work_unit.work_unit.SceneReferenceImage(
                    artifactId="test_artifact_id_1",
                    sourceTopic="test_renderer_key_1",
                    rawImageWidth=10,
                    rawImageHeight=10,
                    renderedCanvasWidth=10,
                    renderedCanvasHeight=10,
                )
            )
        ),
        "test_renderer_key_2": interface.visual_overlay.OrchestratorRenderer(
            scene_reference_image_data=(
                interface.robot_job_work_unit.work_unit.SceneReferenceImage(
                    artifactId="test_artifact_id_2",
                    sourceTopic="test_renderer_key_2",
                    rawImageWidth=20,
                    rawImageHeight=20,
                    renderedCanvasWidth=20,
                    renderedCanvasHeight=20,
                )
            )
        ),
    }

    response = interface_lib.get_visual_overlay_image_as_pil_image(
        renderer_key="test_renderer_key_1"
    )
    self.assertTrue(response.success)
    image = response.visual_overlay_image
    self.assertIsNotNone(image)
    self.assertIsInstance(image, interface.visual_overlay.Image.Image)
    self.assertEqual(image.width, 10)
    self.assertEqual(image.height, 10)

    response = interface_lib.get_visual_overlay_image_as_pil_image(
        renderer_key="test_renderer_key_2"
    )
    self.assertTrue(response.success)
    image = response.visual_overlay_image
    self.assertIsNotNone(image)
    self.assertIsInstance(image, interface.visual_overlay.Image.Image)
    self.assertEqual(image.width, 20)
    self.assertEqual(image.height, 20)

  def test_get_visual_overlay_image_as_pil_image_bad_renderer_key(self):

    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )
    interface_lib._visual_overlay = {
        "test_renderer_key_1": interface.visual_overlay.OrchestratorRenderer(
            scene_reference_image_data=(
                interface.robot_job_work_unit.work_unit.SceneReferenceImage(
                    artifactId="test_artifact_id_1",
                    sourceTopic="test_renderer_key_1",
                    rawImageWidth=10,
                    rawImageHeight=10,
                    renderedCanvasWidth=10,
                    renderedCanvasHeight=10,
                )
            )
        ),
    }

    response = interface_lib.get_visual_overlay_image_as_pil_image(
        renderer_key="test_renderer_key_2"
    )
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_RENDERER_FOUND
    )

  def test_get_visual_overlay_image_as_np_array_good(self):

    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )
    interface_lib._visual_overlay = {
        "test_renderer_key_1": interface.visual_overlay.OrchestratorRenderer(
            scene_reference_image_data=(
                interface.robot_job_work_unit.work_unit.SceneReferenceImage(
                    artifactId="test_artifact_id_1",
                    sourceTopic="test_renderer_key_1",
                    rawImageWidth=10,
                    rawImageHeight=10,
                    renderedCanvasWidth=10,
                    renderedCanvasHeight=10,
                )
            )
        ),
        "test_renderer_key_2": interface.visual_overlay.OrchestratorRenderer(
            scene_reference_image_data=(
                interface.robot_job_work_unit.work_unit.SceneReferenceImage(
                    artifactId="test_artifact_id_2",
                    sourceTopic="test_renderer_key_2",
                    rawImageWidth=20,
                    rawImageHeight=20,
                    renderedCanvasWidth=20,
                    renderedCanvasHeight=20,
                )
            )
        ),
    }

    response = interface_lib.get_visual_overlay_image_as_np_array(
        renderer_key="test_renderer_key_1"
    )
    self.assertTrue(response.success)
    image = response.visual_overlay_image
    self.assertIsNotNone(image)
    self.assertIsInstance(image, interface.visual_overlay.np.ndarray)
    self.assertEqual(image.shape, (10, 10, 3))

    response = interface_lib.get_visual_overlay_image_as_np_array(
        renderer_key="test_renderer_key_2"
    )
    self.assertTrue(response.success)
    image = response.visual_overlay_image
    self.assertIsNotNone(image)
    self.assertIsInstance(image, interface.visual_overlay.np.ndarray)
    self.assertEqual(image.shape, (20, 20, 3))

  def test_get_visual_overlay_image_as_np_array_bad_renderer_key(self):

    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )
    interface_lib._visual_overlay = {
        "test_renderer_key_1": interface.visual_overlay.OrchestratorRenderer(
            scene_reference_image_data=(
                interface.robot_job_work_unit.work_unit.SceneReferenceImage(
                    artifactId="test_artifact_id_1",
                    sourceTopic="test_renderer_key_1",
                    rawImageWidth=10,
                    rawImageHeight=10,
                    renderedCanvasWidth=10,
                    renderedCanvasHeight=10,
                )
            )
        ),
    }

    response = interface_lib.get_visual_overlay_image_as_np_array(
        renderer_key="test_renderer_key_2"
    )
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_RENDERER_FOUND
    )

  def test_get_visual_overlay_image_as_bytes_good(self):

    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )
    interface_lib._visual_overlay = {
        "test_renderer_key_1": interface.visual_overlay.OrchestratorRenderer(
            scene_reference_image_data=(
                interface.robot_job_work_unit.work_unit.SceneReferenceImage(
                    artifactId="test_artifact_id_1",
                    sourceTopic="test_renderer_key_1",
                    rawImageWidth=10,
                    rawImageHeight=10,
                    renderedCanvasWidth=10,
                    renderedCanvasHeight=10,
                )
            )
        ),
        "test_renderer_key_2": interface.visual_overlay.OrchestratorRenderer(
            scene_reference_image_data=(
                interface.robot_job_work_unit.work_unit.SceneReferenceImage(
                    artifactId="test_artifact_id_2",
                    sourceTopic="test_renderer_key_2",
                    rawImageWidth=20,
                    rawImageHeight=20,
                    renderedCanvasWidth=20,
                    renderedCanvasHeight=20,
                )
            )
        ),
    }

    response = interface_lib.get_visual_overlay_image_as_bytes(
        renderer_key="test_renderer_key_1"
    )
    self.assertTrue(response.success)
    image = response.visual_overlay_image
    self.assertIsNotNone(image)
    self.assertIsInstance(image, bytes)

    response = interface_lib.get_visual_overlay_image_as_bytes(
        renderer_key="test_renderer_key_2",
        img_format=interface.visual_overlay.ImageFormat.PNG,
    )
    self.assertTrue(response.success)
    image_png = response.visual_overlay_image
    self.assertIsNotNone(image_png)
    self.assertIsInstance(image_png, bytes)

    response = interface_lib.get_visual_overlay_image_as_bytes(
        renderer_key="test_renderer_key_2",
        img_format=interface.visual_overlay.ImageFormat.JPEG,
    )
    self.assertTrue(response.success)
    image_jpeg = response.visual_overlay_image
    self.assertIsNotNone(image_jpeg)
    self.assertIsInstance(image_jpeg, bytes)

    self.assertNotEqual(image_png, image_jpeg)

  def test_get_visual_overlay_image_as_bytes_bad_renderer_key(self):

    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )
    interface_lib._visual_overlay = {
        "test_renderer_key_1": interface.visual_overlay.OrchestratorRenderer(
            scene_reference_image_data=(
                interface.robot_job_work_unit.work_unit.SceneReferenceImage(
                    artifactId="test_artifact_id_1",
                    sourceTopic="test_renderer_key_1",
                    rawImageWidth=10,
                    rawImageHeight=10,
                    renderedCanvasWidth=10,
                    renderedCanvasHeight=10,
                )
            )
        ),
    }

    response = interface_lib.get_visual_overlay_image_as_bytes(
        renderer_key="test_renderer_key_2"
    )
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_RENDERER_FOUND
    )

  def test_reset_visual_overlay_renderer_good_individual(self):

    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )
    interface_lib._visual_overlay = {
        "test_renderer_key_1": interface.visual_overlay.OrchestratorRenderer(
            scene_reference_image_data=(
                interface.robot_job_work_unit.work_unit.SceneReferenceImage(
                    artifactId="test_artifact_id_1",
                    sourceTopic="test_renderer_key_1",
                    rawImageWidth=10,
                    rawImageHeight=10,
                    renderedCanvasWidth=10,
                    renderedCanvasHeight=10,
                )
            )
        ),
        "test_renderer_key_2": interface.visual_overlay.OrchestratorRenderer(
            scene_reference_image_data=(
                interface.robot_job_work_unit.work_unit.SceneReferenceImage(
                    artifactId="test_artifact_id_2",
                    sourceTopic="test_renderer_key_2",
                    rawImageWidth=20,
                    rawImageHeight=20,
                    renderedCanvasWidth=20,
                    renderedCanvasHeight=20,
                )
            )
        ),
    }

    renderer_1: interface.visual_overlay.OrchestratorRenderer = (
        interface_lib._visual_overlay["test_renderer_key_1"]
    )
    renderer_1_init_image_np = renderer_1._overlay_image_np.copy()
    renderer_1._overlay_image = interface.visual_overlay.Image.new(
        mode="RGB", size=(10, 10), color="black",
    )
    renderer_1._overlay_image_np = interface.visual_overlay.np.array(
        renderer_1._overlay_image
    )
    renderer_1._workunit_objects = [
        mock.Mock(spec=interface.robot_job_work_unit.work_unit.SceneObject),
        mock.Mock(spec=interface.robot_job_work_unit.work_unit.SceneObject),
    ]
    renderer_1._overlay_objects = [
        mock.Mock(
            spec=interface.visual_overlay.visual_overlay_icon.DrawCircleIcon
        ),
        mock.Mock(
            spec=interface.visual_overlay.visual_overlay_icon.DrawCircleIcon
        ),
    ]

    renderer_2 = interface_lib._visual_overlay["test_renderer_key_2"]
    renderer_2_init_image_np = renderer_2._overlay_image_np.copy()
    renderer_2._overlay_image = interface.visual_overlay.Image.new(
        mode="RGB", size=(20, 20), color="black",
    )
    renderer_2._overlay_image_np = interface.visual_overlay.np.array(
        renderer_2._overlay_image
    )
    renderer_2._workunit_objects = [
        mock.Mock(spec=interface.robot_job_work_unit.work_unit.SceneObject),
    ]
    renderer_2._overlay_objects = [
        mock.Mock(
            spec=interface.visual_overlay.visual_overlay_icon.DrawCircleIcon
        ),
    ]

    response = interface_lib.reset_visual_overlay_renderer(
        renderer_key="test_renderer_key_1"
    )
    self.assertTrue(response.success)
    self.assertEqual(
        renderer_1._overlay_image_np.shape, renderer_1_init_image_np.shape
    )
    self.assertTrue(
        interface.visual_overlay.np.array_equal(
            renderer_1._overlay_image_np, renderer_1_init_image_np
        )
    )
    self.assertEmpty(renderer_1._workunit_objects)
    self.assertEmpty(renderer_1._overlay_objects)

    self.assertEqual(
        renderer_2._overlay_image_np.shape, renderer_2_init_image_np.shape
    )
    self.assertFalse(
        interface.visual_overlay.np.array_equal(
            renderer_2._overlay_image_np, renderer_2_init_image_np
        )
    )
    self.assertLen(renderer_2._workunit_objects, 1)
    self.assertLen(renderer_2._overlay_objects, 1)

  def test_reset_visual_overlay_renderer_good_all(self):

    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )
    interface_lib._visual_overlay = {
        "test_renderer_key_1": interface.visual_overlay.OrchestratorRenderer(
            scene_reference_image_data=(
                interface.robot_job_work_unit.work_unit.SceneReferenceImage(
                    artifactId="test_artifact_id_1",
                    sourceTopic="test_renderer_key_1",
                    rawImageWidth=10,
                    rawImageHeight=10,
                    renderedCanvasWidth=10,
                    renderedCanvasHeight=10,
                )
            )
        ),
        "test_renderer_key_2": interface.visual_overlay.OrchestratorRenderer(
            scene_reference_image_data=(
                interface.robot_job_work_unit.work_unit.SceneReferenceImage(
                    artifactId="test_artifact_id_2",
                    sourceTopic="test_renderer_key_2",
                    rawImageWidth=20,
                    rawImageHeight=20,
                    renderedCanvasWidth=20,
                    renderedCanvasHeight=20,
                )
            )
        ),
    }

    renderer_1: interface.visual_overlay.OrchestratorRenderer = (
        interface_lib._visual_overlay["test_renderer_key_1"]
    )
    renderer_1_init_image_np = renderer_1._overlay_image_np.copy()
    renderer_1._overlay_image = interface.visual_overlay.Image.new(
        mode="RGB", size=(10, 10), color="black",
    )
    renderer_1._overlay_image_np = interface.visual_overlay.np.array(
        renderer_1._overlay_image
    )
    renderer_1._workunit_objects = [
        mock.Mock(spec=interface.robot_job_work_unit.work_unit.SceneObject),
        mock.Mock(spec=interface.robot_job_work_unit.work_unit.SceneObject),
    ]
    renderer_1._overlay_objects = [
        mock.Mock(
            spec=interface.visual_overlay.visual_overlay_icon.DrawCircleIcon
        ),
        mock.Mock(
            spec=interface.visual_overlay.visual_overlay_icon.DrawCircleIcon
        ),
    ]

    renderer_2 = interface_lib._visual_overlay["test_renderer_key_2"]
    renderer_2_init_image_np = renderer_2._overlay_image_np.copy()
    renderer_2._overlay_image = interface.visual_overlay.Image.new(
        mode="RGB", size=(20, 20), color="black",
    )
    renderer_2._overlay_image_np = interface.visual_overlay.np.array(
        renderer_2._overlay_image
    )
    renderer_2._workunit_objects = [
        mock.Mock(spec=interface.robot_job_work_unit.work_unit.SceneObject),
    ]
    renderer_2._overlay_objects = [
        mock.Mock(
            spec=interface.visual_overlay.visual_overlay_icon.DrawCircleIcon
        ),
    ]

    response = interface_lib.reset_visual_overlay_renderer(
        renderer_key="", reset_all_renderers=True,
    )
    self.assertTrue(response.success)
    self.assertEqual(
        renderer_1._overlay_image_np.shape, renderer_1_init_image_np.shape
    )
    self.assertTrue(
        interface.visual_overlay.np.array_equal(
            renderer_1._overlay_image_np, renderer_1_init_image_np
        )
    )
    self.assertEmpty(renderer_1._workunit_objects)
    self.assertEmpty(renderer_1._overlay_objects)

    self.assertEqual(
        renderer_2._overlay_image_np.shape, renderer_2_init_image_np.shape
    )
    self.assertTrue(
        interface.visual_overlay.np.array_equal(
            renderer_2._overlay_image_np, renderer_2_init_image_np
        )
    )
    self.assertEmpty(renderer_2._workunit_objects)
    self.assertEmpty(renderer_2._overlay_objects)

  def test_reset_visual_overlay_renderer_bad_renderer_key(self):

    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )
    interface_lib._visual_overlay = {
        "test_renderer_key_1": interface.visual_overlay.OrchestratorRenderer(
            scene_reference_image_data=(
                interface.robot_job_work_unit.work_unit.SceneReferenceImage(
                    artifactId="test_artifact_id_1",
                    sourceTopic="test_renderer_key_1",
                    rawImageWidth=10,
                    rawImageHeight=10,
                    renderedCanvasWidth=10,
                    renderedCanvasHeight=10,
                )
            )
        ),
    }
    response = interface_lib.reset_visual_overlay_renderer(
        renderer_key="test_renderer_key_2"
    )
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_RENDERER_FOUND
    )

  def test_create_single_visual_overlay_renderer_good(self):

    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )
    self.assertEmpty(interface_lib._visual_overlay)

    response = interface_lib.create_single_visual_overlay_renderer(
        renderer_key="test_renderer_key_1",
        image_pixel_width=10,
        image_pixel_height=10,
    )
    self.assertTrue(response.success)
    self.assertLen(interface_lib._visual_overlay, 1)
    self.assertIn("test_renderer_key_1", interface_lib._visual_overlay)
    renderer_1 = interface_lib._visual_overlay["test_renderer_key_1"]
    self.assertEqual(renderer_1._overlay_image_np.shape, (10, 10, 3))

    response = interface_lib.create_single_visual_overlay_renderer(
        renderer_key="test_renderer_key_2",
        image_pixel_width=20,
        image_pixel_height=20,
    )
    self.assertTrue(response.success)
    self.assertLen(interface_lib._visual_overlay, 2)
    self.assertIn("test_renderer_key_1", interface_lib._visual_overlay)
    renderer_1 = interface_lib._visual_overlay["test_renderer_key_1"]
    self.assertEqual(renderer_1._overlay_image_np.shape, (10, 10, 3))
    self.assertIn("test_renderer_key_2", interface_lib._visual_overlay)
    renderer_2 = interface_lib._visual_overlay["test_renderer_key_2"]
    self.assertEqual(renderer_2._overlay_image_np.shape, (20, 20, 3))

  def test_create_single_visual_overlay_renderer_bad_renderer_key(self):

    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )
    self.assertEmpty(interface_lib._visual_overlay)

    response = interface_lib.create_single_visual_overlay_renderer(
        renderer_key="test_renderer_key_1",
        image_pixel_width=10,
        image_pixel_height=10,
    )
    self.assertTrue(response.success)
    self.assertLen(interface_lib._visual_overlay, 1)
    self.assertIn("test_renderer_key_1", interface_lib._visual_overlay)

    response = interface_lib.create_single_visual_overlay_renderer(
        renderer_key="test_renderer_key_1",
        image_pixel_width=40,
        image_pixel_height=40,
    )
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_RENDERER_ALREADY_EXISTS
    )
    self.assertLen(interface_lib._visual_overlay, 1)
    self.assertIn("test_renderer_key_1", interface_lib._visual_overlay)

  def test_add_single_overlay_object_to_visual_overlay_good(self):

    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )
    self.assertEmpty(interface_lib._visual_overlay)

    response = interface_lib.create_single_visual_overlay_renderer(
        renderer_key="test_renderer_key_1",
        image_pixel_width=10,
        image_pixel_height=10,
    )
    self.assertTrue(response.success)
    self.assertLen(interface_lib._visual_overlay, 1)
    self.assertIn("test_renderer_key_1", interface_lib._visual_overlay)
    renderer_1 = interface_lib._visual_overlay["test_renderer_key_1"]
    self.assertEqual(renderer_1._overlay_image_np.shape, (10, 10, 3))
    self.assertEmpty(renderer_1._workunit_objects)
    self.assertEmpty(renderer_1._overlay_objects)

    response = interface_lib.add_single_overlay_object_to_visual_overlay(
        renderer_key="test_renderer_key_1",
        overlay_object=mock.Mock(
            spec=interface.visual_overlay.visual_overlay_icon.DrawCircleIcon
        ),
    )
    self.assertTrue(response.success)
    self.assertEmpty(renderer_1._workunit_objects)
    self.assertLen(renderer_1._overlay_objects, 1)

    response = interface_lib.add_single_overlay_object_to_visual_overlay(
        renderer_key="test_renderer_key_1",
        overlay_object=mock.Mock(
            spec=interface.visual_overlay.visual_overlay_icon.DrawContainer
        ),
    )
    self.assertTrue(response.success)
    self.assertEmpty(renderer_1._workunit_objects)
    self.assertLen(renderer_1._overlay_objects, 2)

  def test_add_single_overlay_object_to_visual_overlay_bad_renderer_key(self):

    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )
    self.assertEmpty(interface_lib._visual_overlay)

    response = interface_lib.create_single_visual_overlay_renderer(
        renderer_key="test_renderer_key_1",
        image_pixel_width=10,
        image_pixel_height=10,
    )
    self.assertTrue(response.success)
    self.assertLen(interface_lib._visual_overlay, 1)
    self.assertIn("test_renderer_key_1", interface_lib._visual_overlay)

    response = interface_lib.add_single_overlay_object_to_visual_overlay(
        renderer_key="test_renderer_key_2",
        overlay_object=mock.Mock(
            spec=interface.visual_overlay.visual_overlay_icon.DrawCircleIcon
        ),
    )
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_RENDERER_FOUND
    )
    renderer_1 = interface_lib._visual_overlay["test_renderer_key_1"]
    self.assertEmpty(renderer_1._workunit_objects)
    self.assertEmpty(renderer_1._overlay_objects)

  def test_render_visual_overlay_good(self):

    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )
    self.assertEmpty(interface_lib._visual_overlay)

    response = interface_lib.create_single_visual_overlay_renderer(
        renderer_key="test_renderer_key_1",
        image_pixel_width=50,
        image_pixel_height=50,
    )
    self.assertTrue(response.success)
    self.assertLen(interface_lib._visual_overlay, 1)
    self.assertIn("test_renderer_key_1", interface_lib._visual_overlay)

    renderer_1 = interface_lib._visual_overlay["test_renderer_key_1"]
    init_image_np = renderer_1._overlay_image_np.copy()
    self.assertEqual(renderer_1._overlay_image_np.shape, (50, 50, 3))
    self.assertEmpty(renderer_1._workunit_objects)
    self.assertEmpty(renderer_1._overlay_objects)

    response = interface_lib.add_single_overlay_object_to_visual_overlay(
        renderer_key="test_renderer_key_1",
        overlay_object=(
            interface.visual_overlay.visual_overlay_icon.DrawCircleIcon(
                object_id="test_object_id_1",
                overlay_text_label="test_overlay_text_label_1",
                rgb_hex_color_value="FF0000",
                layer_order=1,
                x=25,
                y=25,
            )
        ),
    )
    self.assertTrue(response.success)
    self.assertEmpty(renderer_1._workunit_objects)
    self.assertLen(renderer_1._overlay_objects, 1)

    response = interface_lib.render_visual_overlay(
        renderer_key="test_renderer_key_1"
    )
    self.assertTrue(response.success)
    self.assertFalse(
        interface.visual_overlay.np.array_equal(
            init_image_np, renderer_1._overlay_image_np
        )
    )

  def test_render_visual_overlay_bad_renderer_key(self):

    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )
    self.assertEmpty(interface_lib._visual_overlay)

    response = interface_lib.create_single_visual_overlay_renderer(
        renderer_key="test_renderer_key_1",
        image_pixel_width=10,
        image_pixel_height=10,
    )
    self.assertTrue(response.success)
    self.assertLen(interface_lib._visual_overlay, 1)
    self.assertIn("test_renderer_key_1", interface_lib._visual_overlay)

    response = interface_lib.render_visual_overlay(
        renderer_key="test_renderer_key_2",
    )
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_RENDERER_FOUND
    )

    response = interface_lib.render_visual_overlay(
        renderer_key="test_renderer_key_2",
        new_image=mock.Mock(spec=interface.visual_overlay.Image.Image),
    )
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_RENDERER_FOUND
    )

  @mock.patch(
      "safari_sdk.orchestrator.client.libs.robot_job.OrchestratorRobotJob.request_robot_job",
      return_value=interface.robot_job_work_unit._RESPONSE(
          success=True,
          robot_id="test_robot_id",
          robot_job_id="test_robot_job_id",
      ),
  )
  @mock.patch(
      "safari_sdk.orchestrator.client.libs.robot_job_work_unit.OrchestratorRobotJobWorkUnit.start_work_unit_software_asset_prep",
      return_value=interface.robot_job_work_unit._RESPONSE(
          success=True,
          robot_id="test_robot_id",
          robot_job_id="test_robot_job_id",
          work_unit_id="test_work_unit_id",
          work_unit=interface.robot_job_work_unit.work_unit.WorkUnit(
              robotJobId="test_robot_job_id",
              workUnitId="test_work_unit_id",
          ),
      ),
  )
  def test_robot_job_work_unit_start_software_asset_prep_good(self, *_):

    FLAGS.api_key = "mock_test_key"
    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )

    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_build.return_value = mock.Mock(
          spec=interface.auth.discovery.Resource
      )
      response = interface_lib.connect()
      self.assertTrue(response.success)

    response = interface_lib.robot_job_work_unit_start_software_asset_prep()
    self.assertTrue(response.success)
    self.assertEqual(response.robot_id, "test_robot_id")
    self.assertEqual(response.robot_job_id, "test_robot_job_id")
    self.assertEqual(response.work_unit_id, "test_work_unit_id")
    self.assertIsInstance(
        response.work_unit, interface.robot_job_work_unit.work_unit.WorkUnit
    )

  def test_robot_job_work_unit_start_software_asset_prep_bad_active_connection(
      self,
  ):

    FLAGS.api_key = None
    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )
    interface_lib._connection = None
    interface_lib._robot_job_lib = mock.MagicMock(
        spec=interface.robot_job.OrchestratorRobotJob
    )
    interface_lib._robot_job_work_unit_lib = mock.MagicMock(
        spec=interface.robot_job_work_unit.OrchestratorRobotJobWorkUnit
    )

    response = interface_lib.robot_job_work_unit_start_software_asset_prep()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_ACTIVE_CONNECTION
    )

    interface_lib._connection = mock.Mock(
        spec=interface.auth.discovery.Resource
    )
    interface_lib._robot_job_lib = None
    interface_lib._robot_job_work_unit_lib = mock.MagicMock(
        spec=interface.robot_job_work_unit.OrchestratorRobotJobWorkUnit
    )

    response = interface_lib.robot_job_work_unit_start_software_asset_prep()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_ACTIVE_CONNECTION
    )

    interface_lib._connection = mock.Mock(
        spec=interface.auth.discovery.Resource
    )
    interface_lib._robot_job_lib = mock.MagicMock(
        spec=interface.robot_job.OrchestratorRobotJob
    )
    interface_lib._robot_job_work_unit_lib = None

    response = interface_lib.robot_job_work_unit_start_software_asset_prep()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_ACTIVE_CONNECTION
    )

  @mock.patch(
      "safari_sdk.orchestrator.client.libs.robot_job.OrchestratorRobotJob.request_robot_job",
      return_value=interface.robot_job_work_unit._RESPONSE(
          success=True,
          robot_id="test_robot_id",
          robot_job_id="test_robot_job_id",
      ),
  )
  @mock.patch(
      "safari_sdk.orchestrator.client.libs.robot_job_work_unit.OrchestratorRobotJobWorkUnit.start_work_unit_scene_prep",
      return_value=interface.robot_job_work_unit._RESPONSE(
          success=True,
          robot_id="test_robot_id",
          robot_job_id="test_robot_job_id",
          work_unit_id="test_work_unit_id",
          work_unit=interface.robot_job_work_unit.work_unit.WorkUnit(
              robotJobId="test_robot_job_id",
              workUnitId="test_work_unit_id",
          ),
      ),
  )
  def test_robot_job_work_unit_start_scene_prep_good(self, *_):

    FLAGS.api_key = "mock_test_key"
    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )

    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_build.return_value = mock.Mock(
          spec=interface.auth.discovery.Resource
      )
      response = interface_lib.connect()
      self.assertTrue(response.success)

    response = interface_lib.robot_job_work_unit_start_scene_prep()
    self.assertTrue(response.success)
    self.assertEqual(response.robot_id, "test_robot_id")
    self.assertEqual(response.robot_job_id, "test_robot_job_id")
    self.assertEqual(response.work_unit_id, "test_work_unit_id")
    self.assertIsInstance(
        response.work_unit, interface.robot_job_work_unit.work_unit.WorkUnit
    )

  def test_robot_job_work_unit_start_scene_prep_bad_active_connection(self):

    FLAGS.api_key = None
    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )
    interface_lib._connection = None
    interface_lib._robot_job_lib = mock.MagicMock(
        spec=interface.robot_job.OrchestratorRobotJob
    )
    interface_lib._robot_job_work_unit_lib = mock.MagicMock(
        spec=interface.robot_job_work_unit.OrchestratorRobotJobWorkUnit
    )

    response = interface_lib.robot_job_work_unit_start_scene_prep()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_ACTIVE_CONNECTION
    )

    interface_lib._connection = mock.Mock(
        spec=interface.auth.discovery.Resource
    )
    interface_lib._robot_job_lib = None
    interface_lib._robot_job_work_unit_lib = mock.MagicMock(
        spec=interface.robot_job_work_unit.OrchestratorRobotJobWorkUnit
    )

    response = interface_lib.robot_job_work_unit_start_scene_prep()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_ACTIVE_CONNECTION
    )

    interface_lib._connection = mock.Mock(
        spec=interface.auth.discovery.Resource
    )
    interface_lib._robot_job_lib = mock.MagicMock(
        spec=interface.robot_job.OrchestratorRobotJob
    )
    interface_lib._robot_job_work_unit_lib = None

    response = interface_lib.robot_job_work_unit_start_scene_prep()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_ACTIVE_CONNECTION
    )

  @mock.patch(
      "safari_sdk.orchestrator.client.libs.robot_job.OrchestratorRobotJob.request_robot_job",
      return_value=interface.robot_job_work_unit._RESPONSE(
          success=True,
          robot_id="test_robot_id",
          robot_job_id="test_robot_job_id",
      ),
  )
  @mock.patch(
      "safari_sdk.orchestrator.client.libs.robot_job_work_unit.OrchestratorRobotJobWorkUnit.start_work_unit_execution",
      return_value=interface.robot_job_work_unit._RESPONSE(
          success=True,
          robot_id="test_robot_id",
          robot_job_id="test_robot_job_id",
          work_unit_id="test_work_unit_id",
          work_unit=interface.robot_job_work_unit.work_unit.WorkUnit(
              robotJobId="test_robot_job_id",
              workUnitId="test_work_unit_id",
          ),
      ),
  )
  def test_robot_job_work_unit_start_execution_good(self, *_):

    FLAGS.api_key = "mock_test_key"
    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )

    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_build.return_value = mock.Mock(
          spec=interface.auth.discovery.Resource
      )
      response = interface_lib.connect()
      self.assertTrue(response.success)

    response = interface_lib.robot_job_work_unit_start_execution()
    self.assertTrue(response.success)
    self.assertEqual(response.robot_id, "test_robot_id")
    self.assertEqual(response.robot_job_id, "test_robot_job_id")
    self.assertEqual(response.work_unit_id, "test_work_unit_id")
    self.assertIsInstance(
        response.work_unit, interface.robot_job_work_unit.work_unit.WorkUnit
    )

  def test_robot_job_work_unit_start_execution_bad_active_connection(self):

    FLAGS.api_key = None
    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )
    interface_lib._connection = None
    interface_lib._robot_job_lib = mock.MagicMock(
        spec=interface.robot_job.OrchestratorRobotJob
    )
    interface_lib._robot_job_work_unit_lib = mock.MagicMock(
        spec=interface.robot_job_work_unit.OrchestratorRobotJobWorkUnit
    )

    response = interface_lib.robot_job_work_unit_start_execution()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_ACTIVE_CONNECTION
    )

    interface_lib._connection = mock.Mock(
        spec=interface.auth.discovery.Resource
    )
    interface_lib._robot_job_lib = None
    interface_lib._robot_job_work_unit_lib = mock.MagicMock(
        spec=interface.robot_job_work_unit.OrchestratorRobotJobWorkUnit
    )

    response = interface_lib.robot_job_work_unit_start_execution()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_ACTIVE_CONNECTION
    )

    interface_lib._connection = mock.Mock(
        spec=interface.auth.discovery.Resource
    )
    interface_lib._robot_job_lib = mock.MagicMock(
        spec=interface.robot_job.OrchestratorRobotJob
    )
    interface_lib._robot_job_work_unit_lib = None

    response = interface_lib.robot_job_work_unit_start_execution()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_ACTIVE_CONNECTION
    )

  @mock.patch(
      "safari_sdk.orchestrator.client.libs.robot_job.OrchestratorRobotJob.request_robot_job",
      return_value=interface.robot_job_work_unit._RESPONSE(
          success=True,
          robot_id="test_robot_id",
          robot_job_id="test_robot_job_id",
      ),
  )
  @mock.patch(
      "safari_sdk.orchestrator.client.libs.robot_job_work_unit.OrchestratorRobotJobWorkUnit.complete_work_unit",
      return_value=interface.robot_job_work_unit._RESPONSE(
          success=True,
          robot_id="test_robot_id",
          robot_job_id="test_robot_job_id",
          work_unit_id="test_work_unit_id",
          work_unit=interface.robot_job_work_unit.work_unit.WorkUnit(
              robotJobId="test_robot_job_id",
              workUnitId="test_work_unit_id",
          ),
      ),
  )
  def test_robot_job_work_unit_complete_work_unit_good(self, *_):

    FLAGS.api_key = "mock_test_key"
    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )

    with mock.patch("googleapiclient.discovery.build") as mock_build:
      mock_build.return_value = mock.Mock(
          spec=interface.auth.discovery.Resource
      )
      response = interface_lib.connect()
      self.assertTrue(response.success)

    response = interface_lib.robot_job_work_unit_complete_work_unit(
        outcome=interface.WORK_UNIT_OUTCOME.WORK_UNIT_OUTCOME_SUCCESS,
        note="test_note",
    )
    self.assertTrue(response.success)
    self.assertEqual(response.robot_id, "test_robot_id")
    self.assertEqual(response.robot_job_id, "test_robot_job_id")
    self.assertEqual(response.work_unit_id, "test_work_unit_id")
    self.assertIsInstance(
        response.work_unit, interface.robot_job_work_unit.work_unit.WorkUnit
    )

  def test_robot_job_work_unit_complete_work_unit_bad_active_connection(self):

    FLAGS.api_key = None
    interface_lib = interface.OrchestratorInterface(
        robot_id="test_robot_id",
        job_type=interface.JOB_TYPE.ALL,
    )
    interface_lib._connection = None
    interface_lib._robot_job_lib = mock.MagicMock(
        spec=interface.robot_job.OrchestratorRobotJob
    )
    interface_lib._robot_job_work_unit_lib = mock.MagicMock(
        spec=interface.robot_job_work_unit.OrchestratorRobotJobWorkUnit
    )

    response = interface_lib.robot_job_work_unit_complete_work_unit(
        outcome=interface.WORK_UNIT_OUTCOME.WORK_UNIT_OUTCOME_SUCCESS,
        note="test_note",
    )
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_ACTIVE_CONNECTION
    )

    interface_lib._connection = mock.Mock(
        spec=interface.auth.discovery.Resource
    )
    interface_lib._robot_job_lib = None
    interface_lib._robot_job_work_unit_lib = mock.MagicMock(
        spec=interface.robot_job_work_unit.OrchestratorRobotJobWorkUnit
    )

    response = interface_lib.robot_job_work_unit_complete_work_unit(
        outcome=interface.WORK_UNIT_OUTCOME.WORK_UNIT_OUTCOME_SUCCESS,
        note="test_note",
    )
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_ACTIVE_CONNECTION
    )

    interface_lib._connection = mock.Mock(
        spec=interface.auth.discovery.Resource
    )
    interface_lib._robot_job_lib = mock.MagicMock(
        spec=interface.robot_job.OrchestratorRobotJob
    )
    interface_lib._robot_job_work_unit_lib = None

    response = interface_lib.robot_job_work_unit_complete_work_unit(
        outcome=interface.WORK_UNIT_OUTCOME.WORK_UNIT_OUTCOME_SUCCESS,
        note="test_note",
    )
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, interface._ERROR_NO_ACTIVE_CONNECTION
    )


if __name__ == "__main__":
  absltest.main()
