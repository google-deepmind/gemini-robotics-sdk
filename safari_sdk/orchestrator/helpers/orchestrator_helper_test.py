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

"""Unit tests for orchestrator_helper.py."""

from unittest import mock

from absl.testing import absltest

from safari_sdk.orchestrator.helpers import orchestrator_helper


class OrchestratorHelperTest(absltest.TestCase):

  @mock.patch(
      "safari_sdk.orchestrator.client.interface.OrchestratorInterface.connect",
      return_value=orchestrator_helper.interface.RESPONSE(success=True),
  )
  def test_connect_good(self, _):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.connect()
    self.assertTrue(response.success)

  @mock.patch(
      "safari_sdk.orchestrator.client.interface.OrchestratorInterface.connect",
      return_value=orchestrator_helper.interface.RESPONSE(
          success=False, error_message="Error from interface.connect()"
      ),
  )
  def test_connect_bad_without_raise_error(self, _):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="",
        job_type=orchestrator_helper.JOB_TYPE.COLLECTION,
    )

    response = helper_lib.connect()
    self.assertFalse(response.success)
    self.assertEqual(response.error_message, "Error from interface.connect()")

  @mock.patch(
      "safari_sdk.orchestrator.client.interface.OrchestratorInterface.connect",
      return_value=orchestrator_helper.interface.RESPONSE(
          success=False, error_message="Error from interface.connect()"
      ),
  )
  def test_connect_bad_with_raise_error(self, _):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="",
        job_type=orchestrator_helper.JOB_TYPE.EVALUATION,
        raise_error=True,
    )

    with self.assertRaises(ValueError):
      helper_lib.connect()

  def test_get_current_connection_good(self):
    mock_interface = mock.create_autospec(
        spec=orchestrator_helper.interface.OrchestratorInterface, instance=True
    )
    mock_interface.get_current_connection.return_value = (
        orchestrator_helper.interface.RESPONSE(
            success=True,
            server_connection=mock.MagicMock(),
            robot_id="test_robot_id",
        )
    )
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )
    helper_lib._interface = mock_interface

    response = helper_lib.get_current_connection()
    self.assertTrue(response.success)
    self.assertIsNotNone(response.server_connection)
    self.assertEqual(response.robot_id, "test_robot_id")

  def test_get_current_connection_bad_without_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.get_current_connection()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, orchestrator_helper._ERROR_NO_ACTIVE_CONNECTION
    )

  def test_get_current_connection_bad_with_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="",
        raise_error=True,
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    with self.assertRaises(ValueError):
      helper_lib.get_current_connection()

  def test_get_current_robot_info_good(self):
    mock_interface = mock.create_autospec(
        spec=orchestrator_helper.interface.OrchestratorInterface, instance=True
    )
    mock_interface.get_current_robot_info.return_value = (
        orchestrator_helper.interface.RESPONSE(
            success=True,
            robot_id="test_robot_id",
            is_operational=True,
            operator_id="test_operator_id"
        )
    )
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )
    helper_lib._interface = mock_interface

    response = helper_lib.get_current_robot_info()
    self.assertTrue(response.success)
    self.assertEqual(response.robot_id, "test_robot_id")
    self.assertTrue(response.is_operational)
    self.assertEqual(response.operator_id, "test_operator_id")

  def test_get_current_robot_info_bad_without_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.get_current_robot_info()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, orchestrator_helper._ERROR_NO_ACTIVE_CONNECTION
    )

  def test_get_current_robot_info_bad_with_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
        raise_error=True,
    )

    with self.assertRaises(ValueError):
      helper_lib.get_current_robot_info()

  def test_set_current_robot_operator_id_good(self):
    mock_interface = mock.create_autospec(
        spec=orchestrator_helper.interface.OrchestratorInterface, instance=True
    )
    mock_interface.set_current_robot_operator_id.return_value = (
        orchestrator_helper.interface.RESPONSE(
            success=True,
            robot_id="test_robot_id",
            operator_id="test_operator_id"
        )
    )
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )
    helper_lib._interface = mock_interface

    response = helper_lib.set_current_robot_operator_id(
        operator_id="test_operator_id"
    )
    self.assertTrue(response.success)
    self.assertEqual(response.robot_id, "test_robot_id")
    self.assertEqual(response.operator_id, "test_operator_id")

  def test_set_current_robot_operator_id_bad_without_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.set_current_robot_operator_id(
        operator_id="test_operator_id"
    )
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, orchestrator_helper._ERROR_NO_ACTIVE_CONNECTION
    )

  def test_set_current_robot_operator_id_bad_with_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
        raise_error=True,
    )

    with self.assertRaises(ValueError):
      helper_lib.set_current_robot_operator_id(operator_id="test_operator_id")

  def test_request_work_unit_good(self):
    mock_interface = mock.create_autospec(
        spec=orchestrator_helper.interface.OrchestratorInterface, instance=True
    )
    mock_interface.request_robot_job_work_unit.return_value = (
        orchestrator_helper.interface.RESPONSE(success=True)
    )
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )
    helper_lib._interface = mock_interface

    response = helper_lib.request_work_unit()
    self.assertTrue(response.success)

  def test_request_work_unit_bad_without_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.request_work_unit()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, orchestrator_helper._ERROR_NO_ACTIVE_CONNECTION
    )

  def test_request_work_unit_bad_with_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
        raise_error=True,
    )

    with self.assertRaises(ValueError):
      helper_lib.request_work_unit()

  def test_get_current_work_unit_good(self):
    mock_interface = mock.create_autospec(
        spec=orchestrator_helper.interface.OrchestratorInterface, instance=True
    )
    mock_interface.get_current_robot_job_work_unit.return_value = (
        orchestrator_helper.interface.RESPONSE(success=True)
    )
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )
    helper_lib._interface = mock_interface

    response = helper_lib.get_current_work_unit()
    self.assertTrue(response.success)

  def test_get_current_work_unit_bad_without_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.get_current_work_unit()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, orchestrator_helper._ERROR_NO_ACTIVE_CONNECTION
    )

  def test_get_current_work_unit_bad_with_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
        raise_error=True,
    )

    with self.assertRaises(ValueError):
      helper_lib.get_current_work_unit()

  def test_is_visual_overlay_in_current_work_unit_good(self):
    mock_interface = mock.create_autospec(
        spec=orchestrator_helper.interface.OrchestratorInterface, instance=True
    )
    mock_interface.is_visual_overlay_in_current_work_unit.return_value = (
        orchestrator_helper.interface.RESPONSE(
            success=True, is_visual_overlay_found=True
        )
    )
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )
    helper_lib._interface = mock_interface

    response = helper_lib.is_visual_overlay_in_current_work_unit()
    self.assertTrue(response.success)
    self.assertTrue(response.is_visual_overlay_found)

    mock_interface.is_visual_overlay_in_current_work_unit.return_value = (
        orchestrator_helper.interface.RESPONSE(
            success=True, is_visual_overlay_found=False
        )
    )

    response = helper_lib.is_visual_overlay_in_current_work_unit()
    self.assertTrue(response.success)
    self.assertFalse(response.is_visual_overlay_found)

  def test_is_visual_overlay_in_current_work_unit_bad_without_raise_error(
      self
  ):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.is_visual_overlay_in_current_work_unit()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, orchestrator_helper._ERROR_NO_ACTIVE_CONNECTION
    )

  def test_is_visual_overlay_in_current_work_unit_bad_with_raise_error(
      self
  ):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
        raise_error=True,
    )

    with self.assertRaises(ValueError):
      helper_lib.is_visual_overlay_in_current_work_unit()

  def test_create_visual_overlays_for_current_work_unit_good(self):
    mock_interface = mock.create_autospec(
        spec=orchestrator_helper.interface.OrchestratorInterface, instance=True
    )
    mock_interface.create_visual_overlays_for_current_work_unit.return_value = (
        orchestrator_helper.interface.RESPONSE(success=True)
    )
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )
    helper_lib._interface = mock_interface

    response = helper_lib.create_visual_overlays_for_current_work_unit()
    self.assertTrue(response.success)

  def test_create_visual_overlays_for_current_work_unit_bad_without_raise_error(
      self
  ):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.create_visual_overlays_for_current_work_unit()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, orchestrator_helper._ERROR_NO_ACTIVE_CONNECTION
    )

  def test_create_visual_overlays_for_current_work_unit_bad_with_raise_error(
      self
  ):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
        raise_error=True,
    )

    with self.assertRaises(ValueError):
      helper_lib.create_visual_overlays_for_current_work_unit()

  def test_list_visual_overlay_renderer_keys_good(self):
    mock_interface = mock.create_autospec(
        spec=orchestrator_helper.interface.OrchestratorInterface, instance=True
    )
    mock_interface.list_visual_overlay_renderer_keys.return_value = (
        orchestrator_helper.interface.RESPONSE(
            success=True,
            visual_overlay_renderer_keys=["renderer_1", "renderer_2"]
        )
    )
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )
    helper_lib._interface = mock_interface

    response = helper_lib.list_visual_overlay_renderer_keys()
    self.assertTrue(response.success)
    self.assertLen(response.visual_overlay_renderer_keys, 2)
    self.assertSameElements(
        response.visual_overlay_renderer_keys, ["renderer_1", "renderer_2"]
    )

  def test_list_visual_overlay_renderer_keys_bad_without_raise_error(
      self
  ):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.list_visual_overlay_renderer_keys()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, orchestrator_helper._ERROR_NO_ACTIVE_CONNECTION
    )

  def test_list_visual_overlay_renderer_keys_bad_with_raise_error(
      self
  ):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
        raise_error=True,
    )

    with self.assertRaises(ValueError):
      helper_lib.list_visual_overlay_renderer_keys()

  def test_render_visual_overlay_good(self):
    mock_interface = mock.create_autospec(
        spec=orchestrator_helper.interface.OrchestratorInterface, instance=True
    )
    mock_interface.render_visual_overlay.return_value = (
        orchestrator_helper.interface.RESPONSE(success=True)
    )
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )
    helper_lib._interface = mock_interface

    response = helper_lib.render_visual_overlay(renderer_key="renderer_1")
    self.assertTrue(response.success)

  def test_render_visual_overlay_bad_without_raise_error(
      self
  ):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.render_visual_overlay(renderer_key="renderer_1")
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, orchestrator_helper._ERROR_NO_ACTIVE_CONNECTION
    )

  def test_render_visual_overlay_bad_with_raise_error(
      self
  ):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
        raise_error=True,
    )

    with self.assertRaises(ValueError):
      helper_lib.render_visual_overlay(renderer_key="renderer_1")

  def test_get_visual_overlay_image_as_pil_image_good(self):
    mock_interface = mock.create_autospec(
        spec=orchestrator_helper.interface.OrchestratorInterface, instance=True
    )
    mock_interface.get_visual_overlay_image_as_pil_image.return_value = (
        orchestrator_helper.interface.RESPONSE(
            success=True,
            visual_overlay_image=mock.MagicMock(
                spec=orchestrator_helper.interface.api_response.Image.Image
            ),
        )
    )
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )
    helper_lib._interface = mock_interface

    response = helper_lib.get_visual_overlay_image_as_pil_image(
        renderer_key="renderer_1"
    )
    self.assertTrue(response.success)
    self.assertIsInstance(
        response.visual_overlay_image,
        orchestrator_helper.interface.api_response.Image.Image,
    )

  def test_get_visual_overlay_image_as_pil_image_bad_without_raise_error(
      self
  ):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.get_visual_overlay_image_as_pil_image(
        renderer_key="renderer_1"
    )
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, orchestrator_helper._ERROR_NO_ACTIVE_CONNECTION
    )

  def test_get_visual_overlay_image_as_pil_image_bad_with_raise_error(
      self
  ):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
        raise_error=True,
    )

    with self.assertRaises(ValueError):
      helper_lib.get_visual_overlay_image_as_pil_image(
          renderer_key="renderer_1"
      )

  def test_get_visual_overlay_image_as_np_array_good(self):
    mock_interface = mock.create_autospec(
        spec=orchestrator_helper.interface.OrchestratorInterface, instance=True
    )
    mock_interface.get_visual_overlay_image_as_np_array.return_value = (
        orchestrator_helper.interface.RESPONSE(
            success=True,
            visual_overlay_image=mock.MagicMock(
                spec=orchestrator_helper.interface.api_response.np.ndarray
            ),
        )
    )
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )
    helper_lib._interface = mock_interface

    response = helper_lib.get_visual_overlay_image_as_np_array(
        renderer_key="renderer_1"
    )
    self.assertTrue(response.success)
    self.assertIsInstance(
        response.visual_overlay_image,
        orchestrator_helper.interface.api_response.np.ndarray,
    )

  def test_get_visual_overlay_image_as_np_array_bad_without_raise_error(
      self
  ):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.get_visual_overlay_image_as_np_array(
        renderer_key="renderer_1"
    )
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, orchestrator_helper._ERROR_NO_ACTIVE_CONNECTION
    )

  def test_get_visual_overlay_image_as_np_array_bad_with_raise_error(
      self
  ):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
        raise_error=True,
    )

    with self.assertRaises(ValueError):
      helper_lib.get_visual_overlay_image_as_np_array(
          renderer_key="renderer_1"
      )

  def test_get_visual_overlay_image_as_bytes_good(self):
    mock_interface = mock.create_autospec(
        spec=orchestrator_helper.interface.OrchestratorInterface, instance=True
    )
    mock_interface.get_visual_overlay_image_as_bytes.return_value = (
        orchestrator_helper.interface.RESPONSE(
            success=True,
            visual_overlay_image=mock.MagicMock(spec=bytes),
        )
    )
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )
    helper_lib._interface = mock_interface

    response = helper_lib.get_visual_overlay_image_as_bytes(
        renderer_key="renderer_1"
    )
    self.assertTrue(response.success)
    self.assertIsInstance(response.visual_overlay_image, bytes)

  def test_get_visual_overlay_image_as_bytes_bad_without_raise_error(
      self
  ):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.get_visual_overlay_image_as_bytes(
        renderer_key="renderer_1"
    )
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, orchestrator_helper._ERROR_NO_ACTIVE_CONNECTION
    )

  def test_get_visual_overlay_image_as_bytes_bad_with_raise_error(
      self
  ):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
        raise_error=True,
    )

    with self.assertRaises(ValueError):
      helper_lib.get_visual_overlay_image_as_bytes(
          renderer_key="renderer_1"
      )

  def test_reset_visual_overlay_renderer_good(self):
    mock_interface = mock.create_autospec(
        spec=orchestrator_helper.interface.OrchestratorInterface, instance=True
    )
    mock_interface.reset_visual_overlay_renderer.return_value = (
        orchestrator_helper.interface.RESPONSE(success=True)
    )
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )
    helper_lib._interface = mock_interface

    response = helper_lib.reset_visual_overlay_renderer(
        renderer_key="renderer_1"
    )
    self.assertTrue(response.success)

    response = helper_lib.reset_visual_overlay_renderer(
        renderer_key="", reset_all_renderers=True
    )
    self.assertTrue(response.success)

  def test_reset_visual_overlay_renderer_bad_without_raise_error(
      self
  ):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.reset_visual_overlay_renderer(
        renderer_key="renderer_1"
    )
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, orchestrator_helper._ERROR_NO_ACTIVE_CONNECTION
    )

  def test_reset_visual_overlay_renderer_bad_with_raise_error(
      self
  ):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
        raise_error=True,
    )

    with self.assertRaises(ValueError):
      helper_lib.reset_visual_overlay_renderer(
          renderer_key="renderer_1"
      )

  def test_create_single_visual_overlay_renderer_good(self):
    mock_interface = mock.create_autospec(
        spec=orchestrator_helper.interface.OrchestratorInterface, instance=True
    )
    mock_interface.create_single_visual_overlay_renderer.return_value = (
        orchestrator_helper.interface.RESPONSE(success=True)
    )
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )
    helper_lib._interface = mock_interface

    response = helper_lib.create_single_visual_overlay_renderer(
        renderer_key="renderer_1",
        image_pixel_width=10,
        image_pixel_height=10,
        overlay_bg_color="#444444",
    )
    self.assertTrue(response.success)

  def test_create_single_visual_overlay_renderer_bad_without_raise_error(
      self
  ):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.create_single_visual_overlay_renderer(
        renderer_key="renderer_1",
        image_pixel_width=10,
        image_pixel_height=10,
        overlay_bg_color="#444444",
    )
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, orchestrator_helper._ERROR_NO_ACTIVE_CONNECTION
    )

  def test_create_single_visual_overlay_renderer_bad_with_raise_error(
      self
  ):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
        raise_error=True,
    )

    with self.assertRaises(ValueError):
      helper_lib.create_single_visual_overlay_renderer(
          renderer_key="renderer_1",
          image_pixel_width=10,
          image_pixel_height=10,
          overlay_bg_color="#444444",
      )

  def test_add_single_overlay_object_to_visual_overlay_good(self):
    mock_interface = mock.create_autospec(
        spec=orchestrator_helper.interface.OrchestratorInterface, instance=True
    )
    mock_interface.add_single_overlay_object_to_visual_overlay.return_value = (
        orchestrator_helper.interface.RESPONSE(success=True)
    )
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )
    helper_lib._interface = mock_interface

    response = helper_lib.add_single_overlay_object_to_visual_overlay(
        renderer_key="renderer_1",
        overlay_object=orchestrator_helper.DRAW_CIRCLE_ICON(
            object_id="test_object_id_1",
            overlay_text_label="test_overlay_text_label_1",
            rgb_hex_color_value="FF0000",
            layer_order=1,
            x=25,
            y=25,
        ),
    )
    self.assertTrue(response.success)

  def test_add_single_overlay_object_to_visual_overlay_bad_without_raise_error(
      self
  ):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.add_single_overlay_object_to_visual_overlay(
        renderer_key="renderer_1",
        overlay_object=orchestrator_helper.DRAW_CIRCLE_ICON(
            object_id="test_object_id_1",
            overlay_text_label="test_overlay_text_label_1",
            rgb_hex_color_value="FF0000",
            layer_order=1,
            x=25,
            y=25,
        ),
    )
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, orchestrator_helper._ERROR_NO_ACTIVE_CONNECTION
    )

  def test_add_single_overlay_object_to_visual_overlay_bad_with_raise_error(
      self
  ):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
        raise_error=True,
    )

    with self.assertRaises(ValueError):
      helper_lib.add_single_overlay_object_to_visual_overlay(
          renderer_key="renderer_1",
          overlay_object=orchestrator_helper.DRAW_CIRCLE_ICON(
              object_id="test_object_id_1",
              overlay_text_label="test_overlay_text_label_1",
              rgb_hex_color_value="FF0000",
              layer_order=1,
              x=25,
              y=25,
          ),
      )

  def test_start_work_unit_software_asset_prep_good(self):
    mock_interface = mock.create_autospec(
        spec=orchestrator_helper.interface.OrchestratorInterface, instance=True
    )
    mock_interface.robot_job_work_unit_start_software_asset_prep.return_value = orchestrator_helper.interface.RESPONSE(
        success=True
    )
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )
    helper_lib._interface = mock_interface

    response = helper_lib.start_work_unit_software_asset_prep()
    self.assertTrue(response.success)

  def test_start_work_unit_software_asset_prep_bad_without_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.start_work_unit_software_asset_prep()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, orchestrator_helper._ERROR_NO_ACTIVE_CONNECTION
    )

  def test_start_work_unit_software_asset_prep_bad_with_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
        raise_error=True,
    )

    with self.assertRaises(ValueError):
      helper_lib.start_work_unit_software_asset_prep()

  def test_start_work_unit_scene_prep_good(self):
    mock_interface = mock.create_autospec(
        spec=orchestrator_helper.interface.OrchestratorInterface, instance=True
    )
    mock_interface.robot_job_work_unit_start_scene_prep.return_value = (
        orchestrator_helper.interface.RESPONSE(success=True)
    )
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )
    helper_lib._interface = mock_interface

    response = helper_lib.start_work_unit_scene_prep()
    self.assertTrue(response.success)

  def test_start_work_unit_scene_prep_bad_without_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.start_work_unit_scene_prep()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, orchestrator_helper._ERROR_NO_ACTIVE_CONNECTION
    )

  def test_start_work_unit_scene_prep_bad_with_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
        raise_error=True,
    )

    with self.assertRaises(ValueError):
      helper_lib.start_work_unit_scene_prep()

  def test_start_work_unit_execution_good(self):
    mock_interface = mock.create_autospec(
        spec=orchestrator_helper.interface.OrchestratorInterface, instance=True
    )
    mock_interface.robot_job_work_unit_start_execution.return_value = (
        orchestrator_helper.interface.RESPONSE(success=True)
    )
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )
    helper_lib._interface = mock_interface

    response = helper_lib.start_work_unit_execution()
    self.assertTrue(response.success)

  def test_start_work_unit_execution_bad_without_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.start_work_unit_execution()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, orchestrator_helper._ERROR_NO_ACTIVE_CONNECTION
    )

  def test_start_work_unit_execution_bad_with_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
        raise_error=True,
    )

    with self.assertRaises(ValueError):
      helper_lib.start_work_unit_execution()

  def test_complete_work_unit_good(self):
    mock_interface = mock.create_autospec(
        spec=orchestrator_helper.interface.OrchestratorInterface, instance=True
    )
    mock_interface.robot_job_work_unit_complete_work_unit.return_value = (
        orchestrator_helper.interface.RESPONSE(success=True)
    )
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )
    helper_lib._interface = mock_interface

    response = helper_lib.complete_work_unit(
        outcome=orchestrator_helper.WORK_UNIT_OUTCOME.WORK_UNIT_OUTCOME_SUCCESS,
        note="test_note",
    )
    self.assertTrue(response.success)

  def test_complete_work_unit_bad_without_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.complete_work_unit(
        outcome=orchestrator_helper.WORK_UNIT_OUTCOME.WORK_UNIT_OUTCOME_SUCCESS,
        note="test_note",
    )
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, orchestrator_helper._ERROR_NO_ACTIVE_CONNECTION
    )

  def test_complete_work_unit_bad_with_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
        raise_error=True,
    )

    with self.assertRaises(ValueError):
      helper_lib.complete_work_unit(
          outcome=(
              orchestrator_helper.WORK_UNIT_OUTCOME.WORK_UNIT_OUTCOME_SUCCESS
          ),
          note="test_note",
      )


if __name__ == "__main__":
  absltest.main()
