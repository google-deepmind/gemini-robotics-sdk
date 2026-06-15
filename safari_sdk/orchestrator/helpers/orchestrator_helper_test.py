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
      return_value=orchestrator_helper.interface.RESPONSE(
          success=True, robot_id="test_robot_id"
      ),
  )
  def test_connect_robot_id_good(self, _):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.connect()
    self.assertTrue(response.success)
    self.assertEqual(response.robot_id, "test_robot_id")

  @mock.patch(
      "safari_sdk.orchestrator.client.interface.OrchestratorInterface.connect",
      return_value=orchestrator_helper.interface.RESPONSE(
          success=True, robot_id="test_robot_id"
      ),
  )
  def test_connect_hostname_good(self, _):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="",
        hostname="test_hostname",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.connect()
    self.assertTrue(response.success)
    self.assertEqual(response.robot_id, "test_robot_id")

  def test_job_type_codes_only_valid(self):
    helper = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type_codes=["test_code"],
    )
    self.assertEqual(helper._job_type_codes, ["test_code"])
    self.assertIsNone(helper._job_type)

  def test_job_type_only_valid(self):
    helper = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )
    self.assertEqual(helper._job_type, orchestrator_helper.JOB_TYPE.ALL)
    self.assertIsNone(helper._job_type_codes)

  def test_both_job_type_and_job_type_codes_valid(self):
    helper = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
        job_type_codes=["test_code"],
    )
    self.assertEqual(helper._job_type, orchestrator_helper.JOB_TYPE.ALL)
    self.assertEqual(helper._job_type_codes, ["test_code"])

  def test_both_job_type_and_job_type_codes_empty_raises_value_error(self):
    with self.assertRaisesRegex(
        ValueError,
        "OrchestratorHelper: job_type_code must be set.",
    ):
      orchestrator_helper.OrchestratorHelper(
          robot_id="test_robot_id",
      )

  def test_connect_no_robot_id_and_no_hostname_bad(self):
    with self.assertRaisesRegex(
        ValueError,
        "OrchestratorHelper: Either robot_id or hostname must be set.",
    ):
      orchestrator_helper.OrchestratorHelper(
          robot_id="",
          hostname="",
          job_type=orchestrator_helper.JOB_TYPE.ALL,
      )

  def test_connect_both_robot_id_and_hostname_bad(self):
    with self.assertRaisesRegex(
        ValueError,
        "OrchestratorHelper: Only one of robot_id or hostname should be set. If"
        " you wish to use hostname instead of robot_id, please set the value of"
        " robot_id as an empty string.",
    ):
      orchestrator_helper.OrchestratorHelper(
          robot_id="test_robot_id",
          hostname="test_hostname",
          job_type=orchestrator_helper.JOB_TYPE.ALL,
      )

  @mock.patch(
      "safari_sdk.orchestrator.client.interface.OrchestratorInterface.connect",
      return_value=orchestrator_helper.interface.RESPONSE(
          success=False, error_message="Error from interface.connect()"
      ),
  )
  def test_connect_bad_without_raise_error(self, _):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
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
        robot_id="test_robot_id",
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
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.get_current_connection()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, orchestrator_helper._ERROR_NO_ACTIVE_CONNECTION
    )

  def test_get_current_connection_bad_with_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
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
            operator_id="test_operator_id",
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
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.get_current_robot_info()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, orchestrator_helper._ERROR_NO_ACTIVE_CONNECTION
    )

  def test_get_current_robot_info_bad_with_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
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
            operator_id="test_operator_id",
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
        robot_id="test_robot_id",
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
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
        raise_error=True,
    )

    with self.assertRaises(ValueError):
      helper_lib.set_current_robot_operator_id(operator_id="test_operator_id")

  def test_update_robot_hardware_config_good(self):
    mock_interface = mock.create_autospec(
        spec=orchestrator_helper.interface.OrchestratorInterface, instance=True
    )
    mock_interface.update_robot_hardware_config.return_value = (
        orchestrator_helper.interface.RESPONSE(
            success=True,
            robot_id="test_robot_id",
        )
    )
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )
    helper_lib._interface = mock_interface

    response = helper_lib.update_robot_hardware_config(
        components=[
            orchestrator_helper.ROBOT_HARDWARE_COMPONENT(
                component_name="component_a", serial_number="123"
            )
        ]
    )
    self.assertTrue(response.success)
    self.assertEqual(response.robot_id, "test_robot_id")

  def test_update_robot_hardware_config_bad_without_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.update_robot_hardware_config(
        components=[
            orchestrator_helper.ROBOT_HARDWARE_COMPONENT(
                component_name="component_a", serial_number="123"
            )
        ]
    )
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, orchestrator_helper._ERROR_NO_ACTIVE_CONNECTION
    )

  def test_update_robot_hardware_config_bad_with_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
        raise_error=True,
    )

    with self.assertRaises(ValueError):
      helper_lib.update_robot_hardware_config(
          components=[
              orchestrator_helper.ROBOT_HARDWARE_COMPONENT(
                  component_name="component_a", serial_number="123"
              )
          ]
      )

  def test_add_operator_event_battery_level(self):
    mock_interface = mock.create_autospec(
        spec=orchestrator_helper.interface.OrchestratorInterface, instance=True
    )
    mock_interface.add_operator_event.return_value = (
        orchestrator_helper.interface.RESPONSE(success=True)
    )
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )
    helper_lib._interface = mock_interface

    response = helper_lib.add_operator_event(
        operator_event_str="Battery Level",
        operator_id="test_operator_id",
        event_timestamp=123456789,
        resetter_id="test_operator_id",
        event_note="85",
    )
    self.assertTrue(response.success)
    mock_interface.add_operator_event.assert_called_once_with(
        operator_event_type=None,
        operator_event_str="Battery Level",
        operator_id="test_operator_id",
        event_timestamp=123456789,
        resetter_id="test_operator_id",
        event_note="85",
    )

    response = helper_lib.add_operator_event(
        operator_event_type=24,  # OPERATOR_EVENT_TYPE_BATTERY_LEVEL_INFO
        operator_id="test_operator_id",
        event_timestamp=123456789,
        resetter_id="test_operator_id",
        event_note="85",
    )
    self.assertTrue(response.success)
    mock_interface.add_operator_event.assert_called_with(
        operator_event_type=24,  # OPERATOR_EVENT_TYPE_BATTERY_LEVEL_INFO
        operator_event_str="",
        operator_id="test_operator_id",
        event_timestamp=123456789,
        resetter_id="test_operator_id",
        event_note="85",
    )

  def test_add_operator_event_battery_level_default_unknown_level(self):
    mock_interface = mock.create_autospec(
        spec=orchestrator_helper.interface.OrchestratorInterface, instance=True
    )
    mock_interface.add_operator_event.return_value = (
        orchestrator_helper.interface.RESPONSE(success=True)
    )
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )
    helper_lib._interface = mock_interface
    # Default battery level is 0.
    response = helper_lib.add_operator_event(
        operator_event_str="Battery Level",
        operator_id="test_operator_id",
        event_timestamp=123456789,
        resetter_id="test_operator_id",
        event_note="0",
    )
    self.assertTrue(response.success)
    mock_interface.add_operator_event.assert_called_once_with(
        operator_event_type=None,
        operator_event_str="Battery Level",
        operator_id="test_operator_id",
        event_timestamp=123456789,
        resetter_id="test_operator_id",
        event_note="0",
    )

    # Default battery level is 0 with operator event type.
    response = helper_lib.add_operator_event(
        operator_event_type=24,  # OPERATOR_EVENT_TYPE_BATTERY_LEVEL_INFO
        operator_id="test_operator_id",
        event_timestamp=123456789,
        resetter_id="test_operator_id",
        event_note="0",
    )
    self.assertTrue(response.success)
    mock_interface.add_operator_event.assert_called_with(
        operator_event_type=24,  # OPERATOR_EVENT_TYPE_BATTERY_LEVEL_INFO
        operator_event_str="",
        operator_id="test_operator_id",
        event_timestamp=123456789,
        resetter_id="test_operator_id",
        event_note="0",
    )

  def test_add_operator_event_good(self):
    mock_interface = mock.create_autospec(
        spec=orchestrator_helper.interface.OrchestratorInterface, instance=True
    )
    mock_interface.add_operator_event.return_value = (
        orchestrator_helper.interface.RESPONSE(success=True)
    )
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )
    helper_lib._interface = mock_interface

    response = helper_lib.add_operator_event(
        operator_event_str="Other Break",
        operator_id="test_operator_id",
        event_timestamp=123456789,
        resetter_id="test_resetter_id",
        event_note="test_event_note",
    )
    self.assertTrue(response.success)
    mock_interface.add_operator_event.assert_called_once_with(
        operator_event_type=None,
        operator_event_str="Other Break",
        operator_id="test_operator_id",
        event_timestamp=123456789,
        resetter_id="test_resetter_id",
        event_note="test_event_note",
    )

    response = helper_lib.add_operator_event(
        operator_event_type=5,  # OPERATOR_EVENT_TYPE_BREAK_OTHER
        operator_id="test_operator_id",
        event_timestamp=123456789,
        resetter_id="test_resetter_id",
        event_note="test_event_note",
    )
    self.assertTrue(response.success)
    mock_interface.add_operator_event.assert_called_with(
        operator_event_type=5,  # OPERATOR_EVENT_TYPE_BREAK_OTHER
        operator_event_str="",
        operator_id="test_operator_id",
        event_timestamp=123456789,
        resetter_id="test_resetter_id",
        event_note="test_event_note",
    )

  def test_add_operator_event_bad_without_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.add_operator_event(
        operator_event_str="Other Break",
        operator_id="test_operator_id",
        event_timestamp=123456789,
        resetter_id="test_resetter_id",
        event_note="test_event_note",
    )
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, orchestrator_helper._ERROR_NO_ACTIVE_CONNECTION
    )

    response = helper_lib.add_operator_event(
        operator_event_type=5,  # OPERATOR_EVENT_TYPE_BREAK_OTHER
        operator_id="test_operator_id",
        event_timestamp=123456789,
        resetter_id="test_resetter_id",
        event_note="test_event_note",
    )
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, orchestrator_helper._ERROR_NO_ACTIVE_CONNECTION
    )

  def test_add_operator_event_bad_with_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
        raise_error=True,
    )

    with self.assertRaises(ValueError):
      helper_lib.add_operator_event(
          operator_event_str="Other Break",
          operator_id="test_operator_id",
          event_timestamp=123456789,
          resetter_id="test_resetter_id",
          event_note="test_event_note",
      )

    with self.assertRaises(ValueError):
      helper_lib.add_operator_event(
          operator_event_type=5,  # OPERATOR_EVENT_TYPE_BREAK_OTHER
          operator_id="test_operator_id",
          event_timestamp=123456789,
          resetter_id="test_resetter_id",
          event_note="test_event_note",
      )

  def test_add_robot_event_good(self):
    mock_interface = mock.create_autospec(
        spec=orchestrator_helper.interface.OrchestratorInterface, instance=True
    )
    mock_interface.add_robot_event.return_value = (
        orchestrator_helper.interface.RESPONSE(success=True)
    )
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )
    helper_lib._interface = mock_interface

    response = helper_lib.add_robot_event(
        event_type="break_ergo",
        payload={"operator_id": "user123"},
        event_timestamp="2026-05-29T17:11:46Z",
    )
    self.assertTrue(response.success)
    mock_interface.add_robot_event.assert_called_once_with(
        event_type="break_ergo",
        payload={"operator_id": "user123"},
        event_timestamp="2026-05-29T17:11:46Z",
    )

  def test_add_robot_event_bad_without_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.add_robot_event(
        event_type="break_ergo",
        payload={"operator_id": "user123"},
        event_timestamp="2026-05-29T17:11:46Z",
    )
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, orchestrator_helper._ERROR_NO_ACTIVE_CONNECTION
    )

  def test_add_robot_event_bad_with_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
        raise_error=True,
    )

    with self.assertRaises(ValueError):
      helper_lib.add_robot_event(
          event_type="break_ergo",
          payload={"operator_id": "user123"},
          event_timestamp="2026-05-29T17:11:46Z",
      )

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

  def test_observe_latest_work_unit_good(self):
    mock_interface = mock.create_autospec(
        spec=orchestrator_helper.interface.OrchestratorInterface, instance=True
    )
    mock_interface.observe_latest_work_unit.return_value = (
        orchestrator_helper.interface.RESPONSE(success=True)
    )
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )
    helper_lib._interface = mock_interface

    response = helper_lib.observe_latest_work_unit()
    self.assertTrue(response.success)

  def test_observe_latest_work_unit_bad_without_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.observe_latest_work_unit()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, orchestrator_helper._ERROR_NO_ACTIVE_CONNECTION
    )

  def test_observe_latest_work_unit_bad_with_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
        raise_error=True,
    )

    with self.assertRaises(ValueError):
      helper_lib.observe_latest_work_unit()

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

  def test_is_visual_overlay_in_current_work_unit_bad_without_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.is_visual_overlay_in_current_work_unit()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, orchestrator_helper._ERROR_NO_ACTIVE_CONNECTION
    )

  def test_is_visual_overlay_in_current_work_unit_bad_with_raise_error(self):
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
      self,
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
      self,
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
            visual_overlay_renderer_keys=["renderer_1", "renderer_2"],
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

  def test_list_visual_overlay_renderer_keys_bad_without_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.list_visual_overlay_renderer_keys()
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, orchestrator_helper._ERROR_NO_ACTIVE_CONNECTION
    )

  def test_list_visual_overlay_renderer_keys_bad_with_raise_error(self):
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

  def test_render_visual_overlay_bad_without_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.render_visual_overlay(renderer_key="renderer_1")
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, orchestrator_helper._ERROR_NO_ACTIVE_CONNECTION
    )

  def test_render_visual_overlay_bad_with_raise_error(self):
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

  def test_get_visual_overlay_image_as_pil_image_bad_without_raise_error(self):
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

  def test_get_visual_overlay_image_as_pil_image_bad_with_raise_error(self):
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

  def test_get_visual_overlay_image_as_np_array_bad_without_raise_error(self):
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

  def test_get_visual_overlay_image_as_np_array_bad_with_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
        raise_error=True,
    )

    with self.assertRaises(ValueError):
      helper_lib.get_visual_overlay_image_as_np_array(renderer_key="renderer_1")

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

  def test_get_visual_overlay_image_as_bytes_bad_without_raise_error(self):
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

  def test_get_visual_overlay_image_as_bytes_bad_with_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
        raise_error=True,
    )

    with self.assertRaises(ValueError):
      helper_lib.get_visual_overlay_image_as_bytes(renderer_key="renderer_1")

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

  def test_reset_visual_overlay_renderer_bad_without_raise_error(self):
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

  def test_reset_visual_overlay_renderer_bad_with_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
        raise_error=True,
    )

    with self.assertRaises(ValueError):
      helper_lib.reset_visual_overlay_renderer(renderer_key="renderer_1")

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

  def test_create_single_visual_overlay_renderer_bad_without_raise_error(self):
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

  def test_create_single_visual_overlay_renderer_bad_with_raise_error(self):
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
      self,
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
      self,
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

  def test_insert_session_info_good(self):
    mock_interface = mock.create_autospec(
        spec=orchestrator_helper.interface.OrchestratorInterface, instance=True
    )
    mock_interface.robot_job_work_unit_insert_session_info.return_value = (
        orchestrator_helper.interface.RESPONSE(success=True)
    )
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )
    helper_lib._interface = mock_interface

    response = helper_lib.insert_session_info(
        session_log_type="test_session_log_type",
        session_start_time_ns=1764547200000000001,
        session_end_time_ns=1764547210000000002,
        session_note="test_note",
    )
    self.assertTrue(response.success)

  def test_insert_session_info_bad_without_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.insert_session_info(
        session_log_type="test_session_log_type",
        session_start_time_ns=1764547200000000001,
        session_end_time_ns=1764547210000000002,
        session_note="test_note",
    )
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, orchestrator_helper._ERROR_NO_ACTIVE_CONNECTION
    )

  def test_insert_session_info_bad_with_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
        raise_error=True,
    )

    with self.assertRaises(ValueError):
      helper_lib.insert_session_info(
          session_log_type="test_session_log_type",
          session_start_time_ns=1764547200000000001,
          session_end_time_ns=1764547210000000002,
          session_note="test_note",
      )

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

    response = helper_lib.complete_work_unit(
        outcome=orchestrator_helper.WORK_UNIT_OUTCOME.WORK_UNIT_OUTCOME_SUCCESS,
        success_score=0.5,
        success_score_definition="test_success_score_definition",
        note="test_note",
    )
    self.assertTrue(response.success)

    response = helper_lib.complete_work_unit(
        outcome=orchestrator_helper.WORK_UNIT_OUTCOME.WORK_UNIT_OUTCOME_SUCCESS,
        session_start_time_ns=123456789,
        session_end_time_ns=987654321,
        session_log_type="test_session_log_type",
        note="test_note",
    )
    self.assertTrue(response.success)

    response = helper_lib.complete_work_unit(
        outcome=orchestrator_helper.WORK_UNIT_OUTCOME.WORK_UNIT_OUTCOME_SUCCESS,
        success_score=0.5,
        success_score_definition="test_success_score_definition",
        session_start_time_ns=123456789,
        session_end_time_ns=987654321,
        session_log_type="test_session_log_type",
        note="test_note",
    )
    self.assertTrue(response.success)

    response = helper_lib.complete_work_unit(
        outcome=orchestrator_helper.WORK_UNIT_OUTCOME.WORK_UNIT_OUTCOME_SUCCESS,
        note="test_note",
        response_to_questions=[
            orchestrator_helper.WORK_UNIT_QUESTION(
                question="test_question_1",
                whenToAsk=[
                    orchestrator_helper.QUESTION_CONDITION.QUESTION_CONDITION_ALWAYS,
                ],
                answerFormat=orchestrator_helper.QUESTION_ANSWER_TYPE.ANSWER_TYPE_SINGLE_CHOICE,
                allowedAnswers=[
                    "test_allowed_answer_1",
                    "test_allowed_answer_2",
                ],
                userAnswers=["test_user_answer_1"],
                wasDisplayed=True,
            ),
        ],
    )
    self.assertTrue(response.success)

    response = helper_lib.complete_work_unit(
        outcome=orchestrator_helper.WORK_UNIT_OUTCOME.WORK_UNIT_OUTCOME_SUCCESS,
        success_score=0.5,
        success_score_definition="test_success_score_definition",
        session_start_time_ns=123456789,
        session_end_time_ns=987654321,
        session_log_type="test_session_log_type",
        note="test_note",
        response_to_questions=[
            orchestrator_helper.WORK_UNIT_QUESTION(
                question="test_question_1",
                whenToAsk=[
                    orchestrator_helper.QUESTION_CONDITION.QUESTION_CONDITION_ALWAYS,
                ],
                answerFormat=orchestrator_helper.QUESTION_ANSWER_TYPE.ANSWER_TYPE_SINGLE_CHOICE,
                allowedAnswers=[
                    "test_allowed_answer_1",
                    "test_allowed_answer_2",
                ],
                userAnswers=["test_user_answer_1"],
                wasDisplayed=True,
            ),
        ],
    )
    self.assertTrue(response.success)

  def test_complete_work_unit_bad_without_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.complete_work_unit(
        outcome=orchestrator_helper.WORK_UNIT_OUTCOME.WORK_UNIT_OUTCOME_SUCCESS,
        success_score=0.5,
        success_score_definition="test_success_score_definition",
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
          success_score=0.5,
          success_score_definition="test_success_score_definition",
          note="test_note",
      )

  def test_get_artifact_good(self):
    mock_interface = mock.create_autospec(
        spec=orchestrator_helper.interface.OrchestratorInterface, instance=True
    )
    mock_interface.get_artifact.return_value = orchestrator_helper.interface.RESPONSE(
        success=True,
        artifact=orchestrator_helper.interface.api_response.artifact_data.Artifact(
            uri="test_artifact_uri",
            artifactId="test_artifact_id",
            name="test_name",
            artifactObjectType="ARTIFACT_OBJECT_TYPE_IMAGE",
            commitTime="2025-01-01T00:00:00Z",
            tags=["tag1", "tag2"],
            version="1",
            isZipped=False,
        ),
    )
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )
    helper_lib._interface = mock_interface

    response = helper_lib.get_artifact(artifact_id="test_artifact_id")
    self.assertTrue(response.success)
    self.assertEqual(response.artifact.uri, "test_artifact_uri")
    self.assertEqual(response.artifact.artifactId, "test_artifact_id")
    self.assertEqual(response.artifact.name, "test_name")
    self.assertEqual(
        response.artifact.artifactObjectType,
        "ARTIFACT_OBJECT_TYPE_IMAGE",
    )
    self.assertEqual(response.artifact.commitTime, "2025-01-01T00:00:00Z")
    self.assertEqual(response.artifact.tags, ["tag1", "tag2"])
    self.assertEqual(response.artifact.version, "1")
    self.assertFalse(response.artifact.isZipped)

  def test_get_artifact_bad_without_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.get_artifact(artifact_id="test_artifact_id")
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, orchestrator_helper._ERROR_NO_ACTIVE_CONNECTION
    )

  def test_get_artifact_bad_with_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
        raise_error=True,
    )

    with self.assertRaises(ValueError):
      helper_lib.get_artifact(artifact_id="test_artifact_id")

  def test_get_artifact_uri_good(self):
    mock_interface = mock.create_autospec(
        spec=orchestrator_helper.interface.OrchestratorInterface, instance=True
    )
    mock_interface.get_artifact_uri.return_value = (
        orchestrator_helper.interface.RESPONSE(
            success=True,
            artifact_uri="test_artifact_uri",
        )
    )
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )
    helper_lib._interface = mock_interface

    response = helper_lib.get_artifact_uri(artifact_id="test_artifact_id")
    self.assertTrue(response.success)
    self.assertEqual(response.artifact_uri, "test_artifact_uri")

  def test_get_artifact_uri_bad_without_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.get_artifact_uri(artifact_id="test_artifact_id")
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, orchestrator_helper._ERROR_NO_ACTIVE_CONNECTION
    )

  def test_get_artifact_uri_bad_with_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
        raise_error=True,
    )

    with self.assertRaises(ValueError):
      helper_lib.get_artifact_uri(artifact_id="test_artifact_id")

  def test_disconnect(self):
    mock_interface = mock.create_autospec(
        spec=orchestrator_helper.interface.OrchestratorInterface, instance=True
    )
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )
    helper_lib._interface = mock_interface

    helper_lib.disconnect()
    mock_interface.disconnect.assert_called_once()
    self.assertIsNone(helper_lib._interface)

  def test_upload_text_log_artifact_good(self):
    mock_interface = mock.create_autospec(
        spec=orchestrator_helper.interface.OrchestratorInterface, instance=True
    )
    mock_interface.upload_text_log_artifact.return_value = (
        orchestrator_helper.interface.RESPONSE(
            success=True,
            artifact_id="test_uploaded_artifact_id",
        )
    )
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )
    helper_lib._interface = mock_interface

    response = helper_lib.upload_text_log_artifact(
        source_file_name="test_log.txt",
        text_file_bytes=b"test log content",
    )
    self.assertTrue(response.success)
    self.assertEqual(response.artifact_id, "test_uploaded_artifact_id")

  def test_upload_text_log_artifact_bad_without_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.upload_text_log_artifact(
        source_file_name="test_log.txt",
        text_file_bytes=b"test log content",
    )
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, orchestrator_helper._ERROR_NO_ACTIVE_CONNECTION
    )

  def test_upload_text_log_artifact_bad_with_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
        raise_error=True,
    )

    with self.assertRaises(ValueError):
      helper_lib.upload_text_log_artifact(
          source_file_name="test_log.txt",
          text_file_bytes=b"test log content",
      )

  def test_load_rui_workcell_state_good(self):
    mock_interface = mock.create_autospec(
        spec=orchestrator_helper.interface.OrchestratorInterface, instance=True
    )
    mock_interface.load_rui_workcell_state.return_value = (
        orchestrator_helper.interface.RESPONSE(
            success=True,
            workcell_state="RUI_WORKCELL_STATE_AVAILABLE",
        )
    )
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )
    helper_lib._interface = mock_interface

    response = helper_lib.load_rui_workcell_state(robot_id="test_robot_id")
    self.assertTrue(response.success)
    self.assertEqual(response.workcell_state, "RUI_WORKCELL_STATE_AVAILABLE")

  def test_load_rui_workcell_state_bad_without_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.load_rui_workcell_state(robot_id="test_robot_id")
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, orchestrator_helper._ERROR_NO_ACTIVE_CONNECTION
    )

  def test_load_rui_workcell_state_bad_with_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
        raise_error=True,
    )

    with self.assertRaises(ValueError):
      helper_lib.load_rui_workcell_state(robot_id="test_robot_id")

  def test_set_rui_workcell_state_good(self):
    mock_interface = mock.create_autospec(
        spec=orchestrator_helper.interface.OrchestratorInterface, instance=True
    )
    mock_interface.set_rui_workcell_state.return_value = (
        orchestrator_helper.interface.RESPONSE(success=True)
    )
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )
    helper_lib._interface = mock_interface

    response = helper_lib.set_rui_workcell_state(
        robot_id="test_robot_id",
        workcell_state_type=10,
    )
    self.assertTrue(response.success)

  def test_set_rui_workcell_state_bad_without_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )

    response = helper_lib.set_rui_workcell_state(
        robot_id="test_robot_id",
        workcell_state_type=10,
    )
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, orchestrator_helper._ERROR_NO_ACTIVE_CONNECTION
    )

  def test_set_rui_workcell_state_bad_with_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
        raise_error=True,
    )

    with self.assertRaises(ValueError):
      helper_lib.set_rui_workcell_state(
          robot_id="test_robot_id",
          workcell_state_type=10,
      )

  def test_complete_work_unit_with_client_overrides(self):
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

    overrides = [
        orchestrator_helper.KV_MSG(
            key="test_key",
            type=orchestrator_helper.KV_MSG_TYPE.KV_MSG_VALUE_TYPE_STRING,
            value=orchestrator_helper.KV_MSG_VALUE(stringValue="test_value"),
        )
    ]
    response = helper_lib.complete_work_unit(
        outcome=orchestrator_helper.WORK_UNIT_OUTCOME.WORK_UNIT_OUTCOME_SUCCESS,
        note="test_note",
        client_overrides=overrides,
    )
    self.assertTrue(response.success)
    mock_interface.robot_job_work_unit_complete_work_unit.assert_called_once_with(
        outcome=orchestrator_helper.WORK_UNIT_OUTCOME.WORK_UNIT_OUTCOME_SUCCESS,
        success_score=None,
        success_score_definition=None,
        session_start_time_ns=None,
        session_end_time_ns=None,
        session_log_type=None,
        session_note=None,
        response_to_questions=None,
        note="test_note",
        request_retry_bypass=False,
        client_overrides=overrides,
    )

  def test_create_kv_methods_good(self):
    mock_interface = mock.create_autospec(
        spec=orchestrator_helper.interface.OrchestratorInterface, instance=True
    )
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )
    helper_lib._interface = mock_interface

    helper_lib.create_kv_string("k1", "v1")
    mock_interface.create_kv_msg.assert_called_with(
        key="k1",
        kv_type=orchestrator_helper.KV_MSG_TYPE.KV_MSG_VALUE_TYPE_STRING,
        value="v1",
    )

    helper_lib.create_kv_string_list("k2", ["v2"])
    mock_interface.create_kv_msg.assert_called_with(
        key="k2",
        kv_type=orchestrator_helper.KV_MSG_TYPE.KV_MSG_VALUE_TYPE_STRING_LIST,
        value=["v2"],
    )

    helper_lib.create_kv_int("k3", 3)
    mock_interface.create_kv_msg.assert_called_with(
        key="k3",
        kv_type=orchestrator_helper.KV_MSG_TYPE.KV_MSG_VALUE_TYPE_INT,
        value=3,
    )

    helper_lib.create_kv_int_list("k4", [4])
    mock_interface.create_kv_msg.assert_called_with(
        key="k4",
        kv_type=orchestrator_helper.KV_MSG_TYPE.KV_MSG_VALUE_TYPE_INT_LIST,
        value=[4],
    )

    helper_lib.create_kv_float("k5", 5.0)
    mock_interface.create_kv_msg.assert_called_with(
        key="k5",
        kv_type=orchestrator_helper.KV_MSG_TYPE.KV_MSG_VALUE_TYPE_FLOAT,
        value=5.0,
    )

    helper_lib.create_kv_float_list("k6", [6.0])
    mock_interface.create_kv_msg.assert_called_with(
        key="k6",
        kv_type=orchestrator_helper.KV_MSG_TYPE.KV_MSG_VALUE_TYPE_FLOAT_LIST,
        value=[6.0],
    )

    helper_lib.create_kv_bool("k7", True)
    mock_interface.create_kv_msg.assert_called_with(
        key="k7",
        kv_type=orchestrator_helper.KV_MSG_TYPE.KV_MSG_VALUE_TYPE_BOOL,
        value=True,
    )

    helper_lib.create_kv_bool_list("k8", [False])
    mock_interface.create_kv_msg.assert_called_with(
        key="k8",
        kv_type=orchestrator_helper.KV_MSG_TYPE.KV_MSG_VALUE_TYPE_BOOL_LIST,
        value=[False],
    )

    helper_lib.create_kv_json("k9", "{}")
    mock_interface.create_kv_msg.assert_called_with(
        key="k9",
        kv_type=orchestrator_helper.KV_MSG_TYPE.KV_MSG_VALUE_TYPE_JSON,
        value="{}",
    )

  def test_create_kv_methods_bad_without_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
    )
    response = helper_lib.create_kv_string("k1", "v1")
    self.assertFalse(response.success)
    self.assertEqual(
        response.error_message, orchestrator_helper._ERROR_NO_ACTIVE_CONNECTION
    )

  def test_create_kv_methods_bad_with_raise_error(self):
    helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="test_robot_id",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
        raise_error=True,
    )
    with self.assertRaises(ValueError):
      helper_lib.create_kv_string("k1", "v1")


class OrchestratorHelperTestWithMockInterface(absltest.TestCase):
  """Tests for OrchestratorHelper with an overridden interface."""

  helper_lib: orchestrator_helper.OrchestratorHelper | None = None

  class FakeOrchestratorInterface(
      orchestrator_helper.interface.OrchestratorInterface
  ):
    """Fake OrchestratorInterface for testing."""

    def __init__(
        self,
        robot_id: str,
        job_type: orchestrator_helper.JOB_TYPE | None = None,
        job_type_codes: list[str] | None = None,
        hostname: str | None = None,
        observer_mode: bool = False,
    ):
      super().__init__(
          robot_id=robot_id,
          job_type=job_type,
          job_type_codes=job_type_codes,
          hostname=hostname,
          observer_mode=observer_mode,
      )

    def connect(self) -> orchestrator_helper.RESPONSE:
      return orchestrator_helper.RESPONSE(
          success=True,
          robot_id="test_robot_id",
      )

    def disconnect(self) -> None:
      return

  def setUp(self):
    super().setUp()
    self.helper_lib = orchestrator_helper.OrchestratorHelper(
        robot_id="",
        hostname="test_hostname",
        job_type=orchestrator_helper.JOB_TYPE.ALL,
        interface_type=self.FakeOrchestratorInterface,
    )

  def test_connects_and_gets_robot_id(self):
    response = self.helper_lib.connect()
    self.assertTrue(response.success)
    self.assertEqual(response.robot_id, "test_robot_id")


if __name__ == "__main__":
  absltest.main()
