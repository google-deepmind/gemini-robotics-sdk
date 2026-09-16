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

"""Unit tests for operator_event.py."""

from unittest import mock

from absl.testing import absltest
from googleapiclient import errors

from safari_sdk.orchestrator.client.libs import operator_event


class OperatorEventTest(absltest.TestCase):

  def test_add_operator_event_good(self):
    mock_connection = mock.MagicMock()
    response_dict = {
        "success": True,
    }
    mock_connection.orchestrator().addOperatorEvent().execute.return_value = (
        response_dict
    )

    operator_event_lib_instance = operator_event.OrchestratorOperatorEvent(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    response = operator_event_lib_instance.add_operator_event(
        operator_event_type=5,  # OPERATOR_EVENT_TYPE_BREAK_OTHER
        operator_id="test_operator_id",
        event_timestamp=123456789,
        resetter_id="test_resetter_id",
        event_note="test_event_note",
    )
    self.assertTrue(response.success)

  def test_add_operator_event_none_type(self):
    mock_connection = mock.MagicMock()
    response_dict = {
        "success": True,
    }
    mock_connection.orchestrator().addOperatorEvent().execute.return_value = (
        response_dict
    )

    operator_event_lib_instance = operator_event.OrchestratorOperatorEvent(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    response = operator_event_lib_instance.add_operator_event(
        operator_event_type=None,
        operator_id="test_operator_id",
        event_timestamp=123456789,
        resetter_id="test_resetter_id",
        event_note="test_event_note",
    )
    self.assertFalse(response.success)

  def test_add_operator_event_battery(self):
    mock_connection = mock.MagicMock()
    response_dict = {
        "success": True,
    }
    mock_connection.orchestrator().addOperatorEvent().execute.return_value = (
        response_dict
    )

    operator_event_lib_instance = operator_event.OrchestratorOperatorEvent(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    response = operator_event_lib_instance.add_operator_event(
        operator_event_type=24,  # OPERATOR_EVENT_TYPE_BATTERY_LEVEL_INFO
        operator_id="test_operator_id",
        event_timestamp=123456789,
        resetter_id="test_resetter_id",
        event_note="99",
    )
    self.assertTrue(response.success)

  def test_add_operator_event_bad_server_call(self):
    class MockHttpError:

      def __init__(self):
        self.status = "Mock status"
        self.reason = "Mock reason"
        self.error_details = "Mock error details"

    def raise_error_side_effect():
      raise errors.HttpError(MockHttpError(), "Mock failed HTTP call.".encode())

    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().addOperatorEvent().execute.side_effect = (
        raise_error_side_effect
    )

    operator_event_lib_instance = operator_event.OrchestratorOperatorEvent(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    response = operator_event_lib_instance.add_operator_event(
        operator_event_type=5,  # OPERATOR_EVENT_TYPE_BREAK_OTHER
        operator_id="test_operator_id",
        event_timestamp=123456789,
        resetter_id="test_resetter_id",
        event_note="test_event_note",
    )

    self.assertFalse(response.success)
    self.assertIn(
        operator_event._ERROR_RECORD_OPERATOR_EVENT, response.error_message
    )

if __name__ == "__main__":
  absltest.main()
