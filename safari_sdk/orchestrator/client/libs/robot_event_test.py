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

"""Unit tests for robot_event.py."""

import datetime
from unittest import mock

from absl.testing import absltest
from googleapiclient import errors

from safari_sdk.orchestrator.client.libs import robot_event


class RobotEventTest(absltest.TestCase):

  def test_add_robot_event_good(self):
    mock_connection = mock.MagicMock()
    response_dict = {
        "eventId": "test-event-id-123",
    }
    mock_connection.orchestrator().createRobotEvent.return_value.execute.return_value = (
        response_dict
    )

    robot_event_lib = robot_event.OrchestratorRobotEvent(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    response = robot_event_lib.add_robot_event(
        event_type="break_ergo",
        payload={"operator_id": "user123", "resetter_id": "user456"},
    )
    self.assertTrue(response.success)
    mock_connection.orchestrator().createRobotEvent.assert_called_once_with(
        body={
            "event": {
                "robotId": "test_robot_id",
                "eventType": "break_ergo",
                "payloadJson": (
                    '{"operator_id": "user123", "resetter_id": "user456"}'
                ),
            },
            "tracer": mock.ANY,
        }
    )

  def test_add_robot_event_no_note(self):
    mock_connection = mock.MagicMock()
    response_dict = {
        "eventId": "test-event-id-456",
    }
    mock_connection.orchestrator().createRobotEvent.return_value.execute.return_value = (
        response_dict
    )

    robot_event_lib = robot_event.OrchestratorRobotEvent(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    response = robot_event_lib.add_robot_event(
        event_type="battery_level_info",
        payload={"battery_level": 85},
    )
    self.assertTrue(response.success)
    mock_connection.orchestrator().createRobotEvent.assert_called_once_with(
        body={
            "event": {
                "robotId": "test_robot_id",
                "eventType": "battery_level_info",
                "payloadJson": '{"battery_level": 85}',
            },
            "tracer": mock.ANY,
        }
    )

  def test_add_robot_event_no_connection(self):
    robot_event_lib = robot_event.OrchestratorRobotEvent(
        connection=mock.MagicMock(),
        robot_id="test_robot_id",
    )
    robot_event_lib.disconnect()

    response = robot_event_lib.add_robot_event(
        event_type="maintenance",
        payload={"operator_id": "user123"},
    )
    self.assertFalse(response.success)
    self.assertIn(
        robot_event._ERROR_NO_ORCHESTRATOR_CONNECTION, response.error_message
    )

  def test_add_robot_event_server_error(self):
    class MockHttpError:

      def __init__(self):
        self.status = "Mock status"
        self.reason = "Mock reason"
        self.error_details = "Mock error details"

    def raise_error_side_effect():
      raise errors.HttpError(MockHttpError(), "Mock failed HTTP call.".encode())

    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().createRobotEvent.return_value.execute.side_effect = (
        raise_error_side_effect
    )

    robot_event_lib = robot_event.OrchestratorRobotEvent(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    response = robot_event_lib.add_robot_event(
        event_type="break_ergo",
        payload={"operator_id": "user123"},
    )

    self.assertFalse(response.success)
    self.assertIn(robot_event._ERROR_RECORD_ROBOT_EVENT, response.error_message)
    mock_connection.orchestrator().createRobotEvent.assert_called_once_with(
        body={
            "event": {
                "robotId": "test_robot_id",
                "eventType": "break_ergo",
                "payloadJson": '{"operator_id": "user123"}',
            },
            "tracer": mock.ANY,
        }
    )

  def test_add_robot_event_with_timestamp(self):
    mock_connection = mock.MagicMock()
    response_dict = {
        "eventId": "test-event-id-123",
    }
    mock_connection.orchestrator().createRobotEvent.return_value.execute.return_value = (
        response_dict
    )

    robot_event_lib = robot_event.OrchestratorRobotEvent(
        connection=mock_connection,
        robot_id="test_robot_id",
    )

    # 1. Test with datetime.datetime object (naive, defaults to UTC)
    dt = datetime.datetime(2026, 5, 29, 17, 11, 46)
    response = robot_event_lib.add_robot_event(
        event_type="break_ergo",
        payload={"operator_id": "user123"},
        event_timestamp=dt,
    )
    self.assertTrue(response.success)
    mock_connection.orchestrator().createRobotEvent.assert_called_once_with(
        body={
            "event": {
                "robotId": "test_robot_id",
                "eventType": "break_ergo",
                "payloadJson": '{"operator_id": "user123"}',
                "eventTimestamp": "2026-05-29T17:11:46+00:00",
            },
            "tracer": mock.ANY,
        }
    )

    # 2. Test with string timestamp
    mock_connection.reset_mock()
    response = robot_event_lib.add_robot_event(
        event_type="break_ergo",
        payload={"operator_id": "user123"},
        event_timestamp="2026-05-29T17:11:46Z",
    )
    self.assertTrue(response.success)
    mock_connection.orchestrator().createRobotEvent.assert_called_once_with(
        body={
            "event": {
                "robotId": "test_robot_id",
                "eventType": "break_ergo",
                "payloadJson": '{"operator_id": "user123"}',
                "eventTimestamp": "2026-05-29T17:11:46Z",
            },
            "tracer": mock.ANY,
        }
    )

  def test_add_robot_event_missing_event_id_in_response(self):
    mock_connection = mock.MagicMock()
    response_dict = {
        "someOtherField": "value",
    }
    mock_connection.orchestrator().createRobotEvent.return_value.execute.return_value = (
        response_dict
    )

    robot_event_lib = robot_event.OrchestratorRobotEvent(
        connection=mock_connection,
        robot_id="test_robot_id",
    )
    response = robot_event_lib.add_robot_event(
        event_type="maintenance",
        payload={"operator_id": "user123"},
    )
    self.assertFalse(response.success)
    mock_connection.orchestrator().createRobotEvent.assert_called_once_with(
        body={
            "event": {
                "robotId": "test_robot_id",
                "eventType": "maintenance",
                "payloadJson": '{"operator_id": "user123"}',
            },
            "tracer": mock.ANY,
        }
    )


if __name__ == "__main__":
  absltest.main()
