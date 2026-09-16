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

from safari_sdk.orchestrator.client.libs import rui_workcell_state


class RuiWorkcellStateTest(absltest.TestCase):

  def test_load_rui_workcell_state_success(self):
    mock_connection = mock.MagicMock()
    response_dict = {
        "success": True,
        "workcellState": "test_workcell_state",
    }
    mock_connection.orchestrator().loadRuiWorkcellState().execute.return_value = (
        response_dict
    )
    rui_workcell_state_lib = rui_workcell_state.OrchestratorRuiWorkcellState(
        connection=mock_connection,
    )
    response = rui_workcell_state_lib.load_rui_workcell_state(
        robot_id="test_robot_id"
    )
    self.assertTrue(response.success)
    self.assertEqual(response.workcell_state, "test_workcell_state")

  def test_load_rui_workcell_state_bad_server_call(self):
    class MockHttpError:

      def __init__(self):
        self.status = "Mock status"
        self.reason = "Mock reason"
        self.error_details = "Mock error details"

    def raise_error_side_effect():
      raise errors.HttpError(MockHttpError(), "Mock failed HTTP call.".encode())

    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().loadRuiWorkcellState().execute.side_effect = (
        raise_error_side_effect
    )

    rui_workcell_state_lib = rui_workcell_state.OrchestratorRuiWorkcellState(
        connection=mock_connection,
    )
    response = rui_workcell_state_lib.load_rui_workcell_state(
        robot_id="test_robot_id"
    )

    self.assertFalse(response.success)
    self.assertIn(
        rui_workcell_state._ERROR_LOAD_RUI_WORKCELL_STATE,
        response.error_message
    )

  def test_set_rui_workcell_state_success(self):
    mock_connection = mock.MagicMock()
    response_dict = {
        "success": True,
    }
    mock_connection.orchestrator().setRuiWorkcellState().execute.return_value = (
        response_dict
    )
    rui_workcell_state_lib = rui_workcell_state.OrchestratorRuiWorkcellState(
        connection=mock_connection,
    )
    response = rui_workcell_state_lib.set_rui_workcell_state(
        robot_id="test_robot_id", workcell_state_type=10
    )
    self.assertTrue(response.success)

  def test_set_rui_workcell_state_bad_server_call(self):
    class MockHttpError:

      def __init__(self):
        self.status = "Mock status"
        self.reason = "Mock reason"
        self.error_details = "Mock error details"

    def raise_error_side_effect():
      raise errors.HttpError(MockHttpError(), "Mock failed HTTP call.".encode())

    mock_connection = mock.MagicMock()
    mock_connection.orchestrator().setRuiWorkcellState().execute.side_effect = (
        raise_error_side_effect
    )
    rui_workcell_state_lib = rui_workcell_state.OrchestratorRuiWorkcellState(
        connection=mock_connection,
    )
    response = rui_workcell_state_lib.set_rui_workcell_state(
        robot_id="test_robot_id", workcell_state_type=10
    )
    self.assertFalse(response.success)
    self.assertIn(
        rui_workcell_state._ERROR_SET_RUI_WORKCELL_STATE,
        response.error_message
    )

if __name__ == "__main__":
  absltest.main()
