# Copyright 2026 Google LLC
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

from absl.testing import absltest
from safari_sdk.orchestrator.client.dataclass import robot_event


class RobotEventDataclassTest(absltest.TestCase):

  def test_create_robot_event_response_parses_correctly(self):
    input_json = '{"eventId": "test-event-id-123"}'
    response = robot_event.CreateRobotEventResponse.from_json(input_json)
    self.assertEqual(response.eventId, "test-event-id-123")

  def test_create_robot_event_response_parses_with_none(self):
    input_json = '{"eventId": null}'
    response = robot_event.CreateRobotEventResponse.from_json(input_json)
    self.assertIsNone(response.eventId)

  def test_create_robot_event_response_parses_empty(self):
    input_json = "{}"
    response = robot_event.CreateRobotEventResponse.from_json(input_json)
    self.assertIsNone(response.eventId)


if __name__ == "__main__":
  absltest.main()
