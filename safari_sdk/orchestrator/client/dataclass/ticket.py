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

"""Orchestrator ticket information."""

import enum

# pylint: disable=invalid-name


class TicketType(enum.IntEnum):
  TICKET_TYPE_UNSPECIFIED = 0
  TICKET_TYPE_ROBOT_MAINTENANCE = 1
  TICKET_TYPE_ORCHESTRATOR_ISSUE = 2


class RobotFailureReason(enum.IntEnum):
  ROBOT_FAILURE_REASON_UNSPECIFIED = 0
  HARDWARE = 1
  SOFTWARE = 2
  SOFTWARE_EVAL = 7
  ROBOT_BEHAVIOR = 3
  INVESTIGATION = 4
  UPGRADE = 5
