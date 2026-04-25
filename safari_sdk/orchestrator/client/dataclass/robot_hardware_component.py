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

"""Robot hardware component dataclass."""

import dataclasses


@dataclasses.dataclass(kw_only=True)
class RobotHardwareComponent:
  """Information about a robot hardware component.

  Use this to pass robot hardware component updates to the SDK interface for
  merging robot hardware components lists upstream.

  Example:
    robot_hardware_component = RobotHardwareComponent(
        component_name="left_gripper",
        serial_number="12345",
        model="Inspire Hand",
        firmware_number="1.0.0",
    )

  Attributes:
    component_name: The name of the component.
    serial_number: The serial number of the component.
    model: The model of the component.
    firmware_number: The firmware number of the component.
  """

  component_name: str
  serial_number: str
  model: str | None = None
  firmware_number: str | None = None

  def to_dict(self) -> dict[str, str | None]:
    """Converts the dataclass to a dictionary."""
    return dataclasses.asdict(self)
