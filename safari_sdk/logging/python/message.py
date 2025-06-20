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

"""Safari Message dataclass."""

import dataclasses

from google.protobuf import message as message_lib


@dataclasses.dataclass
class Message:
  """Safari Message dataclass."""

  def __init__(
      self,
      topic: str,
      message: message_lib.Message,
      publish_time_nsec: int,
      log_time_nsec: int = 0,
  ):
    """Initializes a Message.

    Args:
      topic: The safari_logging_topic of the message.
      message: The proto message to be written.
      publish_time_nsec: The timestamp of the message (this may be the time the
        message was published, or the time the data in the  message was
        sampled).
      log_time_nsec: The time when the logger received the message.
    """
    self.topic = topic
    self.message = message
    self.publish_time_nsec = publish_time_nsec
    self.log_time_nsec = log_time_nsec
