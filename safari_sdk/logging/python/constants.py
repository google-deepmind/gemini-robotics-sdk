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

"""Constants for logging."""


# Reserved topic names.
FILE_METADATA_TOPIC_NAME: str = '/file_metadata'
SESSION_TOPIC_NAME: str = '/session'
SYNC_TOPIC_NAME: str = '/sync'
TIMESTEP_TOPIC_NAME: str = '/timestep'
ACTION_TOPIC_NAME: str = '/action'
POLICY_EXTRA_TOPIC_NAME: str = '/policy_extra'
DEFAULT_FILE_SHARD_SIZE_LIMIT_BYTES: int = 1 * 1024 * 1024 * 1024
