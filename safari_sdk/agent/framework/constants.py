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

"""Constants used across the framework."""

# Default audio input sample rate. Gemini Live API input audio is natively 16kHz
# according to the following link:
# https://ai.google.dev/gemini-api/docs/live-guide#audio-formats
DEFAULT_AUDIO_INPUT_RATE = 16000

# Reserved key for the camera name in event metadata. Camera streams should
# embed their name under this key for subscriber identification.
STREAM_NAME_METADATA_KEY = "stream_name"
