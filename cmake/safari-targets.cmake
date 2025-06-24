# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

# Reusable targets for the safari package.
#
# Accessing google cloud repositories requires 2 things:
# 1. The gcloud artifact registry keyring must be installed. This can be done
#    automatically by specifying the index's PIP_DEPENDS (see below).
# 2. The user is authenticated with gcloud by running
#    `gcloud auth application-default login`. This must be done manually.

include(cmake/python-packaging.cmake)

# A standard python virtual environment which targets can use if desired.
# The venv directory can be specified with the SAFARI_VENV_DIR variable.
add_python_venv(
  VENV_NAME safari_venv
  OUT_DIR "${SAFARI_VENV_DIR}"
)
