#!/bin/bash
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
# This updates the versions of the dependencies pinned in requirements.txt to
# the latest versions that satisfy the version constraints in pyproject.toml.
# See README.md for more details.
#
# Usage:
#   bash scripts/update_pip_dependencies.sh

set -e
SCRIPTS_DIR="$(realpath "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")"
SAFARI_DIR="$(realpath "${SCRIPTS_DIR}/..")"

# Setup virtual environment and install pip-tools.
VENV_DIR="$(mktemp -d)"
echo "Creating virtual environment and installing pip-tools in ${VENV_DIR}."
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
pip install --require-hashes -r "${SCRIPTS_DIR}/base_tooling_requirements.txt"

# Update requirements.txt file from pyproject.toml.
echo -e "\e[36mGenerating requirements file... (This may take a few minutes.)\e[0m"
pip-compile --upgrade --generate-hashes --allow-unsafe --all-extras \
  ${SAFARI_DIR}/pyproject.toml --output-file=${SAFARI_DIR}/requirements.txt
# Remove any index-url lines from requirements.txt.
sed -i '/^--index-url/d' "${SAFARI_DIR}/requirements.txt"
echo "Saved to ${SAFARI_DIR}/requirements.txt"
deactivate
