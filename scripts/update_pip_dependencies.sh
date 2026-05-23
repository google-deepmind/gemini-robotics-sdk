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

# Exclude safari-sdk-logging from this dependency update.
# The problem with including it here, is that the code for safari-sdk-logging
# is built from this very same repo, while the dependency update here is done
# based on pypi. This could lead to issues if the two don't have the exact
# same dependencies; you want the pinned dependencies to be compatible with both
# safari-sdk and safari-sdk-logging.
if [[ -z "$(pip-compile --strip-extras  "${SAFARI_DIR}/subpackages/logging/pyproject.toml" 2>&1 | grep -Ev '^#')" ]]; then
  # No dependencies - safe to ignore logging.
  echo "Temporarily removing safari-sdk-logging from pyproject for dependency"
  echo "resolution."

  cp ${SAFARI_DIR}/pyproject.toml{,.no-cc-logging}
  # Comment out safari-sdk-logging dependency.
  # This is surprisingly annoying to do - Python's standard library can read
  # toml but not write it. Doing multiline sed replacements is awkward.
  # Adding a dependency on perl would be annoying. awk can do it but it's an
  # annoying amount of code.
  sed -i 's/"safari-sdk-logging"/#"safari-sdk-logging"/' ${SAFARI_DIR}/pyproject.toml
else
  echo "NotImplementedError: safari-sdk-logging has dependencies. These are not"
  echo "in the pypi release yet and would be incorrectly ignored when building"
  echo "the requirements.txt file."
  exit 1
fi

# Update requirements.txt file from pyproject.toml.
echo -e "\e[36mGenerating requirements file... (This may take a few minutes.)\e[0m"
pip-compile --upgrade --generate-hashes --allow-unsafe --all-extras \
  ${SAFARI_DIR}/pyproject.toml --output-file=${SAFARI_DIR}/requirements.txt
# Remove any index-url lines from requirements.txt.
sed -i '/^--index-url/d' "${SAFARI_DIR}/requirements.txt"
echo "Saved to ${SAFARI_DIR}/requirements.txt"
deactivate

# Restore pyproject.yaml
if [[ -f ${SAFARI_DIR}/pyproject.toml.no-cc-logging ]]; then
  echo "Restoring pyproject.yaml"
  mv ${SAFARI_DIR}/pyproject.toml{.no-cc-logging,}
fi
