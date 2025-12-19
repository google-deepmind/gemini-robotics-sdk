#!/bin/bash
# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Fail on any error.
set -e
SAFARI_DIR="$(realpath "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..")"
VENV_DIR="$(mktemp -d)"
UPLOAD_TARGET=""
UPLOAD_WHL=false
RUN_SMOKE_TEST=true

function _usage() {
  echo "Usage: $0 [-h|--help] [--no-smoke-test] [--upload]"
  echo "  -h|--help: Show this help message and exit."
  echo "  --no-smoke-test: Skip the smoke test."
  echo "  --repository-url: Upload the wheel to this repository URL."
  echo "  --upload: Upload the wheel to PyPi."
}

while (( $# > 0 )) ; do
  case "$1" in
    -h|--help) _usage; exit 1 ;;
    --upload) UPLOAD_WHL=true ; shift ;;
    --repository-url) UPLOAD_TARGET="--repository-url $2"; UPLOAD_WHL=true; shift 2 ;;
    --no-smoke-test) RUN_SMOKE_TEST=false; shift ;;
    *) echo "Unknown option: $1"; _usage; exit 1 ;;
  esac
done

echo "Building wheel in ${SAFARI_DIR} with venv: ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
pip install build
python3 -m build ${SAFARI_DIR}
echo Pip wheel is in ${SAFARI_DIR}/dist/*.whl

# Smoke test generated package if desired.
if ${RUN_SMOKE_TEST}; then
  echo "Start smoke test"
  pip install ${SAFARI_DIR}/dist/*.whl

  flywheel-cli help
  python3 -c "from safari_sdk.logging.python import stream_logger"
  python3 -c "from safari_sdk.ui import client"
  python3 -c "from safari_sdk.logging.python import episodic_logger"

  echo "Smoke test done."
fi

# Upload the wheel to gcloud or PyPI if desired.
if [[ ${UPLOAD_TARGET} == *python.pkg.dev* ]]; then
  # Install the keyrings to allow authentication with Artifact Registry.
  pip install keyrings.google-artifactregistry-auth
  echo "Uploading whl to gCloud (${UPLOAD_TARGET})."
elif [[ -n "${UPLOAD_TARGET}" ]]; then
  echo "Uploading whl to ${UPLOAD_TARGET}."
elif ${UPLOAD_WHL}; then
  echo "Uploading whl to PyPI."
fi

if ${UPLOAD_WHL}; then
  pip install twine
  twine upload --verbose ${SAFARI_DIR}/dist/*.whl ${UPLOAD_TARGET}
fi
