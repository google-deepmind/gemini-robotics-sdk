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

function _usage() {
  echo "Usage: $0 [-h|--help] [--no-smoke-test] [--upload]"
  echo "  -h|--help: Show this help message and exit."
  echo "  --all-python-versions: Build the wheel for all supported python"
  echo "    versions (3.11, 3.12, and 3.13)."
  echo "  --python-version <VERSION>: Build the wheel for this python version."
  echo "  --docker-image <IMAGE>: Build the wheel inside this docker <IMAGE>."
  echo "  --build-docker-image: Build a docker image from"
  echo "    kokoro/gcp_ubuntu_docker/Dockerfile and buil the wheel inside it."
  echo "  --repository-url <URL>: Upload the wheel to this repository <URL>."
  echo "  --upload: Upload the wheel to PyPI."
  echo ""
  echo "If either --all-python-versions or --python-version is specified,"
  echo "  using --docker-image or --build-docker-image is recommended, to"
  echo "  ensure the required python version(s) are available."
  echo "If neither --all-python-versions nor --python-version is specified,"
  echo "  the system python version will be used."
}

# Handle the docker-related arguments. All other arguments are handled later.
SCRIPTS_DIR="$(realpath "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")"
SAFARI_DIR="$(realpath "${SCRIPTS_DIR}/..")"
PYTHON_VERSIONS=("3")
DOCKER_IMAGE=""
BUILD_DOCKER_IMAGE=false
UPLOAD_TARGET=()
UPLOAD_WHL=false
# Arguments that are passed through to the docker image.
PASS_THROUGH_ARGS=()

while (( $# > 0 )) ; do
  case "$1" in
    -h|--help) _usage; exit 1 ;;
    --all-python-versions)
      PYTHON_VERSIONS=("3.10" "3.11" "3.12" "3.13")
      PASS_THROUGH_ARGS+=("$1")
      shift ;;
    --python-version)
      PYTHON_VERSIONS=("$2")
      PASS_THROUGH_ARGS+=("$1" "$2")
      shift 2 ;;
    --repository-url)
      UPLOAD_WHL=true
      UPLOAD_TARGET=("$1" "$2")
      PASS_THROUGH_ARGS+=("$1" "$2")
      shift 2 ;;
    --upload)
      UPLOAD_WHL=true
      PASS_THROUGH_ARGS+=("$1")
      shift ;;
    # Docker-related arguments do not pass through into the docker container.
    --docker-image)
      DOCKER_IMAGE=$2;
      shift 2 ;;
    --build-docker-image)
      BUILD_DOCKER_IMAGE=true;
      shift ;;
    *) echo "Unknown option: $1"; _usage; exit 1 ;;
  esac
done

if ${BUILD_DOCKER_IMAGE}; then
  if [[ -z "${DOCKER_IMAGE}" ]]; then
    DOCKER_IMAGE="safari-sdk-wheel-build:latest"
  fi
  docker build -t "${DOCKER_IMAGE}" "${SAFARI_DIR}"
fi
if [[ -n "${DOCKER_IMAGE}" ]]; then
  echo "*** Building wheel inside of docker image ${DOCKER_IMAGE}"
  docker_args=("--rm" "-u" "$(id -u):$(id -g)" "-v" "${SAFARI_DIR}:/safari")
  if [[ "${UPLOAD_TARGET[1]}" == *python.pkg.dev* ]]; then
    adc_path=".config/gcloud/application_default_credentials.json"
    docker_args+=("-e" "HOME=/home/user" "-v" "${HOME}/${adc_path}:/home/user/${adc_path}:ro")
  fi
  docker run "${docker_args[@]}" "${DOCKER_IMAGE}" \
    /safari/scripts/build_wheel.sh "${PASS_THROUGH_ARGS[@]}"
  echo "*** Exited the docker image."
  exit 0
fi

# Clean out old wheels.
rm -rf "${SAFARI_DIR}"/wheelhouse/*

# Generate the revision info file. This is a no-op if not built from a git repository.
if [[ -f ${SAFARI_DIR}/scripts/generate_revision_info.sh ]]; then
  ${SAFARI_DIR}/scripts/generate_revision_info.sh "${SAFARI_DIR}/safari_sdk/revision_info.txt" || true
fi

# Build each python version.
for py_version in "${PYTHON_VERSIONS[@]}"; do
  PY_EXEC="python${py_version}"
  VENV_DIR="$(mktemp -d)"
  echo "*** Building wheels for $(${PY_EXEC} --version) in ${VENV_DIR}"
  "${PY_EXEC}" -m venv "${VENV_DIR}"
  source "${VENV_DIR}/bin/activate"
  pip install --require-hashes -r "${SCRIPTS_DIR}/base_tooling_requirements.txt"
  mkdir -p "${SAFARI_DIR}/wheelhouse"
  python -m build --wheel "${SAFARI_DIR}" --outdir "${SAFARI_DIR}/dist"

  # auditwheel repair: Converts 'linux_x86_64' wheels into portable
  # 'manylinux' wheels by bundling external shared library dependencies
  # (.so) and updating RPATHs. Pure-Python wheels need no repair.
  whl="$(ls "${SAFARI_DIR}"/dist/*.whl)"
  if ! auditwheel repair "${whl}" -w "${SAFARI_DIR}/wheelhouse"; then
    echo "*** auditwheel could not repair $(basename "${whl}"), copying from dist/ directly."
    cp "${whl}" "${SAFARI_DIR}/wheelhouse/"
  fi
  rm -rf "${SAFARI_DIR}/dist"

  # Deactivate the venv.
  deactivate
done

echo "*** Wheel files are in the wheelhouse folder:"
ls -l "${SAFARI_DIR}/wheelhouse"

if ${UPLOAD_WHL}; then
  VENV_DIR="$(mktemp -d)"
  echo "*** Uploading wheels using venv in ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
  source "${VENV_DIR}/bin/activate"
  pip install --require-hashes -r "${SCRIPTS_DIR}/base_tooling_requirements.txt"

  echo "*** Uploading wheels to ${UPLOAD_TARGET[1]:-PyPI}."
  twine upload --verbose ${SAFARI_DIR}/wheelhouse/*.whl "${UPLOAD_TARGET[@]}"
  deactivate
fi
