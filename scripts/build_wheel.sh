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

CMAKE_BUILD_DIR="${SAFARI_DIR}/build"  # cmake build directory
mkdir -p "${CMAKE_BUILD_DIR}"
cd "${CMAKE_BUILD_DIR}"
cmake "${SAFARI_DIR}"
make pip_wheel pip_install

# Smoke test generated package
echo "Start smoke test"
source "${CMAKE_BUILD_DIR}/safari_venv/bin/activate"

flywheel-cli help

python3 -c "from safari_sdk.logging.python import stream_logger"
python3 -c "from safari_sdk.model import saved_model_policy"
python3 -c "from safari_sdk.logging.python import mcap_episodic_logger"

deactivate

echo "Smoke test done."
echo Pip wheel is in ${SAFARI_DIR}/dist/safari_sdk-*-py3-none-any.whl

ln -fs `ls ${SAFARI_DIR}/dist/safari_sdk-*-py3-none-any.whl` /tmp/safari_sdk-lastbuild-py3-none-any.whl
