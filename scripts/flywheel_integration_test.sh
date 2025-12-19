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

# ==============================================================================
# This script is used to test the flywheel-cli.
# ==============================================================================

# Fail on any error.
set -e

if [[ ! -d ${KOKORO_KEYSTORE_DIR} ]]; then
  echo "KOKORO_KEYSTORE_DIR is not set."
  exit 1
fi

API_KEY=$(cat ${KOKORO_KEYSTORE_DIR}/78388_robotics-api-test-consumer_api_key)
START_TIME=$(date +%s)

# Install a newer version of protoc
# Must be protoc 25.4 or earlier to support Python protobuf library v4, which
# is required by TF 2.15.
# See https://protobuf.dev/support/version-support/#python
dir="$(mktemp -d)"
trap "rm -rf '$dir'" EXIT
cd "$dir"
wget https://github.com/protocolbuffers/protobuf/releases/download/v25.4/protoc-25.4-linux-x86_64.zip
unzip protoc-25.4-linux-x86_64.zip
cp bin/protoc /usr/bin/protoc

# Code under repo is checked out to ${KOKORO_ARTIFACTS_DIR}/git.
# The final directory name in this path is determined by the scm name specified
# in the job configuration.
cd "${KOKORO_ARTIFACTS_DIR}/git/safari"

protoc -I=./ --python_out=./ ./safari_sdk/protos/*.proto safari_sdk/protos/ui/*.proto safari_sdk/protos/logging/*.proto

echo "Installing Python 3.11"
pyenv install "3.11"
pyenv global "3.11"

echo "Installing flywheel-cli"
pip install .
apt install -y expect

echo "Testing flywheel-cli"
flywheel-cli help --api_key=${API_KEY}
flywheel-cli version --api_key=${API_KEY}
flywheel-cli list --api_key=${API_KEY}
flywheel-cli data_stats --api_key=${API_KEY}

echo "Training a model"
TRAINING_JOB_ID=$(flywheel-cli train \
  --api_key=${API_KEY} \
  --task_id=putaway_cookware \
  --start_date=20250324 \
  --end_date=20250324 \
  | jq -r '.trainingJobId')

echo "Training job ID: ${TRAINING_JOB_ID}"
echo "Waiting 3 hours before checking on the training job status."
sleep 10800 # 3 hours
TRAINING_COMPLETE=false
CHECKS=0

while [[ ${TRAINING_COMPLETE} == false ]]; do
  sleep 300
  CHECKS=$((CHECKS + 1))
  echo "Checking on the training job status, check ${CHECKS}."
  LIST=$(flywheel-cli list \
    --api_key=${API_KEY} \
    --json_output)
  LATEST_JOB=$(echo "$LIST" | jq '.trainingJobs[0]')
  JOB_STATUS=$(echo "${LATEST_JOB}" | jq -r '.stage')
  echo "JOB_STATUS: ${JOB_STATUS}"

  if [[ ${JOB_STATUS} == "TRAINING_JOB_STAGE_DONE" ]]; then
    TIME_ELAPSED=$(($(date +%s) - ${START_TIME}))
    echo "Training job ${TRAINING_JOB_ID} is done."
    echo "It took ${TIME_ELAPSED} seconds."
    TRAINING_COMPLETE=true
  elif [[ ${JOB_STATUS} == "TRAINING_JOB_STAGE_FAILED" ]]; then
    echo "Training job ${TRAINING_JOB_ID} failed."
    exit 1
  else
    echo "Training job ${TRAINING_JOB_ID} is still running" \
      "at $(date +%H:%M:%S)."
  fi
done

echo "Downloading the model"
mkdir ${KOKORO_ARTIFACTS_DIR}/flywheel-cli-download
cd $KOKORO_ARTIFACTS_DIR/flywheel-cli-download

expect << EOF
set timeout 60
spawn flywheel-cli download --api_key=${API_KEY} --training_job_id=${TRAINING_JOB_ID}
expect "Enter artifact # to download (or press Enter to skip):"
send "0\\r"
expect "Save artifact as (default:"
send "\\r"
expect {
  "Overwrite? (y/n):" {
    send "y\\r"
    exp_continue
  }
  "Download complete!" {
    # Do nothing, just wait for eof
  }
  eof
}
EOF
