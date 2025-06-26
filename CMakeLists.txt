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

# Cmake file for building the Safari SDK.

# Example usage (starting in this directory):
# mkdir build && cd build && cmake .. && make all
# when rebuilding, only `make all` is necessary.

# Version 3.16 ships with Ubuntu 22.04, which is required by ROS noetic.
cmake_minimum_required(VERSION 3.16)
project(safari_sdk)
include(cmake/protobuf-generate.cmake)
include(cmake/python-packaging.cmake)
include(cmake/safari-targets.cmake)

# Generate the python protobufs and add them to the 'all' build target.
# Ouput files are placed in the source tree, along side their corresponding
# proto files. They can be removed with `make clean`.
file(GLOB_RECURSE _PROTO_FILES CONFIGURE_DEPENDS "safari_sdk/protos/*.proto")
protobuf_generate(
  PROTOS ${_PROTO_FILES}
  PROTOC_OUT_DIR ${CMAKE_SOURCE_DIR}
  LANGUAGE python
  OUT_VAR _PB_GENERATED_FILES
)
add_custom_target(py_proto ALL
  DEPENDS ${_PB_GENERATED_FILES}
)

# Define the properies needed to build and/or install the safari package.
add_pip_source_package_info(
  TARGET_NAME safari_sdk_source_package
  PACKAGE_NAME safari_sdk
  IMPORT_ROOT_DIR safari_sdk
  DEPENDS ${_PB_GENERATED_FILES}
  EDITABLE_INSTALL_OUTPUT safari_sdk.pth
  PY_FILE_GLOB
)

# Add targets to install the package from source in regular and editable modes.
# TODO(b/413020601#comment2): Which method should be used for installation?
add_pip_install(
  TARGET_NAME pip_install
  VENV_NAME safari_venv
  PACKAGE safari_sdk_source_package
)
add_pip_editable_install(
  TARGET_NAME pip_install_e
  VENV_NAME safari_venv
  PACKAGE safari_sdk_source_package
)

# Add a target to build a pip wheel.
add_pip_wheel(
  TARGET_NAME pip_wheel
  VENV_NAME safari_venv
  PACKAGE safari_sdk_source_package
)
