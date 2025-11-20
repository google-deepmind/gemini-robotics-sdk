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

# Cmake file for building the Safari SDK protobuf messages. This is invoked by
# pip when it builds the safari_sdk package; it should not be invoked directly.

cmake_minimum_required(VERSION 3.16)
project(safari_sdk)
include(cmake/protobuf-generate.cmake)

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
