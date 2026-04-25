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

# This uses FetchContent_Declare with FIND_PACKAGE_ARGS. With that, CMake will
# try to find a pre-installed version of the dependencies, and if that fails,
# it will download and build the dependency. CMake can be forced to always
# download and build dependencies (therefor making the build more hermetic) by
# setting FETCHCONTENT_TRY_FIND_PACKAGE_MODE to NEVER.

cmake_minimum_required(VERSION 3.26)
# ASM language is required by one dependency when compiled in manylinux.
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX ASM)

set(PYBIND11_FINDPYTHON ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)  # Abseil needs at least C++17.

# Hide all the internal symbols.
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# As we pack a wheel and we need to enforce hermeticity we prefer to build the
# static libraries. We manually control our targets but this variable will
# affect our dependencies.
set(BUILD_SHARED_LIBS OFF)

include(GNUInstallDirs)

# Override the default installation directory for CMake. This will affect only
# the dependent targets as we will specify our own installation directory later.
# This is the only way to block dependent targets that do not expose options to
# install their artefacts (we will later remove them).
# Note that we need to define the single components and not the general install
# prefix as that will also affect our installation.
set(CMAKE_INSTALL_LIBDIR "deps/lib" CACHE PATH "" FORCE)
set(CMAKE_INSTALL_BINDIR "deps/bin" CACHE PATH "" FORCE)
set(CMAKE_INSTALL_INCLUDEDIR "deps/include" CACHE PATH "" FORCE)
set(CMAKE_INSTALL_DATAROOTDIR "deps/share" CACHE PATH "" FORCE)

# =============================================================================
# Dependencies
# =============================================================================

# Explicitly find Python. This is to avoid issues when PyBind tries to find it
# (incorrectly).
find_package(Python3 COMPONENTS Interpreter Development)

include(FetchContent)

# --- Abseil ---
# Always built from source to ensure absl::string_view = std::string_view.
# System abseil (e.g. gLinux's libabsl-dev) may be compiled with
# ABSL_OPTION_USE_STD_STRING_VIEW=0, making absl::string_view a polyfill type
# (absl::debian7::string_view) that is ABI-incompatible with pybind11 wrappers
# and protobuf. This is a build config choice, not a version issue.
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)

FetchContent_Declare(
  abseil-cpp
  GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
  GIT_TAG 20250814.1
  OVERRIDE_FIND_PACKAGE
)

# --- Protobuf ---
# Force protobuf build options before fetching the dependency.
# CACHE BOOL "" FORCE ensures these propagate to the protobuf sub-project.
set(protobuf_INSTALL OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

# Always build protobuf from source using OVERRIDE_FIND_PACKAGE to intercept
# all find_package(Protobuf) calls (including from dependencies like
# pybind11_protobuf). This prevents system protobuf cmake configs from loading,
# which would create IMPORTED targets that conflict with the source build.
FetchContent_Declare(
  Protobuf
  URL https://github.com/protocolbuffers/protobuf/archive/refs/tags/v32.1.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  OVERRIDE_FIND_PACKAGE
)

# --- PyBind11 ---
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v3.0.1
  OVERRIDE_FIND_PACKAGE
)

# --- PyBind11 Abseil ---
FetchContent_Declare(
  pybind11_abseil
  GIT_REPOSITORY https://github.com/pybind/pybind11_abseil.git
  GIT_TAG c55fdc9c53d26af70fa8c2314a683abef62fa3f0  # 25 Feb 2025.
  OVERRIDE_FIND_PACKAGE
)
# pybind11_abseil internally forces interprocedural optimization to off. But
# this is not propagated up to the top level CMakeLists.txt, which means that
# there will be incompatibility between our targets and the one defined by
# pybind11_abseil.
# We force it to on here.
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON CACHE BOOL "" FORCE)
set(PYBIND11_PROTOBUF_BUILD_TESTING OFF)

# --- PyBind11 Protobuf ---
FetchContent_Declare(
  pybind11_protobuf
  GIT_REPOSITORY https://github.com/pybind/pybind11_protobuf.git
  GIT_TAG 4825dca68c8de73f5655fc50ce79c49c4d814652  # 29 Oct 2025.
  OVERRIDE_FIND_PACKAGE
)

# --- MCAP (header-only, we create our own target below) ---
FetchContent_Declare(
  mcap
  GIT_REPOSITORY https://github.com/foxglove/mcap.git
  GIT_TAG releases/mcap-cli/v0.0.60
  OVERRIDE_FIND_PACKAGE
)

# --- LZ4 ---
FetchContent_Declare(
  lz4
  URL https://github.com/lz4/lz4/archive/refs/tags/v1.10.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SOURCE_SUBDIR  build/cmake
  OVERRIDE_FIND_PACKAGE
)

# --- Zstd ---
set(ZSTD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_STATIC ON CACHE BOOL "" FORCE)

FetchContent_Declare(
  zstd
  URL https://github.com/facebook/zstd/archive/refs/tags/v1.5.7.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SOURCE_SUBDIR build/cmake
  OVERRIDE_FIND_PACKAGE
)

# --- OpenCV (minimal: core, imgproc, imgcodecs only) ---
set(WITH_WEBP OFF CACHE BOOL "" FORCE)
set(WITH_TIFF OFF CACHE BOOL "" FORCE)
set(WITH_IPP OFF CACHE BOOL "" FORCE)
set(WITH_V4L OFF CACHE BOOL "" FORCE)
set(WITH_PROTOBUF OFF CACHE BOOL "" FORCE)
set(WITH_IMGCODEC_GIF OFF CACHE BOOL "" FORCE)
set(WITH_IMGCODEC_HDR OFF CACHE BOOL "" FORCE)
set(WITH_IMGCODEC_SUNRASTER OFF CACHE BOOL "" FORCE)
set(WITH_IMGCODEC_PXM OFF CACHE BOOL "" FORCE)
set(WITH_IMGCODEC_PFM OFF CACHE BOOL "" FORCE)
set(WITH_FLATBUFFERS OFF CACHE BOOL "" FORCE)
set(BUILD_JAVA OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_apps OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_PERF_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_python3 OFF CACHE BOOL "" FORCE)
set(INSTALL_PYTHON_EXAMPLES OFF CACHE BOOL "" FORCE)
set(INSTALL_C_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_world OFF CACHE BOOL "" FORCE)
set(BUILD_LIST "core,imgproc,imgcodecs" CACHE STRING "Selected OpenCV modules" FORCE)
set(OPENCV_SETUPVARS_INSTALL_PATH "${CMAKE_INSTALL_BINDIR}" CACHE PATH "" FORCE)

FetchContent_Declare(
  OpenCV
  URL https://github.com/opencv/opencv/archive/refs/tags/4.13.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  OVERRIDE_FIND_PACKAGE
)

# --- TensorFlow (only 2 proto files needed, download directly) ---
set(_TF_PROTO_DIR "${CMAKE_CURRENT_BINARY_DIR}/_tf_protos/tensorflow/core/example")
file(MAKE_DIRECTORY ${_TF_PROTO_DIR})

set(_TF_RAW_URL "https://raw.githubusercontent.com/tensorflow/tensorflow/HEAD/tensorflow/core/example")
file(DOWNLOAD "${_TF_RAW_URL}/example.proto" "${_TF_PROTO_DIR}/example.proto")
file(DOWNLOAD "${_TF_RAW_URL}/feature.proto" "${_TF_PROTO_DIR}/feature.proto")

# --- Make available ---
FetchContent_MakeAvailable(abseil-cpp Protobuf
                           pybind11 pybind11_abseil pybind11_protobuf
                           OpenCV lz4 zstd mcap)


# =============================================================================
# Proto Libraries
# =============================================================================

# When fetched from source, Protobuf_INCLUDE_DIRS isn't set (it's a
# FindProtobuf module variable). Set it so protoc can find the standard
# protos (e.g. google/protobuf/duration.proto).
if(protobuf_SOURCE_DIR AND NOT Protobuf_INCLUDE_DIRS)
  set(Protobuf_INCLUDE_DIRS "${protobuf_SOURCE_DIR}/src")
  # When built from source, the protobuf_generate() function isn't loaded
  # automatically. Include the module that defines it.
  include("${protobuf_SOURCE_DIR}/cmake/protobuf-generate.cmake")
  # Create an imported target for the pre-downloaded protoc binary so that
  # protobuf_generate() can find it (we skipped building protoc from source).
  if(NOT TARGET protobuf::protoc)
    add_executable(protobuf::protoc IMPORTED)
    set_target_properties(protobuf::protoc PROPERTIES
      IMPORTED_LOCATION "${_PROTOC_BINARY}")
  endif()
endif()

# Python proto library for Safari protos.
file(GLOB_RECURSE _SAFARI_PROTO_FILES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/safari_sdk/protos/*.proto")

# Generate Python proto files directly into the source tree so that
# scikit-build-core's packages = ["safari_sdk"] picks them up automatically.
protobuf_generate(LANGUAGE python
                  PROTOS ${_SAFARI_PROTO_FILES}
                  OUT_VAR _SAFARI_PB_GENERATED_FILES
                  IMPORT_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${Protobuf_INCLUDE_DIRS}
                  PROTOC_OUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}
)
add_custom_target(safari_py_protos ALL DEPENDS ${_SAFARI_PB_GENERATED_FILES})

# C++ proto library for Safari protos.
add_library(safari_protos STATIC)
protobuf_generate(
  TARGET safari_protos
  LANGUAGE cpp
  PROTOS ${_SAFARI_PROTO_FILES}
  IMPORT_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${Protobuf_INCLUDE_DIRS}
  )
target_link_libraries(safari_protos protobuf::libprotobuf)
target_include_directories(safari_protos PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

# TF example and feature protos.
set(_TF_PROTO_ROOT "${CMAKE_CURRENT_BINARY_DIR}/_tf_protos")
add_library(tf_protos STATIC)
protobuf_generate(
  TARGET tf_protos
  LANGUAGE cpp
  PROTOS ${_TF_PROTO_ROOT}/tensorflow/core/example/example.proto
         ${_TF_PROTO_ROOT}/tensorflow/core/example/feature.proto
  IMPORT_DIRS ${_TF_PROTO_ROOT}
  )
target_link_libraries(tf_protos protobuf::libprotobuf)
target_include_directories(tf_protos PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

# =============================================================================
# MCAP Library
# =============================================================================

set(_mcap_include_dir ${mcap_SOURCE_DIR}/cpp/mcap/include)
add_library(mcap_lib STATIC)
target_include_directories(mcap_lib PUBLIC
  "$<BUILD_INTERFACE:${_mcap_include_dir}>"
  "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>"
)
# mcap.cpp is accessed via the safari_sdk symlink.
target_sources(mcap_lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/safari_sdk/logging/cc/mcap.cpp)

# When lz4 is found via find_package, the target is lz4::lz4; when fetched from
# source, the target is lz4. Similarly, zstd system package provides
# zstd::libzstd_static while the source build provides libzstd_static.
if(TARGET lz4::lz4)
  set(_LZ4_TARGET lz4::lz4)
else()
  set(_LZ4_TARGET lz4)
endif()
if(TARGET zstd::libzstd_static)
  set(_ZSTD_TARGET zstd::libzstd_static)
elseif(TARGET libzstd_static)
  set(_ZSTD_TARGET libzstd_static)
else()
  set(_ZSTD_TARGET zstd::libzstd_shared)
endif()

target_link_libraries(mcap_lib ${_LZ4_TARGET} ${_ZSTD_TARGET})

# =============================================================================
# Log Writer C++ Library
# =============================================================================

# All C++ source files are accessed via the safari_sdk symlink.
add_library(log_writer_cc STATIC)
target_sources(log_writer_cc PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/safari_sdk/logging/cc/base_mcap_file_handle_factory.h
  ${CMAKE_CURRENT_SOURCE_DIR}/safari_sdk/logging/cc/episode_data.h
  ${CMAKE_CURRENT_SOURCE_DIR}/safari_sdk/logging/cc/episode_data.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/safari_sdk/logging/cc/log_data_serializer_utils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/safari_sdk/logging/cc/log_data_serializer_utils.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/safari_sdk/logging/cc/mcap_file_handle_factory.h
  ${CMAKE_CURRENT_SOURCE_DIR}/safari_sdk/logging/cc/mcap_file_handle_factory.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/safari_sdk/logging/cc/mcap_file_handle.h
  ${CMAKE_CURRENT_SOURCE_DIR}/safari_sdk/logging/cc/mcap_file_handle.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/safari_sdk/logging/cc/mcap_write_op.h
  ${CMAKE_CURRENT_SOURCE_DIR}/safari_sdk/logging/cc/thread_pool_log_writer.h
  ${CMAKE_CURRENT_SOURCE_DIR}/safari_sdk/logging/cc/thread_pool_log_writer.cc
)

# When OpenCV is found via find_package, module include dirs are provided by the
# imported targets. When built from source, we need to add them explicitly.
if(DEFINED OPENCV_MODULE_opencv_core_LOCATION)
  target_include_directories(log_writer_cc PUBLIC
                             ${OPENCV_MODULE_opencv_core_LOCATION}/include
                             ${OPENCV_MODULE_opencv_imgcodecs_LOCATION}/include
                             ${OPENCV_MODULE_opencv_imgproc_LOCATION}/include)
endif()
target_include_directories(log_writer_cc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(log_writer_cc PUBLIC safari_protos
                                        absl::core_headers
                                        absl::any_invocable
                                        absl::status
                                        absl::statusor
                                        absl::strings
                                        absl::string_view
                                        absl::span
                                        absl::log
                                        absl::flat_hash_map
                                        absl::fixed_array
                                        absl::flat_hash_set
                                        absl::memory
                                        absl::synchronization
                                        absl::time
                                        opencv_core
                                        opencv_imgcodecs
                                        opencv_imgproc
                                        mcap_lib
                                        protobuf::libprotobuf
                                        tf_protos)

# =============================================================================
# Python Bindings
# =============================================================================

pybind11_add_module(log_writer ${CMAKE_CURRENT_SOURCE_DIR}/safari_sdk/logging/cc/python/log_writer.cc)
target_link_libraries(log_writer PUBLIC log_writer_cc
                                        safari_protos
                                        absl::log
                                        absl::status
                                        absl::statusor
                                        absl::strings
                                        absl::string_view
                                        absl::absl_check
                                        pybind11_abseil::absl_casters
                                        pybind11_abseil::status_casters
                                        pybind11_protobuf::pybind11_native_proto_caster
                                        )

# =============================================================================
# Installation
# =============================================================================

# As we build with static libraries we do not need to install dependencies nor
# to update the RPATH.
# Install into the safari_sdk namespace (hardcoded, not ${SKBUILD_PROJECT_NAME}).
install(TARGETS log_writer DESTINATION safari_sdk/logging/cc/python)

# TODO: At the moment pybind11_abseil does not provide a Pip package.
# We need to install the targets manually. Note that this might conflict with
# other library that might be installed by the user.
install(TARGETS status_py_extension_stub ok_status_singleton DESTINATION pybind11_abseil)

# Now remove the dependencies that we installed but we do not need.
install(CODE "
    message(STATUS \"Removing dependencies installation artifacts...\")
    file(REMOVE_RECURSE \"\${CMAKE_INSTALL_PREFIX}/deps\")
")
