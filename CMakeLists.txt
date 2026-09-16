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

cmake_minimum_required(VERSION 3.24)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX ASM)

# =============================================================================
# Compiler Configuration & Ccache Integration
# =============================================================================

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    message(STATUS "Found ccache: ${CCACHE_PROGRAM}. Build will be cached.")
else()
    message(STATUS "ccache not found. Proceeding with standard compilation.")
endif()

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

# =============================================================================
# Dependencies Setup
# =============================================================================

find_package(Python3 COMPONENTS Interpreter Development)
include(FetchContent)

if(DEFINED ENV{GDMR_BUILD_DIR})
  set(GDMR_CACHE_DIR "$ENV{GDMR_BUILD_DIR}/cache")
  # FetchContent will look for cached downloads in this directory.
  # Rather than the ephemeral build directory created by skbuild.
  set(FETCHCONTENT_BASE_DIR "$ENV{GDMR_BUILD_DIR}/fetchcontent")
else()
  set(GDMR_CACHE_DIR "${CMAKE_BINARY_DIR}/cache")
endif()

set(GDMR_SDK_WARMUP ON)
set(CMAKE_INSTALL_PREFIX "${GDMR_CACHE_DIR}" CACHE PATH "Warmup cache install directory" FORCE)
message(STATUS "Warmup Mode ON. Dependencies will be installed to: ${CMAKE_INSTALL_PREFIX}")


# =============================================================================
# External Dependencies
# =============================================================================

# --- Abseil ---
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)
set(ABSL_ENABLE_INSTALL ${GDMR_SDK_WARMUP} CACHE BOOL "" FORCE)
FetchContent_Declare(
  abseil-cpp
  URL https://github.com/abseil/abseil-cpp/archive/refs/tags/20250814.1.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  FIND_PACKAGE_ARGS NAMES absl QUIET CONFIG PATHS ${GDMR_CACHE_DIR} NO_DEFAULT_PATH
)

# --- Protobuf ---
# Force protobuf build options before fetching the dependency.
set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(UTF8_RANGE_INSTALL ${GDMR_SDK_WARMUP} CACHE BOOL "" FORCE)
set(protobuf_INSTALL ${GDMR_SDK_WARMUP} CACHE BOOL "" FORCE)

# When building protobuf from source (because the system version is too old),
# skip the expensive protoc compiler build and use a pre-built binary instead.
# This saves ~5-10 minutes of build time.
set(protobuf_BUILD_PROTOC_BINARIES OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  protobuf
  URL https://github.com/protocolbuffers/protobuf/archive/refs/tags/v32.1.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  FIND_PACKAGE_ARGS NAMES protobuf QUIET CONFIG PATHS ${GDMR_CACHE_DIR} NO_DEFAULT_PATH
)

# Download a pre-built protoc binary matching our source version.
set(_PROTOC_VERSION "32.1")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(_PROTOC_PLATFORM "linux-x86_64")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
    set(_PROTOC_PLATFORM "linux-aarch_64")
  endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(_PROTOC_PLATFORM "osx-x86_64")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(_PROTOC_PLATFORM "osx-aarch_64")
  endif()
endif()

set(_PROTOC_DIR "${CMAKE_CURRENT_BINARY_DIR}/_protoc")
set(_PROTOC_BINARY "${_PROTOC_DIR}/bin/protoc")

if(NOT EXISTS "${_PROTOC_BINARY}")
  set(_PROTOC_ZIP "${_PROTOC_DIR}/protoc-${_PROTOC_VERSION}-${_PROTOC_PLATFORM}.zip")
  file(DOWNLOAD
    "https://github.com/protocolbuffers/protobuf/releases/download/v${_PROTOC_VERSION}/protoc-${_PROTOC_VERSION}-${_PROTOC_PLATFORM}.zip"
    "${_PROTOC_ZIP}")
  file(ARCHIVE_EXTRACT INPUT "${_PROTOC_ZIP}" DESTINATION "${_PROTOC_DIR}")
  file(CHMOD "${_PROTOC_BINARY}" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)
endif()

# --- PyBind11 ---
set(PYBIND11_INSTALL ${GDMR_SDK_WARMUP} CACHE BOOL "" FORCE)
set(PYBIND11_PROTOBUF_ENABLE_INSTALL ${GDMR_SDK_WARMUP} CACHE BOOL "" FORCE)
FetchContent_Declare(
  pybind11
  URL https://github.com/pybind/pybind11/archive/refs/tags/v3.0.1.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  FIND_PACKAGE_ARGS QUIET CONFIG PATHS ${GDMR_CACHE_DIR} NO_DEFAULT_PATH
)

# --- PyBind11 Abseil ---
set(PYBIND11_ABSEIL_ENABLE_INSTALL ${GDMR_SDK_WARMUP} CACHE BOOL "" FORCE)
FetchContent_Declare(
  pybind11_abseil
  URL https://github.com/pybind/pybind11_abseil/archive/c55fdc9c53d26af70fa8c2314a683abef62fa3f0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  FIND_PACKAGE_ARGS QUIET CONFIG PATHS ${GDMR_CACHE_DIR} NO_DEFAULT_PATH
)
# pybind11_abseil internally forces interprocedural optimization to off. But
# this is not propagated up to the top level CMakeLists.txt, which means that
# there will be incompatibility between our targets and the one defined by
# pybind11_abseil. We force it to on here.
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON CACHE BOOL "" FORCE)
set(PYBIND11_PROTOBUF_BUILD_TESTING OFF)

# --- PyBind11 Protobuf ---
FetchContent_Declare(
  pybind11_protobuf
  URL https://github.com/pybind/pybind11_protobuf/archive/4825dca68c8de73f5655fc50ce79c49c4d814652.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  FIND_PACKAGE_ARGS QUIET CONFIG PATHS ${GDMR_CACHE_DIR} NO_DEFAULT_PATH
)

# --- MCAP (header-only, we create our own target below) ---
FetchContent_Declare(
  mcap
  URL https://github.com/foxglove/mcap/archive/refs/tags/releases/mcap-cli/v0.0.60.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

# --- LZ4 ---
FetchContent_Declare(
  lz4
  URL https://github.com/lz4/lz4/archive/refs/tags/v1.10.0.tar.gz
  SOURCE_SUBDIR  build/cmake
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
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
  SOURCE_SUBDIR build/cmake
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

# --- TensorFlow Protos ---
# Instead of downloading the 200MB+ tarball,
# just grab the two raw proto files we need.
set(TF_PROTO_DIR "${CMAKE_CURRENT_BINARY_DIR}/tensorflow_protos")

# Create the directory structure
# so protoc can resolve the internal package paths properly
file(MAKE_DIRECTORY "${TF_PROTO_DIR}/tensorflow/core/example")

if(NOT EXISTS "${TF_PROTO_DIR}/tensorflow/core/example/example.proto")
  file(DOWNLOAD
      "https://raw.githubusercontent.com/tensorflow/tensorflow/v2.19.1/tensorflow/core/example/example.proto"
      "${TF_PROTO_DIR}/tensorflow/core/example/example.proto"
      SHOW_PROGRESS
  )
endif()

if(NOT EXISTS "${TF_PROTO_DIR}/tensorflow/core/example/feature.proto")
  file(DOWNLOAD
      "https://raw.githubusercontent.com/tensorflow/tensorflow/v2.19.1/tensorflow/core/example/feature.proto"
      "${TF_PROTO_DIR}/tensorflow/core/example/feature.proto"
      SHOW_PROGRESS
  )
endif()

# --- OpenCV (minimal: core, imgproc, imgcodecs only) ---
enable_language(ASM_NASM)
set(WITH_OPENEXR OFF CACHE BOOL "" FORCE)
set(WITH_EIGEN OFF CACHE BOOL "" FORCE)
set(WITH_FFMPEG OFF CACHE BOOL "" FORCE)
set(WITH_GSTREAMER OFF CACHE BOOL "" FORCE)
set(WITH_ADE OFF CACHE BOOL "" FORCE)
set(WITH_ITT OFF CACHE BOOL "" FORCE)
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
  opencv
  URL https://github.com/opencv/opencv/archive/refs/tags/4.13.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  FIND_PACKAGE_ARGS NAMES OpenCV QUIET CONFIG PATHS ${GDMR_CACHE_DIR} NO_DEFAULT_PATH
)

# Retrieve and build all dependencies
FetchContent_MakeAvailable(abseil-cpp protobuf
                           pybind11 pybind11_abseil pybind11_protobuf
                           opencv lz4 zstd mcap)

# Create an imported target for the pre-downloaded protoc binary if we
# skipped building protoc from source so that protobuf_generate() can find it.
if(NOT TARGET protobuf::protoc)
  add_executable(protobuf::protoc IMPORTED)
  set_target_properties(protobuf::protoc PROPERTIES
    IMPORTED_LOCATION "${_PROTOC_BINARY}")
endif()

# =============================================================================
# Warmup Cache Population
# =============================================================================

# Install specific dependency targets that FetchContent misses
if(GDMR_SDK_WARMUP AND TARGET utf8_validity)
    install(TARGETS utf8_range utf8_validity
            EXPORT protobuf-targets
            DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

# Pip install via scikit-build-core overrides the install prefix locations.
# Thus, we need to manually copy them to the cache.
if(GDMR_SDK_WARMUP)
    install(CODE "
        message(STATUS \"Populating GDMR Cache: Copying dependencies to ${GDMR_CACHE_DIR}...\")
        foreach(dir IN ITEMS ${CMAKE_INSTALL_LIBDIR} ${CMAKE_INSTALL_INCLUDEDIR} ${CMAKE_INSTALL_DATADIR} ${CMAKE_INSTALL_BINDIR})
            if(EXISTS \"\${CMAKE_INSTALL_PREFIX}/\${dir}\")
                file(COPY \"\${CMAKE_INSTALL_PREFIX}/\${dir}\" DESTINATION \"${GDMR_CACHE_DIR}\")
            endif()
        endforeach()
    ")
endif()

# --- C++ and Python Proto Libraries ---
FetchContent_GetProperties(protobuf SOURCE_DIR protobuf_SOURCE_DIR)

if(protobuf_SOURCE_DIR)
    # Protobuf is being built from source.
    # We must manually include the script.
    include(${protobuf_SOURCE_DIR}/cmake/protobuf-generate.cmake)
    set(PROTOBUF_IMPORT_DIR "${protobuf_SOURCE_DIR}/src")
else()
    # Cache Mode: Protobuf was found via find_package().
    # The script is auto-included.
    set(PROTOBUF_IMPORT_DIR "${GDMR_CACHE_DIR}/include")
endif()

# Proto files are accessed via the safari_sdk symlink.
file(GLOB_RECURSE _SAFARI_PROTO_FILES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/safari_sdk/protos/*.proto")

# C++ Protos
add_library(safari_protos STATIC)
protobuf_generate(
  TARGET safari_protos
  LANGUAGE cpp
  PROTOS ${_SAFARI_PROTO_FILES}
  IMPORT_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${PROTOBUF_IMPORT_DIR}
)
target_link_libraries(safari_protos protobuf::libprotobuf)
target_include_directories(safari_protos PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

# Python Protos
# Generate Python proto files directly into the source tree so that
# scikit-build-core's packages = ["safari_sdk"] picks them up automatically.
protobuf_generate(LANGUAGE python
                  PROTOS ${_SAFARI_PROTO_FILES}
                  OUT_VAR _SAFARI_PB_GENERATED_FILES
                  IMPORT_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
                  PROTOC_OUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}
)
add_custom_target(safari_py_protos ALL DEPENDS ${_SAFARI_PB_GENERATED_FILES})

# TF example and feature protos.
FetchContent_GetProperties(tensorflow)
add_library(tf_protos STATIC)
protobuf_generate(
  TARGET tf_protos
  LANGUAGE cpp
  PROTOS "${TF_PROTO_DIR}/tensorflow/core/example/example.proto"
         "${TF_PROTO_DIR}/tensorflow/core/example/feature.proto"
  IMPORT_DIRS "${TF_PROTO_DIR}"
)
target_link_libraries(tf_protos protobuf::libprotobuf)
target_include_directories(tf_protos PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

# =============================================================================
# MCAP Library
# =============================================================================
set(_mcap_include_dir ${mcap_SOURCE_DIR}/cpp/mcap/include)
add_library(mcap STATIC)
target_include_directories(mcap PUBLIC
  "$<BUILD_INTERFACE:${_mcap_include_dir}>"
  "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>"
)
# mcap.cpp is accessed via the safari_sdk symlink.
target_sources(mcap PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/safari_sdk/logging/cc/mcap.cpp)
target_link_libraries(mcap lz4 libzstd_static)

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

target_include_directories(log_writer_cc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                          ${OPENCV_MODULE_opencv_core_LOCATION}/include
                          ${OPENCV_MODULE_opencv_imgcodecs_LOCATION}/include
                          ${OPENCV_MODULE_opencv_imgproc_LOCATION}/include)

target_link_libraries(log_writer_cc PUBLIC
    safari_protos
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
    mcap
    protobuf::libprotobuf
    tf_protos
)

# =============================================================================
# Python Bindings
# =============================================================================

pybind11_add_module(log_writer ${CMAKE_CURRENT_SOURCE_DIR}/safari_sdk/logging/cc/python/log_writer.cc)
target_link_libraries(log_writer PUBLIC
    log_writer_cc
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
# Installation & Wheel Cleanup
# =============================================================================

# As we build with static libraries we do not need to install dependencies nor
# to update the RPATH.
# Install into the safari_sdk namespace (hardcoded, not ${SKBUILD_PROJECT_NAME}).
install(TARGETS log_writer DESTINATION safari_sdk/logging/cc/python)

# TODO: At the moment pybind11_abseil does not provide a Pip package.
# We need to install the targets manually. Note that this might conflict with
# other library that might be installed by the user.
install(TARGETS status_py_extension_stub ok_status_singleton DESTINATION pybind11_abseil)

if(NOT CMAKE_INSTALL_PREFIX STREQUAL GDMR_CACHE_DIR)
    install(CODE "
        message(STATUS \"Removing intermediate installation artifacts to slim the wheel...\")
        file(REMOVE_RECURSE \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}\"
                            \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}\"
                            \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATADIR}\"
                            \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}\"
                            # Fallbacks for poorly-behaved third-party dependencies
                            # that hardcode 'lib' or 'lib64' instead of using GNUInstallDirs
                            \"\${CMAKE_INSTALL_PREFIX}/lib\"
                            \"\${CMAKE_INSTALL_PREFIX}/lib64\")
    ")
endif()
