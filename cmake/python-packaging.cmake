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

# CMake functions for packaging python projects with pip and virtual environments.
#
# The following functions are provided:
# - add_pip_package_info: Adds information about a pip package.
# - add_pip_source_package_info: Adds information about a pip source package.
# - add_pip_index_info: Adds information about a pip index.
# - add_python_venv: Creates a python virtual environment.
# - add_python_venv_command: Adds a custom command to run in a virtual env.
# - add_pip_install: Installs a pip package into a virtual environment.
# - add_pip_editable_install: Installs a pip package into a virtual environment
#   in editable mode.
# - add_pip_wheel: Builds a pip wheel file for a package.
# - add_pip_upload: Uploads a pip wheel file to a index.
#
# Other functions are only intended to be used internally, not externally. This
# sends some debug messages which can be output with `cmake --log-level=DEBUG`.

# The python version can be specified with CMAKE_PYTHON_VERSION. Set a minimum
# (not exact)version if it was not already specified.
if(NOT DEFINED CMAKE_PYTHON_VERSION)
  set(CMAKE_PYTHON_VERSION "3.10")
endif()
find_package(Python ${CMAKE_PYTHON_VERSION} COMPONENTS Interpreter)
if(NOT Python_Interpreter_FOUND)
  message(WARNING "Python not found. Assuming default values.")
  set(Python_EXECUTABLE "python3")
  set(Python_VERSION_MAJOR 3)
  execute_process(
    COMMAND python3 -c "import sys; print(sys.version_info.minor)"
    OUTPUT_VARIABLE Python_VERSION_MINOR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
endif()

include(CMakeParseArguments)

# Formats the given arguments into a string, surrounding keywords with spaces
# and/or new lines, and splitting DEPENDS values into separate lines. The result
# is stored in the OUT_VAR variable in the parent scope. Note: DEPENDS should be
# the last keyword in ARGN; any other keywords that appear after it will
# probably get lumped together with DEPENDS.
function(format_arguments OUT_VAR)
  cmake_parse_arguments(arg "" "" "DEPENDS" ${ARGN})

  # Find keywords (ie, words in ALL_CAPS with underscores), and surround them
  # with spaces and/or new lines.
  string(REGEX REPLACE "(^|;)([A-Z_]+)(;|$)" # The pattern to match
                     "\n       \\2 "         # The replacement string
                     _ARGUMENTS_STRING       # Output variable
                     "${arg_UNPARSED_ARGUMENTS}"  # Input string
  )

  if(arg_DEPENDS)
    # Since there are often many DEPENDS values, put each one on a new line.
    list(PREPEND arg_DEPENDS "DEPENDS")
    list(JOIN arg_DEPENDS "\n         " _DEPENDS_STRING)
    string(APPEND _ARGUMENTS_STRING "\n       ${_DEPENDS_STRING}")
  endif()

  # Output the formatted arguments.
  set(${OUT_VAR} "${_ARGUMENTS_STRING}" PARENT_SCOPE)
endfunction()

# Given a path to a python file, which is usually a package's __init__.py file,
# gets the value of the __version__ variable in the file, and assigns that to
# the given OUT_VAR in the parent scope.
#
# Note that this just returns literal contents of the assignment, which is
# assumed to be surrounded by quotes. Python expressions are NOT evaluated.
function(get_python_package_version OUT_VAR VERSION_FILE_PATH)
  # Read the file content.
  # CMake implicitly tracks files read with file(READ ...) during configuration.
  # If this file changes, re-configuration will be triggered on the next build.
  file(READ "${VERSION_FILE_PATH}" _VERSION_FILE_CONTENT)

  # Extract the version string using a regular expression. This looks for a line
  # starting with optional whitespace, then '__version__', optional whitespace,
  # '=', optional whitespace, a single or double quote, captures everything up
  # to the next matching quote, and then the closing quote.
  string(REGEX MATCH "[\n\r \t]*__version__[ \t]*=[ \t]*[\"']([^\"']*)[\"']"
         _VERSION_MATCH "${_VERSION_FILE_CONTENT}")

  # Check if the regex matched and store the captured version string
  if(NOT _VERSION_MATCH)
    message(FATAL_ERROR "Could not find __version__ assignment line in ${VERSION_FILE_PATH}")
  endif()

  # Explicitly declare the dependency, which is more robust across CMake versions.
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${VERSION_FILE_PATH}")

  # The captured group (the version string itself) is in CMAKE_MATCH_1
  set(${OUT_VAR} ${CMAKE_MATCH_1} PARENT_SCOPE)
endfunction()

# If the given INDEX is a target created with add_pip_index_info, get the URL
# form that target and add a command to install (in the given VENV_NAME) any pip
# packages needed to access that index. The given dependencies are also updated
# to ensure the install commands are run. VENV_NAME and INDEX are inputs and
# passed by value (eg, "${arg_VENV_NAME}"). DEPENDS_VAR is an input/output
# argument (ie, it is read and modified) and URL_VAR is an output. Both
# DEPENDS_VAR and URL_VAR should be passed by reference (eg, "arg_DEPENDS").
function(get_index_url_and_deps URL_VAR DEPENDS_VAR VENV_NAME INDEX)
  if(TARGET "${INDEX}")
    # Get the URL of the index.
    get_property(_URL TARGET "${INDEX}" PROPERTY URL)

    # We cannot append directly to a list in the parent scope, so store the
    # dependencies in a local variable and overwrite the parent scope later.
    get_property(_DEPENDS TARGET "${INDEX}" PROPERTY DEPENDS)
    set(_EXTRA_DEPENDS "${INDEX}" ${_DEPENDS})

    # Install any pip packages needed to access the index.
    get_property(_PIP_DEPENDS TARGET "${INDEX}" PROPERTY PIP_DEPENDS)
    foreach(_PIP_DEPENDENCY IN LISTS _PIP_DEPENDS)
      message(DEBUG "Installing pip dependency '${_PIP_DEPENDENCY}' for index '${INDEX}'")
      add_pip_install(
        VENV_NAME "${VENV_NAME}"
        PACKAGE "${_PIP_DEPENDENCY}"
        OUT_VAR _PIP_DEPENDENCY_OUTPUT
      )
      if(TARGET "${_PIP_DEPENDENCY}")
        list(APPEND _EXTRA_DEPENDS ${_PIP_DEPENDENCY})
      endif()
      list(APPEND _EXTRA_DEPENDS ${_PIP_DEPENDENCY_OUTPUT})
    endforeach()

    # Prepend new dependencies to the list and set it in the parent scope.
    set(${DEPENDS_VAR} ${${DEPENDS_VAR}} ${_EXTRA_DEPENDS} PARENT_SCOPE)
  else()
    set(_URL "${INDEX}")
  endif()

  set(${URL_VAR} "${_URL}" PARENT_SCOPE)
endfunction()

# Adds a custom target with properties defining a pip package. This target can
# then be passed to add_pip_install or add_pip_index_info.
#
# Arguments:
# - PACKAGE_NAME: Required. The name of the package to install.
# - TARGET_NAME: The name of the target to create. If not specified, it will
#   default to the PACKAGE_NAME.
# - INSTALL_SOURCE: The source of the package to install. This can be a wheel
#   file, a directory containing the pyproject.toml file, or anything else that
#   'pip install' accepts. If not specified, the package will be downloaded from
#   an index (eg PyPI) based on its name and version.
# - INSTALL_OUTPUT: The name of a file output when the package is installed.
#   If not specified, "${PACKAGE_NAME}/__init__.py" is assumed. This path is
#   relative to the VENV's site-packages directory.
# - VERSION: If INSTALL_SOURCE is specified, this is the version which is built
#   or installed from that source. Otherwise, this is the version of the package
#   which should be downloaded and installed from a index.
# - INDEX_DEPENDS: Repositories used as possible sources for the package or
#   its dependencies. This can be either the name of a target created with
#   add_pip_index_info, or the url of a index.
# - DEPENDS: Any other dependencies which are satisfied via the build system.
#   This should only be needed if the package is installed from somewhere
#   other than a index (ie, INSTALL_SOURCE is specified).
function(add_pip_package_info)
  cmake_parse_arguments(arg "" "TARGET_NAME;PACKAGE_NAME;INSTALL_SOURCE;INSTALL_OUTPUT;VERSION" "INDEX_DEPENDS;DEPENDS" ${ARGN})

  if(NOT arg_PACKAGE_NAME)
    message(FATAL_ERROR "PACKAGE_NAME is required.")
  endif()
  if(NOT arg_TARGET_NAME)
    set(arg_TARGET_NAME "${arg_PACKAGE_NAME}")
  endif()

  add_custom_target(${arg_TARGET_NAME})
  set_target_properties(${arg_TARGET_NAME} PROPERTIES
    PACKAGE_NAME "${arg_PACKAGE_NAME}"
    VERSION "${arg_VERSION}"
    INSTALL_SOURCE "${arg_INSTALL_SOURCE}"
    INSTALL_OUTPUT "${arg_INSTALL_OUTPUT}"
    INDEX_DEPENDS "${arg_INDEX_DEPENDS}"
    DEPENDS "${arg_DEPENDS}"
  )
endfunction()

# Adds a custom target with properties defining a pip source package- ie, a
# package which is built or installed from source code. This target can
# then be passed to add_pip_install, add_pip_editable_install, or add_pip_wheel.
#
# This takes the same arguments as add_pip_install, with these differences:
# - IMPORT_ROOT_DIR is required and must be the path to a directory containing
#   the __init__.py file, which in turn must set the __version__ variable.
# - SOURCE_FILES are any python source files which are part of the package.
#   These are added as dependencies for a regular install but not for an
#   editable install. A file should be added here if 1. it is not a generated
#   file (eg message files) and 2. its modification does not necessitate
#   re-running an editable install (but *does* necessitate re-running a regular
#   install). Anything else should be added to the regular DEPENDS list.
# - If PY_FILE_GLOB is specified, all .py files under IMPORT_ROOT_DIR are
#   automatically added to the SOURCE_FILES via a recursive glob. If not using
#   this, the SOURCE_FILES list should be manually specified.
# - SOURCE_DIR is used instead of INSTALL_SOURCE and must be a directory
#   containing a pyproject.toml file (which is automatically added as a
#   dependency). It defaults to the current source directory.
# - EDITABLE_INSTALL_OUTPUT is used by add_pip_editable_install instead of
#   INSTALL_OUTPUT. This can include the ${PACKAGE_VERSION} placeholder, which
#   this function will replace with the package's version from __init__.py.
#   This is also relative to the VENV's site-packages directory. It has no
#   default, and add_pip_editable_install cannot be used if it is not specified.
function(add_pip_source_package_info)
  cmake_parse_arguments(arg "PY_FILE_GLOB" "TARGET_NAME;PACKAGE_NAME;SOURCE_DIR;INSTALL_OUTPUT;EDITABLE_INSTALL_OUTPUT;IMPORT_ROOT_DIR" "SOURCE_FILES;INDEX_DEPENDS;DEPENDS" ${ARGN})

  if(NOT arg_IMPORT_ROOT_DIR)
    message(FATAL_ERROR "IMPORT_ROOT_DIR is required.")
  endif()

  # Get the package version from __init__.py.
  get_python_package_version(PACKAGE_VERSION "${arg_IMPORT_ROOT_DIR}/__init__.py")

  # The editable install output can include the ${PACKAGE_VERSION} placeholder.
  if(arg_EDITABLE_INSTALL_OUTPUT)
    string(CONFIGURE "${arg_EDITABLE_INSTALL_OUTPUT}" arg_EDITABLE_INSTALL_OUTPUT)
  endif()

  # The package source defaults to the current source directory.
  if(NOT arg_SOURCE_DIR)
    set(arg_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  endif()

  # Add pyproject.toml as a dependency.
  list(PREPEND arg_DEPENDS "${arg_SOURCE_DIR}/pyproject.toml")

  # Add all .py files under IMPORT_ROOT_DIR to SOURCE_FILES if desired.
  if(arg_PY_FILE_GLOB)
    file(GLOB_RECURSE _PYTHON_FILES CONFIGURE_DEPENDS "${arg_IMPORT_ROOT_DIR}/*.py")
    list(APPEND arg_SOURCE_FILES ${_PYTHON_FILES})
  endif()

  add_pip_package_info(
    TARGET_NAME "${arg_TARGET_NAME}"
    PACKAGE_NAME "${arg_PACKAGE_NAME}"
    VERSION "${PACKAGE_VERSION}"
    INSTALL_SOURCE "${arg_SOURCE_DIR}"
    INSTALL_OUTPUT "${arg_INSTALL_OUTPUT}"
    INDEX_DEPENDS "${arg_INDEX_DEPENDS}"
    DEPENDS "${arg_DEPENDS}"
  )

  # Set properties unique to a pip *source* package target.
  if(NOT arg_TARGET_NAME)
    set(arg_TARGET_NAME "${arg_PACKAGE_NAME}")
  endif()
  set_target_properties(${arg_TARGET_NAME} PROPERTIES
    EDITABLE_INSTALL_OUTPUT "${arg_EDITABLE_INSTALL_OUTPUT}"
    SOURCE_FILES "${arg_SOURCE_FILES}"
  )
endfunction()

# Emits a FATAL_ERROR message if the given PACKAGE is empty, not a target, or
# does not have the INSTALL_SOURCE property set. These conditions are guaranteed
# to be satisfied if the target was created with add_pip_source_package_info.
function(check_is_pip_source_package_info PACKAGE)
  if(NOT PACKAGE)
    message(FATAL_ERROR "PACKAGE is required.")
  endif()
  if(NOT TARGET "${PACKAGE}")
    message(FATAL_ERROR "PACKAGE '${PACKAGE}' must be a target.")
  endif()
  get_property(_INSTALL_SOURCE TARGET "${PACKAGE}" PROPERTY INSTALL_SOURCE)
  if(NOT _INSTALL_SOURCE)
    message(FATAL_ERROR "PACKAGE '${PACKAGE}' must have the INSTALL_SOURCE property set.")
  endif()
endfunction()

# Adds a custom target with properties defining a pip index.
#
# Arguments:
# - TARGET_NAME: Required. The name of the target created.
# - URL: Required. The URL of the index, *without* the trailing '/simple'.
# - PIP_DEPENDS: Any pip packages which must be installed to access the index,
#   such as a keyring. This can be a package name or a target created with
#   add_pip_package_info.
# - DEPENDS: Any other dependencies which are satisfied via the build system.
function(add_pip_index_info)
  cmake_parse_arguments(arg "" "TARGET_NAME;URL" "DEPENDS;PIP_DEPENDS" ${ARGN})
  if(NOT arg_TARGET_NAME)
    message(FATAL_ERROR "TARGET_NAME is required.")
  endif()
  if(NOT arg_URL)
    message(FATAL_ERROR "URL is required.")
  endif()
  if(arg_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unexpected arguments: ${arg_UNPARSED_ARGUMENTS}")
  endif()

  add_custom_target(${arg_TARGET_NAME})
  set_target_properties(${arg_TARGET_NAME} PROPERTIES ${ARGN})

  # Debugging output.
  format_arguments(_PROPERTIES_STRING ${ARGN})
  message(DEBUG "Added pip index info: ${_PROPERTIES_STRING}")
endfunction()

# Create a python virtual environment with the given name at build time.
#
# The virtual environment will be created in the given OUT_DIR, which defaults
# to ${CMAKE_BINARY_DIR}/${VENV_NAME} if not specified.
#
# A custom target will be created with the given VENV_NAME, which depends on the
# virtual environment being created. That target has the following properties:
# - VENV_DIR: The directory containing the virtual environment, ie ${OUT_DIR}.
# - ACTIVATE_PATH: The path to the activate script, ie ${VENV_DIR}/bin/activate.
# - SITE_PACKAGES_DIR: The directory in the virtual environment where packages
#   are installed. Files in this directory azre often used as an output or
#   depedency when installing packages.
function(add_python_venv)
  cmake_parse_arguments(arg "" "VENV_NAME;OUT_DIR" "" ${ARGN})

  if (NOT arg_VENV_NAME)
    message(FATAL_ERROR "VENV_NAME is required.")
  endif()

  # Set the default directory from the name if not specified.
  if(NOT arg_OUT_DIR)
    set(arg_OUT_DIR "${CMAKE_BINARY_DIR}/${arg_VENV_NAME}")
  endif()

  # Create the virtual environment.
  set(_ACTIVATE_PATH "${arg_OUT_DIR}/bin/activate")
  set(_ARGUMENTS
    OUTPUT ${_ACTIVATE_PATH}
    COMMAND "${Python_EXECUTABLE}" -m venv "${arg_OUT_DIR}"
    COMMAND touch ${_ACTIVATE_PATH}
    COMMENT "Creating Python virtual environment ${arg_VENV_NAME}\; to activate use: source ${_ACTIVATE_PATH}"
    VERBATIM
  )
  add_custom_command(${_ARGUMENTS})

  add_custom_target(${arg_VENV_NAME} DEPENDS ${_ACTIVATE_PATH})
  set_target_properties(${arg_VENV_NAME} PROPERTIES
     VENV_DIR "${arg_OUT_DIR}"
     ACTIVATE_PATH "${_ACTIVATE_PATH}"
     SITE_PACKAGES_DIR "${arg_OUT_DIR}/lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages"
  )

  # Debugging output.
  format_arguments(_ARGUMENTS_STRING ${_ARGUMENTS})
  message(DEBUG "Added python venv creation command: ${_ARGUMENTS_STRING}")
endfunction()

# Adds a custom command to run a command in the python virtual environment.
#
# This adds sourcing the virtual environment activation script and a dependency
# on that environment, and the calls add_custom_command or add_custom_target.
#
# Arguments:
# - VENV_NAME: Required. The name of the virtual environment to run the command
#   in. This must have been created with add_python_venv.
# - COMMAND: Required. The command to run in the virtual environment. All
#   command arguments should be specified in a single cmake function argument,
#   and if the python interpreter is used, it should be specified as just
#   'python3', not the full path.
# - OUTPUT: The name of the output file, used for dependencies. If specified,
#   add_custom_command will be used, otherwise add_custom_target.
# - TARGET_NAME: The name of the target to create. If OUTPUT is also specified,
#   that target will only depends on the output file; otherwise it will run
#   the given COMMAND directly. It will also have the VENV_NAME and OUTPUT
#   properties set. At least one of OUTPUT or TARGET_NAME is required.
# - COMMENT: A comment to pass to add_custom_command or add_custom_target.
#   Do not (or properly escape) any semicolons.
# - CONTEXT: The context that this was called from, usually the name of the
#   function that called this. Only used for debugging output.
# - DEPENDS: Any other dependencies which are satisfied via the build system.
#
function(add_python_venv_command)
  cmake_parse_arguments(arg "" "VENV_NAME;COMMAND;OUTPUT;TARGET_NAME;COMMENT;CONTEXT" "DEPENDS" ${ARGN})

  if(NOT arg_VENV_NAME)
    message(FATAL_ERROR "VENV_NAME is required.")
  endif()
  if(NOT arg_COMMAND)
    message(FATAL_ERROR "COMMAND is required.")
  endif()
  if(NOT arg_OUTPUT AND NOT arg_TARGET_NAME)
    message(FATAL_ERROR "OUTPUT or TARGET_NAME is required.")
  endif()

  # Wrap the command with the activation path.
  get_property(_ACTIVATE_PATH TARGET "${arg_VENV_NAME}" PROPERTY ACTIVATE_PATH)
  set(_ARGUMENTS COMMAND bash -c "source '${_ACTIVATE_PATH}' && ${arg_COMMAND}")

  # Add the COMMENT if needed.
  if(arg_COMMENT)
    list(APPEND _ARGUMENTS COMMENT "${arg_COMMENT}")
  endif()

  # Always use VERBATIM.
  list(APPEND _ARGUMENTS VERBATIM)

  # Add the dependencies, including the venv target.
  list(APPEND _ARGUMENTS DEPENDS "${arg_VENV_NAME}" "${_ACTIVATE_PATH}" ${arg_DEPENDS})

  # The OUTPUT and TARGET_NAME keywords determine whether add_custom_command
  # and/or add_custom_target is used.
  if (arg_OUTPUT)
    list(PREPEND _ARGUMENTS OUTPUT "${arg_OUTPUT}")
    # OUTPUT is specified, which implies adding a custom command.
    add_custom_command(${_ARGUMENTS})
    if(arg_TARGET_NAME)
      # Both OUTPUT and TARGET_NAME are specified, so add a target which only
      # depends on the output of the command.
      add_custom_target("${arg_TARGET_NAME}" DEPENDS "${arg_OUTPUT}")
    endif()
  elseif(arg_TARGET_NAME)
    # Only TARGET_NAME is specified, so create a target to run the command.
    list(PREPEND _ARGUMENTS "${arg_TARGET_NAME}")
    add_custom_target(${_ARGUMENTS})
  endif()

  # Set the venv and the output file as properties of the target (if any).
  if(arg_TARGET_NAME)
    set_target_properties("${arg_TARGET_NAME}" PROPERTIES
      VENV_NAME "${arg_VENV_NAME}"
      OUTPUT "${arg_OUTPUT}"
    )
  endif()

  # Debugging output.
  message(DEBUG "In ${arg_CONTEXT}:")
  if(arg_TARGET_NAME)
    message(DEBUG "  Created target: ${arg_TARGET_NAME}")
  endif()
  format_arguments(_ARGUMENTS_STRING ${_ARGUMENTS})
  message(DEBUG "  Added a custom command or target with arguments: ${_ARGUMENTS_STRING}")
endfunction()

# Low level function to add a pip install command. This is mostly an
# implementation detail, and callers should use add_pip_install or
# add_pip_editable_install.
function(add_pip_install_command)
  cmake_parse_arguments(arg "" "VENV_NAME;INSTALL_ARGS;CONTEXT;EXTRA_COMMENT;TARGET_NAME;OUTPUT;OUT_VAR" "INDEX_DEPENDS;DEPENDS" ${ARGN})

  if (NOT arg_VENV_NAME)
    message(FATAL_ERROR "VENV_NAME is required.")
  endif()
  if(NOT arg_INSTALL_ARGS)
    message(FATAL_ERROR "INSTALL_ARGS are required.")
  endif()
  if(NOT arg_OUTPUT)
    message(FATAL_ERROR "OUTPUT is required.")
  endif()
  if(NOT arg_CONTEXT)
    set(arg_CONTEXT "add_pip_install_command")
  endif()

  # Join the output relative path with the site-packages directory.
  get_property(_SITE_PACKAGES_DIR TARGET "${arg_VENV_NAME}" PROPERTY SITE_PACKAGES_DIR)
  string(PREPEND arg_OUTPUT "${_SITE_PACKAGES_DIR}/")

  # Add custom the repositories to the install arguments.
  foreach(_INDEX IN LISTS arg_INDEX_DEPENDS)
    get_index_url_and_deps(_URL arg_DEPENDS "${arg_VENV_NAME}" "${_INDEX}")
    string(APPEND arg_INSTALL_ARGS " --extra-index-url '${_URL}/simple'")
  endforeach()

  get_property(_VENV_DIR TARGET "${arg_VENV_NAME}" PROPERTY VENV_DIR)
  add_python_venv_command(
    COMMAND "pip install ${arg_INSTALL_ARGS}"
    TARGET_NAME "${arg_TARGET_NAME}"
    OUTPUT "${arg_OUTPUT}"
    COMMENT "Installing pip package '${_NAME}' into '${arg_VENV_NAME}' (${_VENV_DIR})${arg_EXTRA_COMMENT}"
    VENV_NAME "${arg_VENV_NAME}"
    CONTEXT "${arg_CONTEXT}"
    DEPENDS ${arg_DEPENDS}
  )

  if(arg_OUT_VAR)
    set(${arg_OUT_VAR} ${arg_OUTPUT} PARENT_SCOPE)
  endif()
endfunction()

# Installs a pip package into the given virtual environment.
#
# Arguments:
# - VENV_NAME: Required. The name of the virtual environment to install the
#   package in. This must have been created with add_python_venv.
# - PACKAGE: Required. The name of the package to install, or the name of a
#   target created with add_pip_package_info or add_pip_source_package_info.
# - OUT_VAR: The name of a variable in the parent scope to assign the path of
#   the output file to, incuding the prepended site-packages directory.
#   eg if the package's INSTALL_OUTPUT is "foo/bar.py" and python 3.12 is used,
#   OUT_VAR is set to "/path/to/venv/lib/python3.12/site-packages/foo/bar.py".
# - TARGET_NAME: If specified, a custom target will be created with that name
#   and a dependency on the output file. Note: Do NOT use this to transitively
#   depend on the output file; use that file directly (retrieved via OUT_VAR or
#   from the OUTPUT property set on that target).
# - DEPENDS: Any other dependencies which are satisfied via the build system.
function(add_pip_install)
  cmake_parse_arguments(arg "" "VENV_NAME;PACKAGE;OUT_VAR;TARGET_NAME" "DEPENDS" ${ARGN})

  if(NOT arg_PACKAGE)
    message(FATAL_ERROR "PACKAGE is required.")
  endif()

  # Ensure that local variables are not inherited from parent scopes.
  set(_NAME "${arg_PACKAGE}")
  unset(_OUTPUT)
  unset(_VERSION)
  unset(_DEPENDS)
  unset(_INDEX_DEPENDS)
  unset(_INSTALL_ARGS)

  if(TARGET "${arg_PACKAGE}")
    # This was created with add_pip_[source_]package_info. Get its properties.
    get_property(_NAME TARGET "${arg_PACKAGE}" PROPERTY PACKAGE_NAME)
    get_property(_OUTPUT TARGET "${arg_PACKAGE}" PROPERTY INSTALL_OUTPUT)
    get_property(_VERSION TARGET "${arg_PACKAGE}" PROPERTY VERSION)
    get_property(_SOURCE_FILES TARGET "${arg_PACKAGE}" PROPERTY SOURCE_FILES)
    get_property(_DEPENDS TARGET "${arg_PACKAGE}" PROPERTY DEPENDS)
    get_property(_INDEX_DEPENDS TARGET "${arg_PACKAGE}" PROPERTY INDEX_DEPENDS)
    get_property(_INSTALL_ARGS TARGET "${arg_PACKAGE}" PROPERTY INSTALL_SOURCE)
    list(APPEND arg_DEPENDS "${arg_PACKAGE}" ${_DEPENDS} ${_SOURCE_FILES})
  endif()

  if(NOT _INSTALL_ARGS)
    # Install the package from the index based on its name.
    set(_INSTALL_ARGS "${_NAME}")

    if(_VERSION)
      # Install the specified version of the package.
      string(APPEND _INSTALL_ARGS "==${_VERSION}")
    endif()
  endif()

  if(NOT _OUTPUT)
    # Use the package's __init__.py file by default.
    set(_OUTPUT "${_NAME}/__init__.py")
  endif()

  add_pip_install_command(
    VENV_NAME "${arg_VENV_NAME}"
    INSTALL_ARGS "${_INSTALL_ARGS}"
    TARGET_NAME "${arg_TARGET_NAME}"
    OUTPUT "${_OUTPUT}"
    OUT_VAR "${arg_OUT_VAR}"
    CONTEXT "add_pip_install"
    INDEX_DEPENDS ${_INDEX_DEPENDS}
    DEPENDS ${arg_DEPENDS}
  )

  if(arg_OUT_VAR)
    set(${arg_OUT_VAR} ${${arg_OUT_VAR}} PARENT_SCOPE)
  endif()
endfunction()

# Installs a pip package into the given virtual environment in editable mode.
#
# In editable mode, a link to the package's source code is installed in the
# virtual environment rather than copying that code, so that changes to the
# code are reflected in the virtual environment without needing to reinstall.
# An editable install will replace a regular install of the same package in the
# same virtual environment, and vice-versa.
#
# This takes the same arguments and has the same requirements as add_pip_install
# except that the PACKAGE must be defined with add_pip_source_package_info, and
# the EDITABLE_INSTALL_OUTPUT property must be set. The SOURCE_FILES are not set
# as a dependency since changes to them do not require a reinstall.
function(add_pip_editable_install)
  cmake_parse_arguments(arg "" "VENV_NAME;PACKAGE;OUT_VAR;TARGET_NAME" "DEPENDS" ${ARGN})

  check_is_pip_source_package_info("${arg_PACKAGE}")

  get_property(_NAME TARGET "${arg_PACKAGE}" PROPERTY PACKAGE_NAME)
  get_property(_OUTPUT TARGET "${arg_PACKAGE}" PROPERTY EDITABLE_INSTALL_OUTPUT)
  get_property(_DEPENDS TARGET "${arg_PACKAGE}" PROPERTY DEPENDS)
  get_property(_INDEX_DEPENDS TARGET "${arg_PACKAGE}" PROPERTY INDEX_DEPENDS)
  get_property(_INSTALL_ARGS TARGET "${arg_PACKAGE}" PROPERTY INSTALL_SOURCE)
  list(APPEND arg_DEPENDS "${arg_PACKAGE}" ${_DEPENDS})
  string(PREPEND _INSTALL_ARGS "--editable ")

  if(NOT _OUTPUT)
    message(FATAL_ERROR "PACKAGE '${arg_PACKAGE}' must have EDITABLE_INSTALL_OUTPUT set.")
  endif()

  add_pip_install_command(
    VENV_NAME "${arg_VENV_NAME}"
    INSTALL_ARGS "${_INSTALL_ARGS}"
    TARGET_NAME "${arg_TARGET_NAME}"
    OUTPUT "${_OUTPUT}"
    OUT_VAR "${arg_OUT_VAR}"
    CONTEXT "add_pip_editable_install"
    EXTRA_COMMENT " in editable mode"
    INDEX_DEPENDS "${_INDEX_DEPENDS}"
    DEPENDS ${arg_DEPENDS}
  )

  if(arg_OUT_VAR)
    set(${arg_OUT_VAR} ${${arg_OUT_VAR}} PARENT_SCOPE)
  endif()
endfunction()

# Builds a pip wheel from package source code. This takes the same arguments and
# has the same requirements as add_pip_install except that the PACKAGE must be
# defined with add_pip_source_package_info.
function(add_pip_wheel)
  cmake_parse_arguments(arg "" "VENV_NAME;PACKAGE;TARGET_NAME;OUT_VAR" "DEPENDS" ${ARGN})

  check_is_pip_source_package_info("${arg_PACKAGE}")
  if(NOT arg_VENV_NAME)
    message(FATAL_ERROR "VENV_NAME is required.")
  endif()

  get_property(_NAME TARGET "${arg_PACKAGE}" PROPERTY PACKAGE_NAME)
  get_property(_VERSION TARGET "${arg_PACKAGE}" PROPERTY VERSION)
  get_property(_INSTALL_SOURCE TARGET "${arg_PACKAGE}" PROPERTY INSTALL_SOURCE)
  get_property(_SOURCE_FILES TARGET "${arg_PACKAGE}" PROPERTY SOURCE_FILES)
  get_property(_DEPENDS TARGET "${arg_PACKAGE}" PROPERTY DEPENDS)
  list(APPEND arg_DEPENDS "${arg_PACKAGE}" ${_DEPENDS} ${_SOURCE_FILES})

  if(NOT _INSTALL_SOURCE)
    message(FATAL_ERROR "Package ${arg_PACKAGE} must have a INSTALL_SOURCE property specified.")
  endif()

  # Get the output wheel file name.
  set(_OUTPUT "${_INSTALL_SOURCE}/dist/${_NAME}-${_VERSION}-py3-none-any.whl")

  # Add a command to install the 'build' package into the virtual env.
  # This is required to build the wheel file.
  add_pip_install(
    VENV_NAME "${arg_VENV_NAME}"
    PACKAGE build
    OUT_VAR _PIP_INSTALL_BUILD_OUTPUT
  )
  list(APPEND arg_DEPENDS "${_PIP_INSTALL_BUILD_OUTPUT}")

  # Add a command to build the wheel file.
  add_python_venv_command(
    COMMAND "python3 -m build ${_INSTALL_SOURCE}"
    TARGET_NAME "${arg_TARGET_NAME}"
    OUTPUT "${_OUTPUT}"
    COMMENT "Building pip wheel for package '${_NAME}', output to '${_OUTPUT}'"
    VENV_NAME "${arg_VENV_NAME}"
    CONTEXT "add_pip_wheel"
    DEPENDS ${arg_DEPENDS}
  )

  if(arg_OUT_VAR)
    set(${arg_OUT_VAR} ${_OUTPUT} PARENT_SCOPE)
  endif()
endfunction()

# Uploads a pip wheel file to an index.
#
# Arguments:
# - VENV_NAME: Required. The name of the virtual environment (created with
#   add_python_venv) to install twine and any other dependencies into.
# - WHEEL: Required. The pip wheel file to upload, or a target created with
#   add_pip_wheel.
# - INDEX: Required. The URL of the index to upload the wheel to, or
#   a target created with add_pip_index_info.
# - TARGET_NAME: The name of the custom target to create. Note that there is no
#   output file created, so upload can only be triggered via this target name.
# - DEPENDS: Any other dependencies which are satisfied via the build system.
function(add_pip_upload)
  cmake_parse_arguments(arg "" "VENV_NAME;WHEEL;INDEX;TARGET_NAME" "DEPENDS" ${ARGN})

  if(NOT arg_VENV_NAME)
    message(FATAL_ERROR "VENV_NAME is required.")
  endif()
  if(NOT arg_WHEEL)
    message(FATAL_ERROR "WHEEL is required.")
  endif()
  if(NOT arg_INDEX)
    message(FATAL_ERROR "INDEX is required.")
  endif()
  if(NOT arg_TARGET_NAME)
    message(FATAL_ERROR "TARGET_NAME is required.")
  endif()

  # Get the wheel file to upload.
  if(TARGET "${arg_WHEEL}")
    # Add the wheel *target* as a dependency. The file will be added below.
    list(APPEND arg_DEPENDS "${arg_WHEEL}")
    get_property(arg_WHEEL TARGET ${arg_WHEEL} PROPERTY OUTPUT)
  endif()
  if (NOT "${arg_WHEEL}" MATCHES ".whl$")
    message(FATAL_ERROR "Wheel file name '${arg_WHEEL}' must end with .whl")
  endif()

  # Add a commands to install the 'twine' package into the virtual env.
  # This is required to upload the wheel file.
  add_pip_install(
    VENV_NAME "${arg_VENV_NAME}"
    PACKAGE twine
    OUT_VAR _PIP_INSTALL_TWINE_OUTPUT
  )

  # Add the wheel file and the output of the twine installation as dependencies.
  list(APPEND arg_DEPENDS "${arg_WHEEL}" "${_PIP_INSTALL_TWINE_OUTPUT}")

  # Get the index url and dependencies.
  get_index_url_and_deps(_URL arg_DEPENDS "${arg_VENV_NAME}" "${arg_INDEX}")

  # Add a command to upload the wheel file to the artifact registry.
  add_python_venv_command(
    COMMAND "twine upload --repository-url '${_URL}' '${arg_WHEEL}' --verbose"
    TARGET_NAME "${arg_TARGET_NAME}"
    COMMENT "Uploading pip wheel '${arg_WHEEL}' to '${_URL}'"
    VENV_NAME "${arg_VENV_NAME}"
    CONTEXT "add_pip_upload"
    DEPENDS ${arg_DEPENDS}
  )
endfunction()
