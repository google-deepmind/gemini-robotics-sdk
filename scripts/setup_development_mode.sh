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

function _usage() {
  echo "Setup development mode for the given python pip source packages."
  echo ""
  echo "Usage: $0 [--venv=<path>] [--pip-arg=<arg>] [-r|--recursive] <src_paths> ..."
  echo ""
  echo "<src_paths> are paths to directories containing pip source packages."
  echo "  These directories contain a pyproject.toml file and all source and"
  echo "  data files; each is installed into the python virtual environment"
  echo "  in editable mode. Multiple directories can be specified as separate"
  echo "  arguments or colon-separated lists and all will be installed. Also"
  echo "  see https://pip.pypa.io/en/stable/topics/pipstall/#editable-mode."
  echo "-r|--recursive: If passed, the <src_paths> are recursively searched for"
  echo "  pyproject.toml files bellow the top level, and all are installed."
  echo "--pip-arg=<arg>: Additional arguments to pass to 'pip install'."
  echo "  This can be passed multiple times to add multiple arguments."
  echo "--venv <path>: The virtual environment to install into. If specified,"
  echo "  <path>/bin/activate will be sourced before installing. If not"
  echo "  specified, a venv must be activated before running this script."
}

# Initialize variables with default values.
venv_dir=""
src_dirs=()
pip_args=()
recursive=false

# Parse command line arguments.
while (( $# > 0 )) ; do
  case "$1" in
    -h|--help) _usage; exit 1 ;;
    -r|--recursive) recursive=true; shift 1 ;;
    --venv) venv_dir="$2"; shift 2 ;;
    --venv=*) venv_dir="${1#*=}"; shift 1 ;;
    --pip-arg) pip_args+=("$2"); shift 2 ;;
    --pip-arg=*) pip_args+=("${1#*=}"); shift 1 ;;
    *)
      IFS=':' read -ra src_dirs_read <<< "$1"
      src_dirs+=("${src_dirs_read[@]}")
      shift 1 ;;
  esac
done

# Collect the source directories to install.
pkg_dirs=()
for src_dir in "${src_dirs[@]}"; do
  if $recursive; then
    # Find all subdirectories which contain a pyproject.toml file.
    for toml_path in $(find $src_dir -name pyproject.toml); do
      pkg_dirs+=($(dirname "$toml_path"))
    done
  else
    pkg_dirs+=("$src_dir")
  fi
done
if [[ -z "${pkg_dirs[@]}" ]]; then
  echo "No pip source packages specified or found."
  exit 1
fi

# Activate the virtual environment or check that one is already active.
if [[ -n "$venv_dir" ]]; then
  source "$venv_dir/bin/activate" || exit 1
elif [[ -z $VIRTUAL_ENV ]]; then
  echo "A virtual environment is required. Source an environment before"
  echo "  running this script or pass --venv."
  exit 1
fi  # else use the already-active virtual environment.

# Install the packages in editable mode.
for pkg_dir in "${pkg_dirs[@]}"; do
  pip install "${pip_args[@]}" -e "${pkg_dir}" || exit 1
done

# The install often has a lot of output, so print a summary after that is done.
echo -e "\n\nSet up development mode in the virtual env: $VIRTUAL_ENV"
if [[ -n "${pip_args[@]}" ]]; then
  echo "With pip args: ${pip_args[@]}"
fi
echo "For the following packages:"
for pkg_dir in "${pkg_dirs[@]}"; do
  echo "  $pkg_dir"
done
