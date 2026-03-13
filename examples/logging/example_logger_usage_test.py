# Copyright 2026 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
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

import glob
import os

from absl.testing import absltest

# Import tensorflow before safari_sdk to work around a pybind11_abseil type
# registration bug in tensorflow/pybind11_abseil.
import tensorflow as tf  # noqa: F401  # pylint: disable=unused-import

import example_logger_usage
from safari_sdk.logging.python import constants
from safari_sdk.logging.python import mcap_parser_utils


class ExampleLoggerUsageTest(absltest.TestCase):

  def test_example_runs_without_error(self):
    output_directory = self.create_tempdir().full_path
    example_logger_usage.write_example_to_mcap(output_directory)
    # Check that there is an mcap file
    mcap_files = glob.glob(
        os.path.join(output_directory, "**/*.mcap"), recursive=True
    )
    self.assertNotEmpty(mcap_files)
    # Print out a summary of the created mcap data
    mcap_proto_data = mcap_parser_utils.read_proto_data(
        output_directory,
        constants.TIMESTEP_TOPIC_NAME,
        constants.ACTION_TOPIC_NAME,
        constants.POLICY_EXTRA_TOPIC_NAME,
    )
    # Confirm we wrote expected number of timesteps and actions
    expected_count = 4
    self.assertLen(mcap_proto_data.timesteps, expected_count)
    self.assertLen(mcap_proto_data.actions, expected_count)


if __name__ == "__main__":
  absltest.main()
