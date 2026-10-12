# Copyright 2025 Google LLC
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

"""Unit tests for utils.py."""

import os
import tempfile
from unittest import mock

from absl import flags
from absl.testing import absltest

from safari_sdk import utils


def _write_file(directory: str, filename: str, content: str) -> None:
  with open(os.path.join(directory, filename), "w", encoding="utf-8") as f:
    f.write(content)


class ConfigDirectoriesTest(absltest.TestCase):

  def test_default_directories_order(self):
    # The first entry is $SAFARI_CONFIG (read at import time).
    self.assertEqual(
        utils._CONFIG_DIRECTORIES[1:],
        ["~/.config/safari_sdk/", "/opt/safari_sdk/"],
    )


class GetValueFromConfigFileTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.dir1 = tempfile.TemporaryDirectory()
    self.dir2 = tempfile.TemporaryDirectory()
    self.addCleanup(self.dir1.cleanup)
    self.addCleanup(self.dir2.cleanup)
    # Empty entries (e.g. unset $SAFARI_CONFIG) must be skipped.
    dirs_patcher = mock.patch.object(
        utils, "_CONFIG_DIRECTORIES", ["", self.dir1.name, "", self.dir2.name]
    )
    dirs_patcher.start()
    self.addCleanup(dirs_patcher.stop)

  def test_strips_whitespace(self):
    _write_file(self.dir1.name, "VALUE", "  my_value \n\t ")
    self.assertEqual(utils._get_value_from_config_file("VALUE"), "my_value")

  def test_reads_from_first_directory_with_file(self):
    _write_file(self.dir1.name, "VALUE", "dir1_value")
    _write_file(self.dir2.name, "VALUE", "dir2_value")
    self.assertEqual(utils._get_value_from_config_file("VALUE"), "dir1_value")

  def test_skips_missing_and_empty_files(self):
    _write_file(self.dir1.name, "VALUE", "  \n")
    _write_file(self.dir2.name, "VALUE", "dir2_value")
    self.assertEqual(utils._get_value_from_config_file("VALUE"), "dir2_value")
    self.assertIsNone(utils._get_value_from_config_file("OTHER"))

  def test_expands_user(self):
    os.mkdir(os.path.join(self.dir1.name, "sub"))
    _write_file(os.path.join(self.dir1.name, "sub"), "VALUE", "home_value")
    with (
        mock.patch.dict(os.environ, {"HOME": self.dir1.name}),
        mock.patch.object(utils, "_CONFIG_DIRECTORIES", ["~/sub"]),
    ):
      self.assertEqual(utils._get_value_from_config_file("VALUE"), "home_value")


class GetConfigValueTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.dir = tempfile.TemporaryDirectory()
    self.addCleanup(self.dir.cleanup)
    dirs_patcher = mock.patch.object(
        utils, "_CONFIG_DIRECTORIES", [self.dir.name]
    )
    dirs_patcher.start()
    self.addCleanup(dirs_patcher.stop)
    env_patcher = mock.patch.dict(
        os.environ, {"ENV_A": "", "ENV_B": " env_b\n"}, clear=True
    )
    env_patcher.start()
    self.addCleanup(env_patcher.stop)
    _write_file(self.dir.name, "VALUE", "file_value")
    # Private FlagValues so the test doesn't register a global flag.
    self.flag_values = flags.FlagValues()
    self.flag = flags.DEFINE_string(
        "value", None, "A value.", flag_values=self.flag_values
    )
    self.flag_values.mark_as_parsed()

  def test_flag_takes_precedence(self):
    self.flag_values.value = " flag_value "
    self.assertEqual(
        utils.get_config_value(self.flag, ("ENV_A", "ENV_B"), "VALUE"),
        "flag_value",
    )

  def test_env_used_when_flag_empty(self):
    self.flag_values.value = "  "
    self.assertEqual(
        utils.get_config_value(self.flag, ("ENV_A", "ENV_B"), "VALUE"),
        "env_b",
    )

  def test_file_used_when_flag_and_env_empty(self):
    self.assertEqual(
        utils.get_config_value(self.flag, ("ENV_A",), "VALUE"), "file_value"
    )

  def test_all_sources_optional(self):
    self.assertIsNone(utils.get_config_value())
    self.assertEqual(utils.get_config_value(file_name="VALUE"), "file_value")
    self.assertIsNone(utils.get_config_value(env_vars=("ENV_A",)))

  def test_required_returns_value(self):
    self.assertEqual(
        utils.get_config_value(env_vars=("ENV_B",), required=True), "env_b"
    )

  def test_required_raises_when_not_found(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "No value found. Set the --value flag, or the $ENV_A or $ENV_C"
        " environment variable, or a OTHER file in a Safari SDK config"
        " directory.",
    ):
      utils.get_config_value(
          self.flag, ("ENV_A", "ENV_C"), "OTHER", required=True
      )


class GetRobotIdFromSystemEnvTest(absltest.TestCase):

  def test_prefers_ga_robot_id(self):
    with mock.patch.dict(
        os.environ,
        {"GA_ROBOT_ID": "ga_robot", "SAFARI_ROBOT_ID": "safari_robot"},
    ):
      self.assertEqual(utils.get_robot_id_from_system_env(), "ga_robot")

  def test_falls_back_to_safari_robot_id(self):
    with mock.patch.dict(
        os.environ, {"GA_ROBOT_ID": "", "SAFARI_ROBOT_ID": "safari_robot"}
    ):
      self.assertEqual(utils.get_robot_id_from_system_env(), "safari_robot")

  def test_returns_none_when_unset(self):
    with mock.patch.dict(os.environ, {}, clear=True):
      self.assertIsNone(utils.get_robot_id_from_system_env())

  def test_strips_and_skips_whitespace_only_value(self):
    with mock.patch.dict(
        os.environ, {"GA_ROBOT_ID": "  ", "SAFARI_ROBOT_ID": " robot\n"}
    ):
      self.assertEqual(utils.get_robot_id_from_system_env(), "robot")


class GetRobotIdTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.dir1 = tempfile.TemporaryDirectory()
    self.dir2 = tempfile.TemporaryDirectory()
    self.addCleanup(self.dir1.cleanup)
    self.addCleanup(self.dir2.cleanup)
    dirs_patcher = mock.patch.object(
        utils, "_CONFIG_DIRECTORIES", [self.dir1.name, self.dir2.name]
    )
    dirs_patcher.start()
    self.addCleanup(dirs_patcher.stop)
    env_patcher = mock.patch.dict(os.environ, {}, clear=True)
    env_patcher.start()
    self.addCleanup(env_patcher.stop)
    # Use private FlagValues rather than the global FLAGS: test runners that
    # share a process across test modules (e.g. pytest) may have already
    # registered a conflicting --robot_id from another module.
    self.flag_values = flags.FlagValues()
    flags.DEFINE_string(
        "robot_id", None, "The robot ID.", flag_values=self.flag_values
    )
    self.flag_values.mark_as_parsed()
    flags_patcher = mock.patch.object(flags, "FLAGS", self.flag_values)
    flags_patcher.start()
    self.addCleanup(flags_patcher.stop)

  def test_flag_takes_precedence(self):
    self.flag_values.robot_id = "flag_robot"
    os.environ["GA_ROBOT_ID"] = "env_robot"
    _write_file(self.dir1.name, "ROBOT_ID", "file_robot")
    self.assertEqual(utils.get_robot_id(), "flag_robot")

  def test_whitespace_flag_falls_back_to_env(self):
    self.flag_values.robot_id = "  \t "
    os.environ["SAFARI_ROBOT_ID"] = "env_robot"
    self.assertEqual(utils.get_robot_id(), "env_robot")

  def test_env_takes_precedence_over_file(self):
    os.environ["GA_ROBOT_ID"] = "env_robot"
    _write_file(self.dir1.name, "ROBOT_ID", "file_robot")
    self.assertEqual(utils.get_robot_id(), "env_robot")

  def test_reads_from_first_directory_with_file(self):
    _write_file(self.dir1.name, "ROBOT_ID", "dir1_robot\n")
    _write_file(self.dir2.name, "ROBOT_ID", "dir2_robot\n")
    self.assertEqual(utils.get_robot_id(), "dir1_robot")

  def test_skips_empty_file(self):
    _write_file(self.dir1.name, "ROBOT_ID", "  \n")
    _write_file(self.dir2.name, "ROBOT_ID", "dir2_robot")
    self.assertEqual(utils.get_robot_id(), "dir2_robot")

  def test_returns_none_when_not_found(self):
    self.assertIsNone(utils.get_robot_id())

  def test_undefined_flag_falls_back_to_env(self):
    os.environ["GA_ROBOT_ID"] = "env_robot"
    with mock.patch.object(flags, "FLAGS", flags.FlagValues()):
      self.assertEqual(utils.get_robot_id(), "env_robot")

  def test_non_string_flag_is_ignored(self):
    # e.g. flywheel_cli defines --robot_id as a list.
    os.environ["GA_ROBOT_ID"] = "env_robot"
    flag_values = flags.FlagValues()
    flags.DEFINE_list(
        "robot_id", ["a", "b"], "Robot IDs.", flag_values=flag_values
    )
    flag_values.mark_as_parsed()
    with mock.patch.object(flags, "FLAGS", flag_values):
      self.assertEqual(utils.get_robot_id(), "env_robot")

  def test_get_required_robot_id_returns_robot_id(self):
    os.environ["GA_ROBOT_ID"] = "env_robot"
    self.assertEqual(utils.get_required_robot_id(), "env_robot")

  def test_get_required_robot_id_raises_when_not_found(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "No value found. Set the $GA_ROBOT_ID or $SAFARI_ROBOT_ID environment"
        " variable, or a ROBOT_ID file in a Safari SDK config directory.",
    ):
      utils.get_required_robot_id()


if __name__ == "__main__":
  absltest.main()
