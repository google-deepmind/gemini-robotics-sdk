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

r"""Launching the agent evals."""

from collections.abc import Sequence

from absl import app
from absl import flags

from safari_sdk import utils
from safari_sdk.agent.framework.eval import agent_eval_lib
from safari_sdk.orchestrator.helpers import orchestrator_helper


_LAUNCH_SUBPROCESS = flags.DEFINE_bool(
    name="launch_subprocess",
    default=True,
    help="Whether to launch agent binary and the robot binary as subprocesses.",
)

_SHOW_SUBPROCESS_LOGS = flags.DEFINE_bool(
    name="show_subprocess_logs",
    default=False,
    help="Whether to show the subprocess stdout and stderr in the console.",
)

_SAVE_SUBPROCESS_LOGS = flags.DEFINE_bool(
    name="save_subprocess_logs",
    default=True,
    help="Whether to save subprocess logs to the subprocess_log_dir.",
)

_SUBPROCESS_LOG_DIR = flags.DEFINE_string(
    name="subprocess_log_dir",
    default="/tmp/agent_eval_logs/",
    help="The directory to store the subprocess logs.",
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  robot_id = utils.get_robot_id_from_system_env()
  if not robot_id:
    raise ValueError(
        f"`{utils.DEFAULT_ROBOT_ID_IN_SYSTEM_ENV}` environment variable must be"
        " set."
    )
  orchestrator_client = orchestrator_helper.OrchestratorHelper(
      robot_id=robot_id,
      job_type=orchestrator_helper.JOB_TYPE.EVALUATION,
  )
  evaluator = agent_eval_lib.AgentEvaluator(
      orchestrator_client=orchestrator_client,
      robot_id=robot_id,
      launch_subprocess=_LAUNCH_SUBPROCESS.value,
      show_subprocess_logs=_SHOW_SUBPROCESS_LOGS.value,
      save_subprocess_logs=_SAVE_SUBPROCESS_LOGS.value,
      subprocess_log_dir=_SUBPROCESS_LOG_DIR.value,
  )
  evaluator.run_eval()


if __name__ == "__main__":
  app.run(main)
