"""A collection of Success detection tools."""

import asyncio
from collections.abc import Awaitable, Callable
import datetime

from absl import logging
from google.genai import types

from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.tools import tool

_TASK_SUCCESS_KEY = "task_success"
_TIMEOUT_KEY = "timeout"

_DEFAULT_FUNCTION_DECLARATION = types.FunctionDeclaration(
    name="run_instruction_for_duration",
    description="""
  Run a natural language instruction on the robot. This will cause the agent on the robot to execute low-level actions (in a run high frequency run loop) needed to complete the language instruction.
  This function will return after {duration} seconds.
  Generally the duration be as short as possible, since the robot might complete the instruction and then undo it.
  The language instruction needs to be specific, unambiguous, atomic (involving single action with one object) instructions, e.g.:
    * "bring the cup to the table"
    * "put the book on the table"
    * "put the red dice in the green tray"
    * "push the bowl to the left"
    """,
    behavior=types.Behavior.NON_BLOCKING,
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "instruction": types.Schema(
                type=types.Type.STRING,
                description="""
  A specific, unambiguous, atomic (involving single action) natural language instruction for the robot (e.g. 'bring the cup to the table', 'put the dice in the green tray').
  This instruction should be specific enough to describe the exact object on the table, e.g. 'the white cup on the center of the table' or 'the green triangluar block on the most left'.
  It probably should not contain the word 'and'.
  """,
            ),
        },
        required=["instruction"],
    ),
)


class RunInstructionForDurationTool(tool.Tool):
  """An augmented version of robot's run_instruction tool.

  Will return a task success signal to the main agent after the robot finishes
  the instruction.
  """

  def __init__(
      self,
      bus: event_bus.EventBus,
      embodiment_run_instruction_tool: tool.Tool,
      embodiment_stop_tool: tool.Tool | None = None,
      stop_on_success: bool = True,
      default_duration: float = 5.0,
      function_declaration: types.FunctionDeclaration = (
          _DEFAULT_FUNCTION_DECLARATION
      ),
      lockstep_timer: (
          Callable[[float], Awaitable[tuple[dict[str, bool], bool]]] | None
      ) = None,
  ):
    """Initializes the tool.

    Args:
      bus: The event bus to subscribe to.
      embodiment_run_instruction_tool: The tool to use for running instructions.
      embodiment_stop_tool: If provided, this tool will be called when
        stop_on_success is True and the instruction completes. If None and
        stop_on_success is True, raises ValueError.
      stop_on_success: If True, the robot will be stopped after the instruction
        completes successfully.
      default_duration: The default duration in seconds to run instructions.
      function_declaration: The function declaration to use for the tool.
      lockstep_timer: Optional callable timer function that takes
        duration_seconds as its argument. If provided, this will be used instead
        of wall-clock sleep. This enables lockstep timing with simulation
        backends.
    """
    desc = function_declaration.description
    if desc is not None:
      desc = desc.format(duration=default_duration)

    declaration = types.FunctionDeclaration(
        name=function_declaration.name,
        description=desc,
        behavior=function_declaration.behavior,
        parameters=function_declaration.parameters,
    )
    super().__init__(
        fn=self.run_instruction_for_duration,
        declaration=declaration,
        bus=bus,
    )
    self._call_id = None
    self._latest_images = {}
    self._start_imgs = []
    self._default_duration = default_duration
    self._stop_on_success = stop_on_success

    self._embodiment_run_instruction_tool = embodiment_run_instruction_tool

    self._embodiment_stop_tool = embodiment_stop_tool
    self._lockstep_timer = lockstep_timer
    if stop_on_success and (embodiment_stop_tool is None):
      raise ValueError(
          "No stop tool from the embodiment found. You must have a stop tool"
          " to use the run_instruction_until_done tool when stop_on_success"
          " is set to True."
      )

  def _handle_tool_call_cancellation_events(self, event: event_bus.Event):
    """Handle the subscribed events."""
    if self._call_id in event.data.ids:
      self._tool_call_cancelled = True
      logging.info(
          "Tool call cancelled for run_instruction_until_done. id: %s.",
          self._call_id,
      )

  async def run_instruction_for_duration(
      self, instruction: str, call_id: str
  ) -> types.FunctionResponse:
    """Runs an instruction, stopping the robot after a duration."""

    duration = self._default_duration
    self._call_id = call_id

    t0 = datetime.datetime.now(tz=datetime.timezone.utc)

    # Send run instruction to the robot using the embodiment's tool.
    await self._embodiment_run_instruction_tool.fn(
        instruction, call_id
    )  # pytype: disable=bad-return-type

    t1 = datetime.datetime.now(tz=datetime.timezone.utc)

    if self._lockstep_timer is not None:
      task_success_dict, _ = await self._lockstep_timer(float(duration))
    else:
      task_success_dict, _ = await _run_instruction_timer(float(duration))

    t2 = datetime.datetime.now(tz=datetime.timezone.utc)

    # Stop the robot if needed.
    if self._stop_on_success and task_success_dict[_TASK_SUCCESS_KEY]:
      assert self._embodiment_stop_tool is not None, (
          "No stop tool from the embodiment found. You must have a stop tool"
          " to use the run_instruction_until_done tool when stop_on_success"
          " is set to True."
      )
      await self._embodiment_stop_tool.fn(call_id)  # pytype: disable=bad-return-type

    t3 = datetime.datetime.now(tz=datetime.timezone.utc)
    logging.info(
        "run_instruction_for_duration timing: start=%.3fs, sleep=%.3fs,"
        " stop=%.3fs, total=%.3fs",
        (t1 - t0).total_seconds(),
        (t2 - t1).total_seconds(),
        (t3 - t2).total_seconds(),
        (t3 - t0).total_seconds(),
    )

    return types.FunctionResponse(
        response={
            # TODO: This does not adhere to GenAI standards which
            # expect an "output" field.
            "subtask": instruction,
            "is_robot_stopped": self._stop_on_success,
        },
        will_continue=False,
    )


async def _run_instruction_timer(time_limit: float):
  await asyncio.sleep(time_limit)
  return {_TASK_SUCCESS_KEY: True, _TIMEOUT_KEY: True}, False
