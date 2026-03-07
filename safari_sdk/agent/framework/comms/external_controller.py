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

"""The Fast API server for the external controller of EAR framework."""

import asyncio
import base64
import enum
import json
import logging
from typing import Any

import fastapi
from fastapi.middleware import cors
import fastapi.staticfiles
import psutil
import uvicorn

from safari_sdk.agent.framework import flags as agentic_flags
from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.ui import terminal_ui


@enum.unique
class EARFrameworkStatus(enum.Enum):
  """Class for tracking the status of the EAR framework and long-horizon task.

  This status is meant to for external entities to know if the framework is
  ready to accept commands, and once a command is sent to the framework, if the
  agent believes that it has finished the long-horizon task. This status is
  streamed by the external controller server to external entities.
  """

  # Framework is just initialized. run() has not been called.
  NOT_READY = "NOT_READY"
  # run() has been called and the framework is ready to accept commands.
  READY = "READY"
  # run() has been called and the agent is executing a long-horizon task.
  RUNNING = "RUNNING"
  # run() has been called and the agent thinks it finished the long-horizon
  # task.
  FINISHED = "FINISHED"


def _postprocess_event(
    event_type: event_bus.EventType,
    source: event_bus.EventSource,
    data: Any,
    metadata: dict[str, Any] | None,
) -> event_bus.Event:
  """Creates an Event from the given inputs.

  For MODEL_TEXT_INPUT events with data, parses user input using terminal_ui
  to handle special prefixes (@d, @s, @f). For all other events, creates a
  standard Event directly.

  Args:
    event_type: The type of event to create.
    source: The source of the event.
    data: The event data.
    metadata: Additional metadata for the event.

  Returns:
    An Event object ready to be published.
  """
  match event_type:
    case event_bus.EventType.MODEL_TEXT_INPUT if data:
      return terminal_ui.parse_user_input_to_event(
          message=str(data), source=source, default_metadata=metadata
      )
    case _:
      return event_bus.Event(
          type=event_type,
          source=source,
          data=data,
          metadata=metadata,
      )


class ExternalControllerFastAPIServer:
  """The Fast API server for the external controller of EAR framework.

  This server is used to control the EAR framework from an external entity, such
  as a multi-episode eval binary, a simple python script, a colab or a web
  browser.
  The server provides a set of canonical and advanced endpoints that can be used
  to control the EAR framework:
  The CANONICAL endpoints are meant to be used in a typical user journey, such
  as executing a long-horizon task, resetting the framework, streaming the
  framework status, etc. These endpoints will always work.
  The ADVANCED endpoints are meant to be used by power users who want to
  publish events directly to the EAR framework's event bus, such as injecting
  debug messages for logging. Note that there is no guarantee that the event
  will be handled/subscribed by the EAR framework correctly.
  """

  def __init__(
      self, bus: event_bus.EventBus, host: str = "127.0.0.1", port: int = 8887
  ):
    self._bus = bus
    self._host = host
    self._port = port
    self._external_controller_server = fastapi.FastAPI()
    self._server_task = None
    self.framework_status = EARFrameworkStatus.NOT_READY
    self.component_health_dict = {}

    # Enable CORS so the web GUI (on a different port) can make requests.
    self._external_controller_server.add_middleware(
        cors.CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Defines the FastAPI endpoints.
    self._setup_endpoints()

  _PROCESS_TERMINATE_TIMEOUT_SECS = 5

  def _kill_process_on_port(self) -> bool:
    """Kills any process using the configured port.

    First attempts graceful termination (SIGTERM), waits up to
    _PROCESS_TERMINATE_TIMEOUT_SECS, then forces kill (SIGKILL) if the
    process is still running.
    Uses psutil for cross-platform compatibility (Linux, macOS, Windows).

    Note: There is a TOCTOU race between killing the old process and binding
    the new server. Another process could theoretically bind to the port in
    that window. In practice this is unlikely for the controller port.

    Returns:
      True if a process was killed, False otherwise.
    """
    killed_any = False
    try:
      for conn in psutil.net_connections(kind="inet"):
        if conn.laddr.port == self._port and conn.status == psutil.CONN_LISTEN:
          try:
            proc = psutil.Process(conn.pid)
            proc_name = proc.name()
            proc.terminate()
            try:
              proc.wait(timeout=self._PROCESS_TERMINATE_TIMEOUT_SECS)
              logging.warning(
                  "\n"
                  "╔══════════════════════════════════════════════════════════════╗\n"
                  "║  🛑 TERMINATED PROCESS ON PORT %s (PID: %s, %s)           "
                  "  ║\n"
                  "╚══════════════════════════════════════════════════════════════╝",
                  self._port,
                  conn.pid,
                  proc_name,
              )
            except psutil.TimeoutExpired:
              proc.kill()
              logging.warning(
                  "\n"
                  "╔══════════════════════════════════════════════════════════════╗\n"
                  "║  🔪 KILLED PROCESS ON PORT %s (PID: %s, %s)               "
                  "  ║\n"
                  "║  (Process did not respond to SIGTERM after %s seconds)    "
                  "  ║\n"
                  "╚══════════════════════════════════════════════════════════════╝",
                  self._port,
                  conn.pid,
                  proc_name,
                  self._PROCESS_TERMINATE_TIMEOUT_SECS,
              )
            killed_any = True
          except psutil.NoSuchProcess:
            pass
          except psutil.AccessDenied:
            logging.warning(
                "Access denied when trying to kill process %s on port %s."
                " Try running with elevated privileges (e.g. sudo).",
                conn.pid,
                self._port,
            )
    except psutil.AccessDenied:
      logging.warning(
          "Access denied when listing network connections on port %s."
          " psutil.net_connections() requires elevated privileges on"
          " some platforms (e.g. sudo on macOS).",
          self._port,
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning(
          "Error checking for processes on port %s: %s", self._port, e
      )
    return killed_any

  def _setup_endpoints(self) -> None:
    """Sets up the external controller FastAPI endpoints."""

    @self._external_controller_server.get("/execute_lh_task/")
    async def execute_lh_task(  # pylint: disable=unused-variable
        lh_task: str,
    ) -> dict[str, str]:
      """CANONICAL endpoint: Tell agent to execute a long-horizon task.

      This is done by publishing a MODEL_TEXT_INPUT event to the event bus.

      Example usage (type in the browser's url bar):
      localhost:8887/execute_lh_task/?lh_task=organize the table, you can decide
      how. Do not ask me questions.

      Args:
        lh_task: The long-horizon task to execute.
      """
      logging.info("EXTERNAL CONTROLLER: Execute lh task: %s", lh_task)
      # Directly tell the agent to execute the task by publishing a
      # MODEL_TEXT_INPUT event to the event bus.
      await self._bus.publish(
          event=event_bus.Event(
              type=event_bus.EventType.MODEL_TEXT_INPUT,
              source=event_bus.EventSource.EXTERNAL_CONTROLLER,
              data=lh_task,
          )
      )
      return {"message": f"Instructed EAR framework to execute task: {lh_task}"}

    @self._external_controller_server.get("/execute_interaction/")
    async def execute_interaction(  # pylint: disable=unused-variable
        interaction: str,
    ) -> dict[str, str]:
      """CANONICAL endpoint: Tell agent to execute an interaction.

      This is done by publishing a MODEL_TEXT_INPUT event to the event bus.

      Example usage (type in the browser's url bar):
      localhost:8887/execute_interaction/?interaction=Hi Appollo, how are you?

      Args:
        interaction: The interaction to execute.
      """
      logging.info("EXTERNAL CONTROLLER: Execute interaction: %s", interaction)
      # Directly tell the agent to execute the task by publishing a
      # MODEL_TEXT_INPUT event to the event bus.
      await self._bus.publish(
          event=event_bus.Event(
              type=event_bus.EventType.MODEL_TEXT_INPUT,
              source=event_bus.EventSource.EXTERNAL_CONTROLLER,
              data=interaction,
          )
      )
      return {"message": f"Instructed EAR framework to execute: {interaction}"}

    @self._external_controller_server.get("/get_agent_session_id/")
    async def get_agent_session_id() -> dict[str, str]:  # pylint: disable=unused-variable
      """CANONICAL endpoint: Gets the agent session id from the event bus.

      This is used to get the agent session id from the event bus.

      Example usage (type in the browser's url bar):
      localhost:8887/get_agent_session_id/
      """
      logging.info("EXTERNAL CONTROLLER: Get agent session id.")
      if self._bus.agent_session_id is None:
        return {"agent_session_id": "Not Found"}
      return {"agent_session_id": str(self._bus.agent_session_id)}

    @self._external_controller_server.get("/terminate/")
    async def terminate() -> dict[str, str]:  # pylint: disable=unused-variable
      """CANONICAL endpoint: Terminates the EAR framework.

      This will cause the EAR framework to terminate all asyncio tasks.
      The server will also be terminated and becomes unresponsive.

      Example usage (type in the browser's url bar):
      localhost:8887/terminate/
      """
      logging.info("EXTERNAL CONTROLLER: Terminate")
      episode_info = self._bus.shutdown()
      for task in asyncio.all_tasks():
        if not task.done():
          task.cancel()
          logging.info("Task %s is cancelled.", task.get_name())

      message = {"message": "Terminating EAR framework."}
      if episode_info is not None and episode_info:
        message.update({k: v for k, v in episode_info.items() if v is not None})
      logging.info("EXTERNAL CONTROLLER terminate message: %s", message)
      return message

    @self._external_controller_server.get("/stop/")
    async def stop() -> dict[str, str]:  # pylint: disable=unused-variable
      """CANONICAL endpoint: Tells the agent to stop what it is doing.

      Also cancels any existing function calls. Note that this is a best effort
      request and the agent may not always stop. This endpoint simply tells the
      agent to stop.

      Example usage (type in the browser's url bar):
      localhost:8887/stop/
      """
      logging.info("EXTERNAL CONTROLLER: Reset")
      await self._bus.publish(
          event=event_bus.Event(
              type=event_bus.EventType.MODEL_TEXT_INPUT,
              source=event_bus.EventSource.EXTERNAL_CONTROLLER,
              data=(
                  "stop what you are doing and cancel existing function calls."
              ),
          )
      )
      return {
          "message": (
              "Instructed the agent to stop what it is doing. This is best"
              " effort and the agent may not always stop."
          )
      }

    @self._external_controller_server.get("/reset/")
    async def reset() -> dict[str, str]:  # pylint: disable=unused-variable
      """CANONICAL endpoint: Resets the EAR framework.

      Example usage (type in the browser's url bar):
      localhost:8887/reset/
      """
      logging.info("EXTERNAL CONTROLLER: Reset")
      await self._bus.publish(
          event=event_bus.Event(
              type=event_bus.EventType.RESET,
              source=event_bus.EventSource.EXTERNAL_CONTROLLER,
              data="",
          )
      )
      return {
          "message": (
              "Resetting EAR framework. The server may also restart. You may"
              " lose the data stream from the /stream_framework_status/"
              " endpoint and may need to call it again."
          )
      }

    @self._external_controller_server.get("/get_framework_status/")
    async def get_framework_status() -> dict[str, Any]:  # pylint: disable=unused-variable
      """CANONICAL endpoint: Gets the framework and component status.

      This endpoint is used to get the framework status without streaming.
      Example usage (type in the browser's url bar):
      localhost:8887/get_framework_status/
      """
      logging.debug("EXTERNAL CONTROLLER: Get framework status.")
      return {
          "framework_status": self.framework_status.value,
          "component_health_dict": self.component_health_dict,
      }

    @self._external_controller_server.get("/stream_framework_status/")
    async def stream_framework_status():  # pylint: disable=unused-variable
      """CANONICAL endpoint: Streams the framework status.

      There are currently 3 statuses: NOT_RAN_YET, RUNNING, FINISHED.
      The status is updated every second. The status is a string representation
      of the EARFrameworkStatus enum.
      This endpoint can be used by clients to monitor:
      1. the health of the framework, e.g., whether the framework is hanging
      (client receives no status updates).
      2. the progress of the long-horizon task, e.g., whether the agent is
      executing a long-horizon task or the agent thinks it finished the task.

      Example usage (type in the browser's url bar):
      localhost:8887/stream_framework_status/
      """

      async def _framework_status_generator():
        while True:
          data = {
              "framework_status": self.framework_status.value,
              "component_health_dict": self.component_health_dict,
          }
          data = f"data: {json.dumps(data)}\n\n"
          yield data
          await asyncio.sleep(1.0)

      logging.info("EXTERNAL CONTROLLER: Stream framework status at 1 Hz.")
      return fastapi.responses.StreamingResponse(
          _framework_status_generator(), media_type="text/event-stream"
      )

    @self._external_controller_server.get("/stream_events/")
    async def stream_events(  # pylint: disable=unused-variable
        event_types: list[str] = fastapi.Query(...),
    ) -> fastapi.responses.StreamingResponse:
      """ADVANCED endpoint: Streams events of specified event types.

      Subscribes to the provided event types and streams them to the client.
      This endpoint can be used to get any event types from the event bus.

      Example usage (type in the browser's url bar):
      localhost:8887/stream_events/?event_types=MODEL_TURN&event_types=TOOL_CALL&event_types=TOOL_RESULT&event_types=MODEL_TEXT_INPUT&event_types=SUCCESS_SIGNAL&event_types=FRAMEWORK_STATUS&event_types=SYSTEM_MESSAGE

      Args:
        event_types: A list of event types to subscribe to. Must be string
          representations of event_bus.EventType enums.
      """

      logging.info(
          "EXTERNAL CONTROLLER: Stream events for types: %s", event_types
      )

      try:
        typed_event_types = [event_bus.EventType[et] for et in event_types]
      except KeyError as e:
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid event_type: {e}. Please choose from "
                f"{list(event_bus.EventType.__members__.keys())}"
            ),
        )

      queue = asyncio.Queue[event_bus.Event]()

      def handler(event: event_bus.Event) -> None:
        try:
          queue.put_nowait(event)
        except asyncio.QueueFull:
          logging.warning("Event queue is full, dropping event: %s", event.type)

      subscription_id = self._bus.subscribe(
          event_types=typed_event_types, handler=handler
      )
      logging.info(
          "Subscribed to event types: %s with subscription ID: %s",
          event_types,
          subscription_id,
      )

      def _extract_audio_data(event_data: Any) -> dict[str, Any] | None:
        """Extract audio data from MODEL_TURN event parts.

        Args:
          event_data: The data of the MODEL_TURN event. This is expected to be a
            Content object with a .parts attribute.

        Returns:
          A dict with base64-encoded audio and mime_type if found,
          or None if no audio data is present.
        """

        # MODEL_TURN data is a Content object with a .parts attribute
        parts = None
        if hasattr(event_data, "parts"):
          parts = event_data.parts
        elif hasattr(event_data, "__iter__"):
          parts = event_data

        if not parts:
          return None

        try:
          for part in parts:
            if hasattr(part, "inline_data") and part.inline_data:
              inline = part.inline_data
              if (
                  hasattr(inline, "mime_type")
                  and inline.mime_type
                  and "audio" in inline.mime_type
                  and hasattr(inline, "data")
                  and inline.data
              ):
                return {
                    "audio_base64": (
                        base64.b64encode(inline.data).decode("utf-8")
                    ),
                    "mime_type": inline.mime_type,
                }
        except (TypeError, AttributeError):
          pass
        return None

      async def _event_generator():
        try:
          while True:
            event = await queue.get()
            event_data_value = event.data
            audio_data = None

            # For MODEL_TURN events, try to extract and base64-encode audio
            if event.type == event_bus.EventType.MODEL_TURN:
              audio_data = _extract_audio_data(event_data_value)

            if isinstance(event_data_value, bytes):
              event_data_value = base64.b64encode(event_data_value).decode(
                  "utf-8"
              )
            elif hasattr(event_data_value, "model_dump"):
              # Pydantic v2 models (e.g., genai_types.Transcription) need
              # explicit JSON serialization. Without this, str() produces
              # Python repr like "text='hello' finished=None" instead of
              # JSON. mode='json' handles nested bytes and other types.
              # Note: Pydantic v2 models also have dict() (deprecated
              # alias), so model_dump is checked first intentionally.
              event_data_value = event_data_value.model_dump(mode="json")
            elif hasattr(event_data_value, "dict"):
              # Fallback for pure Pydantic v1 models (no model_dump).
              event_data_value = event_data_value.dict()
            else:
              event_data_value = str(event_data_value)

            event_data = {
                "type": str(event.type),
                "data": event_data_value,
                "metadata": event.metadata,
                "source": str(event.source),
                "timestamp": event.timestamp.timestamp(),
            }

            # Add extracted audio data as a separate field for easy access
            if audio_data:
              event_data["audio"] = audio_data

            yield f"data: {json.dumps(event_data, default=str)}\n\n"
            queue.task_done()
        except asyncio.CancelledError:
          logging.info("Event streaming cancelled for types: %s", event_types)

      return fastapi.responses.StreamingResponse(
          _event_generator(), media_type="text/event-stream"
      )

    @self._external_controller_server.get("/stream_terminal_output/")
    async def stream_terminal_output() -> fastapi.responses.StreamingResponse:  # pylint: disable=unused-variable
      """CANONICAL endpoint: Streams events typically shown in the terminal.

      This endpoint streams events that are usually processed and displayed
      by the TerminalUI. This way you can use any web browser to monitor the
      progress of the agent.

      Example usage (type in the browser's url bar):
      localhost:8887/stream_terminal_output/
      """
      logging.info("EXTERNAL CONTROLLER: Stream terminal output.")
      event_types = [
          event_bus.EventType.MODEL_TURN.name,
          event_bus.EventType.MODEL_TURN_COMPLETE.name,
          event_bus.EventType.MODEL_TURN_INTERRUPTED.name,
          event_bus.EventType.MODEL_TEXT_INPUT.name,
          event_bus.EventType.GENERATION_COMPLETE.name,
          event_bus.EventType.TOOL_CALL.name,
          event_bus.EventType.TOOL_CALL_CANCELLATION.name,
          event_bus.EventType.TOOL_RESULT.name,
          event_bus.EventType.GO_AWAY.name,
          event_bus.EventType.DEBUG.name,
          event_bus.EventType.OUTPUT_TRANSCRIPT.name,
      ]
      return await stream_events(event_types)

    @self._external_controller_server.post("/publish_event/")
    async def publish_event(event: dict[str, Any]):  # pylint: disable=unused-variable
      """Publishes an event to the agent framework.

      If the event is MODEL_TEXT_INPUT and the message contains special
      prefixes (@d, @s, @f), it will be converted to the appropriate event
      type (DEBUG, SUCCESS_SIGNAL with True/False).

      Example usage (type in the browser's url bar): curl -X POST -H
      "Content-Type: application/json" -d '{"type": "MODEL_TEXT_INPUT",
      "source": "USER", "data": "hello", "metadata": {}}'
      http://localhost:8887/publish_event/

      Args:
        event: The event dict to publish to the agent framework.
      """
      logging.info("EXTERNAL CONTROLLER: Publish event: %s", event)

      event_type_str = event.get("type", "MODEL_TEXT_INPUT")
      source_str = event.get("source", "USER")
      data = event.get("data", "")
      metadata = event.get("metadata", {})

      try:
        event_type = event_bus.EventType[event_type_str]
      except KeyError as exc:
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid event_type: {event_type_str}",
        ) from exc

      try:
        source = event_bus.EventSource[source_str]
      except KeyError:
        source = event_bus.EventSource.USER

      final_event = _postprocess_event(
          event_type=event_type,
          source=source,
          data=data,
          metadata=metadata,
      )
      await self._bus.publish(event=final_event)

    @self._external_controller_server.get("/publish_to_event_bus/")
    async def publish_to_event_bus(  # pylint: disable=unused-variable
        event_type: str,
        data: str,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, str]:
      """ADVANCED endpoint: publishes an event to the event bus directly.

      Publish any event type and data to the event bus. This is an advanced use
      case and should be used with caution, as there is no guarantee that the
      event will be handled/subscribed by the EAR framework correctly.

      Example usage (type in the browser's url bar):
      localhost:8887/publish_to_event_bus/?event_type=MODEL_TEXT_INPUT&data=hello
      this will publish a MODEL_TEXT_INPUT event to the event bus with data
      "hello" (essentially saying "hello" to the agent).

      Args:
        event_type: The type of the event to publish. It must be a string
          representation of an event_bus.EventType enum.
        data: The data of the event to publish.
        metadata: Additional metadata associated with the event. This should be
          a dictionary of key-value pairs.
      """

      logging.info("EXTERNAL CONTROLLER: Publish to event bus")
      try:
        typed_event_type = event_bus.EventType[event_type]
        # Publish an event to the event bus.
        await self._bus.publish(
            event=event_bus.Event(
                type=typed_event_type,
                source=event_bus.EventSource.EXTERNAL_CONTROLLER,
                data=data,
                metadata=metadata,
            )
        )
        return {"message": "Published event to the event bus."}
      except KeyError:
        logging.error("Invalid event type: %s", event_type)
        raise fastapi.HTTPException(  # pylint: disable=raise-missing-from
            status_code=fastapi.status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid event_type: {event_type}. Please choose from the"
                f" following list: {event_bus.EventType.__members__.keys()}"
            ),
        )

  async def _handle_framework_status_events(
      self,
      event: event_bus.Event
  ) -> None:
    """Callback for framework status events."""
    logging.warning(
        "Framework status event received. Status transitioning to: %s",
        event.data,
    )
    self.framework_status = EARFrameworkStatus(event.data)

  async def _handle_orchestrator_health_events(
      self,
      event: event_bus.Event
  ) -> None:
    """Callback for orchestrator health events from any orchestrator type."""
    logging.warning(
        "Health status for orchestrator changed. Status transitioning to: %s",
        event.data["health_status"],
    )
    self.component_health_dict["orchestrator"] = {
        "health_status": event.data["health_status"],
        "exception_message": event.data["exception_message"],
    }

  async def _handle_tool_health_events(
      self,
      event: event_bus.Event
  ) -> None:
    """Callback for tool health events."""
    logging.warning(
        "Health status for tool %s changed. Status transitioning to: %s",
        event.data["tool_name"],
        event.data["health_status"],
    )
    self.component_health_dict[event.data["tool_name"]] = {
        "health_status": event.data["health_status"],
        "exception_message": event.data["exception_message"],
    }

  async def _run_server(self) -> None:
    """Runs the FastAPI server for external control of the EAR framework."""

    config = uvicorn.Config(
        self._external_controller_server,
        host=self._host,
        port=self._port,
        # TODO: Add flag, but Copybara is blocking.
        access_log=False,
    )

    # Custom server class to prevent uvicorn from intercepting keyboard
    # interrupts.
    # This is a hack as it prevents uvicorn from gracefully shutting down the
    # server when it receives a keyboard interrupt.
    # However, it has no impact if we restart the subprocess between episodes.
    class CustomServer(uvicorn.Server):

      def install_signal_handlers(self):
        pass

    server = CustomServer(config)

    try:
      await server.serve()
    except asyncio.CancelledError:
      logging.info("FastAPI server task was explicitly cancelled.")
      await server.shutdown()
    except Exception:  # pylint: disable=broad-exception-caught
      logging.exception("FastAPI server encountered an error.")
      await server.shutdown()
    except KeyboardInterrupt:
      logging.info("FastAPI server was terminated due to a keyboard interrupt.")

  async def connect(self) -> None:
    """Runs the FastAPI server for external control of the EAR framework."""
    # Note: There is a TOCTOU race between killing the port process and
    # starting uvicorn below. See _kill_process_on_port docstring.
    if agentic_flags.AGENTIC_KILL_PORT_PROCESS.value:
      self._kill_process_on_port()

    self._server_task = asyncio.create_task(self._run_server())

    # Subscribe to FRAMEWORK_STATUS events to update the framework status.
    self._bus.subscribe(
        event_types=[event_bus.EventType.FRAMEWORK_STATUS],
        handler=self._handle_framework_status_events,
    )
    # Subscribe to orchestrator health events.
    # - ORCHESTRATOR_CLIENT_HEALTH: published by NonStreamingGenAIHandler
    self._bus.subscribe(
        event_types=[
            event_bus.EventType.ORCHESTRATOR_CLIENT_HEALTH,
        ],
        handler=self._handle_orchestrator_health_events,
    )
    # Subscribe to tool_health_TOOL_CLIENT_HEALTH events to update the tools'
    # health.
    self._bus.subscribe(
        event_types=[event_bus.EventType.TOOL_CLIENT_HEALTH],
        handler=self._handle_tool_health_events,
    )
    self.framework_status = EARFrameworkStatus.READY

  async def disconnect(self) -> None:
    """Stops the FastAPI server for external control of the EAR framework."""
    self.framework_status = EARFrameworkStatus.NOT_READY
    if self._server_task:
      try:
        self._server_task.cancel()
        self._server_task = None
      except asyncio.CancelledError:
        logging.info(
            "External controller fast api server task was already cancelled."
        )
      logging.info("External controller fast api server task cancelled.")
    else:
      logging.info(
          "Cannot disconnect. External controller fast api server task is not"
          " running."
      )
