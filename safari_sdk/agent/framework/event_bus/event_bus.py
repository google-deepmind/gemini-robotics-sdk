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

"""Async event bus."""

import asyncio
import collections
import datetime
import uuid

from absl import logging
import pytz

from google.protobuf import struct_pb2
from safari_sdk.agent.framework import config as framework_config
from safari_sdk.agent.framework import types
from safari_sdk.logging.python import stream_logger as stream_logger_lib
from safari_sdk.protos import label_pb2

Event = types.Event
EventType = types.EventType
EventSource = types.EventSource
EventBusHandlerSignature = types.EventBusHandlerSignature

_LOS_ANGELES_TIMEZONE = pytz.timezone("America/Los_Angeles")
_LOGGING_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
_LOGGING_TOPIC_ROBOT_AGENT_EVENTS = "/robot_agent/events"
_LOGGING_AGENT_SESSION_ID_LABEL_KEY = "agent_session_id"


class EventBus:
  """Async event bus for publishing, subscribing, and dispatching events."""

  def __init__(
      self,
      config: framework_config.AgentFrameworkConfig,
  ):
    self._event_type_to_handlers: dict[
        types.EventType, set[types.EventBusHandlerSignature]
    ] = collections.defaultdict(set)
    self._event_queue: asyncio.Queue[types.Event] = asyncio.Queue()
    self._main_task: asyncio.Task | None = None
    self._handler_tasks: dict[str, asyncio.Task] = {}
    self._is_running = False
    self._agent_session_id: str | None = None
    self._config = config
    self._enable_logging = config.enable_logging
    self._robot_id = config.robot_id
    self._logging_output_directory = config.logging_output_directory
    if self._enable_logging:
      if self._robot_id is None:
        raise ValueError("Specify the logging flag `logging.robot_id`.")
      if self._logging_output_directory is None:
        raise ValueError("Specify the logging flag `logging.output_directory`.")
      self._stream_logger: stream_logger_lib.StreamLogger = (
          stream_logger_lib.StreamLogger(
              agent_id=self._robot_id,
              output_directory=self._logging_output_directory,
              required_topics=[],
              optional_topics=[_LOGGING_TOPIC_ROBOT_AGENT_EVENTS],
          )
      )

  @property
  def agent_session_id(self) -> str | None:
    """The agent session ID."""
    return self._agent_session_id

  def subscribe(
      self,
      event_types: list[types.EventType],
      handler: types.EventBusHandlerSignature,
  ):
    """Register a handler for given event types."""
    for event_type in event_types:
      self._event_type_to_handlers[event_type].add(handler)

  def unsubscribe_all_for_event_types(self, event_types: list[types.EventType]):
    """Unregister all handlers for given event types."""
    for event_type in event_types:
      if event_type in self._event_type_to_handlers:
        del self._event_type_to_handlers[event_type]

  def unsubscribe_all(self):
    """Unregister all handlers for all event types."""
    self._event_type_to_handlers = {}

  def _log_event(self, event: types.Event):
    """Log the event to the stream logger."""
    # All events are logged to the event stream.
    self._log_to_session_stream(event)
    # Also log metadata to the session label.
    if (
        event.type == EventType.LOG_SESSION_METADATA
        or event.type == EventType.DEBUG
    ):
      self._log_to_session_label(event)

  def _log_to_session_label(self, event: types.Event):
    """Log the event as part of the session label.

    Note: the event metadata value will be logged as a string.

    Args:
      event: The event to log.
    """
    for key, value in event.metadata.items():
      if isinstance(value, int) or isinstance(value, float):
        label_value = struct_pb2.Value(number_value=value)
      elif isinstance(value, bool):
        label_value = struct_pb2.Value(bool_value=value)
      elif isinstance(value, str):
        label_value = struct_pb2.Value(string_value=value)
      else:
        logging.warning(
            "Unsupported type: %s for key: %s. Casting to string.",
            type(value),
            key,
        )
        label_value = struct_pb2.Value(string_value=str(value))
      self._stream_logger.add_session_label(
          label_pb2.LabelMessage(
              key=key,
              label_value=label_value,
          )
      )

  def _log_to_session_stream(self, event: types.Event):
    """Log the event as part of the event stream."""
    message = struct_pb2.Struct()
    message.fields["event_type"].string_value = event.type.value
    message.fields["event_source"].string_value = event.source.value
    message.fields["event_timestamp"].string_value = str(
        event.timestamp.timestamp()
    )
    message.fields["event_data"].string_value = str(event.data)
    if event.metadata:
      event_metadata_struct = struct_pb2.Struct()
      for key, value in event.metadata.items():
        event_metadata_struct.fields[key].string_value = str(value)
      message.fields["event_metadata"].struct_value.CopyFrom(
          event_metadata_struct
      )
    publish_time_nsec = datetime.datetime.now(tz=_LOS_ANGELES_TIMEZONE)
    self._stream_logger.update_synchronization_and_maybe_write_message(
        _LOGGING_TOPIC_ROBOT_AGENT_EVENTS,
        message=message,
        publish_time_nsec=int(publish_time_nsec.timestamp() * 1e9),
    )

  async def publish(self, event: types.Event):
    """Publish an event to the event bus."""
    if event.type not in self._event_type_to_handlers:
      logging.log_first_n(
          logging.WARNING,
          "Event type %s is not registered to any handlers.",
          1,
          event.type,
      )
    await self._event_queue.put(event)
    if self._enable_logging:
      self._log_event(event)

  def start(self) -> None:
    """Start the event bus."""
    if self._main_task and not self._main_task.done():
      logging.info("Event queue task already exists and is not done.")
    self._agent_session_id = str(uuid.uuid4())
    if self._enable_logging:
      now = datetime.datetime.now(tz=_LOS_ANGELES_TIMEZONE)
      self._stream_logger.start_session(
          start_nsec=int(now.timestamp() * 1e9),
          task_id="agent_task_id_placeholder",
          output_file_prefix=now.strftime(_LOGGING_TIMESTAMP_FORMAT),
      )
      self._stream_logger.add_session_label(
          label_pb2.LabelMessage(
              key=_LOGGING_AGENT_SESSION_ID_LABEL_KEY,
              label_value=struct_pb2.Value(string_value=self._agent_session_id),
          )
      )
    self._main_task = asyncio.create_task(self._event_queue_loop())
    self._is_running = True

  def shutdown(self) -> None:
    """Shutdown the event bus."""
    if not self._main_task or self._main_task.done():
      logging.info("Event queue task is not existent or already done.")
      return
    logging.info("Shutting down event queue task.")
    self._main_task.cancel()
    self._main_task = None
    self._is_running = False
    if self._enable_logging:
      now = datetime.datetime.now(tz=_LOS_ANGELES_TIMEZONE)
      self._stream_logger.stop_session(stop_nsec=int(now.timestamp() * 1e9))
    self._agent_session_id = None

  async def _event_queue_loop(self) -> None:
    while self._main_task and not self._main_task.done():
      try:
        event = await self._event_queue.get()
        await self._dispatch_event(event)
      except asyncio.CancelledError:
        break

  @property
  def is_running(self) -> bool:
    """Returns whether the event bus is currently running."""
    return self._is_running

  async def _dispatch_event(self, event: types.Event) -> None:
    """Dispatches an event to all registered handlers."""
    handlers = self._event_type_to_handlers.get(event.type)
    if not handlers:
      logging.log_first_n(
          logging.WARNING,
          "Event type %s is not registered to any handlers.",
          1,
          event.type,
      )
      return
    for handler in handlers:
      if asyncio.iscoroutinefunction(handler):
        # For async handlers, create a task directly.
        task = asyncio.create_task(handler(event))
      else:
        # For sync handlers, create a task that runs the handler in a thread.
        task = asyncio.create_task(asyncio.to_thread(handler, event))
      # Add the task to the list of running tasks and remove it when it's done.
      self._handler_tasks[task.get_name()] = task
      task.add_done_callback(lambda t: self._handler_tasks.pop(t.get_name()))
