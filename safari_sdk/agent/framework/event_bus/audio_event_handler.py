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

"""Audio handler using subprocess for recording and playback."""

import asyncio
import subprocess

from absl import logging

from safari_sdk.agent.framework.event_bus import event_bus

# arecord is a command line audio recorder for ALSA soundcard drivers.
# Audio configuration to match Gemini Live API requirements.
# Input: 16kHz, 1-channel, 16-bit signed little-endian PCM.
INPUT_RATE = 16000
ARECORD_CMD = [
    "arecord",
    "-r",
    str(INPUT_RATE),
    "-c",
    "1",  # 6 channels needed for localisation
    "-f",
    "S16_LE",
    "-t",
    "raw",
]
INPUT_CHUNK_SIZE = 1024

# aplay is a command line sound player for ALSA soundcard drivers.
# Output: 24kHz, 1-channel, 16-bit signed little-endian PCM.
OUTPUT_RATE = 24000
APLAY_CMD = [
    "aplay",
    "-r",
    str(OUTPUT_RATE),
    "-c",
    "1",
    "-f",
    "S16_LE",
    "-t",
    "raw",
]


class AudioHandler:
  """Handles audio IO using arecord and aplay subprocesses."""

  def __init__(
      self,
      bus: event_bus.EventBus,
      enable_audio_input: bool,
      enable_audio_output: bool,
      listen_while_speaking: bool = False,
  ):
    self._bus = bus
    # Recording auido from microphone and sending to the model.
    self._enable_audio_input = enable_audio_input
    # Playing back audio from the model to the speaker.
    self._enable_audio_output = enable_audio_output
    # Listening to the audio from the speaker while the model is speaking.
    self._listen_while_speaking = listen_while_speaking

    # Recording attributes
    self._record_proc = None
    self._record_task = None
    self._is_speaking = False

    # Playback attributes
    self._playback_proc = None
    self._playback_task = None
    self._playback_queue = asyncio.Queue()

    # Register event subscribers.
    self.register_event_subscribers()

  async def connect(self):
    """Connects to the audio handler."""
    if self._enable_audio_input:
      await self.start_recording()
    if self._enable_audio_output:
      await self.start_playback()

  async def disconnect(self):
    """Disconnects from the Gemini Live API and robot."""
    if self._enable_audio_input:
      await self.stop_recording()
    if self._enable_audio_output:
      await self.stop_playback()

  def register_event_subscribers(self):
    """Subscribes to MODEL_TURN events to queue audio for playback."""
    self._bus.subscribe(
        event_types=[event_bus.EventType.MODEL_TURN_COMPLETE],
        handler=self._finish_playback,
    )
    if self._enable_audio_output:
      self._bus.subscribe(
          event_types=[event_bus.EventType.MODEL_TURN],
          handler=self._queue_audio_for_playback,
      )
      self._bus.subscribe(
          event_types=[event_bus.EventType.MODEL_TURN_INTERRUPTED],
          handler=self._empty_playback_queue,
      )

  async def start_recording(self):
    """Starts the audio recording task and subprocess."""
    logging.debug("AudioHandler: Starting audio recording stream...")
    try:
      if self._record_proc is None:
        self._record_proc = await asyncio.create_subprocess_exec(
            *ARECORD_CMD, stdout=subprocess.PIPE
        )
        logging.debug(
            "AudioHandler: `arecord` process started with PID: %d.",
            self._record_proc.pid,
        )
      else:
        logging.warning(
            "AudioHandler: Recording process already running with PID: %d.",
            self._record_proc.pid,
        )

      if self._record_task is None or self._record_task.done():
        self._record_task = asyncio.create_task(self._record_loop())
        logging.debug("AudioHandler: New recording task created.")
      else:
        logging.warning("AudioHandler: Recording task already running.")
    except FileNotFoundError as e:
      self._record_proc = None
      logging.exception(
          "AudioHandler: `arecord` command not found. Please ensure ALSA"
          " utils (alsa-utils package) are installed and in your PATH: %s",
          e,
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      self._record_proc = None
      logging.exception("AudioHandler: Failed to start recording: %s", e)

  async def _record_loop(self):
    """Loop for recording audio data."""
    try:
      logging.debug("AudioHandler: Recording loop started.")
      while True:
        if self._record_proc is None or self._record_proc.stdout is None:
          # Wait a moment for the process to be re-established if needed
          await asyncio.sleep(0.01)
          continue

        chunk = await self._record_proc.stdout.read(INPUT_CHUNK_SIZE)

        if not chunk:
          logging.warning("AudioHandler: `arecord` stdout stream ended.")
          break  # End of stream

        logging.debug("AudioHandler: Read chunk of %d bytes.", len(chunk))
        if (not self._is_speaking) or self._listen_while_speaking:
          event = event_bus.Event(
              type=event_bus.EventType.MODEL_AUDIO_INPUT,
              source=event_bus.EventSource.USER,
              data=chunk,
          )
          await self._bus.publish(event)

    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception("AudioHandler: Error in record loop: %s", e)
    finally:
      await self._terminate_process(self._record_proc, "arecord")
      self._record_proc = None
      logging.debug("AudioHandler: Record loop terminated.")

  async def stop_recording(self):
    """Stops the recording subprocess and thread."""

    logging.debug("AudioHandler: Stopping audio recording...")
    # Terminate the process, which will unblock the _record_loop
    await self._terminate_process(self._record_proc, "arecord")
    self._record_proc = None
    logging.debug("AudioHandler: Recording process stopped.")

    if self._record_task and not self._record_task.done():
      self._record_task.cancel()
      try:
        await self._record_task
      except asyncio.CancelledError:
        pass  # Expected cancellation
    # Reset the task reference only when fully killing it.
    self._record_task = None
    logging.debug("AudioHandler: Recording task stopped and reset.")

  async def start_playback(self):
    """Starts the audio playback subprocess and task."""
    logging.debug("AudioHandler: Starting audio playback stream...")
    try:
      if self._playback_proc is None:
        self._playback_proc = await asyncio.create_subprocess_exec(
            *APLAY_CMD, stdin=subprocess.PIPE
        )
        logging.debug(
            "AudioHandler: `aplay` process started with PID: %d",
            self._playback_proc.pid,
        )
      else:
        logging.warning(
            "AudioHandler: Playback process already running with PID: %d",
            self._playback_proc.pid,
        )

      if self._playback_task is None or self._playback_task.done():
        self._playback_task = asyncio.create_task(self._playback_loop())
        logging.debug("AudioHandler: New playback task created.")
      else:
        logging.warning("AudioHandler: Playback task already running.")
    except FileNotFoundError as e:
      self._playback_proc = None
      logging.exception(
          "AudioHandler: `aplay` command not found. Please ensure ALSA"
          " utils are installed and in your PATH: %s",
          e,
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      self._playback_proc = None
      logging.exception("AudioHandler: Failed to start playback: %s", e)

  async def _playback_loop(self):
    """Asynchronously plays audio from the queue to the subprocess."""
    writer = None
    try:
      logging.debug("AudioHandler: Playback loop started.")
      while True:
        if self._playback_proc is None or self._playback_proc.stdin is None:
          await asyncio.sleep(0.01)
          continue

        audio_data = await self._playback_queue.get()
        if audio_data is None:  # Sentinel to stop the loop
          break

        writer = self._playback_proc.stdin
        writer.write(
            audio_data.tobytes()
            if not isinstance(audio_data, bytes)
            else audio_data
        )

        await writer.drain()
        self._playback_queue.task_done()

    except (ConnectionResetError, BrokenPipeError) as e:
      logging.exception(
          "AudioHandler: Playback pipe closed unexpectedly: %s", e
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception("AudioHandler: Error in playback loop: %s", e)
    finally:
      if writer and not writer.is_closing():
        writer.close()
        await writer.wait_closed()

      await self._terminate_process(self._playback_proc, "aplay")
      self._playback_proc = None

      logging.debug("AudioHandler: Playback loop terminated.")

  async def stop_playback(self):
    """Stops the playback task and subprocess."""

    logging.debug("AudioHandler: Stopping audio playback...")
    # Put the sentinel to ensure the task can exit the await cleanly
    await self._playback_queue.put(None)

    # Terminate the process
    await self._terminate_process(self._playback_proc, "aplay")
    self._playback_proc = None
    logging.debug("AudioHandler: Playback process stopped.")

    # Cancel the running task
    if self._playback_task and not self._playback_task.done():
      self._playback_task.cancel()
      try:
        await self._playback_task
      except asyncio.CancelledError:
        pass  # Expected cancellation
    self._playback_task = None
    logging.debug("AudioHandler: Playback task stopped and reset.")

  async def _terminate_process(
      self, proc: asyncio.subprocess.Process | None, name: str
  ):
    """Gracefully terminate a subprocess."""
    if proc and proc.returncode is None:
      logging.debug("AudioHandler: Terminating `%s` process...", name)
      try:
        proc.terminate()
        await asyncio.wait_for(proc.wait(), timeout=2.0)
      except asyncio.TimeoutError:
        logging.warning("AudioHandler: `%s` did not terminate, killing.", name)
        proc.kill()
        await proc.wait()
      logging.debug("AudioHandler: `%s` process stopped.", name)

  async def _queue_audio_for_playback(
      self, event: event_bus.Event
  ):
    """Queues audio chunks from a MODEL_TURN event for playback."""
    self._is_speaking = True
    for part in event.data.parts:
      if part.inline_data and part.inline_data.data:
        await self._playback_queue.put(part.inline_data.data)

  async def _finish_playback(self, event: event_bus.Event):
    """Finishes playback of audio chunks after a MODEL_TURN_COMPLETE event."""
    logging.debug(
        "AudioHandler: finishing playback after receiving event: %s", event
    )
    # Wait for the last audio chunk to finish playing
    await self._playback_queue.join()
    # A small buffer to prevent cutting off the very start of speech
    await asyncio.sleep(0.5)
    self._is_speaking = False

  async def _empty_playback_queue(self, event: event_bus.Event):
    """Empties the playback queue after a MODEL_TURN_INTERRUPTED event."""
    logging.debug(
        "AudioHandler: emptying playback queue after receiving event: %s", event
    )
    while not self._playback_queue.empty():
      try:
        self._playback_queue.get_nowait()
        self._playback_queue.task_done()
      except asyncio.QueueEmpty:
        break
    self._is_speaking = False
