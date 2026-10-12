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

"""Unit tests for bidi_gemini_robotics_policy."""

import collections
import datetime
import json
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import dm_env
from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np

from safari_sdk.model import bidi_gemini_robotics_policy
from safari_sdk.model import bidi_remote_model_interface
from safari_sdk.model import constants

# Value the policy reports for a latency metric it has no measurement for.
_LATENCY_SENTINEL = np.float32(-1.0)


def _chunk_stream(
    *actions: np.ndarray,
) -> bidi_remote_model_interface.ActionHorizon:
  """Returns an ActionHorizon for a whole response, with its last chunk flagged final.

  Args:
    *actions: The actions each chunk of the response carries, in order.
  """
  stream = _FakeStream()
  last = len(actions) - 1
  for i, chunk in enumerate(actions):
    stream.deliver(chunk, final=i == last)
  return stream


class _FakeStream(bidi_remote_model_interface.ActionHorizon):
  """Stands in for the `ActionHorizon` `query_model()` returns.

  Reports emptiness by raising `TimeoutError` when no chunk has been delivered,
  so a step that waits for a chunk it should not have needed fails the test.
  """

  def __init__(self):
    self._pending = collections.deque()
    super().__init__(read_chunk=self._read_chunk)

  def _read_chunk(
      self, blocking: bool = True
  ) -> bidi_remote_model_interface.ActionChunk:
    del blocking
    if not self._pending:
      raise TimeoutError("No ready chunk in _FakeStream.")
    return self._pending.popleft()

  def deliver(self, chunk: np.ndarray, final: bool = False) -> None:
    """Makes `chunk` available to the horizon.

    Args:
      chunk: The actions the chunk carries.
      final: Whether the server marked this the last chunk of its response.
    """
    self._pending.append(
        bidi_remote_model_interface.ActionChunk(actions=chunk, is_final=final)
    )


class BidiGeminiRoboticsPolicyTest(parameterized.TestCase):

  def test_bidi_remote_model_interface_instantiation(self):
    """Verifies that BidiGeminiRoboticsPolicy instantiates BidiRemoteModelInterface."""
    policy = bidi_gemini_robotics_policy.BidiGeminiRoboticsPolicy(
        serve_id="test_serve_id",
        task_instruction_key="test_instruction_key",
        image_observation_keys=("test_camera_1",),
        proprioceptive_observation_keys=("test_joint_1",),
        min_replan_interval=2,
        inference_mode=constants.InferenceMode.SYNCHRONOUS,
    )
    self.assertIsInstance(
        policy._model, bidi_remote_model_interface.BidiRemoteModelInterface
    )

  def test_bidi_unsupported_connection_type(self):
    """Verifies that BidiGeminiRoboticsPolicy rejects non-CLOUD connection types."""
    with self.assertRaisesRegex(ValueError, "only supports.*CLOUD"):
      bidi_gemini_robotics_policy.BidiGeminiRoboticsPolicy(
          serve_id="test_serve_id",
          task_instruction_key="test_instruction_key",
          image_observation_keys=("test_camera_1",),
          proprioceptive_observation_keys=("test_joint_1",),
          min_replan_interval=2,
          robotics_api_connection=constants.RoboticsApiConnectionType.LOCAL,
      )

  def test_bidi_streaming_policy_execution_sync(self):
    """Verifies synchronous execution with BidiRemoteModelInterface returning an iterator."""
    mock_model = mock.create_autospec(
        bidi_remote_model_interface.BidiRemoteModelInterface,
        instance=True,
        spec_set=True,
    )
    mock_model.get_telemetry.return_value = (
        bidi_remote_model_interface.BidiModelTelemetry(
            remote_inference_duration=datetime.timedelta(milliseconds=42.0),
            network_overhead_duration=datetime.timedelta(milliseconds=8.0),
            client_image_encode_duration=datetime.timedelta(milliseconds=5.0),
            rng_key=[1, 2, 3],
        )
    )

    mock_model.query_model.side_effect = [
        _chunk_stream(np.array([[0.0, 0.0], [0.0, 0.0]])),
        _chunk_stream(np.array([[1.0, 2.0], [3.0, 4.0]])),
    ]

    with mock.patch.object(
        bidi_remote_model_interface,
        "BidiRemoteModelInterface",
        return_value=mock_model,
    ):
      policy = bidi_gemini_robotics_policy.BidiGeminiRoboticsPolicy(
          serve_id="test_serve_id",
          task_instruction_key="test_instruction_key",
          image_observation_keys=("test_camera_1",),
          proprioceptive_observation_keys=("test_joint_1",),
          min_replan_interval=2,
          inference_mode=constants.InferenceMode.SYNCHRONOUS,
      )

    timestep_spec = gdmr_types.TimeStepSpec(
        step_type=gdmr_types.STEP_TYPE_SPEC,
        reward={},
        discount={},
        observation={  # pyrefly: ignore[bad-argument-type]
            "test_camera_1": specs.Array(shape=(10, 10, 3), dtype=np.uint8),  # pyrefly: ignore[bad-assignment]
            "test_joint_1": specs.Array(shape=(1,), dtype=np.float32),  # pyrefly: ignore[bad-assignment]
            "test_instruction_key": specs.StringArray(()),  # pyrefly: ignore[bad-assignment]
        },
    )
    policy.step_spec(timestep_spec)
    mock_model.reset.assert_called_once()

    obs = {
        "test_camera_1": np.zeros((10, 10, 3), dtype=np.uint8),
        "test_joint_1": np.array([0.0], dtype=np.float32),
        "test_instruction_key": np.array("task", dtype=np.dtypes.StringDType()),
    }
    policy_state = policy.initial_state()
    (action_1, extra_1), policy_state = policy.step(
        dm_env.restart(obs), policy_state
    )

    np.testing.assert_array_equal(action_1, np.array([1.0, 2.0]))
    self.assertEqual(extra_1["inference_sent"], 1)  # pyrefly: ignore[bad-index]
    self.assertEqual(extra_1["remote_inference_ms"], 42.0)  # pyrefly: ignore[bad-index]
    self.assertEqual(extra_1["network_overhead_ms"], 8.0)  # pyrefly: ignore[bad-index]
    self.assertEqual(extra_1["client_image_encode_ms"], 5.0)  # pyrefly: ignore[bad-index]

    # Step 2: consumes second action without querying model
    (action_2, extra_2), policy_state = policy.step(
        dm_env.transition(reward=0.0, observation=obs), policy_state
    )
    np.testing.assert_array_equal(action_2, np.array([3.0, 4.0]))
    self.assertEqual(extra_2["inference_sent"], 0)  # pyrefly: ignore[bad-index]

  def test_bidi_streaming_policy_execution_async(self):
    """Verifies asynchronous execution with BidiRemoteModelInterface returning an iterator."""
    mock_model = mock.create_autospec(
        bidi_remote_model_interface.BidiRemoteModelInterface,
        instance=True,
        spec_set=True,
    )
    mock_model.get_telemetry.return_value = (
        bidi_remote_model_interface.BidiModelTelemetry(
            remote_inference_duration=datetime.timedelta(milliseconds=40.0),
            network_overhead_duration=datetime.timedelta(milliseconds=10.0),
            client_image_encode_duration=datetime.timedelta(milliseconds=2.0),
        )
    )

    actions_chunk = _FakeStream()
    actions_chunk.deliver(np.ones((10, 2), dtype=np.float32), final=True)
    mock_model.query_model.side_effect = [
        _chunk_stream(np.zeros((10, 2), dtype=np.float32)),
        actions_chunk,
    ]

    with mock.patch.object(
        bidi_remote_model_interface,
        "BidiRemoteModelInterface",
        return_value=mock_model,
    ):
      policy = bidi_gemini_robotics_policy.BidiGeminiRoboticsPolicy(
          serve_id="test_serve_id",
          task_instruction_key="test_instruction_key",
          image_observation_keys=("test_camera_1",),
          proprioceptive_observation_keys=("test_joint_1",),
          min_replan_interval=5,
          inference_mode=constants.InferenceMode.ASYNCHRONOUS,
      )

    timestep_spec = gdmr_types.TimeStepSpec(
        step_type=gdmr_types.STEP_TYPE_SPEC,
        reward={},
        discount={},
        observation={  # pyrefly: ignore[bad-argument-type]
            "test_camera_1": specs.Array(shape=(10, 10, 3), dtype=np.uint8),  # pyrefly: ignore[bad-assignment]
            "test_joint_1": specs.Array(shape=(1,), dtype=np.float32),  # pyrefly: ignore[bad-assignment]
            "test_instruction_key": specs.StringArray(()),  # pyrefly: ignore[bad-assignment]
        },
    )
    policy.step_spec(timestep_spec)

    obs = {
        "test_camera_1": np.zeros((10, 10, 3), dtype=np.uint8),
        "test_joint_1": np.array([0.0], dtype=np.float32),
        "test_instruction_key": np.array("task", dtype=np.dtypes.StringDType()),
    }
    policy_state = policy.initial_state()
    (action_1, extra_1), policy_state = policy.step(
        dm_env.restart(obs), policy_state
    )

    np.testing.assert_array_equal(action_1, np.ones((2,), dtype=np.float32))
    self.assertEqual(extra_1["inference_sent"], 1)  # pyrefly: ignore[bad-index]

  def test_multi_episode_execution(self):
    """Verifies that BidiGeminiRoboticsPolicy resets the model across episode boundaries."""
    mock_session_1 = mock.MagicMock()
    mock_session_1.use_msgpack = False
    mock_session_1.last_backend_request_time = datetime.datetime(
        2026, 5, 15, 1, 0, 0, tzinfo=datetime.timezone.utc
    )
    mock_session_1.last_backend_response_time = datetime.datetime(
        2026, 5, 15, 1, 0, 0, 50000, tzinfo=datetime.timezone.utc
    )

    mock_session_2 = mock.MagicMock()
    mock_session_2.use_msgpack = False
    mock_session_2.last_backend_request_time = datetime.datetime(
        2026, 5, 15, 1, 0, 0, tzinfo=datetime.timezone.utc
    )
    mock_session_2.last_backend_response_time = datetime.datetime(
        2026, 5, 15, 1, 0, 0, 50000, tzinfo=datetime.timezone.utc
    )

    resp_setup = mock.MagicMock()
    resp_setup.output_bytes = json.dumps(
        {"action_chunk": [[0.0, 0.0], [0.0, 0.0]]}
    ).encode("utf-8")
    resp_setup.final_chunk = True
    resp_ep1 = mock.MagicMock()
    resp_ep1.output_bytes = json.dumps(
        {"action_chunk": [[1.0, 2.0], [3.0, 4.0]]}
    ).encode("utf-8")
    resp_ep1.final_chunk = True
    mock_session_1.send_observation.side_effect = [1, 2]
    mock_session_1.get_response.side_effect = [resp_setup, resp_ep1]

    resp_ep2 = mock.MagicMock()
    resp_ep2.output_bytes = json.dumps(
        {"action_chunk": [[10.0, 20.0], [30.0, 40.0]]}
    ).encode("utf-8")
    resp_ep2.final_chunk = True
    mock_session_2.send_observation.side_effect = [1]
    mock_session_2.get_response.side_effect = [resp_ep2]

    with mock.patch.object(
        bidi_remote_model_interface.BidiRemoteModelInterface,
        "_open_bidi_session",
        side_effect=[mock_session_1, mock_session_2],
    ) as mock_open_session:
      policy = bidi_gemini_robotics_policy.BidiGeminiRoboticsPolicy(
          serve_id="test_serve_id",
          task_instruction_key="test_instruction_key",
          image_observation_keys=("test_camera_1",),
          proprioceptive_observation_keys=("test_joint_1",),
          min_replan_interval=2,
          inference_mode=constants.InferenceMode.SYNCHRONOUS,
      )

      timestep_spec = gdmr_types.TimeStepSpec(
          step_type=gdmr_types.STEP_TYPE_SPEC,
          reward={},
          discount={},
          observation={  # pyrefly: ignore[bad-argument-type]
              "test_camera_1": specs.Array(shape=(10, 10, 3), dtype=np.uint8),  # pyrefly: ignore[bad-assignment]
              "test_joint_1": specs.Array(shape=(1,), dtype=np.float32),  # pyrefly: ignore[bad-assignment]
              "test_instruction_key": specs.StringArray(()),  # pyrefly: ignore[bad-assignment]
          },
      )
      # step_spec triggers _setup() which initializes session 1
      policy.step_spec(timestep_spec)
      mock_session_1.start.assert_called_once()
      self.assertEqual(mock_open_session.call_count, 1)

      observation = {
          "test_camera_1": np.zeros((10, 10, 3), dtype=np.uint8),
          "test_joint_1": np.array([0.0], dtype=np.float32),
          "test_instruction_key": np.array(
              "test_instruction", dtype=np.dtypes.StringDType()
          ),
      }

      # Episode 1 initialization: does NOT close stream (reuses session 1)
      policy_state = policy.initial_state()
      mock_session_1.close.assert_not_called()
      self.assertEqual(policy._model._bidi_session, mock_session_1)

      # Episode 1 execution: streams over session 1
      (action_ep1, _), policy_state = policy.step(
          dm_env.restart(observation),
          policy_state,
      )
      np.testing.assert_array_equal(action_ep1, np.array([1.0, 2.0]))

      # Episode boundary: 2nd call to initial_state resets session 1 and starts
      # session 2.
      policy_state = policy.initial_state()
      mock_session_1.close.assert_called_once()
      mock_session_2.start.assert_called_once()
      self.assertEqual(policy._model._bidi_session, mock_session_2)
      self.assertEqual(mock_open_session.call_count, 2)

      # Episode 2: streams over already active session 2
      (action_ep2, _), policy_state = policy.step(
          dm_env.restart(observation),
          policy_state,
      )
      np.testing.assert_array_equal(action_ep2, np.array([10.0, 20.0]))
      self.assertEqual(mock_open_session.call_count, 2)

      # Close policy explicitly
      policy.close()
      mock_session_2.close.assert_called_once()

  def test_bidi_policy_destructor_does_not_close_model(self):
    mock_model = mock.create_autospec(
        bidi_remote_model_interface.BidiRemoteModelInterface,
        instance=True,
        spec_set=True,
    )
    with mock.patch.object(
        bidi_remote_model_interface,
        "BidiRemoteModelInterface",
        return_value=mock_model,
    ):
      policy = bidi_gemini_robotics_policy.BidiGeminiRoboticsPolicy(
          serve_id="test_serve_id",
          task_instruction_key="test_instruction_key",
          image_observation_keys=("test_camera_1",),
          proprioceptive_observation_keys=("test_joint_1",),
          inference_mode=constants.InferenceMode.ASYNCHRONOUS,
      )
      policy.__del__()
      mock_model.close.assert_not_called()


class BidiPolicyStreamingTest(absltest.TestCase):
  """Verifies the policy consumes streamed chunks incrementally."""

  def _make_timestep_spec(self):
    return gdmr_types.TimeStepSpec(
        step_type=gdmr_types.STEP_TYPE_SPEC,
        reward={},
        discount={},
        observation={  # pyrefly: ignore[bad-argument-type]
            "test_camera_1": specs.Array(shape=(10, 10, 3), dtype=np.uint8),  # pyrefly: ignore[bad-assignment]
            "test_joint_1": specs.Array(shape=(1,), dtype=np.float32),  # pyrefly: ignore[bad-assignment]
            "test_instruction_key": specs.StringArray(()),  # pyrefly: ignore[bad-assignment]
        },
    )

  def _make_observation(self):
    return {
        "test_camera_1": np.zeros((10, 10, 3), dtype=np.uint8),
        "test_joint_1": np.array([0.0], dtype=np.float32),
        "test_instruction_key": np.array("task", dtype=np.dtypes.StringDType()),
    }

  def test_policy_accumulates_multiple_chunks_of_one_query(self):
    """A multi-chunk response must extend the action horizon."""
    mock_model = mock.create_autospec(
        bidi_remote_model_interface.BidiRemoteModelInterface,
        instance=True,
        spec_set=True,
    )
    mock_model.get_telemetry.return_value = None

    mock_model.query_model.side_effect = [
        # The setup query returns a 4-action horizon in a single chunk.
        _chunk_stream(np.zeros((4, 2))),
        # The real query streams two chunks of 2 actions each.
        _chunk_stream(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[5.0, 6.0], [7.0, 8.0]]),
        ),
    ]

    with mock.patch.object(
        bidi_remote_model_interface,
        "BidiRemoteModelInterface",
        return_value=mock_model,
    ):
      policy = bidi_gemini_robotics_policy.BidiGeminiRoboticsPolicy(
          serve_id="test_serve_id",
          task_instruction_key="test_instruction_key",
          image_observation_keys=("test_camera_1",),
          proprioceptive_observation_keys=("test_joint_1",),
          min_replan_interval=2,
          inference_mode=constants.InferenceMode.SYNCHRONOUS,
      )

    policy.step_spec(self._make_timestep_spec())
    policy_state = policy.initial_state()

    (action, extra), _ = policy.step(
        dm_env.restart(self._make_observation()), policy_state
    )

    # Both chunks belong to the same query, so the horizon is their
    # concatenation and execution starts at the very first action.
    np.testing.assert_array_equal(action, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(
        extra["action_chunk"],  # pyrefly: ignore[bad-index]
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
    )

  def test_a_response_still_arriving_cannot_leak_into_the_next_episode(self):
    """A response outliving its episode must not be executed by the next one.

    The model keeps answering a query the episode boundary has made
    irrelevant, so the policy has to let go of that response rather than fold
    what is left of it into the horizon of the episode that followed.
    """
    mock_model = mock.create_autospec(
        bidi_remote_model_interface.BidiRemoteModelInterface,
        instance=True,
        spec_set=True,
    )
    mock_model.get_telemetry.return_value = None

    first_episode = _FakeStream()
    # Answered, but the response is not complete when the episode ends.
    first_episode.deliver(
        np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    )
    second_episode = _FakeStream()
    second_episode.deliver(np.array([[7.0, 7.0], [8.0, 8.0]]), final=True)

    mock_model.query_model.side_effect = [
        # Spec probe issued by the policy setup.
        _chunk_stream(np.zeros((4, 2))),
        first_episode,
        second_episode,
    ]

    with mock.patch.object(
        bidi_remote_model_interface,
        "BidiRemoteModelInterface",
        return_value=mock_model,
    ):
      policy = bidi_gemini_robotics_policy.BidiGeminiRoboticsPolicy(
          serve_id="test_serve_id",
          task_instruction_key="test_instruction_key",
          image_observation_keys=("test_camera_1",),
          proprioceptive_observation_keys=("test_joint_1",),
          min_replan_interval=2,
          inference_mode=constants.InferenceMode.ASYNCHRONOUS,
      )

    policy.step_spec(self._make_timestep_spec())

    policy_state = policy.initial_state()
    policy.step(dm_env.restart(self._make_observation()), policy_state)

    # The episode ends while its response is still arriving.
    policy_state = policy.initial_state()
    self.assertEqual(mock_model.reset.call_count, 2)

    # The rest of that response lands after the boundary.
    first_episode.deliver(np.array([[99.0, 99.0]]), final=True)

    (action, _), _ = policy.step(
        dm_env.restart(self._make_observation()), policy_state
    )

    np.testing.assert_array_equal(action, np.array([7.0, 7.0]))
    policy.close()

  def test_step_does_not_wait_for_a_response_while_actions_remain(self):
    """A query in flight must not stall a step the horizon can still serve.

    This is the whole point of issuing the query without waiting for it: the
    robot keeps executing the actions it holds while the next ones are
    computed. `_FakeStream` reports emptiness rather than blocking, so a step
    that waits when it should not fails here instead of hanging.
    """
    mock_model = mock.create_autospec(
        bidi_remote_model_interface.BidiRemoteModelInterface,
        instance=True,
        spec_set=True,
    )
    mock_model.get_telemetry.return_value = None
    answered = _FakeStream()
    answered.deliver(np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]), final=True)
    # The replan query, which never answers for the rest of the test.
    silent = _FakeStream()
    mock_model.query_model.side_effect = [
        # Spec probe issued by the policy setup.
        _chunk_stream(np.zeros((5, 1))),
        answered,
        silent,
    ]

    with mock.patch.object(
        bidi_remote_model_interface,
        "BidiRemoteModelInterface",
        return_value=mock_model,
    ):
      policy = bidi_gemini_robotics_policy.BidiGeminiRoboticsPolicy(
          serve_id="test_serve_id",
          task_instruction_key="test_instruction_key",
          image_observation_keys=("test_camera_1",),
          proprioceptive_observation_keys=("test_joint_1",),
          min_replan_interval=1,
          inference_mode=constants.InferenceMode.ASYNCHRONOUS,
      )

    policy.step_spec(self._make_timestep_spec())
    policy_state = policy.initial_state()
    observation = self._make_observation()

    (action, _), policy_state = policy.step(
        dm_env.restart(observation), policy_state
    )
    np.testing.assert_array_equal(action, np.array([1.0]))

    # The horizon has aged past the replan interval, so the second step issues
    # a query. It is still unanswered on every step that follows, and none of
    # them may wait for it while the first response still has actions left.
    for expected in ([2.0], [3.0], [4.0], [5.0]):
      (action, _), policy_state = policy.step(
          dm_env.transition(0.0, observation), policy_state
      )
      np.testing.assert_array_equal(action, np.array(expected))

    # 1 spec probe + 2 policy queries.
    self.assertEqual(mock_model.query_model.call_count, 3)
    policy.close()

  def test_chunks_of_one_response_are_collected_as_they_arrive(self):
    """A response must extend the horizon chunk by chunk, not in one go."""
    mock_model = mock.create_autospec(
        bidi_remote_model_interface.BidiRemoteModelInterface,
        instance=True,
        spec_set=True,
    )
    mock_model.get_telemetry.return_value = None
    stream = _FakeStream()
    stream.deliver(np.array([[1.0, 2.0], [3.0, 4.0]]))
    mock_model.query_model.side_effect = [
        # Spec probe issued by the policy setup.
        _chunk_stream(np.zeros((4, 2))),
        stream,
    ]

    with mock.patch.object(
        bidi_remote_model_interface,
        "BidiRemoteModelInterface",
        return_value=mock_model,
    ):
      policy = bidi_gemini_robotics_policy.BidiGeminiRoboticsPolicy(
          serve_id="test_serve_id",
          task_instruction_key="test_instruction_key",
          image_observation_keys=("test_camera_1",),
          proprioceptive_observation_keys=("test_joint_1",),
          min_replan_interval=10,
          inference_mode=constants.InferenceMode.ASYNCHRONOUS,
      )

    policy.step_spec(self._make_timestep_spec())
    policy_state = policy.initial_state()
    observation = self._make_observation()

    # Only the head of the response has arrived, and it is enough to act on.
    (action, extra), policy_state = policy.step(
        dm_env.restart(observation), policy_state
    )
    np.testing.assert_array_equal(action, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(
        extra["action_chunk"],  # pyrefly: ignore[bad-index]
        np.array([[1.0, 2.0], [3.0, 4.0]]),
    )

    # The tail lands while actions from the first chunk are being executed.
    stream.deliver(np.array([[5.0, 6.0], [7.0, 8.0]]), final=True)

    # Step 2 immediately folds the arrived tail into the horizon without
    # stalling while executing the second action from chunk 1.
    (action, extra), policy_state = policy.step(
        dm_env.transition(0.0, observation), policy_state
    )
    np.testing.assert_array_equal(action, np.array([3.0, 4.0]))
    np.testing.assert_array_equal(
        extra["action_chunk"],  # pyrefly: ignore[bad-index]
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
    )

    # Step 3 seamlessly continues into chunk 2 with zero action stalls.
    (action, extra), policy_state = policy.step(
        dm_env.transition(0.0, observation), policy_state
    )
    np.testing.assert_array_equal(action, np.array([5.0, 6.0]))
    np.testing.assert_array_equal(
        extra["action_chunk"],  # pyrefly: ignore[bad-index]
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
    )
    self.assertEqual(policy.episode_statistics.action_stall_count, 0)
    policy.close()

  def test_multi_chunk_async_response_triggers_replan_and_adopts_new_horizon(
      self,
  ):
    """Trailing chunks must mark the horizon complete and arrived replans must take over."""
    mock_model = mock.create_autospec(
        bidi_remote_model_interface.BidiRemoteModelInterface,
        instance=True,
        spec_set=True,
    )
    mock_model.get_telemetry.return_value = None

    query_1 = _FakeStream()
    # Query 1 streams two chunks of 3 actions each (6 total).
    query_1.deliver(np.array([[1.0], [2.0], [3.0]]), final=False)
    query_1.deliver(np.array([[4.0], [5.0], [6.0]]), final=True)

    query_2 = _FakeStream()
    mock_model.query_model.side_effect = [
        _chunk_stream(np.zeros((6, 1))),
        query_1,
        query_2,
    ]

    with mock.patch.object(
        bidi_remote_model_interface,
        "BidiRemoteModelInterface",
        return_value=mock_model,
    ):
      policy = bidi_gemini_robotics_policy.BidiGeminiRoboticsPolicy(
          serve_id="test_serve_id",
          task_instruction_key="test_instruction_key",
          image_observation_keys=("test_camera_1",),
          proprioceptive_observation_keys=("test_joint_1",),
          min_replan_interval=2,
          inference_mode=constants.InferenceMode.ASYNCHRONOUS,
      )

    policy.step_spec(self._make_timestep_spec())
    policy_state = policy.initial_state()
    observation = self._make_observation()

    # Steps 1 and 2 execute actions [1.0] and [2.0] from query_1.
    (action_1, _), policy_state = policy.step(
        dm_env.restart(observation), policy_state
    )
    np.testing.assert_array_equal(action_1, np.array([1.0]))

    (action_2, _), policy_state = policy.step(
        dm_env.transition(0.0, observation), policy_state
    )
    np.testing.assert_array_equal(action_2, np.array([2.0]))

    # Step 3: min_replan_interval=2 is reached. Because query_1's second chunk
    # (final=True) was already drained, query_1.is_complete is True and query_2
    # is dispatched on this step while query_1 still has 4 actions left.
    # query_2 has not answered yet, so Step 3 executes [3.0] from query_1.
    (action_3, extra_3), policy_state = policy.step(
        dm_env.transition(0.0, observation), policy_state
    )
    self.assertEqual(extra_3["inference_sent"], 1)  # pyrefly: ignore[bad-index]
    np.testing.assert_array_equal(action_3, np.array([3.0]))
    self.assertEqual(mock_model.query_model.call_count, 3)

    # Now query_2's first chunk arrives while query_1 still has [4.0, 5.0, 6.0].
    # Since 1 step ([3.0]) executed while query_2 was in flight, query_2 should
    # immediately replace query_1 and rebase to index 1 ([20.0]).
    query_2.deliver(np.array([[10.0], [20.0], [30.0]]), final=True)

    (action_4, _), policy_state = policy.step(
        dm_env.transition(0.0, observation), policy_state
    )
    np.testing.assert_array_equal(action_4, np.array([20.0]))
    self.assertEqual(policy.episode_statistics.action_stall_count, 0)
    policy.close()

  def test_a_completed_response_lets_the_next_query_go_out_in_time(self):
    """Draining a response must not leave the policy with nothing to execute.

    A response is complete as soon as its final chunk has been handed over,
    not when a further `next()` reports `StopIteration`. Recognising it only
    at the latter point is one step too late: by then the horizon is already
    empty and the step that discovers it has no query in flight left to serve
    it, which used to surface as a bare `IndexError` from the action buffer.
    """
    mock_model = mock.create_autospec(
        bidi_remote_model_interface.BidiRemoteModelInterface,
        instance=True,
        spec_set=True,
    )
    mock_model.get_telemetry.return_value = None

    first_query = _FakeStream()
    first_query.deliver(np.array([[1.0], [2.0]]), final=True)
    second_query = _FakeStream()
    second_query.deliver(np.array([[3.0], [4.0]]), final=True)
    mock_model.query_model.side_effect = [
        # Spec probe issued by the policy setup.
        _chunk_stream(np.zeros((2, 1))),
        first_query,
        second_query,
    ]

    with mock.patch.object(
        bidi_remote_model_interface,
        "BidiRemoteModelInterface",
        return_value=mock_model,
    ):
      policy = bidi_gemini_robotics_policy.BidiGeminiRoboticsPolicy(
          serve_id="test_serve_id",
          task_instruction_key="test_instruction_key",
          image_observation_keys=("test_camera_1",),
          proprioceptive_observation_keys=("test_joint_1",),
          # High enough that only an exhausted horizon triggers the replan,
          # which is the case the bug lived in.
          min_replan_interval=10,
          inference_mode=constants.InferenceMode.ASYNCHRONOUS,
      )

    policy.step_spec(self._make_timestep_spec())
    policy_state = policy.initial_state()
    observation = self._make_observation()

    (action, _), policy_state = policy.step(
        dm_env.restart(observation), policy_state
    )
    np.testing.assert_array_equal(action, np.array([1.0]))

    (action, _), policy_state = policy.step(
        dm_env.transition(0.0, observation), policy_state
    )
    np.testing.assert_array_equal(action, np.array([2.0]))

    # The horizon is now empty. Because the first query is known to be done,
    # this step issues the second one and executes its first action.
    (action, _), policy_state = policy.step(
        dm_env.transition(0.0, observation), policy_state
    )
    np.testing.assert_array_equal(action, np.array([3.0]))
    # 1 spec probe + 2 policy queries.
    self.assertEqual(mock_model.query_model.call_count, 3)
    policy.close()

  def test_a_response_carrying_no_chunk_is_reported(self):
    """A model that ends a response empty leaves the policy with nothing."""
    mock_model = mock.create_autospec(
        bidi_remote_model_interface.BidiRemoteModelInterface,
        instance=True,
        spec_set=True,
    )
    mock_model.get_telemetry.return_value = None
    # The model ends the response with an empty final chunk.
    empty = _FakeStream()
    empty.deliver(np.zeros((0, 2)), final=True)
    mock_model.query_model.side_effect = [
        # Spec probe issued by the policy setup.
        _chunk_stream(np.zeros((4, 2))),
        empty,
    ]

    with mock.patch.object(
        bidi_remote_model_interface,
        "BidiRemoteModelInterface",
        return_value=mock_model,
    ):
      policy = bidi_gemini_robotics_policy.BidiGeminiRoboticsPolicy(
          serve_id="test_serve_id",
          task_instruction_key="test_instruction_key",
          image_observation_keys=("test_camera_1",),
          proprioceptive_observation_keys=("test_joint_1",),
          min_replan_interval=2,
          inference_mode=constants.InferenceMode.ASYNCHRONOUS,
      )

    policy.step_spec(self._make_timestep_spec())
    policy_state = policy.initial_state()

    with self.assertRaisesRegex(ValueError, "no action left to execute"):
      policy.step(dm_env.restart(self._make_observation()), policy_state)
    policy.close()

  def test_first_query_of_a_new_episode_reports_its_latency(self):
    """Query ids restart per episode, so the stats cursor must restart too.

    The buffer is replaced at an episode boundary and its ids begin again at
    zero. An episode that adopted exactly one query therefore leaves the policy
    holding an id the next episode is about to reuse, which would make the
    first query of that episode look like one whose statistics were already
    reported.
    """
    mock_model = mock.create_autospec(
        bidi_remote_model_interface.BidiRemoteModelInterface,
        instance=True,
        spec_set=True,
    )
    mock_model.get_telemetry.return_value = None
    first_episode = _FakeStream()
    first_episode.deliver(
        np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]), final=True
    )
    second_episode = _FakeStream()
    second_episode.deliver(
        np.array([[7.0, 7.0], [8.0, 8.0], [9.0, 9.0], [10.0, 10.0]]),
        final=True,
    )
    mock_model.query_model.side_effect = [
        # Spec probe issued by the policy setup.
        _chunk_stream(np.zeros((4, 2))),
        first_episode,
        second_episode,
    ]

    with mock.patch.object(
        bidi_remote_model_interface,
        "BidiRemoteModelInterface",
        return_value=mock_model,
    ):
      policy = bidi_gemini_robotics_policy.BidiGeminiRoboticsPolicy(
          serve_id="test_serve_id",
          task_instruction_key="test_instruction_key",
          image_observation_keys=("test_camera_1",),
          proprioceptive_observation_keys=("test_joint_1",),
          min_replan_interval=2,
          inference_mode=constants.InferenceMode.ASYNCHRONOUS,
      )

    policy.step_spec(self._make_timestep_spec())
    observation = self._make_observation()

    # Episode 1 adopts exactly one query, so the policy's stats cursor is left
    # holding the very first id the next episode will hand out.
    policy_state = policy.initial_state()
    (_, extra), _ = policy.step(dm_env.restart(observation), policy_state)
    latency_ms = np.asarray(extra["inference_total_ms"])  # pyrefly: ignore[bad-index]
    np.testing.assert_array_less(_LATENCY_SENTINEL, latency_ms)

    policy_state = policy.initial_state()
    (_, extra), _ = policy.step(dm_env.restart(observation), policy_state)

    latency_ms = np.asarray(extra["inference_total_ms"])  # pyrefly: ignore[bad-index]
    np.testing.assert_array_less(_LATENCY_SENTINEL, latency_ms)
    policy.close()


if __name__ == "__main__":
  absltest.main()
