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

"""Example usage of the Safari EpisodicLogger.

This tutorial script demonstrates how to use the EpisodicLogger to log
timesteps and actions, and reads back data from the generated MCAP file.
"""

import glob
import os
import tempfile
import textwrap
import time

import dm_env
from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np

from safari_sdk.logging.python import constants
from safari_sdk.logging.python import episodic_logger
from safari_sdk.logging.python import mcap_parser_utils


def write_example_to_mcap(output_directory: str):
  """Writes an example episode to an MCAP file.

  This function demonstrates how to use the EpisodicLogger to log
  observation timesteps and actions, and then reads back data from the generated
  MCAP file.

  Args:
      output_directory: The directory to write the MCAP file to.
  """
  print(f"Logging to directory: {output_directory}")
  # Define Specs
  image_key_1 = "my_custom_numpy_image_name"
  proprio_key_1 = "my_custom_proprio_name_1"
  proprio_key_2 = "my_custom_proprio_name_2"
  proprio_key_3 = "my_custom_proprio_name_3"
  proprio_key_4 = "my_custom_proprio_name_4"
  timestamp_key = "my_custom_timestamp_name"
  image_shape = (480, 640, 3)

  example_numpy_image = np.random.randint(0, 255, image_shape, dtype=np.uint8)

  timestep_spec = gdmr_types.TimeStepSpec(
      step_type=gdmr_types.STEP_TYPE_SPEC,
      # Even though reward and discount are not used,
      # they must be specified.
      reward=specs.Array(shape=(), dtype=np.float32),
      discount=specs.Array(shape=(), dtype=np.float32),
      observation={
          "instruction": specs.StringArray(shape=(), name="instruction"),
          # Here we are specifying that our images will be numpy arrays of
          # shape image_shape, and that they will be of type np.uint8.
          image_key_1: specs.Array(shape=image_shape, dtype=np.uint8),
          # You can log multiple proprioceptive keys, which should all be
          # arrays of type np.float64.
          proprio_key_1: specs.Array(shape=(6,), dtype=np.float64),
          proprio_key_2: specs.Array(shape=(6,), dtype=np.float64),
          proprio_key_3: specs.Array(shape=(1,), dtype=np.float64),
          proprio_key_4: specs.Array(shape=(1,), dtype=np.float64),
          # The timestamp should be in nanoseconds, stored as an int64.
          timestamp_key: specs.Array(shape=(), dtype=np.int64),
      },
  )

  action_spec = specs.BoundedArray(
      shape=(14,),
      dtype=np.float32,
      minimum=np.array([-np.inf] * 14, dtype=np.float32),
      maximum=np.array([np.inf] * 14, dtype=np.float32),
  )

  # Create Logger
  logger = episodic_logger.EpisodicLogger.create(
      episodic_logger.EpisodicLoggerConfig(
          agent_id="example_agent",
          task_id="example_task",
          proprioceptive_observation_keys=[
              proprio_key_1,
              proprio_key_2,
              proprio_key_3,
              proprio_key_4,
          ],
          output_directory=output_directory,
          action_spec=action_spec,
          timestep_spec=timestep_spec,
          image_observation_keys=[image_key_1],
          timestamp_key=timestamp_key,
          policy_extra_spec={},
      )
  )

  # Record the FIRST timestep
  # In this example, robot joints (proprio_key) starts at 0.0.
  observation = {
      # The values in the observation dictionary must match the timestep_spec
      # above. This is why everything is cast as numpy arrays of specific
      # dtypes.
      "instruction": np.array("move to the red ball", dtype=object),
      image_key_1: example_numpy_image,
      proprio_key_1: np.zeros((6,), dtype=np.float64),
      proprio_key_2: np.zeros((6,), dtype=np.float64),
      proprio_key_3: np.zeros((1,), dtype=np.float64),
      proprio_key_4: np.zeros((1,), dtype=np.float64),
      timestamp_key: np.array(time.time_ns(), dtype=np.int64),
  }

  initial_timestep = dm_env.TimeStep(
      # The FIRST step_type should be used for the first timestep, but this is
      # not required.
      step_type=dm_env.StepType.FIRST,
      # reward and discount are not used, but they must still
      # match the type specified in timestep_spec.
      reward=np.float32(0.0),
      discount=np.float32(1.0),
      # The observation dictionary contains the important parts of the timestep.
      observation=observation,
  )

  logger.reset(initial_timestep)
  print("Recorded FIRST timestep.")

  for target_joint_position in [0.1, 0.2]:
    ## ####
    # Robot takes action (move to target_joint_position)
    ## ####
    # Logger expects actions as np.arrays matching action_spec.
    action = np.array([target_joint_position] * 14, dtype=np.float32)

    # Record a subsequent timestep (here we say it arrived at new joint
    # position).
    next_observation = {
        # Logger expects observations as np.arrays matching timestep_spec.
        "instruction": np.array("move to the red ball", dtype=object),
        image_key_1: example_numpy_image,
        proprio_key_1: np.array([target_joint_position] * 6, dtype=np.float64),
        proprio_key_2: np.array([target_joint_position] * 6, dtype=np.float64),
        proprio_key_3: np.array([target_joint_position] * 1, dtype=np.float64),
        proprio_key_4: np.array([target_joint_position] * 1, dtype=np.float64),
        timestamp_key: np.array(time.time_ns(), dtype=np.int64),
    }

    next_timestep = dm_env.TimeStep(
        # The MID step_type should be used for intermediate timesteps, but
        # this is not required.
        step_type=dm_env.StepType.MID,
        reward=np.float32(1.0),
        discount=np.float32(0.0),
        observation=next_observation,
    )

    # record_action_and_next_timestep records
    # the action taken at the current step (move to target_joint_position)
    # and the resulting next timestep (arrived at new joint position)
    logger.record_action_and_next_timestep(
        action=action,
        next_timestep=next_timestep,
        policy_extra={},
    )

  ## ####
  # Robot takes action and receives next timestep
  ## ####
  action = np.array([0.5] * 14, dtype=np.float32)

  # Record a subsequent timestep (LAST in this case)
  next_observation = {
      "instruction": np.array("move to the red ball", dtype=object),
      image_key_1: example_numpy_image,
      proprio_key_1: np.array([target_joint_position] * 6, dtype=np.float64),
      proprio_key_2: np.array([target_joint_position] * 6, dtype=np.float64),
      proprio_key_3: np.array([target_joint_position] * 1, dtype=np.float64),
      proprio_key_4: np.array([target_joint_position] * 1, dtype=np.float64),
      timestamp_key: np.array(time.time_ns(), dtype=np.int64),
  }
  next_timestep = dm_env.TimeStep(
      # The LAST step_type should be used when the episode terminates, but
      # this is not required.
      step_type=dm_env.StepType.LAST,
      reward=np.float32(1.0),
      discount=np.float32(0.0),
      observation=next_observation,
  )

  logger.record_action_and_next_timestep(
      action=action,
      next_timestep=next_timestep,
      policy_extra={},
  )
  print("Recorded LAST timestep.")

  # Write the episode to disk
  logger.write()
  logger.stop()
  print(f"Successfully wrote episode to {output_directory}")

  # Find the created .mcap file
  mcap_files = glob.glob(
      os.path.join(output_directory, "**/*.mcap"), recursive=True
  )
  if not mcap_files:
    print("No MCAP files found!")
    return

  mcap_file = mcap_files[0]
  print(f"MCAP file: {mcap_file}")

  # Print out a summary of the created mcap data
  mcap_proto_data = mcap_parser_utils.read_proto_data(
      output_directory,
      constants.TIMESTEP_TOPIC_NAME,
      constants.ACTION_TOPIC_NAME,
      constants.POLICY_EXTRA_TOPIC_NAME,
  )

  print(textwrap.dedent(f"""
        --- MCAP Data Summary ---
        Number of timesteps logged: {len(mcap_proto_data.timesteps)}
        Number of actions logged: {len(mcap_proto_data.actions)}
        Logged timestamps were:
        {np.array([f.features.feature[f'{constants.OBSERVATION_KEY_PREFIX}/my_custom_timestamp_name'].int64_list.value for f in mcap_proto_data.timesteps])}
        Logged proprio were:
        {np.array([f.features.feature[f'{constants.OBSERVATION_KEY_PREFIX}/my_custom_proprio_name_1'].float_list.value for f in mcap_proto_data.timesteps])}
        Logged actions were:
        {np.array([f.features.feature[constants.ACTION_KEY_PREFIX].float_list.value for f in mcap_proto_data.actions])}

        When calling log_action_and_next_timestep, actions are paired with the next observation which make sense for logging. However, note that when the MCAP data is actually stored, observations are paired with the action taken, which makes sense for imitation learning.

        --------------------------
        """).strip())


if __name__ == "__main__":
  # Use a temporary directory
  with tempfile.TemporaryDirectory() as temp_dir:
    write_example_to_mcap(temp_dir)
