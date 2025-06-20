# Safari: An API to Google DeepMind Robotics models 🦓🦄🐘🐒🐍

## Disclaimer

This is not an officially supported Google product.

## Installation

The safari_sdk package can be built and installed in a virtual environment with
cmake. From the root of the repository, run:

```shell
mkdir build && cd build && cmake ..
```

The build directory may be placed elsewhere if desired, but standard practice is
to place it under the repository root.

Note: Only use `&&` to chain commands as done above in an interactive bash
prompt, not in a script. Using `&&` in a script can cause it to continue
executing even if there are errors; instead put each command on its own line.

The safari_sdk package can then be installed in a virtual environment (which is
also created under the `build` folder) with this command:

```shell
make pip_install
```

Alternatively it can be installed in editable mode. In this mode, the virtual
environment will contain a reference to the python files in the repository, so
those files can subsequently be modified and programs rerun *without* needing to
rerun the pip installation.

```shell
make pip_install_e
```

Note: this only applies to python source files, *not* to proto files. If proto
files are modified, they will need to be rebuilt by running `make py_proto`
(rerunning `make pip_install` will also accomplish this).

After running the pip install, enter the virtual environment (which has the
safari_sdk installed in it) by running this from the build directory (or prepend
the path to the build directory to run it from a different directory):

```shell
source safari_venv/bin/activate
```

## Building the Wheel

To build a Python wheel, run the following command from the root of the
repository.

```shell
scripts/build_wheel.sh
```

This script will build a pip installable wheel for the Safari SDK, and print the
file's path to stdout.

## Logging

Note: please obtain the API key and project ID from the Google Deepmind Robotics
team.

### Overview

![Safari Logging Overview](./assets/safari_logging.png)

#### Common Data Schema in Protobuf

To examine the source code,

```shell
unzip <path_to_whl_file>
```

The common data schema in protobuf is found in `safari_sdk/protos`. We required
the users to convert their ROS1/ROS2 messages to the common data schema. Please
reach out to the Google Deepmind Robotics team for any questions on mapping.

In the case where your ROS message type is "composite", we prefer you to split
the composite message into multiple common messages.

In the case where your ROS message type is just not covered, please reach out to
the Google Deepmind Robotics team and we will look into it and may consider add
it and update the SDK.

### Stream Logger Integration

All timestamps are unix time in nanoseconds.

#### Step 0 - Message conversion

Your conversion utils to map your (ROS1, ROS2) messages to the common data
schema, found in `safari_sdk/protos`.

#### Step 1 - logger initialization

```shell
from safari_sdk.logging.python import stream_logger as stream_logger_lib

stream_logger = stream_logger_lib.StreamLogger(
    agent_id=<your agent id string>,
    output_directory=<your output directory string>,
    required_topics=<a set of required topic strings>,
    optional_topics=<a set of optional topic strings>,
)
```

#### Step 2 - logger call to `update_synchronization_and_maybe_write_message`

This should be called in the ROS subscriber callback.

```shell
stream_logger.update_synchronization_and_maybe_write_message(
    topic=<topic string>,
    message=<sensor message in protobuf>,
    publish_time_nsec=<message creation time, ex. message.header.stamp>,
    log_time_nsec=<message arrival time in the data collection app, ex. ROS Subscriber Callback>
)
```

#### Step 3 - logger call to `maybe_write_sync_message`

This should be called during the runloop to log the synchronization message.

```shell
stream_logger.maybe_write_sync_message(
    publish_time_nsec=<current runloop clock time>
)
```

#### Step 4 - logger call to `start_session`, `stop_session` & `add_session_label`

These should be called during the episode start and stop triggers.

```shell
stream_logger.start_session(
    start_nsec=<current runloop clock time>,
    task_id=<your task_id string>,
    output_file_prefix=<your log filename prefix, ex. {task_id}_{datetime_string}>
)
```

```shell
stream_logger.stop_session(stop_nsec=<current runloop clock time>)
```

```shell
# For adding labels to the session.
## from google.protobuf import struct_pb2
## from safari_sdk.protos import label_pb2
stream_logger.add_session_label(
    label_pb2.LabelMessage(
        key=<label key, please share the key strings with the Google Deepmind Robotics team first.>,
        label_value=struct_pb2.Value(string_value=<label value>),
    )
)
```

Note that `start_session` returns a bool value to indicate that all the required
topics have been received. A common paradigm is to poll the return value of
`start_session` in the runloop.

```shell
while not stream_logger.start_session(...):
    print('Waiting for stream logger to receive all required topics.')
    received_topics = set(
        stream_logger.get_latest_sync_message().last_timestamp_by_topic.keys()
    )
    missing_topics = stream_logging_required_topics - received_topics
    for topic in missing_topics:
        print(f'  Missing topic: {topic}')
    time.sleep(2)
```

#### Expected output

Typically, one episode (designated by `start_session` and `stop_session`) is
logged into one mcap log file. If the episode is large, a new file is produced
for ~every 2GB (configurable). Aside from sensor & sync messages in protobuf,
each mcap log file exactly one file metadata message (topic name:
`/file_metadata`). The last file for an episode would contain exactly one
episode message (topic name: `/session`).

Please refer to https://mcap.dev/guides/python/protobuf for how to read the mcap
log file.

### Data Upload

Please use the upload_data command in `flywheel_cli` to upload data.

## Flywheel CLI

Note: please get your GCP project allow-listed by the Google DeepMind Robotics
team and generate an API key from that project. The API key determines the
workspace and controls access, please make sure you use the correct key and keep
it secure.

### Upload Data

```shell
flywheel-cli upload_data --api_key=xxx --upload_data_robot_id robot_id --upload_data_directory=/path/to/data/files
```

The upload data command will upload all *.mcap files in the directory. It will
rename successfully uploaded file to add a ".uploaded" suffix.

### Check Available Data

```shell
flywheel-cli data_stats --api_key=xxx
```

Example output:

```
Robot id     Task id     Date       Count     Success count
robot_1      task_1      20250301   25        23
robot_2      task_2      20250301   32        17
robot_1      task_1      20250225   19        0
```

### Create training job

```shell
flywheel-cli train --api_key=xxx --task_id=data_coll --start_date=20241201  --end_date=20250101 --train_recipe=RECIPE
```

Task id is the task id of the data collection task. The start and end date are
in the format of `YYYYMMDD`.

`RECIPE` can be `narrow` or `gemini_robotics_v1`. If `--train_recipe` is not
included in the command, the default is `narrow`.

Example output:

```json
{
    "training_job_id": "run-xxxxxxxx-yyyyyy-zzzzzz"
}
```

### List existing training jobs

```shell
flywheel-cli list --api_key=xxx
```

Example output:

```
Training jobs id              Status                        Training type                    Task id       robot id        Start date  End date
run-xxxxxxxx-yyyyyy-zzzzz4    TRAINING_JOB_STAGE_PENDING    TRAINING_TYPE_NARROW             ['task_id_1]  ['robot_id_1']  20250402    20250403
run-xxxxxxxx-yyyyyy-zzzzz3    TRAINING_JOB_STAGE_RUNNING    TRAINING_TYPE_GEMINI_ROBOTICS_V1 ['task_id_1]  ['robot_id_1']  20250402    20250403
run-xxxxxxxx-yyyyyy-zzzzz2    TRAINING_JOB_STAGE_DONE       TRAINING_TYPE_NARROW             ['task_id_1]  None            20250402    20250403
run-xxxxxxxx-yyyyyy-zzzzz1    TRAINING_JOB_STAGE_DONE       TRAINING_TYPE_NARROW             ['task_id_1]  None            20250402    20250403
```

### Download model artifact

Note: Please get the training job ID from the previous step.

Note: Download model artifacts only works for Narrow models.

```shell
flywheel-cli download --api_key=xxx --training_job_id=run-xxx
```

Example output:

```
Please download the artifacts from https://storage.googleapis.com/....
```

The downloaded file is a zip file containing an saved model directory. Please
unzip it before use it for eval.

### Start Serving Gemini Robotics Models

```shell
flywheel-cli serve --api_key=xxx --training_job_id=run-xxx --training_job_id=run-xxx --model_checkpoint_number=5
```

```json
{
    "serve_run_id": "serve-xxxxxxxx-yyyyyy-zzzzzz"
}
```

### List Serving Gemini Robotics Models

```shell
flywheel-cli list_serve --api_key=xxx
```

```
Serve job id                  Training jobs id              Training type                     Task id       robot id        Start date  End date
serve-xxxxxxxx-yyyyyy-zzzzz4  run-xxxxxxxx-yyyyyy-zzzzz4    TRAINING_TYPE_GEMINI_ROBOTICS_V1  ['task_id_1]  ['robot_id_1']  20250402    20250403
```
