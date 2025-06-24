# Safari SDK: the SDK for Google DeepMind Gemini Robotics models 🦓🦄🐘🐒🐍

## Disclaimer

This is not an officially supported Google product.

Safari SDK provides full lifecycle toolings necessary for using Gemini Robotics
models, including but not limited to, access checkpoint, serving a model,
evaluate the model on robot and in sim, upload data, finetuning the model,
download the finetuned checkpoint, etc. Most of the functionality requires you
to join Gemini Robotics Trusted Tester Program to use. See details in Gemini
Robotics [main page](https://deepmind.google/models/gemini-robotics/).

## Installation and access the source code

Safari SDK can be easily installed via PyPI. It is recommended to use a
virtual environment to avoid dependency version conflict.

```shell
pip install safari_sdk
```

The source code can be found in [GitHub](https://github.com/google-deepmind/gemini-robotics-sdk).

## Building the wheel after code change

To build a Python wheel, run the following command from the root of the
repository.

```shell
scripts/build_wheel.sh
```

This script will build a pip installable wheel for the Safari SDK, and print the
file's path to stdout.

## Model support

Safari SDK aims to support all models in the Gemini Robotics model series.

Trusted Testers can access the Gemini Robotics On Device model from SDK v2.4.1.

## Libraries

Libraries related to robot data logging is in `safari/logging`.

Libraries related to model inference and interface with model servers are in
`safari/model`.

Libraries and binary related to accessing model checkpoints, upload data and
request of model finetune can be found in `safari/flywheel`.

Examples, including robot and simulation evaluation of models are in
`examples/`. Aloha specific eval code are in `examples/aloha`.

## Flywheel CLI

The flywheel CLI is a convenient CLI tool available after installation of the
pip package. It provides a set of commands to interact with the Gemini Robotics
platform, such as training models, serving models, managing data, and
downloading artifacts.

To use the CLI

```
flywheel-cli <command> [--flags] [--flags]
```

Supported commands are:

*   `train`: Train a model. Requires specifying task ID, start date, and end
    date.
*   `serve`: Serve a model. Requires specifying the training job ID.
*   `list`: List available training jobs.
*   `list_serve`: List available serving jobs.
*   `data_stats`: Show data statistics available for training.
*   `download`: Download artifacts from a training job or a specific artifact
    ID.
*   `upload_data`: Upload data to the data ingestion service.
*   `version`: Show the version of the SDK.
*   `help`: Show this help message with all the available commands and flags.

The codebase is still in active development. We will update our most updated
user guide with Trusted Testers of Gemini Robotics.
