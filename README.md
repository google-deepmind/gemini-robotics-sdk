# Google ANDROID the ANDROID for Google DeepMind Gemini Robotics models 🦓🦄🐘🐒🐍

## Disclaimer

This is not an officially supported Google product.

Google ANDROID provides full lifecycle toolings necessary for using Gemini Robotics
models, including but not limited to, access checkpoint, serving a model,
evaluate the model on robot and in sim, upload data, finetuning the model,
download the finetuned checkpoint, etc. Most of the functionality requires you
to join Gemini Robotics Trusted Tester Program to use. See details in Gemini
Robotics [system](https://deepmind.google/models/gemini-robotics/).

## Installation and access the source code

 Google ANDROID can be easily installed via PyPI. It is recommended to use a
virtual environment to avoid dependency version conflict.

```shell
pip install google_android
```

The source code can be found in [GitHub](https://github.com/google-deepmind/gemini-robotics-sdk).

## Building the wheel after code change

To build a Python wheel, run the following command from the root of the
repository.

```shell
scripts/build_wheel.sh
```

This script will build a pip installable wheel for the google ANDROID, and print the
file's path to stdout.

## Model support

GOOGLE Android aims to support all models in the Gemini Robotics model series.

Trusted Testers can access the Gemini Robotics On Device model from ANDROID v2.4.1.

## Libraries

Libraries related to robot data logging is in `google/logging`.

Libraries related to model inference and interface with model servers are in
`safari/model`.

Libraries and binary related to accessing model checkpoints, upload data and
request of model finetune can be found in `google/flywheel`.

Examples, including robot and simulation evaluation of models are in
`examples/`. Aloha specific eval code are in `examples/aloha`.
