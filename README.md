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

## Agent

The Safari SDK includes a comprehensive agent framework for building interactive
robotics agents powered by Gemini models. See
[YouTube Video: Gemini Robotics 1.5: Using agentic capabilities](https://youtu.be/AMRxbIO04kQ?si=UFILQ9IOgfw7RTus).
The framework is located in `safari/agent/framework` and provides a modular
architecture for creating agents that can perceive their environment, reason
about tasks, and control robot hardware.

### Key Components

**Agents** (`safari/agent/framework/agents/`): Base agent classes that integrate
with the Gemini Live API to provide conversational interaction and tool use
capabilities.

**Embodiments** (`safari/agent/framework/embodiments/`): Hardware-specific
interfaces that connect agents to physical robot systems (e.g., Aloha robot).
Each embodiment provides tools for robot control.

**Tools** (`safari/agent/framework/tools/`): Modular capabilities that agents
can use, including:

*   Run instruction
*   Success detection
*   Scene description
*   etc.

**Event Bus** (`safari/agent/framework/event_bus/`): Asynchronous
publish-subscribe system for communication between agent components.

**Configuration** (`safari/agent/framework/config.py`): Centralized
configuration management using `AgentFrameworkConfig`, supporting both
programmatic configuration and flag-based setup.

### Aloha Agent Example

The `examples/aloha/agent/` directory contains agent implementations for the
Aloha robot platform.

The primary example is `simple_agent.py`, which provides a conversational agent
that can control the Aloha robot using natural language instructions.

To run the Aloha agent, use the provided `run.py` script:

```shell
python examples/aloha/agent/run.py --agent_name=simple_agent
```

The Aloha agent demonstrates integration of vision-based robot control,
multi-camera perception, and conversational interaction with Gemini models.

Alternatively, you build your own agent using the agent framework.

The codebase is still in active development. We will update our most updated
user guide with Trusted Testers of Gemini Robotics.
