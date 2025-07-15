# EvalML

[![Static Badge](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity/emerging_badge.svg)](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity)
[![Snakemake](https://img.shields.io/badge/snakemake-≥8.0.0-brightgreen.svg)](https://snakemake.github.io)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Actions status](https://github.com/meteoswiss/evalml/actions/workflows/ci.yaml/badge.svg)](https://github.com/meteoswiss/evalml/actions)

Run evaluation pipelines for anemoi models.


## Getting started

1. [Installation](#installation)
2. [Credentials setup](#credentials-setup)
3. [Workspace setup](#workspace-setup)

## Example

To launch an experiment, prepare a config file defining your experiment, e.g.

```yaml
# yaml-language-server: $schema=../workflow/tools/config.schema.json
description: |
  This is an experiment to do blabla.

dates:
  start: 2020-01-01T12:00
  end: 2020-01-10T00:00
  frequency: 54h

lead_time: 120h

runs:
  stage_D-cerra-N320:
    run_id: 2f962c89ff644ca7940072fa9cd088ec
    label: Stage D - N320 global grid with CERRA finetuning
  stage_D-cerra-N320-low_lam:
    run_id: d0846032fc7248a58b089cbe8fa4c511
    label: Stage D - N320 global grid with CERRA finetuning - low LAM weight

baseline: COSMO-E

execution:
  run_group_size: 4

locations:
  output_root: output/
  mlflow_uri:
    - https://servicedepl.meteoswiss.ch/mlstore
    - https://mlflow.ecmwf.int

profile:
  executor: slurm
  default-resources:
    slurm_partition: "postproc"
    cpus_per_task: 1
    mem_mb_per_cpu: 1800
    runtime: "1h"
  jobs: 30
```

You can then run it with:

```bash
evalml experiment path/to/experiment/config.yaml
```


## Installation

This project uses [uv](https://github.com/astral-sh/uv). Download and install it with

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

then, install the project and its dependencies with `uv sync` and activate the virtual
environment with `source .venv/bin/activate`.

## Credentials setup

Some experiments are stored on the ECMWF-hosted MLflow server:
[https://mlflow.ecmwf.int](https://mlflow.ecmwf.int). To access these runs in the
evaluation workflow, you need to authenticate using a valid token. Run the following
commands **once** to log in and obtain a token:

```bash
uv pip install anemoi-training --no-deps
anemoi-training mlflow login --url https://mlflow.ecmwf.int
```

You will be prompted to paste a seed token obtained from https://mlflow.ecmwf.int/seed.
After this step, your token is stored locally and used for subsequent runs. Tokens are
valid for 30 days. Every training or evaluation run within this period automatically
extends the token by another 30 days. It’s good practice to run the login command before
executing the workflow to ensure your token is still valid.

## Workspace setup

By default, data produced by the workflow will be stored under `output/` in your working directory.
We suggest that you set up a symlink to a directory on your scratch:

```bash
mkdir -p $SCRATCH/evalenv/output
ln -s $SCRATCH/evalenv/output output
```

This way data will be written to your scratch, but you will still be able to browse it with your IDE.

If you are using VSCode, we advise that you install the YAML extension, which will enable config validation, autocompletion, hovering support, and more.
