# Model Evaluation Pipeline

[![Snakemake](https://img.shields.io/badge/snakemake-≥8.0.0-brightgreen.svg)](https://snakemake.github.io)
[![run with conda](http://img.shields.io/badge/run%20with-conda-3EB049?labelColor=000000&logo=anaconda)](https://docs.conda.io/en/latest/)

A Snakemake workflow for anemoi models verification and performance auditing.

## Before you start

This project uses `conda`, download and install it through
[miniforge](https://github.com/conda-forge/miniforge) with:

    curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash Miniforge3-Linux-x86_64.sh

Some experiments are stored on the ECMWF-hosted MLflow server:
[https://mlflow.ecmwf.int](https://mlflow.ecmwf.int). To access these runs in the
evaluation workflow, you need to authenticate using a valid token. Run the following
command **once** to log in and obtain a token:

```bash
uv run --with 'anemoi-training' anemoi-training mlflow login --url https://mlflow.ecmwf.int
```

You will be prompted to paste a seed token obtained from https://mlflow.ecmwf.int/seed.
After this step, your token is stored locally and used for subsequent runs.

This will reuse the previously stored URL and token. Tokens are valid for 30 days.
Every training or evaluation run within this period automatically extends the token by
another 30 days. It’s good practice to run the login command before executing the
workflow to ensure your token is still valid.

By default, data produced by the workflow will be stored under `output/` in your working directory.
We suggest that you set up a symlink to a directory on your scratch:

```bash
mkdir -p $SCRATCH/mch-anemoi-evaluation/output
ln -s $SCRATCH/mch-anemoi-evaluation/output output
```

This way data will be written to your scratch, but you will still be able to browse it with your IDE.

## Installation

Clone the `mch-anemoi-evaluation` repository and navigate to the project root directory:

    git clone git@github.com:MeteoSwiss/mch-anemoi-evaluation.git
    cd mch-anemoi-evaluation

Create and activate the conda environment for Snakemake with:

    mamba env create -f environment.yaml
    conda activate evalml

Install the project CLI with:

    pip install -e .

To see available commands, use:

    evalml --help

## Execution

To launch an experiment, prepare a config file defining your experiment, e.g.

```yaml
description: |
  This is an experiment to do blabla.

init_times: 
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
  software-deployment-method: conda
  executor: slurm
  default-resources:
    slurm_partition: "postproc"
    cpus_per_task: 1
    mem_mb_per_cpu: 1800
    runtime: "1h"
  jobs: 30
  use-conda: true

```

You can then run it with:

```bash
evalml launch experiment path/to/experiment/config.yaml
```
