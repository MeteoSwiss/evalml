# EvalML

[![Static Badge](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity/emerging_badge.svg)](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity)
[![Snakemake](https://img.shields.io/badge/snakemake-≥8.0.0-brightgreen.svg)](https://snakemake.github.io)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Actions status](https://github.com/meteoswiss/evalml/actions/workflows/ci.yaml/badge.svg)](https://github.com/meteoswiss/evalml/actions)

Run evaluation pipelines for data-driven weather models built with [Anemoi](https://anemoi.readthedocs.io/).


## Getting started

1. [Installation](#installation)
2. [Credentials setup](#credentials-setup)
3. [Workspace setup](#workspace-setup)

## Features:
- [Experiments](#experiment): compare model performance via standard and diagnostic verification
- [Showcasing](#showcase): produce visual material for specific events
- [Sandboxing](#sandbox): generate an isolated inference development environments for any model

## Quick example

To run an experiment, prepare a demo config file like the one below and adapt it to your setup:

```yaml
# yaml-language-server: $schema=../workflow/tools/config.schema.json
description: |
  Demo experiment: compare two forecaster checkpoints against the same baseline and truth data.

# Optional: used in the output directory name. If omitted, the config file name is used.
config_label: co2-forecasters-demo

# Choose one date style:
# 1. A regular range with a run frequency (shown here)
# 2. An explicit list of ISO timestamps for case studies or showcases
dates:
  start: 2020-01-01T00:00
  end: 2020-01-10T00:00
  frequency: 60h

runs:
  # Each item is either `forecaster`, `interpolator` or `baseline`
  - forecaster:
      # `checkpoint` may point to a supported MLflow run URL, a Hugging Face `.ckpt` URL, or a local checkpoint path.
      checkpoint: https://servicedepl.meteoswiss.ch/mlstore#/experiments/228/runs/2f962c89ff644ca7940072fa9cd088ec
      # Labels are what appear in plots, tables, and reports.
      label: Stage D - N320 global grid with CERRA finetuning
      # Lead times follow start/end/step in hours.
      steps: 0/120/6
      # `config` points to the inference config template for the run. If omitted, evalml uses the bundled default for the run type.
      config: resources/inference/configs/sgm-forecaster-global.yaml
      # Optional extra dependencies needed by this checkpoint at inference time.
      extra_requirements:
        - git+https://github.com/ecmwf/anemoi-inference.git@0.8.3
  - forecaster:
      checkpoint: https://mlflow.ecmwf.int/#/experiments/103/runs/d0846032fc7248a58b089cbe8fa4c511
      label: M-1 forecaster
      steps: 0/120/6
      config: resources/inference/configs/sgm-forecaster-global_trimedge.yaml
  - baseline:
      label: COSMO-E
      root: /store_new/mch/msopr/ml/COSMO-E
      steps: 0/120/6

truth:
  label: COSMO KENDA
  root: /scratch/mch/fzanetta/data/anemoi/datasets/mch-co2-an-archive-0p02-2015-2020-6h-v3-pl13.zarr

stratification:
  regions:
    - jura
    - mittelland
    - voralpen
    - alpennordhang
    - innerealpentaeler
    - alpensuedseite
  root: /scratch/mch/bhendj/regions/Prognoseregionen_LV95_20220517

locations:
  # All workflow outputs are written under this root.
  output_root: output/

profile:
  # Passed through to Snakemake. Tune this block to match your cluster or local executor.
  executor: slurm
  global_resources:
    # Limits total concurrent GPU use across submitted jobs.
    gpus: 16
  default_resources:
    slurm_partition: "postproc"
    cpus_per_task: 1
    mem_mb_per_cpu: 1800
    runtime: "1h"
  jobs: 50
  batch_rules:
    # Group many small plotting jobs into fewer submissions.
    plot_forecast_frame: 32
```

The `runs` list accepts both `forecaster` and `interpolator` entries. For `dates`, you can either provide a `start` / `end` / `frequency` block as above or an explicit list of ISO timestamps for case-study style runs.

You can then run it with:

```bash
evalml experiment path/to/experiment/config.yaml --report
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

## Workflow development guidelines

This section defines the conventions to follow when adding or modifying rules, scripts, and outputs inside `workflow/`.

### Rule names

Rules use the pattern `{module}_{operation}[_{sub_operation}]` in snake_case. The module prefix groups rules by their place in the pipeline:

| Module | Rules prefix | Purpose |
|---|---|---|
| {module}.smk | `data_` | Input data preparation |
| {module}.smk | `inference_` | Model checkpoint retrieval, environment setup, and execution |
| {module}.smk | `verification_` | Metrics calculation, aggregation, and metric plots |
| {module}.smk | `plot_` | Forecast visualisation (frames, animations, meteograms) |
| {module}.smk | `report_` | Dashboard and HTML report generation |

Aggregate target rules follow the pattern `{module}_all` (e.g. `inference_all`, `verification_metrics_all`). Top-level entry points are named after the workflow mode: `experiment_all`, `showcase_all`, `sandbox_all`.

### Script names

Scripts live in `workflow/scripts/` and mirror the rule that calls them:

```
{module}_{operation}.py         # standard Python scripts
{module}_{operation}.mo.py      # interactive Marimo notebooks
```

Examples: `inference_prepare.py`, `verification_metrics.py`, `verification_aggregation.py`, `plot_forecast_frame.mo.py`.

In some cases, a single script may be shared by more than one rule (e.g. `inference_prepare.py` is used by both `inference_prepare_forecaster` and `inference_prepare_interpolator`).

### Log file paths

Every rule declares a log file under:

```
{OUT_ROOT}/logs/{rule_name}/{wildcards}.log
```

When a rule produces a sentinel file (`.ok`) to mark successful completion, it is placed alongside the log:

```
{OUT_ROOT}/logs/{rule_name}/{wildcards}.ok
```

Multiple wildcards are joined with a hyphen: `{run_id}-{init_time}.log`.

### Output directory layout

All outputs are rooted at `OUT_ROOT` (from `locations.output_root` in the config):

```
{OUT_ROOT}/
├── data/
│   ├── runs/{env_id}/                        # per-environment artefacts (shared across config changes)
│   │   ├── inference-last.ckpt
│   │   ├── requirements.txt
│   │   ├── venv.squashfs
│   │   └── {config_hash}/                    # per-run-config artefacts
│   │       ├── summary.md
│   │       ├── verif_aggregated.nc
│   │       └── {init_time}/                  # per-initialisation-time artefacts
│   │           ├── config.yaml
│   │           ├── grib/
│   │           └── verif.nc
│   ├── baselines/{baseline_id}/
│   │   └── {init_time}/verif.nc
│   └── observations/
├── logs/                                      # one sub-directory per rule
└── results/{experiment_name}/               # final products
    ├── dashboard/
    └── plots/
```

### Wildcard conventions

| Wildcard | Format | Example |
|---|---|---|
| `{env_id}` | `{type}-{model_id}-{env_hash}` | `forecaster-1a2b-c3d4` |
| `{run_id}` | `{env_id}/{config_hash}` | `forecaster-1a2b-c3d4/e5f6` |
| `{baseline_id}` | slug derived from root path | `COSMO-E` |
| `{init_time}` | `%Y%m%d%H%M` | `202001011200` |
| `{experiment}` | `{YYYYMMDD}_{label}_{hash}` | `20260331_demo_a1b2` |
| `{param}` | variable name | `T_2M`, `TOT_PREC` |
| `{region}` | geographic region slug | `switzerland`, `globe` |
| `{leadtime}` | zero-padded hours | `000`, `006`, `024` |
