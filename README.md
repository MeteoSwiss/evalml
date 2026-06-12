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
  # Optional: skip specific initialisation dates
  # blacklist:
  #   - 2020-01-05T00:00

runs:
  # Each item is either `forecaster`, `temporal_downscaler` or `baseline`
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
  # A temporal_downscaler entry can optionally embed its driving forecaster config inline.
  # - temporal_downscaler:
  #     checkpoint: /path/to/temporal_downscaler.ckpt
  #     label: My temporal downscaler
  #     steps: 0/120/1
  #     forecaster:
  #       checkpoint: https://...
  #       steps: 0/120/6
  - baseline:
      label: COSMO-E
      root: /store_new/mch/msopr/ml/COSMO-E
      steps: 0/120/6

truth:
  label: COSMO KENDA
  root: /scratch/mch/fzanetta/data/anemoi/datasets/mch-co2-an-archive-0p02-2015-2020-6h-v3-pl13.zarr

experiment:
  stratification:
    regions:
      - jura
      - mittelland
      - voralpen
      - alpennordhang
      - innerealpentaeler
      - alpensuedseite
    root: /store_new/mch/msopr/ml/regions/Prognoseregionen_LV95_20220517
  # Optional: categorical thresholds used for binary skill scores, one entry per parameter.
  thresholds:
    TOT_PREC:
      gt: [0.0, 1, 5]
    T_2M:
      lt: [273.15]
      gt: [288.15, 298.15]
  dashboard:
    # Stratification dimensions to include in the experiment dashboard (any of season, region, init_hour).
    stratification:
      - season
  # Optional: named scorecards comparing each forecaster against a chosen baseline.
  scorecards:
    enabled: true
    sections:
      short_range:
        # Baseline label — must match the `label` field of a baseline entry in `runs`.
        baseline: COSMO-E
        # Lead-time range as start/stop/step (hours).
        lead_times: "0/120/6"
        # Stratification dimension to use as scorecard columns (e.g. region, season).
        stratification: region
        # Variables and metrics as scorecard rows. Format: VAR:METRIC1,METRIC2,...
        # Supported metrics: RMSE, R2, ETS, POD, FAR (categorical metrics require thresholds).
        variables:
          - "T_2M:RMSE,R2"
          - "TOT_PREC:RMSE,ETS"

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

The `runs` list accepts `forecaster`, `temporal_downscaler`, and `baseline` entries. For `dates`, you can either provide a `start` / `end` / `frequency` block as above or an explicit list of ISO timestamps for case-study style runs. Stratification, thresholds, dashboard, and scorecard settings are all grouped under the `experiment` key.

You can then run it with:

```bash
evalml experiment path/to/experiment/config.yaml --report
```

### Truth sources

The `truth.root` value selects how the ground truth is loaded:

- **Analysis Zarr** — a path ending in `.zarr` (anemoi analysis dataset).
- **PeakWeather** — a path containing `peakweather` (SwissMetNet station obs from Hugging Face).
- **DWH / jretrievedwh** — a `jretrievedwh:` marker string fetching SwissMetNet (SMN)
  surface observations live from the MeteoSwiss data warehouse. Variables are mapped to
  ICON names in SI units (temperatures in K, pressure in Pa, precipitation as the hourly
  sum); wind `U_10M`/`V_10M` are derived from speed + direction.

  Marker syntax (station selection is required; pick one of group/locations/bbox):

  ```yaml
  truth:
    label: SwissMetNet (DWH)
    root: jretrievedwh:SwissMetNet                  # stn_group (default group)
    # root: jretrievedwh:locations=ARO,KLO,LUG      # explicit nat_abbr list
    # root: jretrievedwh:bbox=45.8,47.8,5.9,10.5    # minlat,maxlat,minlon,maxlon
    # append ;stage=devt to target a non-prod DWH stage (prod|depl|devt)
  ```

  **Prerequisites:** `jretrievedwh.py` must be on `$PATH` (e.g.
  `/oprusers/osm/opr.inn/bin`) and `$OPR_HOME` set with a readable
  `.jretrievedwh-conf.<stage>.py` conf file. No data is pre-downloaded — the obs are
  queried at verification time.


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

In some cases, a single script may be shared by more than one rule (e.g. `inference_prepare.py` is used by both `inference_prepare_forecaster` and `inference_prepare_temporal_downscaler`).

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
    ├── plots/
    └── scorecards/
```

### Wildcard conventions

| Wildcard | Format | Example |
|---|---|---|
| `{env_id}` | `{type}-{model_id}-{env_hash}` | `forecaster-1a2b-c3d4` |
| `{run_id}` | `{env_id}/{config_hash}` | `forecaster-1a2b-c3d4/e5f6` |
| `{baseline_id}` | `baseline-{hash}` | `baseline-3f9a` |
| `{init_time}` | `%Y%m%d%H%M` | `202001011200` |
| `{experiment}` | `{YYYYMMDD}_{label}_{hash}` | `20260331_demo_a1b2` |
| `{param}` | variable name | `T_2M`, `TOT_PREC` |
| `{region}` | geographic region slug | `switzerland`, `globe` |
| `{leadtime}` | zero-padded hours | `000`, `006`, `024` |


## earthkit v1.0rc support

The adoption of the first upcoming stable release of earthkit has started.
Some issues may still be present. A known issue that is affecting us is that the automatic download and caching of files required
by earthkit to support the ICON-CH grids is currently broken.

The workflow now works around this automatically: the `data_download_eckit_geo_grids` rule fetches
the required ICON-CH1/CH2 grid-definition files into eckit's default geo cache
(`$HOME/.local/share/eckit/geo/grid/icon`) before any plotting or verification step that needs them.
No manual setup is required.
