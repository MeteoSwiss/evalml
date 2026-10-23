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
      - icon  # first item will be used as default verification region
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
- **DWH / jretrievedwh** — a `jretrievedwh:` marker string fetching surface observations
  (e.g. SMN) live from the MeteoSwiss data warehouse. Variables are mapped to
  ICON names in SI units (temperatures in K, pressure in Pa, precipitation as the hourly
  sum); wind `U_10M`/`V_10M` are derived from speed + direction.

  Marker syntax (station selection is required; pick one of group/locations/bbox):

  ```yaml
  truth:
    label: SwissMetNet
    root: jretrievedwh:SwissMetNet                  # station group (default if none is given)
    # root: jretrievedwh:locations=ARO,KLO,LUG      # explicit nat_abbr list
    # root: jretrievedwh:bbox=45.8,47.8,5.9,10.5    # minlat,maxlat,minlon,maxlon
  ```

  Optional keys are appended with `;`, e.g. `jretrievedwh:bbox=45.7,48.0,5.8,10.8;use_limitation=40`.
  `use_limitation` is passed to jretrieve's `--use-limitation` and selects which stations
  are returned (default: 40).

  **Prerequisites:** `jretrievedwh.py` must be on `$PATH` (falls back to
  `/oprusers/osm/opr.inn/bin/jretrievedwh.py`) and DWH credentials must be
  available — see [Credentials setup](#credentials-setup). No data is
  pre-downloaded — the obs are queried at verification time.

### Holdout stations

With station truth (`jretrievedwh:`), you can name stations to verify as a separate
group, for example stations withheld from observation nudging, to get an independent check:

```yaml
experiment:
  station_holdout: [CHM, FRE, RUE, BAS]   # station nat_abbr
```

Every model and baseline is then verified separately on three station groups:
`all`, `holdout` (the listed stations) and `holdin` (all others). The dashboard
gets a station-group selector; scorecards and the standard plots use `all`.

- Holdout stations missing from the truth are ignored; if none are present, or the truth
  is gridded (`.zarr`), only `all` is computed and a warning is logged.
- The list is written to a CSV file that the workflow tracks, so changing it reruns the
  affected verification and inference.
- An inference config can read the same list from `holdout_stations.csv` in its working
  directory (e.g. `holdout_station_file: holdout_stations.csv` for the
  `nudge_toward_observation` filter).


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

### DWH / jretrieve credentials

To use `jretrievedwh:` as a truth source, provide a client ID and secret for
the MeteoSwiss service account. Set them as environment variables:

```bash
export JRETRIEVE_CLIENT_ID=your-client-id
export JRETRIEVE_CLIENT_SECRET=your-client-secret
```

or place them in a `.env` file at the project root (next to `.jretrievedwh-conf.prod.py`):

```
JRETRIEVE_CLIENT_ID=your-client-id
JRETRIEVE_CLIENT_SECRET=your-client-secret
```

The `.env` file is listed in `.gitignore` and is never committed.

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


## Testing documentation

For details about the CI/CD setup, see the [the CI/CD README](ci/README.md).

### Inference fixture for integration tests

In order to reduce the time needed to run tests, and avoid unnecessary
GPU usage, some integration tests use a test fixture for inference.
The implementation of the fixture feature in the EvalML CLI can be found in
[`src/evalml/fixtures.py`](src/evalml/fixtures.py)

#### Using test fixtures: `test_showcase_meteogram` example

This section describes how test fixture is used in
`test_meteogram_small.py:test_showcase_meteogram`. Most other integration
tests use a similar process to bypass inference.

##### How it works

The showcase longtest replays a frozen copy of the forecaster inference GRIB
instead of running inference, so it needs no GPU/MLflow/sandbox build (it still
needs DWH access for truth). `meteogram_small.yaml` **always** replays: it sets
`fixture_root`, so any `evalml showcase` of this config uses the fixture. The
longtest reads that path from the config and is skipped if the fixture is not
populated (no `MANIFEST.yaml`). When `fixture_root` is active, the workflow's
start banner prints `Inference: REPLAYED FROM FIXTURE <path>`.

##### Fixture setup
Creating or refreshing the fixture is needed on a one-time basis for new test fixtures,
and needs to be re-run whenever the checkpoint or config changes. Because
it runs inference, it requires a GPU node. Use the following steps
(meteogram used as an example):

1. Temporarily comment out `fixture_root` in `meteogram_small.yaml` (otherwise
   the config forces replay and won't run real inference), then run a real
   showcase: `evalml showcase tests/integration/configs/meteogram_small.yaml`
2. Capture it (leave `fixture_root` commented for this step, or pass the config
   as-is — capture ignores `fixture_root`):
   `evalml capture-fixture tests/integration/configs/meteogram_small.yaml <FIXTURE_ROOT>`
3. Restore (uncomment) `fixture_root` in `meteogram_small.yaml`.
4. Clear the real inference output the capture run produced, so replay does not
   collide with it: `rm -rf output/data/runs/<forecaster-run> output/logs/inference_execute/<forecaster-run>`
   (or just start replay from a clean `output/`). Replay deliberately refuses to
   overwrite a real, Snakemake-owned `grib/` directory.

Subsequent longtest runs replay from the fixture automatically (CI checks out a
   clean tree, so the last step only matters when capturing and replaying in the same
  `output/`).

`capture-fixture` only snapshots GRIB dirs whose init time matches the config's
`dates`, so an unrelated experiment sharing the same `output/` tree is not swept
in. `MANIFEST.yaml` records the checkpoint(s), captured dates, capture time, the
`evalml` commit, and a **SHA-256 per GRIB dir**. At replay the workflow
re-checks each fixture GRIB against its recorded checksum and fails loudly if it
has drifted (corrupted, partial, or hand-edited); fixtures captured before
checksums existed simply skip the check.

**Important note:** The test fixtures must always be regenerated, even when you
just wish to move the location of the test fixture files. The `MANIFEST.yaml` file
that is generated contains absolute paths to the files, so moving them without
generating a new manifest will cause an error.

#### MEC and fixture tests

Most fixture tests will only work for the non-MEC verification path.
The MEC verification path (`--mec`, `verif_obs.smk`) depends on GRIB at
*derived* source init times (`init_time − lead`), which most fixtures (for
example, the showcase fixture) do not contain, so with `--mec` enabled, fixture
mode would still trigger a real inference (checkpoint download). The MEC
feedback file generation path is tested in `mec_small.py`/`mec_small.yaml`; its
config is deliberately set up for the fixture capture to produce derived source
init times.

#### Versioning and multiple fixtures

A fixture is **per-config**, not global: each fixture-backed test reads
`fixture_root` from its own config (there is no shared fixture constant). To add
another fixture-backed test, add a config with its own `fixture_root` and a test
that reads it, reusing the skip-if-no-`MANIFEST.yaml` pattern.

**Layout** — one directory per test-config under a shared parent, with each
capture in a dated subfolder:

```
/store_new/mch/msopr/cmerker/evalml_test_fixtures/
  meteogram-small/
    v_20240801_b30a/   # MANIFEST.yaml + data/runs/<run_id>/<init_time>/grib
    v_20260729_b30a/
  <other-test>/
    v_<date>_<ckpt>/
```

Each config's `fixture_root` points at its own `<test>/<version>` directory.

Versioning needs **no code change**: `fixture_root` is a self-contained,
relocatable bundle — `MANIFEST.yaml`, the `data/` tree, and the per-GRIB
checksums are all anchored relative to `fixture_root`. To refresh, capture into a
new `v_<date>` directory and bump the `fixture_root` line; the old version stays
for rollback. The pointer lives in git (the config), the data lives in
`/store_new` — so the commit that bumps `fixture_root` is tied to the
code/checkpoint change that required re-capturing.

The version tag is a **human pointer only**; a fixture's true identity
(checkpoint, dates, `evalml` commit) is recorded in `MANIFEST.yaml`. Date-only
(`v_20260729`) is fine for infrequent captures — append the checkpoint
short-hash (`v_20260729_b30a`) if you capture more than once a day or want the
reason visible at a glance.

Notes as this grows:

- **Isolation.** `run_id` is a hash of the forecaster config, so tests with
  different checkpoints/configs get different `run_id`s and never collide.
  Prefer one bundle per test — re-capturing a shared bundle would silently
  affect every test pointing at it.
- **Capture scope.** `capture-fixture` only snapshots GRIB dirs whose init time
  matches the config's `dates`, so capturing one fixture from a shared `output/`
  tree will not sweep in another config's runs. Refreshing all fixtures means
  running `capture-fixture` once per config.
- **Storage.** N tests x M versions of GRIB adds up on `/store_new`; prune stale
  versions. The git history of the `fixture_root` bumps shows which versions
  were ever active.

### Regenerating expected metrics for `test_experiment_metrics`

`test_configs.py` runs each config in `CONFIGS` and compares every metric against
`expected/<config>.yaml`; the test only ever asserts. To refresh those references
after an intentional pipeline change, run the generator script — it runs the same
experiment, then writes the reference file instead of comparing:

```
python tests/integration/regenerate_expected.py varda-single-1.0.yaml
```

Name several configs to regenerate several, or pass `all` for every config in
`CONFIGS`. Review the resulting diff before committing it. CI runs pytest, which
has no way to regenerate anything.

The script **takes exact config file names**, which is normally what you want —
each config costs hours of GPU time and a change rarely invalidates all of them. A
partial name such as `varda-single` is rejected with the valid names listed,
before any experiment starts: it would otherwise also match a future
`varda-single-2.0.yaml`, silently overwriting a reference nobody meant to touch.

**Shared with the test, not copied.** `expected_metrics.py` holds everything both
paths need — `find_nc_files()`, `run_experiment()`, `run_sources()`,
`source_key()` and `metric_value()`. The references therefore cannot be generated
with a different selection, or from a different experiment invocation, than the
one compared against; an earlier generator that duplicated this logic drifted out
of sync unnoticed.

**Only the files the run rewrote are used.** `output/data/runs/` is shared across
configs, so a plain glob would write one config's runs into another's reference
file. The script snapshots the `verif_aggregated_*.nc` mtimes before the
experiment and keeps only the files that changed. A consequence: if snakemake
considers the outputs up to date it rewrites nothing, and regeneration fails with
that explanation rather than writing references from a stale tree — remove the
run's directory under `output/data/runs/` (or the whole `output/`) and re-run.
This is also why the script runs the experiment itself rather than reading an
existing tree: `run_id` is a hash computed inside snakemake, so outside a run
there is nothing to tell one config's outputs from another's.

**Format.** The generated file maps each run's source key (the part before the
`/`, e.g. `forecaster-b30a-4d02`) to its list of `{sel, metrics}` entries, so a
config defining several runs — `forecasters-ich1.yaml` has two forecasters — gets
all of them checked. Metrics that are NaN or ±inf (too few samples, or a
degenerate score such as FBI when no events are forecast) and per-source
statistics (`.max`/`.mean`/`.min`/`.std`) are excluded. `varda-single-1.0.yaml` is
still in the older flat-list format, which the test reads for backward
compatibility; regenerating it converts it to the keyed format.
