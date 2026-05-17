# Configuration

Every EvalML run is driven by a YAML file that is validated against the
Pydantic models defined in [src/evalml/config.py](../../../src/evalml/config.py).
This page is the conceptual walkthrough; the full machine-readable schema is
in [reference/config_schema](../reference/config_schema.md), and the model
classes are auto-documented under [modules/evalml](../modules/evalml.md).

## Top-level shape

```yaml
description: ...
config_label: ...     # optional; defaults to the YAML stem
dates: ...
runs: ...
truth: ...
stratification: ...
locations: ...
profile: ...
```

`extra: forbid` is set on `ConfigModel`, so misspelled keys at this level fail
validation immediately.

## `dates`

Two equivalent shapes are accepted, both producing a list of initialisation
times:

```yaml
# Range form
dates:
  start: 2020-01-01T00:00
  end:   2020-01-10T00:00
  frequency: 60h          # validated by regex `^\d+[hd]$`
```

```yaml
# Explicit list form, useful for case studies
dates:
  - 2020-01-01T00:00
  - 2020-01-03T12:00
  - 2020-02-14T06:00
```

The range form is parsed into a list internally by `parse_reference_times` in
`workflow/rules/common.smk`.

## `runs`

`runs` is a heterogeneous list. Each item is a single-key mapping whose key is
one of `forecaster`, `interpolator` (temporal downscaler), or `baseline`, and whose value is the corresponding model.

### Forecaster / Temporal downscaler

```yaml
- forecaster:
    checkpoint: https://mlflow.ecmwf.int/#/experiments/103/runs/d0846032fc...
    label: M-1 forecaster
    steps: 0/120/6                                         # start/end/step in hours
    config: resources/inference/configs/sgm-forecaster-global.yaml
    extra_requirements:
      - git+https://github.com/ecmwf/anemoi-inference.git@0.8.3
    inference_resources:                                   # optional
      slurm_partition: long-shared
      gpu: 2
      runtime: 1h
    disable_local_eccodes_definitions: false
```

Field meanings:

- **`checkpoint`** — MLflow URL, Hugging Face URL ending in `.ckpt`, or a
  local path that exists. The URL host determines the retrieval mechanism
  (see [Inference workflow](../workflow/inference.md)).
- **`label`** — the human-readable name shown on plots and the dashboard.
  Excluded from the run's identity hash.
- **`steps`** — `start/end/step` in hours, validated by `validate_steps`.
- **`config`** — either a Python dict to override the inference config inline
  or a path to a YAML template under `resources/inference/configs/`. If
  omitted, defaults to `resources/inference/configs/forecaster.yaml` for
  forecasters and `interpolator.yaml` for temporal downscaling.
- **`extra_requirements`** — additional pip-installable dependencies merged
  into the auto-generated `requirements.txt` for the inference venv.
- **`inference_resources`** — overrides the SLURM defaults for this run only.
- **`disable_local_eccodes_definitions`** — skip pointing
  `ECCODES_DEFINITION_PATH` at the venv-bundled COSMO definitions.

`InterpolatorConfig` adds a single optional field:

```yaml
- interpolator:
    checkpoint: ...
    config: resources/inference/configs/interpolator.yaml
    forecaster:                  # nested ForecasterConfig
      checkpoint: ...
      steps: 0/120/6
      ...
```

The temporal downscaling pulls forecasts from `forecaster` instead of from analysis data. If `forecaster` is omitted, the temporal downscaling runs on analysis input.

### Baseline

```yaml
- baseline:
    label: COSMO-E
    root: /store_new/mch/msopr/ml/COSMO-E
    steps: 10/120/1
```

The `baseline_id` used in output paths is derived from `Path(root).stem`. The
top-level `baselines:` key is still accepted for backwards compatibility but
deprecated — prefer entries inside `runs:`.

## `truth`

```yaml
truth:
  label: COSMO KENDA
  root: /scratch/.../mch-co2-an-archive-0p02-2015-2020-6h-v3-pl13.zarr
```

If `root` ends in `peakweather`, EvalML will trigger
`data_download_obs_from_peakweather` to fetch station observations from
Hugging Face.

## `stratification`

```yaml
stratification:
  regions:
    - jura
    - mittelland
    - voralpen
  root: /scratch/mch/bhendj/regions/Prognoseregionen_LV95_20220517
```

Each entry is interpreted as `<root>/<region>.shp`. Verification scripts read
these shapefiles via `ShapefileSpatialAggregationMasks` and produce per-region
metrics in addition to the always-on `all` aggregate.

## `locations`

```yaml
locations:
  output_root: output/
```

`output_root` becomes `OUT_ROOT` in the Snakefile and roots all
intermediate and final paths.

## `profile`

`profile` is forwarded to Snakemake via `Profile.parsable()`. Tune it to match
your executor:

```yaml
profile:
  executor: slurm
  global_resources:
    gpus: 16              # max concurrent GPUs across all jobs
  default_resources:
    slurm_partition: postproc
    cpus_per_task: 1
    mem_mb_per_cpu: 1800
    runtime: 1h
  jobs: 50                # max parallel job submissions
  batch_rules:
    plot_forecast_frame: 32   # group 32 invocations into one submission
```

`batch_rules` becomes a pair of `--groups` / `--group-components` arguments
to Snakemake, and is the simplest way to keep small plotting jobs from
flooding the scheduler.

## Run identity: how `env_id` and `run_id` are derived

EvalML is careful to *only* rebuild the inference environment when something
that genuinely affects the environment changes. The hashing logic lives in
[workflow/rules/common.smk](../../../workflow/rules/common.smk) and uses
two field sets exposed by `RunConfig`:

| Constant | Fields |
| --- | --- |
| `RunConfig.ENV_FIELDS` | `checkpoint`, `extra_requirements`, `disable_local_eccodes_definitions` |
| `RunConfig.HASH_EXCLUDE` | `label`, `inference_resources` |

From those:

- **`env_id`** = `{model_type}-{model_id}-{env_hash}` (with `-on-{forecaster_env}`
  appended for temporal downscaling). Determines which venv / squashfs is built.
- **`run_id`** = `{env_id}/{run_hash}`. Adds a hash of the inference config
  YAML and `steps`. Determines where outputs land.

Practical consequences:

- Changing only the `label` does not invalidate any cache.
- Changing `steps` or the inference config creates a new run directory but
  reuses the existing venv/squashfs.
- Changing `checkpoint` or `extra_requirements` triggers a full environment
  rebuild.

The full mechanism is documented in [Outputs and wildcards](outputs.md) and
[Inference workflow](../workflow/inference.md).
