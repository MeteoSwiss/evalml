---
name: new-config
description: Create a new evalML experiment config YAML file in config/. Use when the user wants to set up a new evaluation experiment.
user-invocable: true
argument-hint: [config-filename]
---

# Create a New evalML Experiment Config

Your task is to create a new evalML experiment config YAML file in `config/`.

## Step 1 – Gather information

Ask the user for each piece of information **one at a time**, in order, using the `AskUserQuestion` tool where possible (to offer sensible options based on existing configs). Do not ask for multiple things at once. Collect the following, inferring from $ARGUMENTS where possible:

1. **Filename** – what to name the config file (e.g. `forecasters-myexp.yaml`)
2. **Description** – a one-sentence description of the experiment
3. **Dates** – either:
   - A date range: `start`, `end` (ISO-8601), and `frequency` (e.g. `6h`, `54h`, `1d`)
   - Or explicit dates as a list (e.g. storm events)
4. **Runs** – one or more model runs. For each run, collect:
   - Type: `forecaster` or `interpolator`
   - `mlflow_id` (32-character hex string)
   - `label` (human-readable name for plots/reports)
   - `steps` (lead times, format `start/end/step`, e.g. `0/120/6`)
   - `config` (path to anemoi inference config YAML, relative to repo root)
   - `extra_dependencies` (list of pip-installable packages, often anemoi-inference)
   - For interpolators only: optionally a nested `forecaster` block
5. **Baselines** – one or more reference forecasts. For each:
   - `baseline_id` (e.g. `COSMO-E`, `ICON-CH1-EPS`)
   - `label`
   - `root` (path to baseline data directory)
   - `steps`
6. **Analysis dataset**:
   - `label` (e.g. `COSMO KENDA`, `REA-L-CH1`)
   - `analysis_zarr` (path to zarr dataset)

For each question, first read existing configs (Step 2) to offer sensible defaults as `AskUserQuestion` options. After each answer, confirm what was captured before moving to the next question. For information you don't have, look at existing configs in `config/` to suggest sensible defaults. The stratification regions and root path are almost always the same (the six Swiss forecast regions); reuse them unless the user says otherwise. The `locations` and `profile` sections are nearly identical across all configs — copy them from an existing config unless the user has specific overrides.

## Step 2 – Look at existing configs for context

Read one or two existing configs from `config/` that are similar to the user's experiment (e.g. same model type or domain) to pick sensible defaults for any missing fields.

## Step 3 – Write the config file

Write the config to `config/<filename>`. Always include the schema reference as the first line:

```yaml
# yaml-language-server: $schema=../workflow/tools/config.schema.json
description: |
  <description>

dates:
  # Use ONE of the two forms below:

  # Form A – date range:
  start: <ISO-8601>
  end: <ISO-8601>
  frequency: <Nh or Nd>

  # Form B – explicit list:
  # - 2020-02-03T00:00  # Storm Petra
  # - 2020-02-07T00:00  # Storm Sabine

runs:
  - forecaster:          # or "interpolator:"
      mlflow_id: <32-char hex>
      label: <label>
      steps: <start/end/step>
      config: <path/to/inference/config.yaml>
      extra_dependencies:
        - <pip-package>

baselines:
  - baseline:
      baseline_id: <id>
      label: <label>
      root: <path>
      steps: <start/end/step>

analysis:
  label: <label>
  analysis_zarr: <path>

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
  output_root: output/
  mlflow_uri:
    - https://servicedepl.meteoswiss.ch/mlstore
    - https://mlflow.ecmwf.int

profile:
  executor: slurm
  global_resources:
    gpus: 16
  default_resources:
    slurm_partition: "postproc"
    cpus_per_task: 1
    mem_mb_per_cpu: 1800
    runtime: "1h"
    gpus: 0
  jobs: 50
  batch_rules:
    plot_forecast_frame: 32
```

## Step 4 – Confirm and validate

After writing the file, tell the user the path and remind them to:
- Validate it with their YAML editor (schema is wired via the first-line comment)
- Or run `pre-commit run --all-files` which includes schema validation
- Run a dry-run with `evalml experiment config/<filename>.yaml -- --dry-run` to check for issues
