# Example configs

The `config/` directory ships eight ready-to-run experiment configurations,
covering the most common scenarios. They double as living tests: every
example is loaded by `tests/conftest.py` and validated against the schema.

| File | Scenario |
| --- | --- |
| `forecasters-co2.yaml` | COSMO-2 forecaster comparison, 3 storm events |
| `forecasters-co2-disentangled.yaml` | "Disentangled" COSMO-2 forecaster variant |
| `forecasters-co1e.yaml` | COSMO-1E emulator fine-tuned on analysis (1 km, Jan 2020) |
| `forecasters-ich1.yaml` | ICON-CH1 single forecaster (stage_C, 1 km) |
| `forecasters-ich1-oper.yaml` | Operational ICON-CH1 configuration |
| `forecasters-ich1-oper-fixed.yaml` | Fixed operational ICON-CH1 variant |
| `interpolators-co2.yaml` | M-2 temporal downscaler on COSMO-2 |
| `interpolators-ich1.yaml` | ICON-CH1 temporal downscaler with multi-dataset support |

## Picking a starting point

If you are building a new evaluation:

- **Comparing two forecasters** — copy `forecasters-co2.yaml` and replace the
  two `forecaster:` entries with your checkpoints. Adjust `dates` and
  `truth` as needed.
- **Evaluating a temporal downscaler** — copy `interpolators-co2.yaml`. Note the
  `forecaster:` block nested inside `interpolator:`; remove it to run the
  temporal downscaler on analysis input instead of forecaster output.
- **Operational ICON-CH1** — start from `forecasters-ich1-oper.yaml`. It uses
  the operational inference config templates under `resources/inference/`.

## Reading an example

A condensed walkthrough of `forecasters-co2.yaml`:

```yaml
description: |
  COSMO-E forecaster emulator on three storm events.

dates:
  - 2020-02-09T00:00
  - 2021-01-13T12:00
  - 2023-08-25T00:00

runs:
  - forecaster:
      checkpoint: https://mlflow.ecmwf.int/.../d0846032fc7248a58b089cbe8fa4c511
      label: M-1 forecaster
      steps: 0/120/6
      config: resources/inference/configs/sgm-forecaster-global.yaml
  - baseline:
      label: COSMO-E
      root: /store_new/mch/msopr/ml/COSMO-E
      steps: 10/120/1

truth:
  label: COSMO KENDA
  root: /scratch/.../mch-co2-an-archive-0p02-2015-2020-6h-v3-pl13.zarr

stratification:
  regions:
    - jura
    - mittelland
    - voralpen
  root: /scratch/mch/bhendj/regions/Prognoseregionen_LV95_20220517

locations:
  output_root: output/

profile:
  executor: slurm
  global_resources: { gpus: 16 }
  default_resources:
    slurm_partition: postproc
    cpus_per_task: 1
    mem_mb_per_cpu: 1800
    runtime: 1h
  jobs: 50
  batch_rules:
    plot_forecast_frame: 32
```

Things worth highlighting:

- `dates` is in **explicit-list form** because the storms are hand-picked.
- The `forecaster:` `config` points at a bundled inference template under
  `resources/inference/configs/`. Inline dict overrides are supported but
  rarely needed.
- The `baseline:` entry uses `steps: 10/120/1`, which means the baseline
  starts at lead time 10 h with hourly cadence; this is independent of the
  forecaster's `0/120/6`.
- `batch_rules.plot_forecast_frame: 32` keeps SLURM happy when generating
  hundreds of frames.

## Schema validation in your editor

All shipped examples include a YAML language-server hint:

```yaml
# yaml-language-server: $schema=../workflow/tools/config.schema.json
```

If you copy an example into a new location, keep that comment so VSCode and
other YAML-aware editors can pull descriptions and autocompletion from the
generated schema.
