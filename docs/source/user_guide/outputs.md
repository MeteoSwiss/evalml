# Outputs and wildcards

All workflow outputs are rooted at `OUT_ROOT`, which is taken from
`locations.output_root` in the config. Three top-level subdirectories live
underneath:

```text
{OUT_ROOT}/
├── data/                                  # intermediate artefacts
│   ├── runs/{env_id}/                     # per-environment, shared across configs
│   │   ├── inference-last.ckpt
│   │   ├── requirements.txt
│   │   ├── venv.squashfs
│   │   └── {config_hash}/                 # per-run-config artefacts
│   │       ├── summary.md
│   │       ├── verif_aggregated.nc
│   │       └── {init_time}/               # per-initialisation-time artefacts
│   │           ├── config.yaml
│   │           ├── grib/
│   │           └── verif.nc
│   ├── baselines/{baseline_id}/
│   │   └── {init_time}/verif.nc
│   └── observations/                      # PeakWeather and other obs caches
├── logs/                                  # one sub-directory per rule
└── results/{experiment_name}/             # final products
    ├── dashboard/
    └── plots/
```

The split between `{env_id}/` and `{env_id}/{config_hash}/` reflects the
identity contract documented in [Configuration](configuration.md): expensive
artefacts (the venv and squashfs) live at the env level so they are reused
across runs that only differ in inference config.

```{note}
`{run_id}` in the wildcard table below is exactly
`{env_id}/{config_hash}` — i.e. it spans **two** directory components,
not one. In the tree above, anything rooted at `data/runs/{env_id}/{config_hash}/`
is equivalent to `data/runs/{run_id}/`; the docs and the Snakemake rules
use both forms interchangeably. The
`wildcard_constraints: showcase=r"[^/]+"` block in the Snakefile exists
specifically because `{run_id}` contains a `/`.
```

(whats-inside-a-verif-nc)=
## What's inside a `verif.nc`

Each per-init `verif.nc` is an `xarray.Dataset` keyed by `region` and
`parameter`. Continuous metrics (`BIAS`, `MSE`, `MAE`, `CORR`) and
statistics (`mean`, `var`, `min`, `max`) appear as scalar data variables
per region/parameter. When `thresholds:` is set in the config, an
additional `contingency_table` variable appears on a `threshold`
dimension; its values are encoded `{op}_{value}` (e.g. `gt_0p001` =
"forecast/obs > 0.001"). Use
[`verification.decode_metric`](../modules/verification.md) to render
the encoded labels back to human-readable form.

## Wildcard reference

| Wildcard | Format | Example |
| --- | --- | --- |
| `{env_id}` | `{type}-{model_id}-{env_hash}` (`-on-{forecaster_env}` for temporal downscalers) | `forecaster-1a2b-c3d4` |
| `{run_id}` | `{env_id}/{config_hash}` | `forecaster-1a2b-c3d4/e5f6` |
| `{baseline_id}` | `Path(root).stem` | `COSMO-E` |
| `{init_time}` | `%Y%m%d%H%M` | `202001011200` |
| `{experiment}` | `{YYYYMMDD}_{label}_{hash}` | `20260331_demo_a1b2` |
| `{param}` | variable name | `T_2M`, `TOT_PREC` |
| `{region}` | geographic region slug | `switzerland`, `globe` |
| `{leadtime}` | zero-padded hours | `000`, `006`, `024` |
| `{showcase}` | same as `{experiment}`; constrained to a single path component | `20260331_demo_a1b2` |

The `wildcard_constraints: showcase=r"[^/]+"` block in the Snakefile is
deliberate: because `run_id` already contains a `/`, Snakemake would otherwise
greedily absorb part of `run_id` into `showcase` when matching paths like
`results/{showcase}/{run_id}/...`.

## Logs and sentinel files

Every rule declares a log file under:

```text
{OUT_ROOT}/logs/{rule_name}/{wildcards}.log
```

When a rule produces a sentinel `.ok` file to mark successful completion, it
sits alongside the log:

```text
{OUT_ROOT}/logs/{rule_name}/{wildcards}.ok
```

Multiple wildcards are joined with a hyphen: `{run_id}-{init_time}.log`. The
forecast / temporal downscaler preparation rules write a `.ok` so downstream rules can depend on the log, not on a directory whose timestamp is unstable.

## What `experiment_name` means

`experiment_name` (used as `{experiment}` and `{showcase}`) is built once per
invocation in the Snakefile:

```python
EXPERIMENT_NAME = f"{WHEN}_{CONFIG_LABEL}_{CONFIG_HASH}"
```

- `WHEN` — today's date in `YYYYMMDD`.
- `CONFIG_LABEL` — `config_label` from the YAML, falling back to the YAML
  filename stem.
- `CONFIG_HASH` — the master hash of every run's `env` and `run` hashes
  (see `master_hash` in `common.smk`).

This means re-running the same config on the same day reuses the same
`results/{experiment_name}/` directory, but a meaningful change to any run
produces a fresh directory rather than overwriting old results.
