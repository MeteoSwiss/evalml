# Quickstart

This page walks through running your first experiment end-to-end. It assumes
you have already followed [Installation](installation.md) and
[Credentials setup](auth.md).

## 1. Pick an example config

The repository ships several end-to-end configurations under `config/`:

| File | What it does |
| --- | --- |
| `config/forecasters-co2.yaml` | COSMO-2 forecaster comparison on storm cases |
| `config/forecasters-ich1.yaml` | ICON-CH1 single-forecaster evaluation |
| `config/interpolators-co2.yaml` | M-2 temporal downscaler on COSMO-2 data |

A walkthrough of each is in [Example configs](../user_guide/examples.md).

## 2. Run the experiment

```bash
evalml experiment config/forecasters-co2.yaml --report
```

What `evalml` does, step by step:

1. **Validate** the YAML against the Pydantic `ConfigModel` (see
   [Configuration](../user_guide/configuration.md)).
2. **Build** a Snakemake invocation using the `profile:` block.
3. **Launch** Snakemake with the `experiment_all` target.

`--report` additionally tells Snakemake to produce a self-contained HTML
report after a successful run.

## 3. Useful flags

```bash
# Don't execute, just show what would run.
evalml experiment config.yaml --dry-run

# Render the dependency DAG to dag.svg via kroki.io.
evalml experiment config.yaml --dag

# Render only the rule graph.
evalml experiment config.yaml --rulegraph

# Pass arbitrary options to Snakemake after `--`.
evalml experiment config.yaml -- --jobs 1 --use-conda
```

The full option matrix is documented in [The evalml CLI](../user_guide/cli.md).

## 4. Inspect the results

```bash
ls output/results/
```

Each run produces a directory named
`{YYYYMMDD}_{config_label}_{config_hash}/` containing the rendered dashboard
(`dashboard/dashboard.html`) and the verification plots (`plots/*.png`).
Forecast animations and meteograms only appear if you used
`evalml showcase` instead of `evalml experiment`.

## 5. What lives under `output/data/`

`output/results/` only holds the polished, shareable artefacts. Almost
everything else lands in `output/data/`, which is roughly three trees:

```text
output/data/
├── runs/<env_id>/                       # one per checkpoint + dep set
│   ├── inference-last.ckpt              # symlink / copy of the checkpoint
│   ├── requirements.txt                 # auto-derived from MLflow metadata
│   ├── venv.squashfs                    # built once, mounted on compute nodes
│   └── <config_hash>/                   # one per inference config + steps
│       ├── summary.md                   # human-readable run description
│       ├── verif_aggregated.nc          # aggregated metrics across init times
│       └── <init_time>/                 # one per forecast init
│           ├── config.yaml              # rendered inference config
│           ├── grib/                    # raw GRIB forecast output
│           └── verif.nc                 # per-init metrics
├── baselines/<baseline_id>/             # one per `baseline:` entry
│   ├── verif_aggregated.nc
│   └── <init_time>/verif.nc
└── observations/                        # cached PeakWeather etc.
```

A few things worth knowing as you start poking around:

- `data/runs/<env_id>/` is **shared** across every run that has the same
  checkpoint and the same extra dependencies. Two runs that only differ
  in `steps` or in the inference config will share the (expensive)
  `venv.squashfs` but each get their own `<config_hash>/` subdirectory.
- `verif.nc` files are plain `xarray` datasets and can be opened with
  `xr.open_dataset` for ad-hoc analysis. The full schema is documented
  in {ref}`Outputs and wildcards → What's inside a verif.nc <whats-inside-a-verif-nc>`.
- `grib/` directories are large. If disk pressure matters, prune
  `data/runs/<env_id>/<config_hash>/<init_time>/grib/` for old init
  times once `verif.nc` has been written — the metrics rule does not
  need them again.
- The full layout (logs, sentinel `.ok` files, every wildcard) is in
  [Outputs and wildcards](../user_guide/outputs.md).

(iterate-without-re-running-everything)=
## 6. Iterate without re-running everything

Snakemake caches outputs based on file hashes, so re-running with a tweaked
config will only redo affected rules. If you change something that EvalML
hashes into `run_id` (steps, inference config, dependencies), a new
sub-directory is created and the existing one is left untouched. See
[Outputs and wildcards](../user_guide/outputs.md) for the rules behind this.

**Caveat:** only the `checkpoint` **path/URL** enters the hash — not its
contents. If a checkpoint is mutated in place while the path stays the
same, EvalML reuses the cached environment and outputs. To force a
rebuild:

```bash
# Force everything to re-run, even if the outputs look up-to-date.
evalml experiment config.yaml -- -F

# Force just one rule (and its downstream dependents) to re-run.
evalml experiment config.yaml -- -R verification_metrics
```

See [CLI → Forcing re-runs](../user_guide/cli.md#forcing-re-runs) for the
full list of force flags.
