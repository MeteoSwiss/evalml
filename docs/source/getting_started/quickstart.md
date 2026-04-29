# Quickstart

This page walks through running your first experiment end-to-end. It assumes
you have already followed [Installation](installation.md) and
[Credentials setup](credentials.md).

## 1. Pick an example config

The repository ships several end-to-end configurations under `config/`:

| File | What it does |
| --- | --- |
| `config/forecasters-co2.yaml` | COSMO-2 forecaster comparison on storm cases |
| `config/forecasters-ich1.yaml` | ICON-CH1 single-forecaster evaluation |
| `config/interpolators-co2.yaml` | M-2 spatial interpolator on COSMO-2 data |

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

## 5. Iterate without re-running everything

Snakemake caches outputs based on file hashes, so re-running with a tweaked
config will only redo affected rules. If you change something that EvalML
hashes into `run_id` (steps, inference config, dependencies), a new
sub-directory is created and the existing one is left untouched. See
[Outputs and wildcards](../user_guide/outputs.md) for the rules behind this.
