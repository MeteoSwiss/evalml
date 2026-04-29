# Glossary

```{glossary}
candidate
    A run that participates in an experiment's verification, dashboard,
    and plots. Set internally by `register_run(... as_candidate=True)`.
    Forecasters and interpolators added directly under `runs:` are
    candidates by default; nested upstream forecasters registered as
    dependencies of an interpolator are *not* candidates unless they
    were also explicitly listed.

dependency run
    A run registered indirectly because another run depends on it (e.g.
    the forecaster nested inside an `interpolator:` block). Its env and
    output directories are built but it does not appear in the
    dashboard.

env_id
    `{model_type}-{model_id}-{env_hash}` (with `-on-{forecaster_env}`
    appended for interpolators). Identifies the inference environment
    (venv, squashfs). Two runs that differ only in inference config or
    `steps` share an `env_id`.

run_id
    `{env_id}/{config_hash}`. Identifies a specific run configuration.
    Each unique `run_id` has its own output directory under
    `data/runs/`.

baseline
    A reference forecast (typically COSMO-E or another operational NWP
    archive) that is read from disk rather than computed by EvalML. The
    `baseline_id` is `Path(root).stem`.

truth
    The ground-truth dataset used in verification. Either an analysis
    Zarr or a PeakWeather observations cache.

OUT_ROOT
    The path-based shorthand for `locations.output_root` from the YAML
    config. Used everywhere in the Snakefile.

experiment_name
    `{YYYYMMDD}_{config_label}_{config_hash}`. Identifies one
    invocation of the workflow. Used as the `{experiment}` (and
    `{showcase}`) wildcard.

config_hash
    Short SHA-256 of the merged config plus every run's env- and
    run-specific hashes. Computed by `master_hash()` in
    `workflow/rules/common.smk`.

env_hash
    Short SHA-256 of the fields in `RunConfig.ENV_FIELDS`
    (`checkpoint`, `extra_requirements`,
    `disable_local_eccodes_definitions`). Computed by
    `env_entry_hash()`.

run_hash
    Short SHA-256 of `steps` plus the inference config YAML contents.
    Computed by `run_specific_hash()`.

inference sandbox
    A zip produced by `inference_create_sandbox` that bundles a
    checkpoint, a `requirements.txt`, an inference config, and a
    rendered README. Suitable for handing a checkpoint plus a
    reproducible runtime to an external collaborator.

squashfs image
    A read-only filesystem image (`venv.squashfs`) of the inference
    venv. `inference_execute` mounts it on the SLURM compute node via
    `squashfs-mount` and activates it as `/user-environment`.

candidate filtering
    `collect_all_candidates()` returns only runs with
    `_is_candidate=True`. The `experiment_all` and `showcase_all` target
    rules iterate over candidates only, so adding a dependency run does
    not produce extra dashboard entries.

stratification region
    A polygon defined by a shapefile under `stratification.root`, used
    to compute regional verification scores. The hardcoded `all` region
    covers the entire grid and is always present.
```
