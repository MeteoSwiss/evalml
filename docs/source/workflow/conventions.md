# Conventions

These conventions apply to every rule and script under `workflow/`. They are
enforced informally during code review and (in part) by `snakefmt` and
`pre-commit`.

## Rule names

Rules use the pattern `{module}_{operation}[_{sub_operation}]` in snake_case.
The module prefix groups rules by their place in the pipeline:

| Module | Rules prefix | Purpose |
| --- | --- | --- |
| `data.smk` | `data_` | Input data preparation |
| `inference.smk` | `inference_` | Checkpoint retrieval, environment setup, execution |
| `verification.smk` | `verification_` | Metrics calculation, aggregation, metric plots |
| `plot.smk` | `plot_` | Forecast visualisation (frames, animations, meteograms) |
| `report.smk` | `report_` | Dashboard and HTML report generation |

Aggregate target rules use the pattern `{module}_all`:
`inference_all`, `verification_metrics_all`, etc.
Top-level entry points are named after the workflow mode: `experiment_all`,
`showcase_all`, `sandbox_all`.

## Script names

Scripts live in `workflow/scripts/` and mirror the rule that calls them:

```text
{module}_{operation}.py         # standard Python scripts
{module}_{operation}.mo.py      # interactive Marimo notebooks
```

Examples: `inference_prepare.py`, `verification_metrics.py`,
`verification_aggregation.py`, `plot_forecast_frame.mo.py`.

A script may be shared between multiple rules — `inference_prepare.py` is
used by both `inference_prepare_forecaster` and
`inference_prepare_interpolator`. When this happens, behaviour is selected
through the rule's `params:` block and read from `snakemake.params` inside
the script.

## Log file paths

Every rule declares a log file under:

```text
{OUT_ROOT}/logs/{rule_name}/{wildcards}.log
```

Multiple wildcards are joined with a hyphen: `{run_id}-{init_time}.log`.

When a rule produces a sentinel `.ok` file to mark successful completion,
it sits alongside the log:

```text
{OUT_ROOT}/logs/{rule_name}/{wildcards}.ok
```

Sentinel files exist because Snakemake otherwise depends on the timestamp of
an output directory, which is unstable; the `.ok` is touched only on
success and is therefore a reliable trigger for downstream rules.

## When to use `localrule: True`

Mark a rule `localrule: True` when:

- It runs a sub-second shell command (symlinking, file copy, JSON parsing).
- It only orchestrates other rules (e.g. `make_forecast_animation` calls
  `convert`).
- It must run on the workflow head node for credentials reasons (MLflow
  authentication is on the head node, not the SLURM compute nodes).

Do *not* mark rules `localrule` if they do real work — `inference_execute`,
`verification_metrics`, and the plotting rules submit through SLURM via
`profile.executor`.

## Resources

When a rule submits to SLURM, declare:

```python
resources:
    slurm_partition="postproc",
    cpus_per_task=24,
    mem_mb=50_000,
    runtime="60m",
```

For inference, the per-run override is plumbed through
`inference_resources` in the YAML config and resolved by `get_resource(wc,
field, default)` in `inference.smk`.

## Adding a new rule

1. Pick the right module (or create a new `.smk` if it's a wholly new
   pipeline stage; remember to `include:` it in the Snakefile).
2. Name the rule `{module}_{operation}` and the script
   `scripts/{module}_{operation}.py`.
3. Add a log path under `logs/{rule_name}/...`.
4. If it submits to SLURM, declare `resources` and avoid `localrule: True`.
5. Add a sentinel `.ok` file if the output is a directory.
6. Run `snakefmt workflow` to apply formatting; commit only after
   `pre-commit run --all-files` is green.
