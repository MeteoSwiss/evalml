# Workflow overview

The Snakemake project lives entirely under `workflow/`. The CLI does not
contain any workflow logic — it just constructs a Snakemake invocation and
delegates. Reading this section will let you debug rules, add new ones, and
reason about why a particular file was (or was not) regenerated.

## Snakefile structure

The top of [workflow/Snakefile](../../../workflow/Snakefile) is small and
ordered deliberately:

1. **Validate** the merged config through `ConfigModel` and write the result
   back into Snakemake's `config` dict via `update_config(...)`.
2. **Include** the rule modules in dependency order:
   - `rules/common.smk` — utilities, hashing, run/env registration.
   - `rules/summary.smk` — human-readable run summary.
   - `rules/data.smk` — observation data acquisition.
   - `rules/inference.smk` — checkpoint retrieval, env build, inference.
   - `rules/verification.smk` — metrics computation and aggregation.
   - `rules/report.smk` — dashboard generation.
   - `rules/plot.smk` — frames, animations, meteograms.
3. Read `.evalml_snakemake_cmd.txt` (written by the CLI) to surface the
   invoked command in the `onstart:` banner.
4. Compute `EXPERIMENT_NAME = f"{WHEN}_{CONFIG_LABEL}_{CONFIG_HASH}"` and the
   list of `CANDIDATES` (runs marked `_is_candidate=True`).
5. Define the wildcard constraint `showcase=r"[^/]+"` — necessary because
   `run_id` contains `/`.
6. Define `onstart:` / `onsuccess:` / `onerror:` banners and the target rules.

## Top-level target rules

| Target | Triggered by | Inputs |
| --- | --- | --- |
| `experiment_all` | `evalml experiment` | `report_experiment_dashboard` + `verification_metrics_plot` for `EXPERIMENT_NAME` |
| `showcase_all` | `evalml showcase` | `make_forecast_animation` + `plot_meteogram` for selected params, regions, stations |
| `sandbox_all` | `evalml sandbox` | `inference_create_sandbox` outputs for all candidates |
| `inference_all` | `evalml make … inference_all` | Per-candidate, per-init-time `data/runs/{run_id}/{init_time}/raw` |
| `verification_metrics_all` | `evalml make …` | `verification_metrics` outputs for all candidates × init times |
| `verification_metrics_plot_all` | `evalml make …` | `verification_metrics_plot` for the current experiment |

## Module map

```text
workflow/
├── Snakefile                 # entry point: includes, target rules, banners
├── rules/
│   ├── common.smk            # configuration parsing, hashing, registries
│   ├── summary.smk           # write_summary
│   ├── data.smk              # data_download_obs_from_peakweather
│   ├── inference.smk         # inference_*, sandbox creation
│   ├── verification.smk      # verification_metrics{,_baseline,_aggregation,_plot}
│   ├── report.smk            # report_experiment_dashboard
│   └── plot.smk              # plot_meteogram, plot_forecast_frame, animation
├── scripts/
│   ├── data_*.py             # called by data.smk
│   ├── inference_*.py        # called by inference.smk
│   ├── verification_*.py     # called by verification.smk
│   ├── report_*.py           # called by report.smk
│   └── plot_*.mo.py          # called by plot.smk (Marimo notebooks)
└── tools/
    └── config.schema.json    # generated from src/evalml/config.py
```

Everything in `rules/` is imported by the Snakefile. Scripts in `scripts/`
are invoked through `script:` directives or `shell:` blocks; they import the
src-layout packages (`evalml`, `verification`, `data_input`, `plotting`) at
runtime.

## How rules find each other

`common.smk` populates several module-level dicts that all later rules read:

- `RUN_CONFIGS` — every registered run, keyed by `run_id`.
- `ENV_CONFIGS` — unique inference environments, keyed by `env_id`.
- `BASELINE_CONFIGS` — baselines, keyed by `baseline_id`.
- `EXPERIMENT_PARTICIPANTS` — `dict[label, path-to-verif_aggregated.nc]`,
  used to wire dashboard and metric plot inputs.
- `REGIONS` — comma-separated string of shapefile paths.
- `REFTIMES` — list of `datetime` objects.

These are computed at parse time from `config["runs"]`, then kept stable
throughout the run. If you add a rule that needs to enumerate participants,
read from the registries rather than re-walking the YAML.

## Reading the rest of this section

- [Data](data.md) — how observations are pulled from PeakWeather and how
  baseline data is wired into verification.
- [Inference](inference.md) — checkpoint retrieval, venv/squashfs build,
  sandbox packaging, and `inference_execute`.
- [Verification](verification.md) — metric computation, aggregation, and
  plotting.
- [Plotting](plotting.md) — Marimo-based frame and meteogram rules,
  animation assembly.
- [Reporting](reporting.md) — the dashboard rule and its Jinja2 template.
- [Conventions](conventions.md) — naming, log paths, sentinel files.
