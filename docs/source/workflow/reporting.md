# Reporting

`workflow/rules/report.smk` defines a single rule,
`report_experiment_dashboard`, that renders an interactive HTML dashboard
summarising one experiment.

## `report_experiment_dashboard`

| Property | Value |
| --- | --- |
| Local rule | yes |
| Output | `results/{experiment}/dashboard/` (directory; `htmlindex="dashboard.html"`) |
| Log | `logs/report_experiment_dashboard/{experiment}.log` |

The rule consumes:

- `EXPERIMENT_PARTICIPANTS.values()` — every aggregated `verif_aggregated.nc`
  for the experiment (one per run + one per baseline).
- `resources/report/dashboard/template.html.jinja2` — the dashboard
  template.
- `resources/report/dashboard/script.js` — the front-end JavaScript.
- The original config file (used to embed a copy in the dashboard).

It then runs:

```bash
python scripts/report_experiment_dashboard.py \
    --verif_files {verif} \
    --template {template} \
    --script {js_script} \
    --header_text "{header_text}" \
    --configfile "{configfile}" \
    --stratification {stratification} \
    --output {output}
```

`--stratification` is a space-separated list of dashboard facets drawn
from `config["dashboard"]["stratification"]`. The Pydantic `Dashboard`
model accepts any of `season`, `region`, `init_hour`; whatever's enabled
appears as selectable axes in the dashboard UI.

`header_text` is computed by `make_header_text()` at parse time:

- Explicit-list `dates` →
  `"Explicit initializations from N runs have been used."`
- Range `dates` →
  `"Verification against {truth} with initializations from {start} to {end} by {frequency}"`

## Where dashboard logic lives

The script in `workflow/scripts/report_experiment_dashboard.py` is
responsible for shaping the netCDF inputs into JSON suitable for the
front-end. The front-end JavaScript in
`resources/report/dashboard/script.js` then builds the interactive plots
(parameter/region/leadtime selectors, etc.).

If you need to extend the dashboard, the typical change is:

- Add a new metric in `verification.spatial.verify` so it lands in
  `verif.nc`.
- Update `verification_aggregation.py` to keep the metric through the
  aggregation step.
- Update `report_experiment_dashboard.py` to surface the metric in the
  JSON payload.
- Update `template.html.jinja2` and/or `script.js` to render it.

## Snakemake's HTML report

This is distinct from the dashboard. When `evalml experiment --report` is
passed, EvalML appends `--report-after-run --report FILE` to Snakemake.
That instructs Snakemake itself to produce a self-contained HTML run
report — execution times, DAG, per-rule logs — independent of the EvalML
dashboard. It's useful for debugging slow rules and reviewing what was
executed.
