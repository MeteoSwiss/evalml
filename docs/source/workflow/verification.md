# Verification

`workflow/rules/verification.smk` turns inference outputs (per-init GRIB
directories) and baselines (Zarr) into per-init `verif.nc` files, then
aggregates and plots them. The metric implementations live in
[src/verification/spatial.py](../../../src/verification/spatial.py); see the
[verification module reference](../modules/verification.md) for the
docstrings.

## The four-rule pipeline

```text
    verification_metrics            verification_metrics_baseline
    (per run, per init)             (per baseline, per init)
              │                              │
              ▼                              ▼
    verification_metrics_aggregation         verification_metrics_aggregation_baseline
              │                              │
              └──────────────┬───────────────┘
                             ▼
                  verification_metrics_plot
```

### `verification_metrics`

Inputs: the inference `.ok` file plus `truth.root`. Output:
`data/runs/{run_id}/{init_time}/verif.nc`.

The shell command is:

```bash
uv run scripts/verification_metrics.py \
    --forecast {grib_out_dir} \
    --truth {truth} \
    --reftime {init_time} \
    --steps {fcst_steps} \
    --label {fcst_label} \
    --truth_label {truth_label} \
    --regions {regions} \
    --threshold_dict "{threshold_dict}" \
    --output {output}
```

`uv run` is used so the script picks up the project's environment without
requiring a manual activation step.

`--threshold_dict` is the literal repr of `config["thresholds"]` and may be
the empty dict, in which case only continuous metrics are computed.

### `verification_metrics_baseline`

Same script and arguments, but the input is a baseline Zarr expanded to
`{root}/FCST{YY}.zarr`. Output:
`data/baselines/{baseline_id}/{init_time}/verif.nc`.

### `verification_metrics_aggregation`

Calls `scripts/verification_aggregation.py` over all per-init `verif.nc`
files for one `run_id` (filtered through `_restrict_reftimes_to_hours`,
which currently passes everything through unchanged but exists as a hook
for hour-of-day stratification). Output:
`data/runs/{run_id}/verif_aggregated.nc`.

`verification_metrics_aggregation_baseline` is defined via Snakemake's
`use rule … with:` syntax, reusing the same body but pointing at baseline
inputs and outputs.

### `verification_metrics_plot`

Takes every aggregated `verif_aggregated.nc` for the experiment
(`EXPERIMENT_PARTICIPANTS.values()`) and produces a directory of PNGs at
`results/{experiment}/plots/`. The directory is wrapped in a
`report(directory(...), patterns=["{name}.png"])` so the rendered HTML
report can include the figures.

## Metric computation

The heavy lifting is in `verification.verify(...)`:

- Computes per-parameter continuous scores (BIAS, MSE, MAE, CORR) via the
  [`scores`](https://scores.readthedocs.io/) library (≥ 2.0).
- Computes per-parameter statistics (mean, var, min, max).
- Aggregates per region, including a hardcoded `all` region.
- When `threshold_dict` is provided, also computes 2×2 contingency tables
  per `(parameter, operator, threshold)` triple via
  `_binary_confusion_matrix` and `scores.categorical.ThresholdEventOperator`.
  The result is stored as a `contingency_table` variable on a `threshold`
  dimension whose values are encoded as `{op}_{value}` (e.g. `gt_0p001`).
- Optionally runs in parallel via Dask, given a `num_workers`.

The CORR metric still uses `xr.corr` under the hood for backwards
compatibility, and `R²` / `VAR` are no longer emitted (they were derivable
from CORR and the statistics dataset). To translate an encoded threshold
label back into human-readable form, use
[`verification.decode_metric`](../modules/verification.md).

For coordinate alignment between forecast and truth on different grids,
`map_forecast_to_truth(fcst, truth)` first does a spherical nearest-neighbour
mapping (`spherical_nearest_neighbor_indices`) so we don't suffer from the
distortions that come with naive lat/lon Euclidean distance.

## Region masks

Region masks are produced by `ShapefileSpatialAggregationMasks` in the same
module. It takes:

- `shp` — a single shapefile or a list of shapefiles.
- `src_crs` and `dst_crs` — the source and destination CRS.

It exposes `get_masks(lat, lon)` which returns an `xarray.DataArray` with a
`region` dimension. The `all` region is always present and covers the whole
grid; the named regions correspond one-to-one with the shapefiles passed in.
