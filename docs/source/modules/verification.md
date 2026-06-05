# `verification`

The `verification` package implements forecast-vs-observation metrics and
the spatial machinery (shapefile masks, nearest-neighbour mapping) needed
to align them. It is consumed by
[verification_metrics](../workflow/verification.md) and
[verification_aggregation](../workflow/verification.md), and is also
unit-tested in `tests/unit/test_spatial_mapping.py`.

The package is split into two modules:

- **`verification`** (package `__init__.py`) — shapefile masks, the public
  `verify` entry point, continuous and categorical score helpers, and the
  `decode_metric` label translator.
- **`verification.spatial`** — spherical nearest-neighbour mapping
  utilities (`spherical_nearest_neighbor_indices`,
  `nearest_grid_yx_indices`, `map_forecast_to_truth`).

## Top-level package (`verification`)

```{eval-rst}
.. automodule:: verification
   :members:
   :show-inheritance:
```

## Spatial mapping (`verification.spatial`)

```{eval-rst}
.. automodule:: verification.spatial
   :members:
   :show-inheritance:
```

## Notes on the API

- `verify(...)` returns a single `xarray.Dataset` with named regions on a
  `region` dimension. A reserved `all` region is always present. Pass
  `regions=None` to compute over the full grid only.
- Continuous metrics (BIAS, MSE, MAE, CORR) are computed via the
  [`scores`](https://scores.readthedocs.io/) library. Statistics
  (mean, var, min, max) come from xarray directly.
- Pass `threshold_dict={"param": {"gt": [v1, v2], ...}, ...}` to
  additionally produce per-`(parameter, operator, value)` contingency
  tables. They land on a `threshold` dimension whose values are encoded
  as `{op}_{value}` (e.g. `gt_0p001`). Use `decode_metric` to render
  the encoded labels back to human form (`gt 0.001`).
- `map_forecast_to_truth(...)` is the right function to align a forecast
  expressed on grid A onto the truth grid B before metric computation.
  It uses `spherical_nearest_neighbor_indices` under the hood, so it does
  not suffer from the lat/lon distortion of a plain Euclidean
  nearest-neighbour search.
- `ShapefileSpatialAggregationMasks` is the only currently-shipped
  concrete subclass of `SpatialAggregationMasks`. To support a different
  region geometry (e.g. raster masks), subclass `AggregationMasks` and
  implement `get_masks`.

## Adding a metric

1. **Continuous**: extend the dataset built in `_compute_scores` with a
   new key (typically using a `scores.continuous.*` helper).
2. **Categorical**: extend `_binary_confusion_matrix` or add a sibling
   helper that produces a (`threshold`,)-shaped DataArray, then attach
   it to the result in `_compute_scores`.
3. Update `workflow/scripts/verification_aggregation.py` so the new
   field survives aggregation across initialisation times.
4. Update `report_experiment_dashboard.py` and the dashboard template if
   the metric should be plottable.
5. Add a test under `tests/unit/`. The existing
   `tests/unit/test_spatial_mapping.py` covers the spatial helpers; new
   metric tests should live in a new file (e.g.
   `tests/unit/test_metrics.py`).
