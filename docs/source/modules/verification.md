# `verification`

The `verification` package implements forecast-vs-observation metrics and
the spatial machinery (shapefile masks, nearest-neighbour mapping) needed
to align them. It is consumed by
[verification_metrics](../workflow/verification.md) and
[verification_aggregation](../workflow/verification.md), and is also
unit-tested in `tests/unit/test_spatial_mapping.py`.

## Package surface

```{eval-rst}
.. automodule:: verification
   :members:
```

## Spatial verification (`verification.spatial`)

```{eval-rst}
.. automodule:: verification.spatial
   :members:
   :show-inheritance:
```

## Notes on the API

- `verify(...)` returns a single `xarray.Dataset` with named regions on a
  `region` dimension. A reserved `all` region is always present. Pass
  `regions=None` to compute over the full grid only.
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

1. Implement the metric inside `_compute_scores` (or a sibling helper if it
   is statistical rather than skill-based).
2. Make sure the metric name is included in `verify`'s output dataset.
3. Update `workflow/scripts/verification_aggregation.py` so the new field
   survives aggregation across initialisation times.
4. Update `report_experiment_dashboard.py` and the dashboard template if
   the metric should be plottable.
5. Add a test under `tests/unit/test_spatial_mapping.py` (or a new file
   under `tests/unit/`).
