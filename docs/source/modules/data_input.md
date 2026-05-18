# `data_input`

`data_input` is the central place for loading forecasts and ground-truth
into the verification pipeline. Nearly every loader returns an
`xarray.Dataset` with a consistent schema (parameter names, units, time
coordinates), so downstream code can treat forecast and truth uniformly.

## Package contents

```{eval-rst}
.. automodule:: data_input
   :members:
```

## Loader dispatch

The package exposes two top-level dispatchers:

- `load_truth_data(root, reftime, steps, params)` — picks
  `load_analysis_data_from_zarr` for `.zarr` paths and
  `load_obs_data_from_peakweather` for PeakWeather caches.
- `load_forecast_data(root, reftime, steps, params)` — picks
  `load_fct_data_from_grib` for GRIB directories (EvalML inference
  output) and `load_baseline_from_zarr` for Zarr-based baselines.

These dispatchers are what verification scripts call; the underlying
loaders are fine to use directly when you need to short-circuit the
dispatch (e.g. inside a notebook).

## `parse_steps`

`parse_steps("0/120/6")` returns `[0, 6, 12, ..., 120]`. The same
`start/end/step` format is validated by `RunConfig.steps`.

## Conventions and pitfalls

- **Variable renaming**: `load_analysis_data_from_zarr` renames Anemoi
  variables to their COSMO equivalents (`t_2m → T_2M`, etc.). Add new
  mappings to the script-local rename table when you need them.
- **TOT_PREC unit conversion**: the analysis Zarr stores precipitation in
  metres; `load_analysis_data_from_zarr` multiplies by 1000 to convert to
  mm, the canonical unit downstream.
- **TOT_PREC disaggregation**: GRIB and Zarr loaders both expect
  cumulative-from-start precipitation (the
  `accumulate_from_start_of_forecast` post-processor must be enabled in
  anemoi-inference) and disaggregate it with `.diff("lead_time")`. A
  sanity check raises if `min(.diff())` is significantly negative —
  that's the signature of data that's already per-step accumulated and
  would be garbled by a second disaggregation.
- **Lead-time selection up-front**: both forecast and baseline loaders
  now restrict to the requested lead times *before* disaggregation, so
  sub-step baselines (e.g. hourly baseline against 6-hourly forecast)
  produce the correct accumulation window.
- **Unit conversions**: `T_2M` from PeakWeather is converted to Kelvin in
  `load_obs_data_from_peakweather`, matching the ML model output. Other
  variables retain their source units.
- **Missing valid times**: `_select_valid_times(ds, times)` warns instead
  of raising when a requested step is missing, which is intentional —
  baseline archives sometimes have gaps and forcing a hard error would
  block whole experiments. If you need stricter behaviour, do an
  explicit check on the returned dataset.
