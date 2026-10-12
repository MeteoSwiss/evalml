"""Helpers for accumulated fields (e.g. precipitation) shared by the data loaders.

Kept in a separate module so that both ``data_input`` and ``data_input.inca`` can
use them without importing each other.
"""

import numpy as np
import xarray as xr


def _accumulate_from_hourly(
    da_1h: xr.DataArray, steps: list[int], skipna: bool = True
) -> xr.DataArray:
    """Reconstruct cumulative-from-start from 1H period-accumulated zarr values.

    da_1h must have a 'step' dimension with hourly timedelta64 coordinates
    covering steps 1h through max(steps). Computes the cumulative sum,
    prepends step=0 with value 0 (IC), then subsets to steps.

    With ``skipna=False`` a missing (NaN) hour makes all later cumulative
    values NaN instead of counting as zero precipitation.
    """
    max_step = max(steps)
    all_1h = da_1h.sel(step=[np.timedelta64(h, "h") for h in range(1, max_step + 1)])
    cumul = all_1h.cumsum(dim="step", skipna=skipna)
    # expand_dims ensures step is the leading dimension in step0 before concat.
    # Also assign the correct time coord for step=0 (= time[step=1] - 1h = reftime)
    # so it agrees with plain-param DataArrays that have time=reftime at step=0,
    # preventing a MergeError when xr.Dataset(vars_out) is called.
    step0 = xr.zeros_like(cumul.isel(step=0)).expand_dims(
        step=[np.timedelta64(0, "ns")]
    )
    if "time" in all_1h.coords:
        ref_time = all_1h["time"].values[0] - np.timedelta64(1, "h")
        step0 = step0.assign_coords(time=("step", np.array([ref_time])))
    cumul = xr.concat([step0, cumul], dim="step")
    return cumul.sel(step=[np.timedelta64(s, "h") for s in steps])
