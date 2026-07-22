"""Pure data derivations for the publication meteogram.

No plotting, no file IO — kept importable and unit-testable in isolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

# Display parameter -> base GRIB parameters required to compute it.
_DERIVED: dict[str, list[str]] = {
    "SP_10M": ["U_10M", "V_10M"],
    "DD_10M": ["U_10M", "V_10M"],
}


def expand_to_base_params(display_params: list[str]) -> list[str]:
    """Expand display params (may include derived SP_10M/DD_10M) to the base
    params that must be loaded. Order-preserving and de-duplicated."""
    base: list[str] = []
    for p in display_params:
        for bp in _DERIVED.get(p, [p]):
            if bp not in base:
                base.append(bp)
    return base


def wind_speed(u, v):
    """10 m wind speed from u/v components (same units as the inputs)."""
    return np.sqrt(u**2 + v**2)


def wind_direction_deg(u, v):
    """Meteorological wind direction in degrees, the direction the wind blows
    FROM (0=N, 90=E, 180=S, 270=W). Returns values in [0, 360)."""
    return (270.0 - np.rad2deg(np.arctan2(v, u))) % 360.0


def add_derived(ds: xr.Dataset, display_params: list[str]) -> xr.Dataset:
    """Add requested derived variables (SP_10M, DD_10M) to a dataset that holds
    U_10M/V_10M. Base params are left untouched.

    Raises ValueError if a derived wind parameter is requested but U_10M/V_10M
    are not present to compute it (fail loudly rather than silently skipping).
    """
    ds = ds.copy()
    wanted = [p for p in ("SP_10M", "DD_10M") if p in display_params]
    if wanted and not ("U_10M" in ds and "V_10M" in ds):
        raise ValueError(
            f"Cannot derive {wanted} without U_10M and V_10M "
            f"(dataset has: {list(ds.data_vars)})."
        )
    if "SP_10M" in display_params:
        ds["SP_10M"] = wind_speed(ds["U_10M"], ds["V_10M"])
    if "DD_10M" in display_params:
        ds["DD_10M"] = wind_direction_deg(ds["U_10M"], ds["V_10M"])
    return ds


def station_timeseries_to_long(
    ds: xr.Dataset, source: str, display_params: list[str]
) -> pd.DataFrame:
    """Flatten a single-station dataset to long form.

    Returns columns [source, valid_time, param, value]. The time coordinate is
    `valid_time` if present, else `time`.

    Raises KeyError if the dataset has no time coordinate, or if any requested
    parameter is absent (fail loudly rather than silently skipping).
    """
    if "valid_time" in ds.coords or "valid_time" in ds.dims:
        tcoord = "valid_time"
    elif "time" in ds.coords or "time" in ds.dims:
        tcoord = "time"
    else:
        raise KeyError(
            f"{source}: dataset has no 'valid_time' or 'time' coordinate "
            f"(coords: {list(ds.coords)})."
        )
    missing = [p for p in display_params if p not in ds]
    if missing:
        raise KeyError(
            f"{source}: requested parameters absent from dataset: {missing} "
            f"(available: {list(ds.data_vars)})."
        )
    frames: list[pd.DataFrame] = []
    for p in display_params:
        da = ds[p].squeeze()
        times = np.asarray(ds[tcoord].squeeze().values).reshape(-1)
        values = np.asarray(da.values).reshape(-1)
        frames.append(
            pd.DataFrame(
                {"source": source, "valid_time": times, "param": p, "value": values}
            )
        )
    if not frames:
        return pd.DataFrame(columns=["source", "valid_time", "param", "value"])
    return pd.concat(frames, ignore_index=True)
