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
    U_10M/V_10M. Base params are left untouched."""
    ds = ds.copy()
    have_uv = "U_10M" in ds and "V_10M" in ds
    if "SP_10M" in display_params and have_uv:
        ds["SP_10M"] = wind_speed(ds["U_10M"], ds["V_10M"])
    if "DD_10M" in display_params and have_uv:
        ds["DD_10M"] = wind_direction_deg(ds["U_10M"], ds["V_10M"])
    return ds


def station_timeseries_to_long(
    ds: xr.Dataset, source: str, display_params: list[str]
) -> pd.DataFrame:
    """Flatten a single-station dataset to long form.

    Returns columns [source, valid_time, param, value]. The time coordinate is
    `valid_time` if present, else `time`. Params absent from `ds` are skipped.
    """
    tcoord = "valid_time" if "valid_time" in ds.coords or "valid_time" in ds.dims else "time"
    frames: list[pd.DataFrame] = []
    for p in display_params:
        if p not in ds:
            continue
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
