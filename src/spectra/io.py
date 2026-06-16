"""Thin evalml I/O for spectra: variable->component mapping and native-field
extraction. Field loading is delegated to data_input loaders."""

from __future__ import annotations

import numpy as np
import xarray as xr

# Logical spectra variable -> (component GRIB params, scale factor).
# WIND_KE combines U/V as kinetic energy E = 0.5*(|u_hat|^2 + |v_hat|^2).
VARIABLE_COMPONENTS: dict[str, tuple[list[str], float]] = {
    "T_2M": (["T_2M"], 1.0),
    "WIND_KE": (["U_10M", "V_10M"], 0.5),
    "TOT_PREC": (["TOT_PREC"], 1.0),
}


def required_params(variables: list[str]) -> list[str]:
    """All GRIB params needed to compute the requested spectra variables."""
    params: list[str] = []
    for var in variables:
        if var not in VARIABLE_COMPONENTS:
            raise KeyError(
                f"Unknown spectra variable {var!r}. "
                f"Known: {sorted(VARIABLE_COMPONENTS)}."
            )
        for p in VARIABLE_COMPONENTS[var][0]:
            if p not in params:
                params.append(p)
    return params


def native_field(ds: xr.Dataset, param: str, step: int) -> np.ndarray:
    """1-D native field for `param` at lead-time `step` (hours).

    Handles both timedelta64 step coordinates (as produced by the data_input
    GRIB/baseline loaders) and plain integer-hour step coordinates.
    """
    da = ds[param]
    if "step" in da.dims:
        step_coord = da.coords["step"]
        if np.issubdtype(step_coord.dtype, np.timedelta64):
            da = da.sel(step=np.timedelta64(step, "h"))
        else:
            da = da.sel(step=step)
    # Drop singleton non-spatial dims left over from the loaders (e.g. a size-1
    # forecast_reference_time, or a single ensemble `number`), keeping only the
    # native grid axis. A genuine multi-member dim survives and trips the guard.
    da = da.squeeze(drop=True)
    if da.ndim != 1:
        raise ValueError(
            f"Expected a 1-D native field for {param!r} after step selection, "
            f"got shape {da.shape} with dims {da.dims}. Power spectra require "
            f"data on the ICON native (unstructured) grid."
        )
    return np.asarray(da.values).reshape(-1)


def native_components(
    ds: xr.Dataset, variable: str, step: int
) -> tuple[list[np.ndarray], float]:
    """Native component arrays + scale factor for a spectra variable at `step`."""
    if variable not in VARIABLE_COMPONENTS:
        raise KeyError(
            f"Unknown spectra variable {variable!r}. "
            f"Known: {sorted(VARIABLE_COMPONENTS)}."
        )
    params, factor = VARIABLE_COMPONENTS[variable]
    return [native_field(ds, p, step) for p in params], factor
