"""Thin evalml I/O for spectra: variable->component mapping and native-field
extraction. Field loading is delegated to data_input loaders."""

from __future__ import annotations

import numpy as np
import xarray as xr

# Spectra variables built from several GRIB params: name -> (component params,
# scale factor). WIND_KE combines U/V as kinetic energy
# E = 0.5*(|u_hat|^2 + |v_hat|^2). Any other variable is a single param, loaded
# as such by the data_input loaders (including derived params such as SP_10M
# and period params such as TOT_PREC1/TOT_PREC6).
VARIABLE_COMPONENTS: dict[str, tuple[list[str], float]] = {
    "WIND_KE": (["U_10M", "V_10M"], 0.5),
}


def variable_components(variable: str) -> tuple[list[str], float]:
    """Component params and scale factor of a spectra variable."""
    return VARIABLE_COMPONENTS.get(variable, ([variable], 1.0))


def required_params(variables: list[str]) -> list[str]:
    """All params to load to compute the requested spectra variables."""
    params: list[str] = []
    for var in variables:
        for p in variable_components(var)[0]:
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
    params, factor = variable_components(variable)
    return [native_field(ds, p, step) for p in params], factor
