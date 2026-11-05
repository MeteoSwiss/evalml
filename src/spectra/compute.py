"""Orchestration: per-source spectra computation and over-init aggregation.
Defines the spectra.nc schema."""

from __future__ import annotations

import numpy as np
import xarray as xr

from spectra import core, io, regrid


def compute_source_spectra(ds: xr.Dataset, variables, lead_times, method, label):
    """Compute spectra for one source (already-loaded native dataset).

    Returns an xr.Dataset with dims (variable, leadtime, wavenumber) and a shared
    `wavelength` coordinate. The native grid is detected from the field length.
    """
    if not variables or not lead_times:
        raise ValueError(
            "compute_source_spectra requires non-empty `variables` and `lead_times`."
        )
    spectrum_fn = core.SPECTRUM_FUNCS.get(method)
    if spectrum_fn is None:
        raise ValueError(
            f"Unknown spectrum method {method!r}. Valid: {list(core.SPECTRUM_FUNCS)}."
        )
    npoints = io.native_field(
        ds, io.required_params(variables)[0], lead_times[0]
    ).shape[0]
    matrix, ny, nx, dx_km = regrid.load_regridder(npoints)

    wavelength = None
    power = np.full((len(variables), len(lead_times), min(ny, nx) // 2), np.nan)
    for vi, var in enumerate(variables):
        for li, step in enumerate(lead_times):
            comps, factor = io.native_components(ds, var, step)
            grids = [regrid.regrid(c, matrix, ny, nx) for c in comps]
            wl, p = core.combined_spectrum(spectrum_fn, grids, dx_km, factor=factor)
            wavelength = wl
            power[vi, li, :] = p

    return xr.Dataset(
        {"power": (("variable", "leadtime", "wavenumber"), power)},
        coords={
            "variable": list(variables),
            "leadtime": list(lead_times),
            "wavenumber": np.arange(power.shape[-1]),
            "wavelength": ("wavenumber", wavelength),
        },
        attrs={
            "dx_km": float(dx_km),
            "npoints": int(npoints),
            "label": label,
            "method": method,
        },
    )


def aggregate_spectra(spectra_files) -> xr.Dataset:
    """Average power over init times. All inputs share one grid.

    NaN propagates: a value missing in any input is missing in the average.
    """
    datasets = [xr.open_dataset(f) for f in spectra_files]
    try:
        stacked = xr.concat(datasets, dim="init")
        agg = stacked.mean(dim="init", skipna=False)
        agg.attrs = datasets[0].attrs
        agg["wavelength"] = datasets[0]["wavelength"].copy()
        agg.load()
    finally:
        for ds in datasets:
            ds.close()
    return agg
