"""Orchestration: per-source spectra computation, over-init aggregation, and the
experiment overlay plot. Defines the spectra.nc schema."""

from __future__ import annotations

from itertools import cycle
from pathlib import Path

import numpy as np
import xarray as xr

from spectra import core, io, regrid

_PALETTE = ["#1f3b73", "#c1272d", "#1b7837", "#6a3d9a", "#e08214", "#4d4d4d"]


def compute_source_spectra(ds: xr.Dataset, variables, lead_times, method, label):
    """Compute spectra for one source (already-loaded native dataset).

    Returns an x.Dataset with dims (variable, leadtime, wavenumber) and a shared
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
    """Average power over init times (nanmean). All inputs share one grid."""
    datasets = [xr.open_dataset(f) for f in spectra_files]
    try:
        stacked = xr.concat(datasets, dim="init")
        agg = stacked.mean(dim="init", skipna=True)
        agg.attrs = datasets[0].attrs
        agg["wavelength"] = datasets[0]["wavelength"].copy()
        agg.load()
    finally:
        for ds in datasets:
            ds.close()
    return agg


def plot_experiment_spectra(
    truth_file, participant_files, out_dir, variables, lead_times
):
    """One overlay figure per (variable, lead time): truth + all participants."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    participants = [xr.open_dataset(f) for f in participant_files]
    try:
        with xr.open_dataset(truth_file) as truth:
            for var in variables:
                for step in lead_times:
                    spectra = {}
                    t_wl = truth["wavelength"].values
                    t_p = truth["power"].sel(variable=var, leadtime=step).values
                    spectra[f"{truth.attrs['label']} (truth)"] = (
                        t_wl,
                        t_p,
                        "k",
                        "-",
                        1.0,
                        2.0,
                    )
                    color = cycle(_PALETTE)
                    for ds in participants:
                        wl = ds["wavelength"].values
                        p = ds["power"].sel(variable=var, leadtime=step).values
                        spectra[ds.attrs["label"]] = (wl, p, next(color), "-", 0.9, 1.6)
                    out = out_dir / f"spectrum_{var}_{step:03d}.png"
                    # Reference grid/eff-res/Nyquist lines use the TRUTH grid's dx (labelled
                    # "truth"); each participant curve keeps its own wavelength axis.
                    core.plot_power_spectra(
                        spectra,
                        out,
                        f"Power spectrum — {var} @ +{step}h",
                        grid_dx_km=float(truth.attrs["dx_km"]),
                        model_short="truth",
                        model_color="k",
                    )
    finally:
        for ds in participants:
            ds.close()
