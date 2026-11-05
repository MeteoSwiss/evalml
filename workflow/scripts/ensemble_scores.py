"""Helpers for consumers of aggregated verification files (ensemble scores).

Shared by the dashboard, the scorecards and the metric plots. Importing this module
has no side effects (no plotting style, no logging setup).
"""

import numpy as np
import xarray as xr

# Variables carrying one of these dims are counts/tables kept for plots, not scores.
NON_SCORE_DIMS = frozenset({"rank", "n_members_yes", "obs"})

CRPS_NOTE = "CRPS = MAE for deterministic forecasts"


def is_score_var(da: xr.DataArray) -> bool:
    """True if the variable is a score that can be tabulated or drawn as a curve."""
    return not (set(da.dims) & NON_SCORE_DIMS)


def drop_non_score_vars(ds: xr.Dataset) -> xr.Dataset:
    """Drop the rank histogram and probability tables (not scores)."""
    drop = [v for v in ds.data_vars if not is_score_var(ds[v])]
    return ds.drop_vars(drop) if drop else ds


def _spread_vars(ds: xr.Dataset) -> list[str]:
    return [v for v in ds.data_vars if v.endswith(".SPREAD")]


def ensemble_sources(ds: xr.Dataset) -> list:
    """Sources of *ds* that carry ensemble scores (any finite SPREAD)."""
    found: set = set()
    for v in _spread_vars(ds):
        if "source" not in ds[v].dims:
            continue
        other = [d for d in ds[v].dims if d != "source"]
        has = ds[v].notnull().any(other) if other else ds[v].notnull()
        found.update(ds["source"].values[has.values].tolist())
    return [s for s in ds["source"].values.tolist() if s in found]


def add_deterministic_crps(ds: xr.Dataset) -> tuple[xr.Dataset, bool]:
    """Multi-source dataset: give deterministic sources ``<p>.CRPS = <p>.MAE``.

    Only when the dataset contains an ensemble source; otherwise returned unchanged.
    The result is for display only. Returns ``(dataset, filled)``.
    """
    ens = ensemble_sources(ds)
    if not ens:
        return ds, False
    is_det = xr.DataArray(
        ~np.isin(ds["source"].values, ens),
        dims="source",
        coords={"source": ds["source"]},
    )
    ds = ds.copy()
    filled = False
    for mae in [v for v in ds.data_vars if v.endswith(".MAE")]:
        if "source" not in ds[mae].dims:
            continue
        name = mae[: -len("MAE")] + "CRPS"
        base = ds[name] if name in ds else xr.full_like(ds[mae], np.nan)
        out = xr.where(is_det, ds[mae], base)
        ds[name] = out.transpose(*ds[mae].dims)
        ds[name].attrs = dict(ds[mae].attrs)
        filled = True
    return ds, filled


def fill_crps_from_mae(ds: xr.Dataset) -> tuple[xr.Dataset, bool]:
    """Single-participant dataset: if it is deterministic, set ``<p>.CRPS = <p>.MAE``."""
    if any(bool(ds[v].notnull().any()) for v in _spread_vars(ds)):
        return ds, False
    ds = ds.copy()
    filled = False
    for mae in [v for v in ds.data_vars if v.endswith(".MAE")]:
        name = mae[: -len("MAE")] + "CRPS"
        if name in ds and bool(ds[name].notnull().any()):
            continue
        ds[name] = ds[mae]
        filled = True
    return ds, filled
