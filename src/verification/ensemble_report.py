"""Ensemble scores in the reports: helpers for consumers of aggregated verification
files (dashboard, scorecards, metric plots).

Importing this module has no side effects (no plotting style, no logging setup).
"""

import numpy as np
import xarray as xr

# Variables carrying one of these dims are counts/tables kept for plots, not scores.
NON_SCORE_DIMS = frozenset({"rank", "n_members_yes", "obs"})

CRPS_NOTE = "CRPS = MAE for deterministic forecasts"

# Source id prefix of the truth (``truth-<hash>``), which only has statistics.
TRUTH_SOURCE_PREFIX = "truth-"


def is_score_var(da: xr.DataArray) -> bool:
    """True if the variable is a score that can be tabulated or drawn as a curve."""
    return not (set(da.dims) & NON_SCORE_DIMS)


def drop_non_score_vars(ds: xr.Dataset) -> xr.Dataset:
    """Drop the rank histogram and probability tables (not scores)."""
    drop = [v for v in ds.data_vars if not is_score_var(ds[v])]
    return ds.drop_vars(drop) if drop else ds


def _ensemble_vars(ds: xr.Dataset) -> list[str]:
    """Rank histograms and probability tables: only an ensemble has them."""
    return [v for v in ds.data_vars if ".rank_hist" in v or ".prob_table" in v]


def _sources_with_values(ds: xr.Dataset, names: list[str]) -> set:
    """Sources with a value (not NaN) in any of the variables *names*."""
    found: set = set()
    for name in names:
        da = ds[name]
        if "source" not in da.dims:
            continue
        other = [d for d in da.dims if d != "source"]
        has = da.notnull().any(other) if other else da.notnull()
        found.update(ds["source"].values[has.values].tolist())
    return found


def ensemble_sources(ds: xr.Dataset) -> list:
    """Sources of *ds* that are ensembles: those with a rank histogram or probability
    table (deterministic sources have NaN there after concatenation)."""
    found = _sources_with_values(ds, _ensemble_vars(ds))
    return [s for s in ds["source"].values.tolist() if s in found]


def is_ensemble(ds: xr.Dataset) -> bool:
    """True if a single-participant dataset holds ensemble scores."""
    return any(bool(ds[v].notnull().any()) for v in _ensemble_vars(ds))


def _mae_vars(ds: xr.Dataset) -> list[str]:
    return [v for v in ds.data_vars if v.endswith(".MAE")]


def add_deterministic_crps(ds: xr.Dataset) -> tuple[xr.Dataset, bool]:
    """Multi-source dataset: give deterministic forecasts ``<p>.CRPS = <p>.MAE``.

    Only when the dataset contains an ensemble source; otherwise returned unchanged.
    Deterministic forecasts are the sources (candidate runs, baselines) with an MAE
    and no ensemble scores; the truth source is never filled. The result is for
    display only. Returns ``(dataset, filled)``, *filled* True if a source was filled.
    """
    ens = ensemble_sources(ds)
    if not ens:
        return ds, False
    with_mae = _sources_with_values(ds, _mae_vars(ds))
    deterministic = [
        s
        for s in ds["source"].values.tolist()
        if s in with_mae and s not in ens and not str(s).startswith(TRUTH_SOURCE_PREFIX)
    ]
    if not deterministic:
        return ds, False
    is_det = xr.DataArray(
        np.isin(ds["source"].values, deterministic),
        dims="source",
        coords={"source": ds["source"]},
    )
    ds = ds.copy()
    for mae in _mae_vars(ds):
        if "source" not in ds[mae].dims:
            continue
        name = mae[: -len("MAE")] + "CRPS"
        base = ds[name] if name in ds else xr.full_like(ds[mae], np.nan)
        out = xr.where(is_det, ds[mae], base)
        ds[name] = out.transpose(*ds[mae].dims)
        ds[name].attrs = dict(ds[mae].attrs)
    return ds, True


def fill_crps_from_mae(ds: xr.Dataset) -> tuple[xr.Dataset, bool]:
    """Single-participant dataset: if it is deterministic, set ``<p>.CRPS = <p>.MAE``."""
    if is_ensemble(ds):
        return ds, False
    ds = ds.copy()
    filled = False
    for mae in _mae_vars(ds):
        name = mae[: -len("MAE")] + "CRPS"
        if name in ds and bool(ds[name].notnull().any()):
            continue
        ds[name] = ds[mae]
        filled = True
    return ds, filled
