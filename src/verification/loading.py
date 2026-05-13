"""Helpers for loading aggregated verification netCDFs into long-form DataFrames."""

from pathlib import Path

import pandas as pd
import xarray as xr


def _ensure_unique_lead_time(ds: xr.Dataset) -> xr.Dataset:
    """Drop duplicate lead_time entries within a Dataset (keep first occurrence)."""
    try:
        idx = ds.get_index("lead_time")
    except Exception:
        idx = pd.Index(ds["lead_time"].values)
    if getattr(idx, "has_duplicates", False):
        keep = ~idx.duplicated(keep="first")
        ds = ds.isel(lead_time=keep)
    return ds


def _select_best_sources(dfs: list[xr.Dataset]) -> list[xr.Dataset]:
    """For sources present in multiple datasets, keep the one with the most lead_times."""
    src_sets = [set(d.source.values.tolist()) for d in dfs]
    all_sources = set().union(*src_sets)

    best: dict[str, int] = {}
    for s in all_sources:
        candidates = []
        for i, d in enumerate(dfs):
            if s in d.source.values:
                di = d.sel(source=s)
                try:
                    n = pd.Index(di["lead_time"].values).unique().size
                except Exception:
                    n = len(pd.unique(di["lead_time"].values))
                candidates.append((i, n))
        if candidates:
            best_idx, _ = max(candidates, key=lambda t: t[1])
            best[s] = best_idx

    out = []
    for i, d in enumerate(dfs):
        drop_src = [s for s, b in best.items() if b != i and s in d.source.values]
        if drop_src:
            d = d.drop_sel(source=drop_src)
        out.append(d)
    return out


def load_long_df(verif_files: list[Path]) -> pd.DataFrame:
    """Open verification netCDFs and return a long-form DataFrame.

    Columns: source, lead_time (hours, float), region, season, init_hour,
    param, metric, value.
    """
    dfs = [xr.open_dataset(f) for f in verif_files]
    dfs = [_ensure_unique_lead_time(d) for d in dfs]
    dfs = _select_best_sources(dfs)
    ds = xr.concat(dfs, dim="source", join="outer")

    nonspatial_vars = [d for d in ds.data_vars if "spatial" not in d]
    df = ds[nonspatial_vars].to_array("stack").to_dataframe(name="value").reset_index()
    df[["param", "metric"]] = df["stack"].str.split(".", n=1, expand=True)
    df.drop(columns=["stack"], inplace=True)
    df["lead_time"] = df["lead_time"].dt.total_seconds() / 3600
    return df


def subset_df(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Return rows of `df` matching every column=value (or column in [values]) constraint."""
    mask = pd.Series([True] * len(df))
    for key, value in kwargs.items():
        if isinstance(value, (list, tuple, set)):
            mask &= df[key].isin(value)
        else:
            mask &= df[key] == value
    return df[mask]
