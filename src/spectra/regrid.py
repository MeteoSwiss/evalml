# src/spectra/regrid.py
"""ICON native (unstructured) -> regular rotlatlon regridding via iconremap RBF
weights. The weights file is selected at runtime from the native point count."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.sparse
import xarray as xr

from spectra import core

# Hardcoded iconremap RBF weight files on Balfrin, keyed by native cell count.
# CH1 native count is 1,147,980 (per the icon-power-spectra skill).
# TODO confirm CH2 count (see plan Task 2 Step 1).
_WEIGHTS_ROOT = Path("/store_new/mch/msopr/icon_workflow_2/iconremap-weights")
CH1_NPOINTS = 1_147_980  # confirmed vs ICON-CH1-EPS native T_2M field
CH2_NPOINTS = 283_876  # confirmed vs ICON-CH2-EPS native T_2M field
ICON_WEIGHTS = {
    CH1_NPOINTS: _WEIGHTS_ROOT / "icon-ch1-eps-rotlatlon.nc",
    CH2_NPOINTS: _WEIGHTS_ROOT / "icon-ch2-eps-rotlatlon.nc",
}
# TODO: prefer uuidOfHGrid as the registry key if point counts ever collide.
# TODO: eventually consume precomputed .npz sparse matrices directly.


def weights_path_for_npoints(npoints: int) -> Path:
    """Resolve the iconremap weights file for a native grid of `npoints` cells."""
    try:
        return ICON_WEIGHTS[npoints]
    except KeyError as exc:
        raise KeyError(
            f"No iconremap weights registered for native grid with {npoints} "
            f"cells. Known grids: {sorted(ICON_WEIGHTS)}."
        ) from exc


def build_truncation_matrix(iconremap_nc, n_source):
    """Sparse native(n_source) -> regular(ny*nx) matrix from iconremap weights."""
    with xr.open_dataset(iconremap_nc) as ds:
        idx = ds["rbf_B_glbidx"].values
        wgt = ds["rbf_B_wgt"].values
        nx, ny = int(ds.attrs["nx"]), int(ds.attrs["ny"])
    npoints = idx.shape[-1]
    idx = idx.reshape(ny * nx, npoints)
    wgt = wgt.reshape(ny * nx, npoints)
    n_targets = ny * nx

    # iconremap indices are 0-based and used as-is (cf. meteodata-lab's
    # `_icon2regular`, which indexes with `np.take(field, indices)`).
    rows = np.broadcast_to(np.arange(n_targets)[:, None], idx.shape).ravel()
    cols = idx.ravel()
    vals = wgt.ravel().astype(np.float32)

    M = scipy.sparse.coo_matrix(
        (vals, (rows, cols)), shape=(n_targets, n_source)
    ).tocsr()
    return M, (ny, nx)


def grid_geometry(iconremap_nc):
    """(ny, nx, dx_km) of the regular target grid from the iconremap weights."""
    with xr.open_dataset(iconremap_nc) as ds:
        return (
            int(ds.attrs["ny"]),
            int(ds.attrs["nx"]),
            float(ds.attrs["dx"]) * core.KM_PER_DEG,
        )


def regrid(native_1d, matrix, ny, nx):
    """Apply a truncation matrix to a native field -> (ny, nx) regular array."""
    return (matrix @ np.asarray(native_1d)).reshape(ny, nx)


@lru_cache(maxsize=4)
def load_regridder(npoints: int):
    """Cached regridder for a native grid: (matrix, ny, nx, dx_km)."""
    weights = weights_path_for_npoints(npoints)
    matrix, (ny, nx) = build_truncation_matrix(weights, n_source=npoints)
    g_ny, g_nx, dx_km = grid_geometry(weights)
    if (g_ny, g_nx) != (ny, nx):
        raise ValueError(
            f"Grid geometry mismatch for {npoints} cells: "
            f"matrix target {(ny, nx)} vs grid_geometry {(g_ny, g_nx)}."
        )
    return matrix, ny, nx, dx_km
