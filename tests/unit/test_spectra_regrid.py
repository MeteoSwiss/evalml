# tests/unit/test_spectra_regrid.py
import numpy as np
import pytest
import xarray as xr

from spectra import regrid


def _fake_weights(tmp_path, ny=4, nx=5, n_source=20, npoints=3):
    # Realistic iconremap file: 0-based source indices, used as-is. Index 0 is a
    # valid source cell (here paired with a nonzero weight), not padding.
    rng = np.arange(ny * nx * npoints) % n_source
    idx = rng.reshape(ny * nx, npoints).astype("int64")  # 0-based, 0..n_source-1
    wgt = np.full((ny * nx, npoints), 1.0 / npoints, dtype="float64")
    ds = xr.Dataset(
        {
            "rbf_B_glbidx": (("cell", "stencil"), idx),
            "rbf_B_wgt": (("cell", "stencil"), wgt),
        },
        attrs={"nx": nx, "ny": ny, "dx": 0.01},
    )
    path = tmp_path / "weights.nc"
    ds.to_netcdf(path)
    return path, ny, nx, n_source


def test_build_truncation_matrix_applies_weights(tmp_path):
    path, ny, nx, n_source = _fake_weights(tmp_path)
    M, (rny, rnx) = regrid.build_truncation_matrix(path, n_source=n_source)
    assert (rny, rnx) == (ny, nx)
    native = np.ones(n_source)
    out = regrid.regrid(native, M, ny, nx)
    assert out.shape == (ny, nx)
    np.testing.assert_allclose(out, 1.0, rtol=1e-6)  # average of ones is one


def test_grid_geometry_reads_dx_in_km(tmp_path):
    path, ny, nx, _ = _fake_weights(tmp_path)
    g_ny, g_nx, dx_km = regrid.grid_geometry(path)
    assert (g_ny, g_nx) == (ny, nx)
    assert dx_km == pytest.approx(0.01 * regrid.core.KM_PER_DEG)


def test_weights_path_for_npoints_unknown_raises():
    with pytest.raises(KeyError, match="No iconremap weights"):
        regrid.weights_path_for_npoints(999_999)


def test_weights_path_for_npoints_known():
    assert "ch1" in str(regrid.weights_path_for_npoints(1_147_980)).lower()


def _fake_weights_zero_based_no_zeros(tmp_path, ny=4, nx=5, n_source=20):
    # 0-based file (as the real ICON-CH1 weights): no zero entries at all, and
    # the smallest *referenced* cell is >= 1, yet max index == n_source - 1. A
    # min-index heuristic wrongly calls this 1-based and shifts every reference
    # by one -- the off-by-one that injected grid-scale noise into the spectra.
    # Single-point stencils (weight 1) so the regridded value IS the referenced
    # native cell, making any shift directly observable.
    idx = ((np.arange(ny * nx) % (n_source - 5)) + 5).reshape(ny * nx, 1)
    assert idx.min() >= 1 and idx.max() == n_source - 1 and (idx != 0).all()
    wgt = np.ones((ny * nx, 1), dtype="float64")
    ds = xr.Dataset(
        {
            "rbf_B_glbidx": (("cell", "stencil"), idx.astype("int64")),
            "rbf_B_wgt": (("cell", "stencil"), wgt),
        },
        attrs={"nx": nx, "ny": ny, "dx": 0.01},
    )
    path = tmp_path / "weights_zero_no_pad.nc"
    ds.to_netcdf(path)
    return path, ny, nx, n_source, idx.ravel()


def test_build_truncation_matrix_zero_based_no_zeros_not_shifted(tmp_path):
    # Regression: a 0-based, zero-padding-free file must NOT be treated as 1-based.
    path, ny, nx, n_source, ref_cells = _fake_weights_zero_based_no_zeros(tmp_path)
    M, _ = regrid.build_truncation_matrix(path, n_source=n_source)
    native = np.arange(n_source, dtype="float64")  # value == cell index
    out = regrid.regrid(native, M, ny, nx).ravel()
    # No off-by-one: each target picks up its referenced native cell as-is.
    np.testing.assert_array_equal(out, ref_cells)
