import numpy as np
import pytest

from evalml.config import ConfigModel, SalConfig
from verification.sal import (
    DEFAULT_THR_FACTOR,
    build_regular_grid,
    compute_sal,
    remap_field,
    remap_indices,
)


def _blob(shape, cy, cx, amp, sigma):
    """A single Gaussian precipitation blob on a 2-D grid."""
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2)))


# ---------------------------------------------------------------------------
# Grid construction and remapping
# ---------------------------------------------------------------------------


def test_build_regular_grid_shape_bounds_and_orientation():
    lats, lons, lat2d, lon2d = build_regular_grid((5.0, 11.0, 45.0, 48.0), 0.5, 0.5)

    assert lat2d.shape == lon2d.shape == (lats.size, lons.size)
    # Lower bounds are exact; upper bounds are included but not overshot.
    assert lats[0] == 45.0 and 48.0 - 0.5 < lats[-1] <= 48.0 + 1e-9
    assert lons[0] == 5.0 and 11.0 - 0.5 < lons[-1] <= 11.0 + 1e-9
    # latitude varies along axis 0, longitude along axis 1.
    assert np.allclose(lat2d[:, 0], lats)
    assert np.allclose(lon2d[0, :], lons)


def test_remap_indices_and_field_are_nearest_neighbour():
    src_lat = np.array([46.0, 46.0, 47.0, 47.0])
    src_lon = np.array([7.0, 8.0, 7.0, 8.0])
    tgt_lat2d = np.array([[46.1, 46.9]])
    tgt_lon2d = np.array([[7.1, 7.9]])

    idx = remap_indices(src_lat, src_lon, tgt_lat2d, tgt_lon2d)
    assert np.array_equal(idx, np.array([0, 3]))

    field = np.array([10.0, 20.0, 30.0, 40.0])
    out = remap_field(field, idx, tgt_lat2d.shape)
    assert np.array_equal(out, np.array([[10.0, 40.0]]))


def test_remap_field_fills_nan_with_zero():
    out = remap_field(np.array([np.nan, 5.0]), np.array([0, 1]), (1, 2))
    assert np.array_equal(out, np.array([[0.0, 5.0]]))


# ---------------------------------------------------------------------------
# SAL wrapper
# ---------------------------------------------------------------------------


def test_compute_sal_identical_fields_is_zero():
    f = _blob((60, 60), 30, 30, 10.0, 6.0)
    s, a, ell = compute_sal(f, f)
    assert abs(s) < 1e-6
    assert abs(a) < 1e-6
    assert abs(ell) < 1e-6


def test_compute_sal_dry_window_returns_nan():
    dry = np.zeros((40, 40))
    wet = _blob((40, 40), 20, 20, 5.0, 4.0)
    assert all(np.isnan(v) for v in compute_sal(dry, wet))
    assert all(np.isnan(v) for v in compute_sal(wet, dry))
    assert all(np.isnan(v) for v in compute_sal(dry, dry))


def test_compute_sal_amplitude_positive_when_overforecast():
    obs = _blob((60, 60), 30, 30, 10.0, 6.0)
    pred = 2.0 * obs  # uniform doubling → amplitude bias only
    s, a, ell = compute_sal(pred, obs)
    assert a > 0.5  # analytic value ~0.667
    assert abs(s) < 1e-6  # structure unchanged by uniform scaling
    assert abs(ell) < 1e-6  # location unchanged


def test_compute_sal_location_positive_when_displaced():
    obs = _blob((80, 80), 40, 25, 10.0, 5.0)
    pred = _blob((80, 80), 40, 55, 10.0, 5.0)  # same blob shifted east
    _, _, ell = compute_sal(pred, obs)
    assert ell > 0.0


def test_compute_sal_is_invariant_to_common_rescaling():
    obs = _blob((80, 80), 40, 25, 10.0, 5.0)
    pred = _blob((80, 80), 38, 50, 7.0, 6.0)  # differs in S, A and L
    base = compute_sal(pred, obs)
    scaled = compute_sal(10.0 * pred, 10.0 * obs)
    assert np.allclose(base, scaled, atol=1e-9)
    assert np.isfinite(base).all()


# ---------------------------------------------------------------------------
# Config model
# ---------------------------------------------------------------------------


def test_sal_config_defaults_and_extra_forbid():
    s = SalConfig()
    assert s.enabled is False
    assert s.params == ["TOT_PREC6"]
    assert s.thr_factor == pytest.approx(DEFAULT_THR_FACTOR)
    assert s.grid_extent == [-1.0, 18.0, 42.0, 50.5]
    with pytest.raises(ValueError):
        SalConfig(unknown_key=1)


def test_sal_config_grid_extent_validation():
    with pytest.raises(ValueError, match="lon_min, lon_max, lat_min, lat_max"):
        SalConfig(grid_extent=[1.0, 2.0, 3.0])  # wrong length
    with pytest.raises(ValueError, match="lon_min < lon_max"):
        SalConfig(grid_extent=[18.0, -1.0, 42.0, 50.5])  # lon_min >= lon_max


def test_sal_leadtime_validator_rejects_unproducible(example_config):
    example_config["experiment"]["sal"] = {"enabled": True, "leadtimes": [9999]}
    with pytest.raises(ValueError, match="sal.leadtimes"):
        ConfigModel.model_validate(example_config)


def test_sal_disabled_skips_leadtime_validation(example_config):
    # Disabled SAL must not trigger leadtime validation even with a bogus value.
    example_config["experiment"]["sal"] = {"enabled": False, "leadtimes": [9999]}
    ConfigModel.model_validate(example_config)
