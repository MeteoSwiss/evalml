import numpy as np
import pytest

from evalml.config import ConfigModel, SalConfig
from verification.sal import GRID_EXTENT, compute_sal, remap_field, sal_raster


def _blob(shape, cy, cx, amp, sigma):
    """A single Gaussian precipitation blob on a 2-D grid."""
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2)))


# ---------------------------------------------------------------------------
# Grid construction and remapping
# ---------------------------------------------------------------------------


def test_sal_raster_shape_bounds_and_orientation():
    lat2d, lon2d = sal_raster()
    lon_min, lon_max, lat_min, lat_max = GRID_EXTENT

    assert lat2d.shape == lon2d.shape == (851, 1311)
    # Lower bounds are exact; upper bounds are included but not overshot.
    assert lat2d[0, 0] == lat_min and lat_max - 0.01 < lat2d[-1, 0] <= lat_max + 1e-9
    assert lon2d[0, 0] == lon_min and lon_max - 0.0145 < lon2d[0, -1] <= lon_max + 1e-9
    # latitude varies along axis 0, longitude along axis 1 (pysteps' L term).
    assert np.all(lat2d[0, :] == lat2d[0, 0]) and np.all(lon2d[:, 0] == lon2d[0, 0])


def test_remap_field_gathers_and_fills_nan_with_zero():
    # NaN source cells fill with 0 (missing precip = no precip).
    out = remap_field(np.array([np.nan, 5.0, 7.0]), np.array([2, 0, 1]), (1, 3))
    assert np.array_equal(out, np.array([[7.0, 0.0, 5.0]]))


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


# ---------------------------------------------------------------------------
# Config model
# ---------------------------------------------------------------------------


def test_sal_config_rejects_non_precip_params():
    with pytest.raises(ValueError, match="TOT_PREC"):
        SalConfig(params=["T_2M"])
    with pytest.raises(ValueError, match="TOT_PREC"):
        SalConfig(params=["TOT_PREC6", "SP_10M"])  # one bad among valid


def test_sal_leadtime_validator_rejects_unproducible(example_config):
    example_config["experiment"]["sal"] = {"enabled": True, "leadtimes": [9999]}
    with pytest.raises(ValueError, match="sal.leadtimes"):
        ConfigModel.model_validate(example_config)
