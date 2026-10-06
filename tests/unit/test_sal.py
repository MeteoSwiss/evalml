import importlib.util
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from evalml.config import ConfigModel, SalConfig
from verification.sal import (
    DEFAULT_GRID_EXTENT,
    DEFAULT_GRID_STEP_LAT,
    DEFAULT_GRID_STEP_LON,
    compute_sal,
    remap_field,
    sal_raster,
)

SCRIPT = Path(__file__).resolve().parents[2] / "workflow/scripts/verification_sal.py"


def _blob(shape, cy, cx, amp, sigma):
    """A single Gaussian precipitation blob on a 2-D grid."""
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2)))


# ---------------------------------------------------------------------------
# Grid construction and remapping
# ---------------------------------------------------------------------------


def test_sal_raster_shape_bounds_and_orientation():
    lat2d, lon2d = sal_raster()
    lon_min, lon_max, lat_min, lat_max = DEFAULT_GRID_EXTENT

    assert lat2d.shape == lon2d.shape == (851, 1311)
    # Lower bounds are exact; the raster never leaves the extent and comes within
    # one step of its upper bounds -- exactly 50.5 in latitude (8.5 deg is a whole
    # multiple of 0.01), 0.005 deg short of 18.0 in longitude.
    assert lat2d[0, 0] == lat_min and lat_max - 0.01 < lat2d[-1, 0] <= lat_max
    assert lon2d[0, 0] == lon_min and lon_max - 0.0145 < lon2d[0, -1] <= lon_max
    # latitude varies along axis 0, longitude along axis 1 (pysteps' L term).
    assert np.all(lat2d[0, :] == lat2d[0, 0]) and np.all(lon2d[:, 0] == lon2d[0, 0])


def test_sal_raster_honours_custom_extent_and_steps():
    lat2d, lon2d = sal_raster((5.0, 6.0, 45.0, 46.0), step_lat=0.5, step_lon=0.25)
    assert lat2d.shape == lon2d.shape == (3, 5)
    assert lat2d[:, 0].tolist() == [45.0, 45.5, 46.0]
    assert lon2d[0].tolist() == [5.0, 5.25, 5.5, 5.75, 6.0]


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


def test_sal_config_grid_defaults_match_the_module_constants():
    # The config carries its own literals (importing verification.sal would pull
    # pysteps into every config validation), so pin the two together.
    s = SalConfig()
    assert tuple(s.grid_extent) == DEFAULT_GRID_EXTENT
    assert (s.grid_step_lat, s.grid_step_lon) == (
        DEFAULT_GRID_STEP_LAT,
        DEFAULT_GRID_STEP_LON,
    )


def test_sal_config_grid_validation():
    with pytest.raises(ValueError, match="lon_min, lon_max, lat_min, lat_max"):
        SalConfig(grid_extent=[1.0, 2.0, 3.0])  # wrong length
    with pytest.raises(ValueError, match="increasing"):
        SalConfig(grid_extent=[18.0, -1.0, 42.0, 50.5])  # lon_min >= lon_max
    with pytest.raises(ValueError, match="finite"):
        SalConfig(grid_extent=[float("nan"), 18.0, 42.0, 50.5])  # non-finite
    with pytest.raises(ValueError, match="lat in"):
        SalConfig(grid_extent=[-1.0, 18.0, 42.0, 100.0])  # lat_max out of range
    with pytest.raises(ValueError, match="grid_step_lat"):
        SalConfig(grid_step_lat=0.0)  # a non-positive spacing yields no raster


def test_sal_config_rejects_non_precip_params():
    with pytest.raises(ValueError, match="TOT_PREC"):
        SalConfig(params=["T_2M"])
    with pytest.raises(ValueError, match="TOT_PREC"):
        SalConfig(params=["TOT_PREC6", "SP_10M"])  # one bad among valid


def test_sal_leadtime_validator_rejects_unproducible(example_config):
    example_config["experiment"]["sal"] = {"enabled": True, "leadtimes": [9999]}
    with pytest.raises(ValueError, match="sal.leadtimes"):
        ConfigModel.model_validate(example_config)


# ---------------------------------------------------------------------------
# Script pre-flight guards
#
# Both fire before any data is read, so they are unit-testable: no archive, no
# truth zarr, no snakemake. The end-to-end path is covered by the longtest
# (tests/integration/test_sal_small.py), which needs /store_new.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sal_script():
    """verification_sal.py loaded by path (workflow/scripts is not a package)."""
    spec = importlib.util.spec_from_file_location("verification_sal", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_rejects_step_below_accumulation(sal_script):
    # A 6 h accumulation cannot be evaluated at a 3 h lead time.
    with pytest.raises(ValueError, match="accumulation"):
        sal_script.main(Namespace(param="TOT_PREC6", step=3, truth=Path("truth.zarr")))


def test_script_rejects_non_zarr_truth(sal_script):
    # SAL needs a resolved gridded field; station observations and other
    # non-zarr roots are rejected up front rather than failing on read.
    with pytest.raises(ValueError, match="zarr"):
        sal_script.main(
            Namespace(param="TOT_PREC6", step=6, truth=Path("jretrievedwh:surface"))
        )
