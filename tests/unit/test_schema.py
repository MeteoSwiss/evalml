import numpy as np
import pytest
import xarray as xr

from data.schema import XARRAY_ENGINE_PROFILE, validate_canonical, wrap_longitude


def test_wrap_longitude_wraps_above_180():
    lon = np.array([0.0, 90.0, 180.0, 270.0, 359.0])
    wrapped = wrap_longitude(lon)
    assert wrapped.max() <= 180.0
    # 180 sits on the boundary and maps to -180 (same meridian), matching the
    # prior inline ((lon + 180) % 360) - 180 behaviour.
    np.testing.assert_allclose(wrapped, [0.0, 90.0, -180.0, -90.0, -1.0])


def test_wrap_longitude_noop_when_in_range():
    lon = np.array([-180.0, -1.0, 0.0, 179.0])
    np.testing.assert_array_equal(wrap_longitude(lon), lon)


def test_wrap_longitude_idempotent():
    lon = np.array([0.0, 270.0, 359.0])
    once = wrap_longitude(lon)
    np.testing.assert_array_equal(wrap_longitude(once), once)


def test_engine_profile_has_expected_keys():
    assert set(XARRAY_ENGINE_PROFILE) == {
        "ensure_dims",
        "add_valid_time_coord",
        "global_attrs",
    }


def _canonical_ds():
    return xr.Dataset(
        {"T_2M": (("step", "y", "x"), np.zeros((1, 2, 2)))},
        coords={
            "step": [np.timedelta64(0, "h")],
            "forecast_reference_time": np.datetime64("2024-01-01T00:00:00"),
            "valid_time": ("step", [np.datetime64("2024-01-01T00:00:00")]),
            "latitude": (("y", "x"), np.zeros((2, 2))),
            "longitude": (("y", "x"), np.zeros((2, 2))),
        },
    )


def test_validate_canonical_accepts_conformant_dataset():
    ds = _canonical_ds()
    assert validate_canonical(ds) is ds


def test_validate_canonical_rejects_missing_coord():
    ds = _canonical_ds().drop_vars("valid_time")
    with pytest.raises(ValueError, match="valid_time"):
        validate_canonical(ds)
