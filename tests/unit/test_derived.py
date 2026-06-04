import numpy as np
import pytest
import xarray as xr

from data.derived import accumulate, deaccumulate, interval, uv_components, wind_speed


def test_wind_speed_numpy():
    np.testing.assert_allclose(wind_speed(np.array([3.0]), np.array([4.0])), [5.0])


def test_wind_speed_xarray_preserves_coords():
    u = xr.DataArray([3.0], dims="values", coords={"values": [0]}, name="U_10M")
    v = xr.DataArray([4.0], dims="values", coords={"values": [0]}, name="V_10M")
    out = wind_speed(u, v)
    assert isinstance(out, xr.DataArray)
    assert list(out.coords) == ["values"]
    np.testing.assert_allclose(out.values, [5.0])


def test_uv_components_known_values():
    # wind FROM north (dd=0) blows toward south: u≈0, v=-speed
    u, v = uv_components(10.0, 0.0)
    assert np.isclose(u, 0.0, atol=1e-9) and np.isclose(v, -10.0)
    # wind FROM east (dd=90) blows toward west: u=-speed, v≈0
    u, v = uv_components(10.0, 90.0)
    assert np.isclose(u, -10.0) and np.isclose(v, 0.0, atol=1e-9)


def test_uv_components_xarray_preserves_type():
    ff = xr.DataArray([10.0], dims="values", name="FF_10M")
    dd = xr.DataArray([0.0], dims="values", name="DD_10M")
    u, v = uv_components(ff, dd)
    assert isinstance(u, xr.DataArray) and isinstance(v, xr.DataArray)
    np.testing.assert_allclose(v.values, [-10.0])


def _cumulative_da(values):
    return xr.DataArray(
        np.array(values, dtype=float),
        dims="step",
        coords={"step": np.arange(len(values))},
        name="TOT_PREC",
    )


def test_deaccumulate_cumulative_to_per_step():
    out = deaccumulate(_cumulative_da([0.0, 1.0, 3.0, 6.0]))
    assert out.sizes["step"] == 4
    np.testing.assert_allclose(out.values[1:], [1.0, 2.0, 3.0])
    assert np.isnan(out.values[0])


def test_deaccumulate_fills_missing_step0():
    out = deaccumulate(_cumulative_da([np.nan, 1.0, 3.0]))
    np.testing.assert_allclose(out.values[1:], [1.0, 2.0])


def test_deaccumulate_raises_on_period_accumulated():
    with pytest.raises(ValueError, match="period-accumulated"):
        deaccumulate(_cumulative_da([0.0, 5.0, 1.0, 4.0]))


def _series(values):
    return xr.DataArray(
        np.array(values, dtype=float),
        dims="step",
        coords={"step": np.arange(len(values))},
        name="TOT_PREC",
    )


def test_accumulate_is_cumsum_over_step():
    out = accumulate(_series([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(out.values, [1.0, 3.0, 6.0])
    assert out.name == "TOT_PREC" and list(out.coords) == ["step"]


def test_interval_window_is_difference():
    # cumulative series; window of 2 steps -> cumul - cumul.shift(2)
    out = interval(_series([0.0, 1.0, 3.0, 6.0, 10.0]), 2)
    np.testing.assert_allclose(out.values[2:], [3.0, 5.0, 7.0])
    assert np.isnan(out.values[0]) and np.isnan(out.values[1])


def test_interval_window_one_recovers_per_step():
    out = interval(_series([0.0, 1.0, 3.0, 6.0]), 1)
    np.testing.assert_allclose(out.values[1:], [1.0, 2.0, 3.0])
    assert np.isnan(out.values[0])


def test_accumulate_interval_round_trip():
    per_interval = _series([2.0, 1.0, 4.0])
    cum = accumulate(per_interval)
    np.testing.assert_allclose(interval(cum, 1).values[1:], per_interval.values[1:])
