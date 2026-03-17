import numpy as np
import xarray as xr

from verification import _compute_categorical_scores


def test_ets_perfect_forecast():
    # H=2, FA=0, M=0, CN=1, N=3 → H_r=4/3, denom=2/3, ETS=1.0
    fcst = xr.DataArray([2.0, 0.5, 2.0], dims=["values"])
    obs = xr.DataArray([2.0, 0.5, 2.0], dims=["values"])
    ds = _compute_categorical_scores(
        fcst, obs, threshold=1.0, dim=["values"], prefix="TOT_PREC."
    )
    assert float(ds["TOT_PREC.ETS_ge1"]) == 1.0


def test_ets_known_contingency_table():
    # H=2, FA=1, M=1, CN=0, N=4
    # H_r = 3*3/4 = 2.25, denom = 2+1+1-2.25 = 1.75
    # ETS = (2 - 2.25) / 1.75 = -1/7
    fcst = xr.DataArray([2.0, 2.0, 0.5, 2.0], dims=["values"])
    obs = xr.DataArray([2.0, 2.0, 2.0, 0.5], dims=["values"])
    ds = _compute_categorical_scores(
        fcst, obs, threshold=1.0, dim=["values"], prefix="TOT_PREC."
    )
    assert np.isclose(float(ds["TOT_PREC.ETS_ge1"]), -1.0 / 7.0)


def test_ets_all_miss():
    # H=0, FA=0, M=3, CN=0, N=3 → H_r=0, denom=3, ETS=0.0
    fcst = xr.DataArray([0.5, 0.5, 0.5], dims=["values"])
    obs = xr.DataArray([2.0, 2.0, 2.0], dims=["values"])
    ds = _compute_categorical_scores(
        fcst, obs, threshold=1.0, dim=["values"], prefix="TOT_PREC."
    )
    assert float(ds["TOT_PREC.ETS_ge1"]) == 0.0


def test_ets_all_false_alarms():
    # H=0, FA=3, M=0, CN=0, N=3 → H_r=0, denom=3, ETS=0/3=0.0
    fcst = xr.DataArray([2.0, 2.0, 2.0], dims=["values"])
    obs = xr.DataArray([0.5, 0.5, 0.5], dims=["values"])
    ds = _compute_categorical_scores(
        fcst, obs, threshold=1.0, dim=["values"], prefix="TOT_PREC."
    )
    assert float(ds["TOT_PREC.ETS_ge1"]) == 0.0


def test_ets_higher_threshold_produces_separate_variable():
    fcst = xr.DataArray([6.0, 0.5, 6.0], dims=["values"])
    obs = xr.DataArray([6.0, 0.5, 6.0], dims=["values"])
    ds = _compute_categorical_scores(
        fcst, obs, threshold=5.0, dim=["values"], prefix="TOT_PREC."
    )
    assert "TOT_PREC.ETS_ge5" in ds.data_vars
    assert float(ds["TOT_PREC.ETS_ge5"]) == 1.0
