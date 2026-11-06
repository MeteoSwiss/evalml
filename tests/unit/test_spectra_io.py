import numpy as np
import pandas as pd
import pytest
import xarray as xr

from spectra import io


def _toy_dataset():
    # native grid of 6 cells, two steps
    return xr.Dataset(
        {
            "T_2M": (("step", "values"), np.arange(12.0).reshape(2, 6)),
            "U_10M": (("step", "values"), np.ones((2, 6))),
            "V_10M": (("step", "values"), 2 * np.ones((2, 6))),
        },
        coords={"step": [0, 6], "values": np.arange(6)},
    )


def test_variable_components():
    assert io.variable_components("WIND_KE") == (["U_10M", "V_10M"], 0.5)
    assert io.variable_components("T_2M") == (["T_2M"], 1.0)
    assert io.variable_components("TOT_PREC1") == (["TOT_PREC1"], 1.0)


def test_native_field_squeezes_singleton_reftime_dim():
    # Real GRIB loads carry a size-1 forecast_reference_time dim; native_field
    # must squeeze it so the field reduces to the 1-D native grid axis.
    ds = xr.Dataset(
        {
            "T_2M": (
                ("forecast_reference_time", "step", "values"),
                np.arange(12.0).reshape(1, 2, 6),
            )
        },
        coords={"step": [0, 6], "values": np.arange(6)},
    )
    arr = io.native_field(ds, "T_2M", step=6)
    assert arr.shape == (6,)
    np.testing.assert_array_equal(arr, np.arange(6, 12))


def test_native_field_extracts_1d_for_step():
    ds = _toy_dataset()
    arr = io.native_field(ds, "T_2M", step=6)
    assert arr.shape == (6,)
    np.testing.assert_array_equal(arr, np.arange(6, 12))


def test_native_components_returns_list_and_factor():
    ds = _toy_dataset()
    comps, factor = io.native_components(ds, "WIND_KE", step=0)
    assert factor == 0.5
    assert len(comps) == 2
    assert comps[0].shape == (6,)


def test_native_components_single_param():
    comps, factor = io.native_components(_toy_dataset(), "T_2M", step=6)
    assert factor == 1.0
    assert len(comps) == 1
    np.testing.assert_array_equal(comps[0], np.arange(6, 12))


def test_missing_variable_raises():
    with pytest.raises(KeyError):
        io.native_components(_toy_dataset(), "NOPE", step=0)


def test_native_field_handles_timedelta_step():
    ds = xr.Dataset(
        {"T_2M": (("step", "values"), np.arange(12.0).reshape(2, 6))},
        coords={"step": pd.to_timedelta([0, 6], unit="h"), "values": np.arange(6)},
    )
    arr = io.native_field(ds, "T_2M", step=6)
    np.testing.assert_array_equal(arr, np.arange(6, 12))


def test_native_field_rejects_non_1d():
    ds = xr.Dataset(
        {"T_2M": (("step", "y", "x"), np.zeros((1, 3, 4)))},
        coords={"step": [0], "y": np.arange(3), "x": np.arange(4)},
    )
    with pytest.raises(ValueError, match="1-D native field"):
        io.native_field(ds, "T_2M", step=0)


def test_required_params_deduplicates():
    assert io.required_params(["WIND_KE", "T_2M", "WIND_KE", "TOT_PREC1"]) == [
        "U_10M",
        "V_10M",
        "T_2M",
        "TOT_PREC1",
    ]


def test_period_param_from_loader_disaggregation():
    # TOT_PREC1 goes through the same disaggregation as in verification.
    from data_input import _disaggregated_and_derived_params

    cumulative = np.repeat(np.arange(4.0)[:, None], 5, axis=1)
    ds = xr.Dataset(
        {"TOT_PREC": (("step", "values"), cumulative)},
        coords={"step": pd.to_timedelta(np.arange(4), unit="h")},
    )
    ds = _disaggregated_and_derived_params(ds, [1, 3], ["TOT_PREC1"])
    comps, factor = io.native_components(ds, "TOT_PREC1", step=3)
    assert factor == 1.0
    np.testing.assert_array_equal(comps[0], np.ones(5))
