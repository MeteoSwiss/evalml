import numpy as np
import pandas as pd
import pytest
import xarray as xr

from verification import verify
from verification.valid_mask import (
    apply_valid_mask,
    combine_valid_masks,
    forecast_valid_mask,
)

STEPS = pd.to_timedelta([0, 1], unit="h")


def _field(values, steps=STEPS):
    """(forecast_reference_time, step, y, x) field on a 1 x 3 grid."""
    data = np.asarray(values, dtype=float).reshape(1, len(steps), 1, 3)
    return xr.DataArray(
        data,
        dims=["forecast_reference_time", "step", "y", "x"],
        coords={
            "forecast_reference_time": [np.datetime64("2025-05-01T06:00")],
            "step": steps,
            "y": [0.0],
            "x": [0.0, 1.0, 2.0],
        },
    )


def test_forecast_valid_mask_marks_points_with_a_value():
    fcst = xr.Dataset({"HZEROCL": _field([[1, np.nan, 3], [1, 2, 3]])})
    mask = forecast_valid_mask(fcst, ["HZEROCL"])
    assert mask["HZEROCL"].dims == ("step", "y", "x")
    np.testing.assert_array_equal(
        mask["HZEROCL"].values[:, 0], [[True, False, True], [True, True, True]]
    )


def test_forecast_valid_mask_ignores_missing_params_and_empty_lead_times():
    """A participant without data must not remove points for the others."""
    fcst = xr.Dataset({"HZEROCL": _field([[np.nan] * 3, [1, np.nan, 3]])})
    mask = forecast_valid_mask(fcst, ["HZEROCL", "SNOWLMT"])
    assert list(mask.data_vars) == ["HZEROCL"]
    np.testing.assert_array_equal(
        mask["HZEROCL"].values[:, 0], [[True, True, True], [True, False, True]]
    )


def test_combine_valid_masks_intersects_participants():
    a = forecast_valid_mask(
        xr.Dataset({"CIN_MU": _field([[np.nan, 2, 3], [1, 2, 3]])}), ["CIN_MU"]
    )
    b = forecast_valid_mask(
        xr.Dataset({"CIN_MU": _field([[1, 2, np.nan], [1, 2, 3]])}), ["CIN_MU"]
    )
    combined = combine_valid_masks([a, b])
    np.testing.assert_array_equal(
        combined["CIN_MU"].values[:, 0], [[False, True, False], [True, True, True]]
    )


def test_combine_valid_masks_lead_time_of_one_participant_only():
    """A lead time is restricted only by the participants that have it."""
    a = forecast_valid_mask(
        xr.Dataset({"CIN_MU": _field([[np.nan, 2, 3], [1, 2, 3]])}), ["CIN_MU"]
    )
    b = forecast_valid_mask(
        xr.Dataset({"CIN_MU": _field([[1, 2, np.nan]], steps=STEPS[:1])}), ["CIN_MU"]
    )
    combined = combine_valid_masks([a, b])
    np.testing.assert_array_equal(
        combined["CIN_MU"].values[:, 0], [[False, True, False], [True, True, True]]
    )


def test_combine_valid_masks_rejects_different_points():
    a = forecast_valid_mask(xr.Dataset({"CIN_MU": _field([[1, 2, 3]] * 2)}), ["CIN_MU"])
    b = a.assign_coords(x=[0.0, 1.0, 5.0])
    with pytest.raises(ValueError):
        combine_valid_masks([a, b])


def test_apply_valid_mask_masks_forecast_and_truth_of_listed_params_only():
    fcst = xr.Dataset(
        {"HZEROCL": _field([[1, 2, 3], [1, 2, 3]]), "T_2M": _field([[1, 2, 3]] * 2)}
    )
    truth = xr.Dataset(
        {"HZEROCL": _field([[4, 5, 6], [4, 5, 6]]), "T_2M": _field([[4, 5, 6]] * 2)}
    )
    other = forecast_valid_mask(
        xr.Dataset({"HZEROCL": _field([[1, np.nan, 3], [1, 2, 3]])}), ["HZEROCL"]
    )
    mask = combine_valid_masks([forecast_valid_mask(fcst, ["HZEROCL"]), other])

    fcst_m, truth_m = apply_valid_mask(fcst, truth, mask)

    expected = [[1.0, np.nan, 3.0], [1.0, 2.0, 3.0]]
    np.testing.assert_array_equal(fcst_m["HZEROCL"].values[0, :, 0], expected)
    assert np.isnan(truth_m["HZEROCL"].values[0, 0, 0, 1])
    xr.testing.assert_identical(fcst_m["T_2M"], fcst["T_2M"])
    xr.testing.assert_identical(truth_m["T_2M"], truth["T_2M"])


def test_apply_valid_mask_masks_truth_without_the_forecast_param():
    """The truth statistics must be the same in every participant's file."""
    fcst = xr.Dataset({"T_2M": _field([[1, 2, 3]] * 2)})
    truth = xr.Dataset({"HZEROCL": _field([[4, 5, 6]] * 2)})
    mask = forecast_valid_mask(
        xr.Dataset({"HZEROCL": _field([[1, np.nan, 3], [1, 2, 3]])}), ["HZEROCL"]
    )
    _, truth_m = apply_valid_mask(fcst, truth, mask)
    assert np.isnan(truth_m["HZEROCL"].values[0, 0, 0, 1])
    assert np.isfinite(truth_m["HZEROCL"].values[0, 1]).all()


_REGION = [{"type": "bbox", "name": "all", "bbox": [1.5, 16, 43, 49.5]}]


def _station_dataset(name, values):
    """One lead time at stations inside _REGION."""
    values = np.asarray(values, dtype=float)[None, :]
    n = values.shape[1]
    return xr.Dataset(
        {name: (("step", "values"), values)},
        coords={
            "step": STEPS[:1],
            "values": np.arange(n),
            "longitude": ("values", np.linspace(5.0, 10.0, n)),
            "latitude": ("values", np.linspace(46.0, 47.0, n)),
            "forecast_reference_time": np.datetime64("2025-05-01T06:00", "ns"),
        },
    )


def test_common_points_lets_a_forecast_with_gaps_be_scored():
    """Without the mask, one undefined point makes all scores of the forecast NaN.
    With it, both forecasts are scored on the points where both have a value, and
    the gap-free forecast keeps its scores when nobody has gaps."""
    truth = _station_dataset("HZEROCL", [1, 2, 3, 4])
    icon = _station_dataset("HZEROCL", [2, np.nan, 4, 5])  # bias +1, one gap
    ml = _station_dataset("HZEROCL", [1, 2, 3, 10])  # exact except at the last point

    def bias(fcst, obs):
        result = verify(fcst, obs, "fcst", "obs", regions=_REGION, num_workers=1)
        return result["HZEROCL.BIAS"].sel(region="all", source="fcst").item()

    assert np.isnan(bias(icon, truth))

    mask = combine_valid_masks(
        [forecast_valid_mask(f, ["HZEROCL"]) for f in (icon, ml)]
    )
    assert bias(*apply_valid_mask(icon, truth, mask)) == pytest.approx(1.0)
    # the ML forecast is scored on the same three points: (0 + 0 + 6) / 3
    assert bias(*apply_valid_mask(ml, truth, mask)) == pytest.approx(2.0)

    no_gaps = combine_valid_masks([forecast_valid_mask(ml, ["HZEROCL"])])
    assert bias(*apply_valid_mask(ml, truth, no_gaps)) == bias(ml, truth)
