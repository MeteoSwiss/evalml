import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).parents[2] / "workflow" / "scripts"))
from verification_aggregation import aggregate_results

from verification import decode_metric, apply_lapse_rate_correction


@pytest.mark.parametrize(
    "label, expected",
    [
        # happy path: no extra 'p' outside decimal markers, no operator substring in words
        ("ETS_gt_10p5", "ETS > 10.5"),
        ("ETS_ge_10p5", "ETS >= 10.5"),
        ("FAR_lt_5p0", "FAR < 5.0"),
        ("ETS_le_10p5", "ETS <= 10.5"),
        ("POD_eq_0p0", "POD == 0.0"),
        ("FAR_ne_5p5", "FAR != 5.5"),
        # edge cases: 'p' in variable name
        ("precip_gt_1p0", "precip > 1.0"),
        ("temp_lt_0p0", "temp < 0.0"),
        # edge case: operator abbreviation inside a word
        ("simple_label", "simple_label"),
    ],
)
def test_decode_metric(label, expected):
    assert decode_metric(label) == expected


def _make_verif_dataset(forecast_reference_times):
    """Minimal verification dataset with a single continuous metric."""
    n = len(forecast_reference_times)
    return xr.Dataset(
        {
            "T_2M.BIAS": (
                ["forecast_reference_time", "region", "source"],
                np.ones((n, 1, 1)),
            )
        },
        coords={
            "forecast_reference_time": forecast_reference_times,
            "region": ["all"],
            "source": ["fcst"],
        },
    )


def test_aggregate_results_n_samples_global():
    forecast_reference_times = np.array(
        ["2024-01-01T00", "2024-04-01T00", "2024-07-01T00", "2024-10-01T00"],
        dtype="datetime64[ns]",
    )
    out = aggregate_results(_make_verif_dataset(forecast_reference_times))

    assert "n_samples" in out.data_vars
    assert int(out["n_samples"].sel(season="all", init_hour=-999)) == 4


def test_aggregate_results_n_samples_by_season():
    # 2 DJF (Jan), 1 MAM (Apr), 1 JJA (Jul)
    forecast_reference_times = np.array(
        ["2024-01-01T00", "2024-01-15T00", "2024-04-01T00", "2024-07-01T00"],
        dtype="datetime64[ns]",
    )
    out = aggregate_results(_make_verif_dataset(forecast_reference_times))

    assert int(out["n_samples"].sel(season="DJF", init_hour=-999)) == 2
    assert int(out["n_samples"].sel(season="MAM", init_hour=-999)) == 1
    assert int(out["n_samples"].sel(season="JJA", init_hour=-999)) == 1


def test_aggregate_results_n_samples_by_init_hour():
    # 3 times at 00Z, 1 time at 12Z
    forecast_reference_times = np.array(
        ["2024-01-01T00", "2024-04-01T00", "2024-07-01T00", "2024-10-01T12"],
        dtype="datetime64[ns]",
    )
    out = aggregate_results(_make_verif_dataset(forecast_reference_times))

    assert int(out["n_samples"].sel(season="all", init_hour=0)) == 3
    assert int(out["n_samples"].sel(season="all", init_hour=12)) == 1


def test_aggregate_results_n_samples_by_season_and_init_hour():
    # DJF/00Z: 2, DJF/12Z: 1, JJA/00Z: 1
    forecast_reference_times = np.array(
        ["2024-01-01T00", "2024-01-15T00", "2024-01-20T12", "2024-07-01T00"],
        dtype="datetime64[ns]",
    )
    out = aggregate_results(_make_verif_dataset(forecast_reference_times))

    assert int(out["n_samples"].sel(season="DJF", init_hour=0)) == 2
    assert int(out["n_samples"].sel(season="DJF", init_hour=12)) == 1
    assert int(out["n_samples"].sel(season="JJA", init_hour=0)) == 1


# ---------------------------------------------------------------------------
# apply_lapse_rate_correction
# ---------------------------------------------------------------------------


def _make_lapse_rate_datasets(fcst_elev, obs_elev, t2m=280.0, td2m=270.0):
    """Return (fcst, obs) pair with elevation coordinates and T/TD data vars."""
    n = len(fcst_elev)
    fcst = xr.Dataset(
        {
            "T_2M": (["step", "values"], np.full((3, n), t2m, dtype=np.float32)),
            "TD_2M": (["step", "values"], np.full((3, n), td2m, dtype=np.float32)),
        },
        coords={"elevation": ("values", np.array(fcst_elev, dtype=np.float32))},
    )
    obs = xr.Dataset(
        coords={"elevation": ("values", np.array(obs_elev, dtype=np.float32))}
    )
    return fcst, obs


def test_lapse_rate_correction_temperature():
    # Station 500 m above forecast grid cell → T should decrease by 0.0065 * 500 = 3.25 K
    fcst, obs = _make_lapse_rate_datasets(fcst_elev=[500.0], obs_elev=[1000.0])
    result = apply_lapse_rate_correction(fcst, obs, ["T_2M", "TD_2M"])
    np.testing.assert_allclose(result["T_2M"].values, 280.0 - 0.0065 * 500.0, atol=1e-4)


def test_lapse_rate_correction_dewpoint_unchanged():
    # TD_2M is not corrected — only T_2M gets the lapse-rate adjustment
    fcst, obs = _make_lapse_rate_datasets(fcst_elev=[500.0], obs_elev=[1000.0])
    result = apply_lapse_rate_correction(fcst, obs, ["T_2M", "TD_2M"])
    np.testing.assert_array_equal(result["TD_2M"].values, fcst["TD_2M"].values)


def test_lapse_rate_correction_station_below_grid():
    # Station 300 m below forecast grid → T should increase by 0.0065 * 300 = 1.95 K
    fcst, obs = _make_lapse_rate_datasets(fcst_elev=[800.0], obs_elev=[500.0])
    result = apply_lapse_rate_correction(fcst, obs, ["T_2M"])
    np.testing.assert_allclose(result["T_2M"].values, 280.0 + 0.0065 * 300.0, atol=1e-4)


def test_lapse_rate_correction_raises_without_forecast_elevation():
    fcst, obs = _make_lapse_rate_datasets(fcst_elev=[500.0], obs_elev=[1000.0])
    fcst_no_elev = fcst.drop_vars("elevation")
    with pytest.raises(ValueError, match="forecast"):
        apply_lapse_rate_correction(fcst_no_elev, obs, ["T_2M", "TD_2M"])


def test_lapse_rate_correction_raises_without_obs_elevation():
    fcst, obs = _make_lapse_rate_datasets(fcst_elev=[500.0], obs_elev=[1000.0])
    obs_no_elev = obs.drop_vars("elevation")
    with pytest.raises(ValueError, match="observations"):
        apply_lapse_rate_correction(fcst, obs_no_elev, ["T_2M", "TD_2M"])


def test_lapse_rate_correction_only_requested_params():
    # Pass only T_2M in params — TD_2M should not be corrected
    fcst, obs = _make_lapse_rate_datasets(fcst_elev=[500.0], obs_elev=[1000.0])
    result = apply_lapse_rate_correction(fcst, obs, ["T_2M"])
    np.testing.assert_allclose(result["T_2M"].values, 280.0 - 0.0065 * 500.0, atol=1e-4)
    np.testing.assert_array_equal(result["TD_2M"].values, fcst["TD_2M"].values)
