import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).parents[2] / "workflow" / "scripts"))
from verification_aggregation import aggregate_results

from verification import decode_metric, apply_lapse_rate_correction_inplace, verify


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


@pytest.fixture
def make_lapse_rate_datasets():
    def _make(fcst_elev, obs_elev, t2m=280.0, td2m=270.0):
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

    return _make


def test_lapse_rate_correction_temperature(make_lapse_rate_datasets):
    # Station 500 m above forecast grid cell → T should decrease by 0.0065 * 500 = 3.25 K
    fcst, obs = make_lapse_rate_datasets(fcst_elev=[500.0], obs_elev=[1000.0])
    apply_lapse_rate_correction_inplace(fcst, obs, ["T_2M", "TD_2M"])
    np.testing.assert_allclose(fcst["T_2M"].values, 280.0 - 0.0065 * 500.0, atol=1e-4)


def test_lapse_rate_correction_dewpoint_unchanged(make_lapse_rate_datasets):
    # TD_2M is not corrected — only T_2M gets the lapse-rate adjustment
    fcst, obs = make_lapse_rate_datasets(fcst_elev=[500.0], obs_elev=[1000.0])
    apply_lapse_rate_correction_inplace(fcst, obs, ["T_2M", "TD_2M"])
    np.testing.assert_array_equal(fcst["TD_2M"].values, 270.0)


def test_lapse_rate_correction_station_below_grid(make_lapse_rate_datasets):
    # Station 300 m below forecast grid → T should increase by 0.0065 * 300 = 1.95 K
    fcst, obs = make_lapse_rate_datasets(fcst_elev=[800.0], obs_elev=[500.0])
    apply_lapse_rate_correction_inplace(fcst, obs, ["T_2M"])
    np.testing.assert_allclose(fcst["T_2M"].values, 280.0 + 0.0065 * 300.0, atol=1e-4)


def test_lapse_rate_correction_raises_without_forecast_elevation(
    make_lapse_rate_datasets,
):
    fcst, obs = make_lapse_rate_datasets(fcst_elev=[500.0], obs_elev=[1000.0])
    fcst_no_elev = fcst.drop_vars("elevation")
    with pytest.raises(ValueError, match="forecast"):
        apply_lapse_rate_correction_inplace(fcst_no_elev, obs, ["T_2M", "TD_2M"])


def test_lapse_rate_correction_raises_without_obs_elevation(make_lapse_rate_datasets):
    fcst, obs = make_lapse_rate_datasets(fcst_elev=[500.0], obs_elev=[1000.0])
    obs_no_elev = obs.drop_vars("elevation")
    with pytest.raises(ValueError, match="observations"):
        apply_lapse_rate_correction_inplace(fcst, obs_no_elev, ["T_2M", "TD_2M"])


def test_lapse_rate_correction_only_requested_params(make_lapse_rate_datasets):
    # Pass only T_2M in params — TD_2M should not be corrected
    fcst, obs = make_lapse_rate_datasets(fcst_elev=[500.0], obs_elev=[1000.0])
    apply_lapse_rate_correction_inplace(fcst, obs, ["T_2M"])
    np.testing.assert_allclose(fcst["T_2M"].values, 280.0 - 0.0065 * 500.0, atol=1e-4)
    np.testing.assert_array_equal(fcst["TD_2M"].values, 270.0)


# ---------------------------------------------------------------------------
# verify — missing-fraction masking
# ---------------------------------------------------------------------------


_FRT = np.datetime64("2024-01-01T00", "ns")


def _station_coords(n):
    """Coordinates for n stations inside the 'all' mask region (lon 1.5–16, lat 43–49.5)."""
    return {
        "longitude": ("values", np.linspace(5.0, 10.0, n)),
        "latitude": ("values", np.linspace(46.0, 47.0, n)),
        "forecast_reference_time": _FRT,
    }


def test_verify_missing_fraction_varies_by_parameter():
    """Parameters with fewer valid obs stations must still yield non-NaN metrics.

    T_2M has obs at only 5 of 10 stations; TOT_PREC has obs at all 10.
    The forecast is complete for both parameters.  Because missing fraction is
    normalised by the number of obs-valid points (not total stations), the
    fraction of missing forecast values is 0 for both parameters and metrics
    must not be masked.
    """
    n = 10
    coords = _station_coords(n)
    fcst_vals = np.ones(n, dtype=np.float32)

    obs_t2m = np.ones(n, dtype=np.float32)
    obs_t2m[:5] = np.nan  # only half of T_2M stations are reporting

    fcst = xr.Dataset(
        {"T_2M": ("values", fcst_vals), "TOT_PREC": ("values", fcst_vals)},
        coords=coords,
    )
    obs = xr.Dataset(
        {"T_2M": ("values", obs_t2m), "TOT_PREC": ("values", fcst_vals)},
        coords=coords,
    )

    result = verify(fcst, obs, "fcst", "obs", num_workers=1)

    t2m_bias = result["T_2M.BIAS"].sel(region="all", source="fcst").values.item()
    prec_bias = result["TOT_PREC.BIAS"].sel(region="all", source="fcst").values.item()
    assert not np.isnan(t2m_bias), "T_2M BIAS should not be NaN"
    assert not np.isnan(prec_bias), "TOT_PREC BIAS should not be NaN"


def test_verify_missing_fraction_varies_by_lead_time():
    """Steps with fewer valid obs stations must still yield non-NaN metrics.

    At step 0 all 10 stations have obs; at step 1 only 5 do.  The forecast is
    complete at every step.  Missing fraction normalised by obs-valid count is
    0 at both steps, so metrics must not be masked at either lead time.
    """
    n = 10
    coords = _station_coords(n)
    steps = np.array([0, 1])

    fcst_vals = np.ones((2, n), dtype=np.float32)

    obs_vals = np.ones((2, n), dtype=np.float32)
    obs_vals[1, :5] = np.nan  # step 1: only half the stations are reporting

    fcst = xr.Dataset(
        {"T_2M": (["step", "values"], fcst_vals)},
        coords={"step": steps, **coords},
    )
    obs = xr.Dataset(
        {"T_2M": (["step", "values"], obs_vals)},
        coords={"step": steps, **coords},
    )

    result = verify(fcst, obs, "fcst", "obs", num_workers=1)

    bias = result["T_2M.BIAS"].sel(region="all", source="fcst")
    assert not np.any(np.isnan(bias.values)), (
        f"T_2M BIAS should be non-NaN at every step, got {bias.values}"
    )


def test_verify_obs_stats_not_masked_by_forecast_gaps():
    """Obs statistics must remain non-NaN in regions where the forecast has gaps.

    Half of the forecast values are NaN (simulating extrapolation masking), so
    the missing fraction exceeds the default threshold and scores are masked.
    Obs statistics should still be valid because they do not depend on forecast
    coverage.
    """
    n = 10
    coords = _station_coords(n)

    fcst_vals = np.ones(n, dtype=np.float32)
    fcst_vals[:5] = (
        np.nan
    )  # forecast missing at half the obs-valid stations → masked region

    fcst = xr.Dataset({"T_2M": ("values", fcst_vals)}, coords=coords)
    obs = xr.Dataset({"T_2M": ("values", np.ones(n, dtype=np.float32))}, coords=coords)

    result = verify(fcst, obs, "fcst", "obs", num_workers=1)

    # Score should be NaN (too many missing forecasts)
    bias = result["T_2M.BIAS"].sel(region="all", source="fcst").values.item()
    assert np.isnan(bias), "BIAS should be NaN when forecast has too many gaps"

    # Obs statistics must survive regardless
    obs_mean = result["T_2M.mean"].sel(region="all", source="obs").values.item()
    assert not np.isnan(obs_mean), (
        "Obs mean should not be NaN when obs data is complete"
    )
