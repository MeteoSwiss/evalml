"""Unit tests for data_input derivation primitives and TOT_PREC de-accumulation."""

from datetime import datetime

import earthkit.meteo.thermo as ekmt
import numpy as np
import pytest
import xarray as xr

from data_input import (
    accumulate_from_hourly,
    _disaggregate_accum,
    _disaggregated_and_derived_params,
    _ensure_accum_ic,
    _mask_icon_missing_values,
    compute_derived,
    get_base_params,
    get_grib_load_params,
    get_steps,
    load_truth_data,
    parse_aggregated_param,
)


def _make_cumul(steps_h, values):
    """Build a cumulative DataArray over `steps_h` (hours)."""
    step = np.array([np.timedelta64(h, "h") for h in steps_h]).astype("timedelta64[ns]")
    return xr.DataArray(
        np.array(values, dtype=np.float64), dims=("step",), coords={"step": step}
    )


def _make_cumul_with_time(steps_h, values, reftime="2025-03-01"):
    """Build a cumulative DataArray that also carries a datetime64 'time' aux coord.

    Mirrors the real loader output where time = reftime + step.
    """
    step = np.array([np.timedelta64(h, "h") for h in steps_h]).astype("timedelta64[ns]")
    ref = np.datetime64(reftime, "ns")
    time = ref + step
    return xr.DataArray(
        np.array(values, dtype=np.float64),
        dims=("step",),
        coords={"step": step, "time": ("step", time)},
    )


# ---------------------------------------------------------------------------
# parse_aggregated_param
# ---------------------------------------------------------------------------


def test_parse_aggregated_known():
    assert parse_aggregated_param("TOT_PREC6") == ("TOT_PREC", 6)
    assert parse_aggregated_param("TOT_PREC1") == ("TOT_PREC", 1)
    assert parse_aggregated_param("TOT_PREC24") == ("TOT_PREC", 24)


def test_parse_aggregated_non_accum_returns_none_hours():
    assert parse_aggregated_param("T_2M") == ("T_2M", None)
    assert parse_aggregated_param("SP_10M") == ("SP_10M", None)
    assert parse_aggregated_param("FOO6") == (
        "FOO6",
        None,
    )  # FOO not in _ACCUMULATABLE_PARAMS


# ---------------------------------------------------------------------------
# get_base_params
# ---------------------------------------------------------------------------


def test_expand_aggregated():
    assert set(get_base_params(["TOT_PREC6"])) == {"TOT_PREC"}


def test_expand_derived():
    assert set(get_base_params(["SP_10M"])) == {"U_10M", "V_10M"}


def test_expand_mixed():
    assert set(get_base_params(["T_2M", "SP_10M", "TOT_PREC6"])) == {
        "T_2M",
        "U_10M",
        "V_10M",
        "TOT_PREC",
    }


def test_expand_no_duplicates():
    assert set(get_base_params(["U_10M", "V_10M", "SP_10M"])) == {"U_10M", "V_10M"}


def test_expand_native_derived_params_not_expanded():
    params = ["SP_10M", "RELHUM_2M", "TOT_SNOW6"]
    assert set(get_base_params(params, native={"SP_10M", "TOT_SNOW"})) == {
        "SP_10M",
        "T_2M",
        "TD_2M",
        "TOT_SNOW",
    }


def test_grib_load_params_include_native_and_components():
    """GRIB loading asks for a derived param both as such and as its components."""
    params = ["TW_2M", "DD_10M", "TOT_SNOW6", "T_2M"]
    assert set(get_grib_load_params(params)) == {
        "TW_2M",
        "T_2M",
        "TD_2M",
        "PS",
        "DD_10M",
        "U_10M",
        "V_10M",
        "TOT_SNOW",
        "SNOW_GSP",
    }


# ---------------------------------------------------------------------------
# get_steps
# ---------------------------------------------------------------------------


def test_get_steps_aggregated():
    assert get_steps([6, 12, 18], ["TOT_PREC6"]) == [0, 6, 12, 18]


def test_get_steps_short_step_not_added():
    # step 3 < n=6 → 3-6=-3 not added; step 6 → adds 0
    assert get_steps([3, 6, 12], ["TOT_PREC6"]) == [0, 3, 6, 12]


def test_get_steps_non_aggregated():
    assert get_steps([6, 12], ["T_2M"]) == [6, 12]


def test_get_steps_multi_agg():
    # TOT_PREC1 adds 5,11; TOT_PREC6 adds 0,6
    result = get_steps([6, 12], ["TOT_PREC1", "TOT_PREC6"])
    assert set(result) == {0, 5, 6, 11, 12}


# ---------------------------------------------------------------------------
# compute_derived
# ---------------------------------------------------------------------------


def test_compute_derived_sp10m():
    ds = xr.Dataset({"U_10M": xr.DataArray([3.0]), "V_10M": xr.DataArray([4.0])})
    np.testing.assert_allclose(compute_derived(ds, "SP_10M").values, [5.0])


def test_compute_derived_unknown_raises():
    with pytest.raises(ValueError, match="No recipe"):
        compute_derived(xr.Dataset(), "FOO")


def test_compute_derived_dd10m_meteorological_convention():
    """Direction the wind blows from, clockwise from North."""
    # from N (blowing south), from E (blowing west), from S, from W
    ds = xr.Dataset(
        {
            "U_10M": xr.DataArray([0.0, -5.0, 0.0, 5.0]),
            "V_10M": xr.DataArray([-5.0, 0.0, 5.0, 0.0]),
        }
    )
    np.testing.assert_allclose(
        compute_derived(ds, "DD_10M").values, [0.0, 90.0, 180.0, 270.0], atol=1e-9
    )


def test_compute_derived_relhum_and_wet_bulb_match_earthkit():
    """Same formulas as the derive-diagnostic post-processor (earthkit-meteo)."""
    t, td, ps = np.array([290.0, 275.0]), np.array([285.0, 270.0]), np.array([1e5, 9e4])
    # 2-D (step, values) and dask-backed, like the loaded data
    ds = xr.Dataset(
        {
            name: (("step", "values"), np.stack([a, a]))
            for name, a in (("T_2M", t), ("TD_2M", td), ("PS", ps))
        }
    ).chunk({"step": 1})
    rh = compute_derived(ds, "RELHUM_2M").compute()
    tw = compute_derived(ds, "TW_2M").compute()
    assert rh.dims == tw.dims == ("step", "values")
    np.testing.assert_allclose(
        rh.values[1], ekmt.relative_humidity_from_dewpoint(t, td)
    )
    np.testing.assert_allclose(
        tw.values[1], ekmt.wet_bulb_temperature_from_dewpoint(t, td, ps)
    )
    assert np.all((rh.values > 0) & (rh.values < 100))
    assert np.all((tw.values > td) & (tw.values < t))
    assert rh.attrs["parameter"]["units"] == "%"


def test_compute_derived_tot_snow_from_grid_scale_snow():
    ds = xr.Dataset({"SNOW_GSP": xr.DataArray([1.0, 2.0], attrs={"units": "kg m-2"})})

    tot_snow = compute_derived(ds, "TOT_SNOW")

    np.testing.assert_allclose(tot_snow.values, [1.0, 2.0])
    assert "parameter" not in ds["SNOW_GSP"].attrs  # input left untouched


# ---------------------------------------------------------------------------
# _disaggregate_accum
# ---------------------------------------------------------------------------


def test_disaggregate_returns_nan_for_short_steps():
    # step=3 < n=6 → NaN; step=6 → valid (cumul[6] - cumul[0] = 10)
    cumul = _make_cumul([0, 3, 6], [0.0, 3.0, 10.0])
    result = _disaggregate_accum(cumul, steps=[3, 6], n=6)
    assert np.isnan(result.sel(step=np.timedelta64(3, "h")).item())
    assert result.sel(step=np.timedelta64(6, "h")).item() == pytest.approx(10.0)


def test_disaggregate_accum_coarser_source_returns_all_nan():
    """When the preceding boundary step (s-n) is absent from cumul (e.g. 6-hourly
    source, n=1), the result for every affected step must be NaN, not a KeyError."""
    cumul = _make_cumul([0, 6, 12], [0.0, 6.0, 12.0])
    result = _disaggregate_accum(cumul, steps=[6, 12], n=1)
    assert np.isnan(result.sel(step=np.timedelta64(6, "h")).item())
    assert np.isnan(result.sel(step=np.timedelta64(12, "h")).item())


def test_disaggregate_accum_with_time_coord_does_not_raise():
    """Regression: when cumul carries a datetime64 'time' aux coordinate, reindexing
    to add missing steps must not raise DTypePromotionError (float NaN vs datetime64)."""
    cumul = _make_cumul_with_time([0, 6, 12], [0.0, 6.0, 12.0])
    # n=1 forces reindex to add steps 5h and 11h which are absent → NaN via NaT fill
    result = _disaggregate_accum(cumul, steps=[6, 12], n=1)
    assert np.isnan(result.sel(step=np.timedelta64(6, "h")).item())
    assert np.isnan(result.sel(step=np.timedelta64(12, "h")).item())


# ---------------------------------------------------------------------------
# _ensure_accum_ic
# ---------------------------------------------------------------------------


def test_ensure_accum_ic_prepends_zero_when_step0_missing():
    cumul = _make_cumul([6], [3.0])
    out = _ensure_accum_ic(cumul, steps=[0, 6])
    assert np.timedelta64(0, "h") in out["step"].values
    np.testing.assert_allclose(out.sel(step=np.timedelta64(0, "h")).item(), 0.0)


def test_ensure_accum_ic_fills_nan_step0_with_zero():
    cumul = _make_cumul([0, 6], [np.nan, 3.0])
    out = _ensure_accum_ic(cumul, steps=[0, 6])
    np.testing.assert_allclose(out.sel(step=np.timedelta64(0, "h")).item(), 0.0)


def test_ensure_accum_ic_noop_when_step0_not_requested():
    cumul = _make_cumul([18, 24], [5.0, 7.0])
    out = _ensure_accum_ic(cumul, steps=[18, 24])
    assert list(out["step"].values) == list(cumul["step"].values)


# ---------------------------------------------------------------------------
# Round-trip: accumulate_from_hourly ↔ _disaggregate_accum
# ---------------------------------------------------------------------------


def _make_hourly_da(values: list[float]) -> xr.DataArray:
    """1H period-accumulated DataArray with step coords 1h, 2h, …, len(values)h."""
    steps = np.array(
        [np.timedelta64(h, "h") for h in range(1, len(values) + 1)],
        dtype="timedelta64[ns]",
    )
    return xr.DataArray(
        np.array(values, dtype=np.float64), dims=("step",), coords={"step": steps}
    )


def test_roundtrip_n1_recovers_hourly_values():
    """accumulate hourly then disaggregate with n=1 must recover the original values."""
    hourly_vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    da_1h = _make_hourly_da(hourly_vals)
    # load_steps includes 0 so disaggregation has an IC at step 0
    load_steps = list(range(0, len(hourly_vals) + 1))
    steps = list(range(1, len(hourly_vals) + 1))

    cumul = accumulate_from_hourly(da_1h, load_steps)
    result = _disaggregate_accum(cumul, steps=steps, n=1)

    for h, expected in zip(steps, hourly_vals):
        np.testing.assert_allclose(
            result.sel(step=np.timedelta64(h, "h")).item(),
            expected,
            err_msg=f"step={h}h",
        )


def test_roundtrip_n6_recovers_block_sums():
    """accumulate hourly then disaggregate with n=6 must give the 6h block sums."""
    block1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # sum = 21
    block2 = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]  # sum = 210
    da_1h = _make_hourly_da(block1 + block2)

    cumul = accumulate_from_hourly(da_1h, [0, 6, 12])
    result = _disaggregate_accum(cumul, steps=[6, 12], n=6)

    np.testing.assert_allclose(
        result.sel(step=np.timedelta64(6, "h")).item(), sum(block1)
    )
    np.testing.assert_allclose(
        result.sel(step=np.timedelta64(12, "h")).item(), sum(block2)
    )


# ---------------------------------------------------------------------------
# _disaggregated_and_derived_params – TOT_PREC6 from cumulative dataset
# ---------------------------------------------------------------------------


def _make_cumul_ds(
    steps_h: list[int], values: list[float], var: str = "TOT_PREC"
) -> xr.Dataset:
    """Dataset with cumulative-from-start values at the given step hours."""
    return xr.Dataset({var: _make_cumul(steps_h, values)})


def test_disaggregated_and_derived_params_tot_prec6_from_cumulative():
    """Given cumulative TOT_PREC, _disaggregated_and_derived_params must return
    correct 6h period sums as TOT_PREC6 and drop the base variable."""
    # IC=0, then 21 mm by hour 6, 231 mm by hour 12
    ds = _make_cumul_ds([0, 6, 12], [0.0, 21.0, 231.0])

    result = _disaggregated_and_derived_params(ds, steps=[6, 12], params=["TOT_PREC6"])

    assert "TOT_PREC6" in result.data_vars
    assert "TOT_PREC" not in result.data_vars
    np.testing.assert_allclose(
        result["TOT_PREC6"].sel(step=np.timedelta64(6, "h")).item(), 21.0
    )
    np.testing.assert_allclose(
        result["TOT_PREC6"].sel(step=np.timedelta64(12, "h")).item(), 210.0
    )


def test_disaggregated_and_derived_params_coarser_source_returns_nan():
    """TOT_PREC1 requested from a 6-hourly cumulative source must yield all-NaN
    with correct step coords — not a KeyError."""
    ds = _make_cumul_ds([0, 6, 12, 18, 24], [0.0, 6.0, 12.0, 18.0, 24.0])

    result = _disaggregated_and_derived_params(
        ds, steps=[6, 12, 18, 24], params=["TOT_PREC1"]
    )

    assert "TOT_PREC1" in result.data_vars
    assert "TOT_PREC" not in result.data_vars
    assert set(
        int(s.astype("timedelta64[h]").astype(int))
        for s in result["TOT_PREC1"]["step"].values
    ) == {6, 12, 18, 24}
    assert result["TOT_PREC1"].isnull().all()


def test_full_roundtrip_hourly_zarr_to_disaggregated():
    """End-to-end: hourly zarr values → accumulate_from_hourly →
    _disaggregated_and_derived_params → correct 6h period sums."""
    block1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # sum = 21
    block2 = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]  # sum = 210
    da_1h = _make_hourly_da(block1 + block2)

    # accumulate_from_hourly is called with load_steps (includes the preceding
    # boundary step 0 so disaggregation has an IC)
    cumul = accumulate_from_hourly(da_1h, [0, 6, 12])
    ds = xr.Dataset({"TOT_PREC": cumul})

    result = _disaggregated_and_derived_params(ds, steps=[6, 12], params=["TOT_PREC6"])

    np.testing.assert_allclose(
        result["TOT_PREC6"].sel(step=np.timedelta64(6, "h")).item(), sum(block1)
    )
    np.testing.assert_allclose(
        result["TOT_PREC6"].sel(step=np.timedelta64(12, "h")).item(), sum(block2)
    )


def test_derived_param_provided_natively_is_used_as_is():
    """A native field (e.g. from a post-processor) wins over its derivation."""
    ds = xr.Dataset(
        {
            "DD_10M": _make_cumul([0, 6], [123.0, 45.0]),
            "U_10M": _make_cumul([0, 6], [0.0, 0.0]),
            "V_10M": _make_cumul([0, 6], [-5.0, -5.0]),
        }
    )

    result = _disaggregated_and_derived_params(ds, steps=[0, 6], params=["DD_10M"])

    np.testing.assert_allclose(result["DD_10M"].values, [123.0, 45.0])
    assert set(result.data_vars) == {"DD_10M"}  # components dropped


def test_derived_param_from_components_when_not_native():
    ds = xr.Dataset(
        {
            "U_10M": _make_cumul([0, 6], [0.0, -5.0]),
            "V_10M": _make_cumul([0, 6], [-5.0, 0.0]),
        }
    )

    result = _disaggregated_and_derived_params(ds, steps=[0, 6], params=["DD_10M"])

    np.testing.assert_allclose(result["DD_10M"].values, [0.0, 90.0], atol=1e-9)
    assert set(result.data_vars) == {"DD_10M"}


def test_tot_snow6_disaggregated_from_derived_grid_scale_snow():
    """ICON baselines: TOT_SNOW6 from the cumulative SNOW_GSP."""
    ds = _make_cumul_ds([0, 6, 12], [0.0, 2.0, 7.0], var="SNOW_GSP")

    result = _disaggregated_and_derived_params(ds, steps=[6, 12], params=["TOT_SNOW6"])

    np.testing.assert_allclose(result["TOT_SNOW6"].values, [2.0, 5.0])
    assert set(result.data_vars) == {"TOT_SNOW6"}


# ---------------------------------------------------------------------------
# ICON missing values and unavailable params
# ---------------------------------------------------------------------------


def test_undefined_cin_is_masked():
    ds = xr.Dataset(
        {
            "CIN_MU": ("values", [-999.9, 0.0, 35.0]),
            "T_2M": ("values", [-999.9, 280.0, 281.0]),
        }
    )

    out = _mask_icon_missing_values(ds)

    assert np.isnan(out["CIN_MU"].values[0])
    np.testing.assert_allclose(out["CIN_MU"].values[1:], [0.0, 35.0])
    np.testing.assert_allclose(out["T_2M"].values, [-999.9, 280.0, 281.0])


def test_derived_param_without_components_is_skipped(caplog):
    """E.g. TOT_SNOW for a source without SNOW_GSP: warn instead of failing."""
    ds = xr.Dataset({"T_2M": _make_cumul([0, 6], [280.0, 281.0])})

    with caplog.at_level("WARNING"):
        result = _disaggregated_and_derived_params(
            ds, steps=[0, 6], params=["T_2M", "TOT_SNOW6"]
        )

    assert set(result.data_vars) == {"T_2M"}
    assert "Cannot derive TOT_SNOW" in caplog.text


def _write_analysis_zarr(path, variables: dict[str, float], n_times=3) -> None:
    """A tiny anemoi analysis dataset: hourly dates from 2025-11-01, 4 grid cells."""
    n_cells = 4
    data = np.stack(
        [np.full((n_times, n_cells), v, dtype=np.float32) for v in variables.values()],
        axis=1,
    )[:, :, None, :]  # (time, variable, ensemble, cell)
    dates = np.datetime64("2025-11-01T00", "ns") + np.arange(n_times).astype(
        "timedelta64[h]"
    )
    xr.Dataset(
        {
            "data": (("time", "variable", "ensemble", "cell"), data),
            "dates": ("time", dates),
            "latitudes": ("cell", np.linspace(46.0, 47.0, n_cells)),
            "longitudes": ("cell", np.linspace(7.0, 8.0, n_cells)),
        },
        attrs={"variables": list(variables)},
    ).to_zarr(path, consolidated=False)


def test_analysis_zarr_skips_unavailable_params_and_masks_cin(tmp_path, caplog):
    """KENDA: params the zarr lacks are skipped; ICON's CIN marker becomes NaN."""
    root = tmp_path / "kenda.zarr"
    _write_analysis_zarr(
        root, {"T_2M": 280.0, "TD_2M": 275.0, "PS": 9e4, "CIN_MU": -999.9, "FIS": 0.0}
    )

    with caplog.at_level("WARNING"):
        truth = load_truth_data(
            root,
            datetime(2025, 11, 1),
            [0, 1],
            ["T_2M", "TW_2M", "CIN_MU", "HZEROCL", "TOT_SNOW1"],
        )

    assert set(truth.data_vars) == {"T_2M", "TW_2M", "CIN_MU"}
    assert np.isnan(truth["CIN_MU"]).all()
    assert truth.sizes["time"] == 2
    assert "HZEROCL" in caplog.text and "Cannot derive TOT_SNOW" in caplog.text
