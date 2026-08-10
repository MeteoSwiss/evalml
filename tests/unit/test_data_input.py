"""Unit tests for data_input derivation primitives and TOT_PREC de-accumulation."""

import numpy as np
import pytest
import xarray as xr

from data_input import (
    _accumulate_from_hourly,
    _disaggregate_accum,
    _disaggregated_and_derived_params,
    _ensure_accum_ic,
    compute_derived,
    get_base_params,
    get_steps,
    parse_aggregated_param,
)


def _make_cumul(steps_h, values):
    """Build a cumulative DataArray over `steps_h` (hours)."""
    step = np.array([np.timedelta64(h, "h") for h in steps_h]).astype("timedelta64[ns]")
    return xr.DataArray(
        np.array(values, dtype=np.float64), dims=("step",), coords={"step": step}
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
        compute_derived(xr.Dataset(), "DD_10M")


# ---------------------------------------------------------------------------
# _disaggregate_accum
# ---------------------------------------------------------------------------


def test_disaggregate_returns_nan_for_short_steps():
    # step=3 < n=6 → NaN; step=6 → valid (cumul[6] - cumul[0] = 10)
    cumul = _make_cumul([0, 3, 6], [0.0, 3.0, 10.0])
    result = _disaggregate_accum(cumul, steps=[3, 6], n=6)
    assert np.isnan(result.sel(step=np.timedelta64(3, "h")).item())
    assert result.sel(step=np.timedelta64(6, "h")).item() == pytest.approx(10.0)


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
# Round-trip: _accumulate_from_hourly ↔ _disaggregate_accum
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

    cumul = _accumulate_from_hourly(da_1h, load_steps)
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

    cumul = _accumulate_from_hourly(da_1h, [0, 6, 12])
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


def test_full_roundtrip_hourly_zarr_to_disaggregated():
    """End-to-end: hourly zarr values → _accumulate_from_hourly →
    _disaggregated_and_derived_params → correct 6h period sums."""
    block1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # sum = 21
    block2 = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]  # sum = 210
    da_1h = _make_hourly_da(block1 + block2)

    # _accumulate_from_hourly is called with load_steps (includes the preceding
    # boundary step 0 so disaggregation has an IC)
    cumul = _accumulate_from_hourly(da_1h, [0, 6, 12])
    ds = xr.Dataset({"TOT_PREC": cumul})

    result = _disaggregated_and_derived_params(ds, steps=[6, 12], params=["TOT_PREC6"])

    np.testing.assert_allclose(
        result["TOT_PREC6"].sel(step=np.timedelta64(6, "h")).item(), sum(block1)
    )
    np.testing.assert_allclose(
        result["TOT_PREC6"].sel(step=np.timedelta64(12, "h")).item(), sum(block2)
    )
