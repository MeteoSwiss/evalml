"""Unit tests for data_input derivation primitives and TOT_PREC de-accumulation."""

import numpy as np
import pytest
import xarray as xr

from data_input import (
    _disaggregate_accum,
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
