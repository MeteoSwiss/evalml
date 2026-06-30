"""Unit tests for data_input TOT_PREC de-accumulation."""

import numpy as np
import pytest
import xarray as xr

from data_input import _tot_prec_handling


def _cumulative_tp(steps_h, values):
    """Build a cumulative-from-start TOT_PREC DataArray over `steps_h` (hours)."""
    step = np.array([np.timedelta64(h, "h") for h in steps_h]).astype("timedelta64[ns]")
    data = np.array(values, dtype=np.float64)[:, np.newaxis] * np.ones((1, 4))
    return xr.DataArray(
        data, dims=("step", "values"), coords={"step": step}, name="TOT_PREC"
    )


def test_tot_prec_missing_step0_synthesised_when_requested():
    """Step 0 requested but absent from the GRIB -> zero IC is synthesised."""
    tp = _cumulative_tp([6], [3.0])
    out = _tot_prec_handling(tp, requested_steps=[0, 6])
    np.testing.assert_allclose(out.sel(step=np.timedelta64(6, "h")).values, 3.0)


def test_tot_prec_full_range_missing_step0():
    """Full-range load with missing step 0 keeps the first lead time."""
    tp = _cumulative_tp([6, 12, 18], [3.0, 5.0, 5.5])
    out = _tot_prec_handling(tp, requested_steps=[0, 6, 12, 18])
    np.testing.assert_allclose(out.sel(step=np.timedelta64(6, "h")).values, 3.0)
    np.testing.assert_allclose(out.sel(step=np.timedelta64(12, "h")).values, 2.0)
    np.testing.assert_allclose(out.sel(step=np.timedelta64(18, "h")).values, 0.5)


def test_tot_prec_window_without_step0_untouched():
    """A [18, 24] window must not be treated as starting at an IC."""
    tp = _cumulative_tp([18, 24], [5.0, 7.0])
    out = _tot_prec_handling(tp, requested_steps=[18, 24])
    np.testing.assert_allclose(out.sel(step=np.timedelta64(24, "h")).values, 2.0)
    # First window step has no preceding accumulation -> NaN after reindex.
    assert np.isnan(out.sel(step=np.timedelta64(18, "h")).values).all()


def test_tot_prec_step0_present_but_nan_is_zero_filled():
    """Step 0 present as an all-NaN hole (earthkit allow_holes) -> zero-filled."""
    tp = _cumulative_tp([0, 6], [np.nan, 3.0])
    out = _tot_prec_handling(tp, requested_steps=[0, 6])
    np.testing.assert_allclose(out.sel(step=np.timedelta64(6, "h")).values, 3.0)


def test_tot_prec_single_step_without_step0_raises():
    """A single loaded step with step 0 not requested cannot be de-accumulated."""
    tp = _cumulative_tp([6], [3.0])
    with pytest.raises(ValueError, match="Cannot de-accumulate TOT_PREC"):
        _tot_prec_handling(tp, requested_steps=[6])
