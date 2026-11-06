"""Unit tests for the scorecard variable/metric selection."""

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).parents[2] / "workflow" / "scripts"))
from report_scorecard import _filter_diff

ALL_METRICS = ["RMSE", "MAE", "STDE", "R2", "ETS", "POD", "FAR"]


def _diff(*names: str) -> xr.Dataset:
    step = np.array([np.timedelta64(6, "h")]).astype("timedelta64[ns]")
    return xr.Dataset({n: ("step", [1.0]) for n in names}, coords={"step": step})


def test_scorecard_selects_inca_variables_by_exact_name():
    diff = _diff(
        "TOT_PREC6.RMSE",
        "TOT_PREC_RADAR6.RMSE",
        "CAPE_MU.RMSE",
        "CAPE_MU.ETS_gt_100p0",
        "HZEROCL.MAE",
    )
    cfg = {
        "all_metrics": ALL_METRICS,
        "lead_times": None,
        "variables": {"TOT_PREC6": ["RMSE"], "CAPE_MU": None, "HZEROCL": ["MAE"]},
    }

    result = _filter_diff(diff, cfg)
    selected = {v for v in result.data_vars if not np.isnan(result[v]).all()}

    assert selected == {
        "TOT_PREC6.RMSE",
        "CAPE_MU.RMSE",
        "CAPE_MU.ETS_gt_100p0",
        "HZEROCL.MAE",
    }


def test_scorecard_keeps_requested_variables_without_data_as_empty_rows():
    diff = _diff("T_2M.RMSE", "T_2M.R2")
    cfg = {
        "all_metrics": ALL_METRICS,
        "lead_times": None,
        "variables": {"CAPE_MU": ["RMSE"], "T_2M": ["RMSE", "R2"]},
    }

    result = _filter_diff(diff, cfg)

    assert list(result.data_vars) == ["CAPE_MU.RMSE", "T_2M.RMSE", "T_2M.R2"]
    assert np.isnan(result["CAPE_MU.RMSE"]).all()
    np.testing.assert_allclose(result["T_2M.RMSE"], 1.0)


def test_scorecard_without_any_data_raises():
    cfg = {
        "all_metrics": ALL_METRICS,
        "lead_times": None,
        "variables": {"CAPE_MU": None},
    }

    with pytest.raises(ValueError, match="No variables left"):
        _filter_diff(_diff("T_2M.RMSE"), cfg)
