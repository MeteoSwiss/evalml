import pytest

from evalml.resolution import (
    leadtime_producible,
    resolve_baseline_id,
    resolve_leadtimes,
    steps_to_leadtimes,
)


def test_resolve_leadtimes_all():
    assert resolve_leadtimes("0/33/6", "all") == [0, 6, 12, 18, 24, 30]


def test_resolve_leadtimes_explicit_filters_unproducible():
    # 36 is not produced by 0/33/6; it is silently dropped.
    assert resolve_leadtimes("0/33/6", [6, 24, 36]) == [6, 24]


def test_resolve_leadtimes_accumulated_drops_short_leads():
    # Accumulated params have no value below one step spacing.
    assert resolve_leadtimes("0/120/6", [0, 6, 12], param="TOT_PREC") == [6, 12]
    assert resolve_leadtimes("0/120/6", [0, 6, 12], param="T_2M") == [0, 6, 12]


def test_leadtime_producible():
    assert leadtime_producible("0/120/1", 24)
    assert not leadtime_producible("0/33/6", 36)
    # accumulated param: 0h window is not producible
    assert not leadtime_producible("0/120/6", 0, param="TOT_PREC")


def test_steps_to_leadtimes():
    assert steps_to_leadtimes("0/12/6") == [0, 6, 12]


def test_resolve_baseline_id_found():
    cfgs = {"baseline-7342": {"label": "ICON-CH1-CTRL"}, "baseline-ce47": {"label": "ICON-CH2-CTRL"}}
    assert resolve_baseline_id("ICON-CH2-CTRL", cfgs) == "baseline-ce47"


def test_resolve_baseline_id_missing_lists_available():
    cfgs = {"baseline-7342": {"label": "ICON-CH1-CTRL"}}
    with pytest.raises(ValueError, match="ICON-CH1-CTRL"):
        resolve_baseline_id("IFS", cfgs)
