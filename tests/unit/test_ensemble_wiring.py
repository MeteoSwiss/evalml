"""Wiring of ensemble verification: the common.smk helpers of the verification rules,
member loading and scoring in verification_metrics.py, and the valid mask of an
ensemble in verification_valid_mask.py."""

import ast
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "workflow" / "scripts"))

import verification_metrics  # noqa: E402
import verification_valid_mask  # noqa: E402
from verification import verify  # noqa: E402

COMMON = ROOT / "workflow/rules/common.smk"
FUNCTIONS = {
    "run_grib_root",
    "run_grib_dir",
    "ensemble_members",
    "members_flag",
    "ensemble_scaled",
}


def _common_namespace(run_configs, out_root):
    tree = ast.parse(COMMON.read_text())
    body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS
    ]
    namespace = {"Path": Path, "RUN_CONFIGS": run_configs, "OUT_ROOT": out_root}
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(COMMON), "exec"), namespace
    )
    return namespace


RUN_CONFIGS = {
    "forecaster-aaaa-bbbb/cccc": {"steps": "0/24/6"},
    "forecaster-aaaa-bbbb/dddd": {
        "steps": "0/24/6",
        "ensemble": {"members": 4, "seed": 0},
    },
}
DET, ENS = RUN_CONFIGS


def test_verification_reads_grib_root_deterministic_unchanged(tmp_path):
    """Deterministic runs: the same grib/ directory as before, and no members flag."""
    ns = _common_namespace(RUN_CONFIGS, tmp_path)
    grib = ns["run_grib_root"](DET, "202406150000")
    assert grib == tmp_path / f"data/runs/{DET}/202406150000/grib"
    assert grib == ns["run_grib_dir"](DET, "202406150000")
    assert ns["members_flag"](Namespace(run_id=DET)) == ""


def test_verification_reads_all_members_of_an_ensemble(tmp_path):
    ns = _common_namespace(RUN_CONFIGS, tmp_path)
    assert ns["run_grib_root"](ENS, "202406150000") == (
        tmp_path / f"data/runs/{ENS}/202406150000/grib"
    )
    assert ns["members_flag"](Namespace(run_id=ENS)) == "--members 4"
    # plots, SAL, score maps and MEC keep member 001
    assert ns["run_grib_dir"](ENS, "202406150000") == (
        tmp_path / f"data/runs/{ENS}/202406150000/grib/001"
    )


def test_ensemble_scaled_resources(tmp_path):
    ns = _common_namespace(RUN_CONFIGS, tmp_path)
    mem = ns["ensemble_scaled"](80_000, 2_000)
    runtime = ns["ensemble_scaled"](60, 6, unit="m")
    assert mem(Namespace(run_id=DET)) == 80_000
    assert runtime(Namespace(run_id=DET)) == "60m"
    assert mem(Namespace(run_id=ENS)) == 88_000
    assert runtime(Namespace(run_id=ENS)) == "84m"


# --- member loading -----------------------------------------------------------


def _member_dataset(member, n_values=3):
    values = np.full((1, 2, n_values), float(member))
    return xr.Dataset(
        {"T_2M": (("forecast_reference_time", "step", "values"), values)},
        coords={
            "forecast_reference_time": [np.datetime64("2024-06-15T00:00", "ns")],
            "step": pd.to_timedelta([0, 6], unit="h"),
            "latitude": ("values", np.linspace(46.0, 47.0, n_values)),
            "longitude": ("values", np.linspace(6.0, 9.0, n_values)),
            "valid_time": (
                ("forecast_reference_time", "step"),
                np.array(
                    [["2024-06-15T00:00", "2024-06-15T06:00"]], dtype="datetime64[ns]"
                ),
            ),
            # differs per member on purpose: member 001's must be kept
            "elevation": ("values", np.full(n_values, 100.0 * member)),
        },
    )


@pytest.fixture
def member_dirs(tmp_path, monkeypatch):
    grib = tmp_path / "grib"
    for m in (1, 2, 3):
        (grib / f"{m:03d}").mkdir(parents=True)
        (grib / f"{m:03d}" / "x_0.grib").touch()
    calls = []

    def fake_load(root, reftime, steps, params, member="000"):
        calls.append((Path(root).name, params))
        return _member_dataset(int(Path(root).name))

    monkeypatch.setattr(verification_metrics, "load_forecast_data", fake_load)
    return grib, calls


def test_load_ensemble_forecast_concats_members(member_dirs):
    grib, calls = member_dirs
    fcst = verification_metrics.load_ensemble_forecast(
        grib, None, [0, 6], ["T_2M"], members=3
    )
    assert [c[0] for c in calls] == ["001", "002", "003"]
    assert fcst["member"].values.tolist() == [1, 2, 3]
    assert fcst["T_2M"].sel(member=2).values.tolist() == [[[2.0] * 3] * 2]
    assert "member" not in fcst["elevation"].dims
    np.testing.assert_array_equal(fcst["elevation"].values, 100.0)


def test_load_ensemble_forecast_fails_on_missing_member(member_dirs):
    grib, _ = member_dirs
    with pytest.raises(FileNotFoundError, match="member 004 of 4"):
        verification_metrics.load_ensemble_forecast(
            grib, None, [0, 6], ["T_2M"], members=4
        )


def test_load_aligned_data_routes_members(member_dirs, monkeypatch):
    grib, _ = member_dirs
    truth = xr.Dataset(
        {"T_2M": (("time", "values"), np.zeros((2, 3)))},
        coords={
            "time": np.array(
                ["2024-06-15T00:00", "2024-06-15T06:00"], dtype="datetime64[ns]"
            )
        },
    )
    monkeypatch.setattr(verification_metrics, "load_truth_data", lambda *a: truth)
    monkeypatch.setattr(verification_metrics, "map_forecast_to_truth", lambda f, t: f)
    args = Namespace(
        forecast=grib, reftime=None, steps=[0, 6], truth=None, members=3, member="000"
    )
    fcst, _ = verification_metrics.load_aligned_data(args, ["T_2M"])
    assert fcst.sizes["member"] == 3


# --- scoring --------------------------------------------------------------------

_REGION = [{"type": "bbox", "name": "all", "bbox": [1.5, 16, 43, 49.5]}]


def _stations(values):
    """(step, values) station field inside _REGION; values (..., n)."""
    values = np.asarray(values, dtype=float)
    n = values.shape[-1]
    dims = ("member", "step", "values") if values.ndim == 3 else ("step", "values")
    coords = {
        "step": pd.to_timedelta([6], unit="h"),
        "values": np.arange(n),
        "longitude": ("values", np.linspace(5.0, 10.0, n)),
        "latitude": ("values", np.linspace(46.0, 47.0, n)),
        "forecast_reference_time": np.datetime64("2024-06-15T00:00", "ns"),
    }
    if values.ndim == 3:
        coords["member"] = np.arange(1, values.shape[0] + 1)
    return xr.Dataset({"T_2M": (dims, values)}, coords=coords)


TRUTH = _stations([[1.0, 2.0, 3.0, 4.0]])
MEMBERS = _stations(
    [
        [[0.0, 2.0, 2.0, 6.0]],
        [[2.0, 4.0, 3.0, 2.0]],
        [[4.0, 0.0, 7.0, 4.0]],
    ]
)


def _run(fcst, truth=TRUTH, **kwargs):
    return verification_metrics.verify_ensemble_run(
        fcst,
        truth,
        "ens",
        "truth",
        regions=_REGION,
        threshold_dict={"T_2M": {"gt": [2.5]}},
        holdout_stations=None,
        **kwargs,
    )


def test_verify_ensemble_run_scores_the_ensemble_mean():
    out = _run(MEMBERS)
    ref = verify(
        MEMBERS.mean("member"),
        TRUTH,
        "ens",
        "truth",
        regions=_REGION,
        threshold_dict={"T_2M": {"gt": [2.5]}},
        num_workers=1,
    )
    for name in ("T_2M.BIAS", "T_2M.MSE", "T_2M.MAE", "T_2M.contingency_table"):
        xr.testing.assert_allclose(out[name], ref[name])
    # truth statistics are those of verify()
    for name in ("T_2M.mean", "T_2M.var", "T_2M.min", "T_2M.max"):
        xr.testing.assert_allclose(
            out[name].sel(source="truth"), ref[name].sel(source="truth")
        )
    for name in ("T_2M.CRPS", "T_2M.ENS_VAR", "T_2M.MEMBER_MSE", "T_2M.rank_hist"):
        assert name in out
    assert "T_2M.prob_table" in out
    assert out.attrs["n_members"] == 3
    assert out["source"].values.tolist() == ["ens", "truth"]


def test_verify_ensemble_run_member_mean_statistics():
    out = _run(MEMBERS)
    sel = dict(source="ens", region="all", station_group="all")
    values = MEMBERS["T_2M"].values[:, 0]  # member, values
    assert out["T_2M.mean"].sel(**sel).item() == pytest.approx(values.mean())
    assert out["T_2M.var"].sel(**sel).item() == pytest.approx(values.var(axis=1).mean())
    assert out["T_2M.min"].sel(**sel).item() == pytest.approx(values.min(axis=1).mean())
    assert out["T_2M.max"].sel(**sel).item() == pytest.approx(values.max(axis=1).mean())


def test_verify_ensemble_run_missing_member_value():
    """A point where one member is missing makes the scores NaN, as in verify()."""
    members = MEMBERS.copy(deep=True)
    members["T_2M"][1, 0, 2] = np.nan
    out = _run(members)
    sel = dict(source="ens", region="all", station_group="all")
    for name in ("T_2M.MSE", "T_2M.CRPS", "T_2M.mean", "T_2M.var"):
        assert np.isnan(out[name].sel(**sel).item()), name
    assert np.isfinite(out["T_2M.mean"].sel(source="truth", region="all").item())


def test_verify_ensemble_run_circular_ensemble_mean():
    """The ensemble mean of wind direction is circular: 350 and 10 deg give 0 deg."""
    members = _stations([[[350.0, 90.0, 180.0, 270.0]], [[10.0, 90.0, 180.0, 270.0]]])
    members = members.rename({"T_2M": "DD_10M"})
    truth = _stations([[0.0, 90.0, 180.0, 270.0]]).rename({"T_2M": "DD_10M"})
    out = verification_metrics.verify_ensemble_run(
        members,
        truth,
        "ens",
        "truth",
        regions=_REGION,
        threshold_dict=None,
        holdout_stations=None,
    )
    sel = dict(source="ens", region="all", station_group="all")
    assert out["DD_10M.MAE"].sel(**sel).item() == pytest.approx(0.0, abs=1e-9)


# --- valid mask -----------------------------------------------------------------


def test_valid_mask_of_an_ensemble_is_the_and_over_members(tmp_path, monkeypatch):
    members = MEMBERS.copy(deep=True)
    members["T_2M"][1, 0, 2] = np.nan  # member 2 missing at point 2
    monkeypatch.setattr(
        verification_valid_mask, "load_aligned_data", lambda args, p: (members, None)
    )
    args = Namespace(params=["T_2M"], output=tmp_path / "mask.nc", members=3)
    verification_valid_mask.main(args)
    mask = xr.open_dataset(args.output)
    assert "member" not in mask.dims
    assert mask["T_2M"].values.ravel().tolist() == [True, True, False, True]


def test_verify_ensemble_run_categorical_has_no_ensemble_mean_scores():
    """PTYPE: the mean of class codes is no class (rain 1 + snow 5 -> 3, freezing
    rain), so the run's deterministic scores are NaN; prob_table scores the ensemble."""
    members = _stations([[[1.0, 5.0, 0.0, 1.0]], [[5.0, 1.0, 0.0, 1.0]]])
    members = members.rename({"T_2M": "PTYPE"})
    truth = _stations([[3.0, 1.0, 0.0, 1.0]]).rename({"T_2M": "PTYPE"})
    out = verification_metrics.verify_ensemble_run(
        members,
        truth,
        "ens",
        "truth",
        regions=_REGION,
        threshold_dict={"PTYPE": {"eq": [1, 3, 5]}},
        holdout_stations=None,
    )
    sel = dict(source="ens", region="all", station_group="all")
    for name in ("PTYPE.BIAS", "PTYPE.MSE", "PTYPE.MAE", "PTYPE.CORR"):
        assert np.isnan(out[name].sel(**sel)).all(), name
    table = out["PTYPE.contingency_table"]
    assert "source" in table.dims
    assert np.isnan(table.sel(**sel)).all()
    # the ensemble's categorical score: 2 members, eq_3 never forecast but observed once
    prob = out["PTYPE.prob_table"].sel(**sel)
    assert prob.sel({"PTYPE.threshold": "eq_3", "n_members_yes": 0, "obs": 1}) == 1
    assert int(prob.sum()) == 3 * 4
    # forecast statistics (averaged over members) and truth statistics are kept
    assert out["PTYPE.mean"].sel(**sel).item() == pytest.approx(14 / 8)
    assert np.isfinite(out["PTYPE.mean"].sel(source="truth", region="all").item())


def test_verify_ensemble_run_keeps_continuous_contingency_tables():
    out = _run(MEMBERS)
    sel = dict(source="ens", region="all", station_group="all")
    assert np.isfinite(out["T_2M.contingency_table"].sel(**sel)).all()


def _reference_member_mean_statistics(fcst, truth, regions, holdout_stations, params):
    """member_mean_statistics before the memory fix (all params in one compute)."""
    from verification import (
        CIRCULAR_PARAMS,
        _apply_masks,
        _compute_statistics,
        _region_and_station_masks,
        _too_many_missing,
    )

    dim = ["values"]
    fcst, truth = xr.align(fcst, truth, join="inner", copy=False)
    masks, station_masks = _region_and_station_masks(truth, regions, holdout_stations)
    stats = []
    for param in params:
        members = fcst[param].where(fcst[param].notnull().all("member"))
        too_many_missing = _too_many_missing(
            members.isnull().any("member"),
            _apply_masks(truth[param], masks, station_masks),
            dim,
            0.0,
        )
        member_stats = _compute_statistics(
            _apply_masks(members, masks, station_masks),
            prefix=param + ".",
            dim=dim,
            circular=param in CIRCULAR_PARAMS,
        )
        stats.append(
            member_stats.mean("member")
            .squeeze("source", drop=True)
            .where(~too_many_missing)
        )
    return xr.merge(stats).compute()


def test_member_mean_statistics_unchanged_by_chunking():
    """Members chunked one by one, one param at a time: same statistics as before."""
    rng = np.random.default_rng(1)
    n_members, n_steps, n = 4, 3, 50
    values = rng.normal(280.0, 5.0, (n_members, n_steps, n))
    values[2, 1, 7] = np.nan  # one member missing at one point and step
    coords = {
        "member": np.arange(1, n_members + 1),
        "step": pd.to_timedelta(np.arange(n_steps) * 6, unit="h"),
        "values": [f"S{i:02d}" for i in range(n)],
        "longitude": ("values", np.linspace(5.0, 10.0, n)),
        "latitude": ("values", np.linspace(45.9, 47.5, n)),
    }
    fcst = xr.Dataset(
        {
            "T_2M": (("member", "step", "values"), values),
            "TD_2M": (("member", "step", "values"), values - 5.0),
        },
        coords=coords,
    ).chunk({"step": 1})
    truth = fcst.isel(member=0, drop=True).compute() + 0.5
    regions = _REGION + [{"type": "bbox", "name": "west", "bbox": [5.0, 7.5, 43, 49.5]}]
    holdout = ["S00", "S03", "S10"]
    params = ["T_2M", "TD_2M"]
    new = verification_metrics.member_mean_statistics(
        fcst, truth, regions, holdout, params=params, num_workers=2
    )
    ref = _reference_member_mean_statistics(fcst, truth, regions, holdout, params)
    xr.testing.assert_allclose(new, ref, rtol=1e-12)
    # S07 is missing in one member at step 1: NaN where it counts, not in holdout
    assert np.isnan(new["T_2M.mean"].isel(step=1).sel(station_group="all")).all()
    assert np.isfinite(new["T_2M.mean"].isel(step=1).sel(station_group="holdout")).all()
    assert np.isfinite(new["T_2M.mean"].isel(step=0)).all()


def test_verification_rules_pass_members_flag():
    """verification_metrics and verification_valid_mask read grib/ and pass the members
    flag as a plain param (empty for deterministic runs)."""
    text = (ROOT / "workflow/rules/verification.smk").read_text()
    for rule in ("verification_valid_mask", "verification_metrics"):
        body = text.split(f"rule {rule}:", 1)[1].split("\nrule ", 1)[0]
        assert (
            "grib_out_dir=lambda wc: run_grib_root(wc.run_id, wc.init_time).resolve()"
            in body
        )
        assert "members_flag=members_flag" in body
        assert "--forecast {params.grib_out_dir}" in body
        assert "{params.members_flag}" in body
