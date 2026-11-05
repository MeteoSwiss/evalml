"""Consumers of aggregated verification (dashboard, scorecard, plots) and ensemble scores."""

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "workflow" / "scripts"))
import report_experiment_dashboard as dash  # noqa: E402
import report_scorecard as sc  # noqa: E402
import verification_plot_metrics as vpm  # noqa: E402
from verification.ensemble_report import (  # noqa: E402
    add_deterministic_crps,
    drop_non_score_vars,
    ensemble_sources,
    fill_crps_from_mae,
)

N = 3
STEPS = np.array([6, 12], dtype="timedelta64[h]").astype("timedelta64[ns]")


def _coords(source):
    return {
        "source": [source],
        "region": ["icon"],
        "season": ["all"],
        "init_hour": [-999],
        "station_group": ["all"],
        "step": STEPS,
    }


def _var(val, extra=None):
    dims = ["source", "region", "season", "init_hour", "station_group", "step"]
    shape = [1, 1, 1, 1, 1, 2]
    arr = np.full(shape, val, dtype=float)
    if extra:
        for d, n in extra.items():
            dims.append(d)
            shape.append(n)
        arr = np.random.default_rng(0).uniform(1, 5, shape)
    return dims, arr


def make_file(source, ensemble=False, n=N):
    ds = xr.Dataset(coords=_coords(source))
    for name, val in [("T_2M.RMSE", 1.0), ("T_2M.MAE", 0.8), ("T_2M.BIAS", 0.1)]:
        ds[name] = _var(val)
    ds["n_samples"] = (("season", "init_hour"), [[4]])
    if ensemble:
        for name, val in [
            ("T_2M.CRPS", 0.5),
            ("T_2M.SPREAD", 0.9),
            ("T_2M.SSR", 1.1),
            ("T_2M.MEMBER_RMSE", 1.2),
            ("T_2M.BS_gt_0p1", 0.2),
        ]:
            ds[name] = _var(val)
        ds["T_2M.rank_hist"] = _var(0, {"rank": n + 1})
        ds["T_2M.prob_table_gt_0p1"] = _var(0, {"n_members_yes": n + 1, "obs": 2})
        ds = ds.assign_coords(rank=np.arange(n + 1), n_members_yes=np.arange(n + 1))
        ds.attrs["n_members"] = n
    return ds


def test_drop_non_score_vars():
    ds = drop_non_score_vars(make_file("ens", ensemble=True))
    assert "T_2M.rank_hist" not in ds and "T_2M.prob_table_gt_0p1" not in ds
    assert {"T_2M.CRPS", "T_2M.BS_gt_0p1", "T_2M.SSR"} <= set(ds.data_vars)


def test_deterministic_crps_is_mae_for_display_only():
    ens, det = make_file("ens", True), make_file("det")
    ds = vpm._concat_sources([ens, det])
    assert ensemble_sources(ds) == ["ens"]
    out, filled = add_deterministic_crps(ds)
    assert filled
    assert out["T_2M.CRPS"].sel(source="det").isel(step=0).item() == 0.8
    assert out["T_2M.CRPS"].sel(source="ens").isel(step=0).item() == 0.5
    assert "T_2M.CRPS" not in det  # input file untouched


def test_deterministic_only_is_unchanged():
    ds = vpm._concat_sources([make_file("a"), make_file("b")])
    out, filled = add_deterministic_crps(ds)
    assert not filled and out is ds
    single, filled = fill_crps_from_mae(make_file("a"))
    assert filled and single["T_2M.CRPS"].equals(single["T_2M.MAE"])
    ens, filled = fill_crps_from_mae(make_file("e", True))
    assert not filled and float(ens["T_2M.CRPS"].max()) == 0.5


def _run_dashboard(tmp_path, files):
    paths = []
    for i, f in enumerate(files):
        p = tmp_path / f"f{i}.nc"
        f.to_netcdf(p)
        paths.append(p)
    out = tmp_path / "out"
    dash.main(
        Namespace(
            verif_files=paths,
            template=ROOT / "resources/report/dashboard/template.html.jinja2",
            script=ROOT / "resources/report/dashboard/script.js",
            header_text="",
            stratification=[],
            station_holdout=False,
            configfile=tmp_path / "none.yaml",
            output=out,
            label_map={},
        )
    )
    return (out / "dashboard.html").read_text()


def test_dashboard(tmp_path):
    html = _run_dashboard(tmp_path, [make_file("ens", True), make_file("det")])
    for m in ["CRPS", "SPREAD", "SSR", "MEMBER_RMSE", "BS &gt; 0.1"]:
        assert f'data-metric="{m}"' in html
    assert "rank_hist" not in html and "prob_table" not in html
    assert "CRPS = MAE for deterministic forecasts" in html


def test_dashboard_deterministic_only_has_no_ensemble_content(tmp_path):
    html = _run_dashboard(tmp_path, [make_file("a"), make_file("b")])
    assert "CRPS" not in html
    assert "deterministic forecasts" not in html


def test_scorecard_crps(tmp_path):
    ens, det = make_file("ens", True), make_file("det")
    pe, pd_ = tmp_path / "e.nc", tmp_path / "d.nc"
    ens.to_netcdf(pe)
    det.to_netcdf(pd_)
    cfg = {
        "model": {"path": str(pe), "source": "ens", "label": "ens"},
        "baseline": {"path": str(pd_), "source": "det", "label": "det"},
        "stratification": "region",
        "lead_times": "6/12/6",
        "all_metrics": sc.ALL_METRICS if hasattr(sc, "ALL_METRICS") else ["RMSE"],
        "variables": {"T_2M": ["CRPS"]},
    }
    diff = sc._load_relative_diff(cfg)
    assert cfg["crps_filled"]
    assert "T_2M.rank_hist" not in diff
    # CRPS 0.5 vs MAE 0.8 -> -37.5 %
    assert float(diff["T_2M.CRPS"].isel(step=0)) == pytest.approx(-37.5)
    assert sc._is_model_better(-37.5, "CRPS", ["CRPS"], []) is True


def test_plot_ensemble_diagnostics(tmp_path):
    ds = vpm._concat_sources([make_file("ens", True), make_file("det")])
    ds = ds.sel(station_group="all", drop=True)
    args = Namespace(output_dir=tmp_path, label_map={})
    vpm.plot_ensemble_diagnostics(ds, args)
    names = {p.name for p in tmp_path.iterdir()}
    assert names == {"rank_hist_T_2M.png", "reliability_T_2M_gt_0p1.png"}


def test_plot_ensemble_diagnostics_per_source_members(tmp_path, monkeypatch):
    """Two ensembles of 3 and 5 members: each keeps its own N + 1 ranks, 1/(N+1)
    reference line and probabilities k/N (the outer join pads the smaller one)."""
    ds = vpm._concat_sources(
        [make_file("small", True, n=3), make_file("big", True, n=5)]
    )
    ds = ds.sel(station_group="all", drop=True)
    assert ds.sizes["rank"] == 6
    figures, close = [], vpm.plt.close
    monkeypatch.setattr(vpm.plt, "close", figures.append)
    vpm.plot_ensemble_diagnostics(ds, Namespace(output_dir=tmp_path, label_map={}))
    rank_fig, rel_fig = figures
    lines = rank_fig.axes[0].get_lines()
    curves = {ln.get_label(): ln for ln in lines if not ln.get_label().startswith("_")}
    assert set(curves) == {"small (N=3)", "big (N=5)"}
    assert curves["small (N=3)"].get_xdata().tolist() == [0, 1, 2, 3]
    assert curves["big (N=5)"].get_xdata().tolist() == [0, 1, 2, 3, 4, 5]
    assert sum(curves["small (N=3)"].get_ydata()) == pytest.approx(1.0)
    references = sorted(
        ln.get_ydata()[0] for ln in lines if ln.get_label().startswith("_")
    )
    assert references == pytest.approx([1 / 6, 1 / 4])
    rel = {
        ln.get_label(): ln.get_xdata().tolist()
        for ln in rel_fig.axes[0].get_lines()
        if not ln.get_label().startswith("_")
    }
    assert rel["small (N=3)"] == pytest.approx([0, 1 / 3, 2 / 3, 1])
    assert rel["big (N=5)"] == pytest.approx([0, 0.2, 0.4, 0.6, 0.8, 1])
    for fig in figures:
        close(fig)


def make_truth(source="truth-c305"):
    """The truth source of an aggregated file: statistics only, no scores."""
    ds = xr.Dataset(coords=_coords(source))
    ds["T_2M.mean"] = _var(285.0)
    return ds


def test_ensemble_detected_by_ensemble_variables_not_spread():
    """An ensemble without a finite SPREAD (e.g. only PTYPE, or step 0) is an ensemble."""
    ens = make_file("ens", True)
    ens["T_2M.SPREAD"][:] = np.nan
    ds = vpm._concat_sources([ens, make_file("det")])
    assert ensemble_sources(ds) == ["ens"]
    single, filled = fill_crps_from_mae(ens)
    assert not filled and float(single["T_2M.CRPS"].max()) == 0.5


def test_truth_never_gets_crps():
    """Ensemble-only experiment: the truth source is not a deterministic forecast."""
    ds = vpm._concat_sources([make_file("ens", True), make_truth()])
    out, filled = add_deterministic_crps(ds)
    assert not filled
    assert np.isnan(out["T_2M.CRPS"].sel(source="truth-c305")).all()
    # with a deterministic participant, only that one is filled
    ds = vpm._concat_sources([make_file("ens", True), make_file("det"), make_truth()])
    out, filled = add_deterministic_crps(ds)
    assert filled
    assert out["T_2M.CRPS"].sel(source="det").isel(step=0).item() == 0.8
    assert np.isnan(out["T_2M.CRPS"].sel(source="truth-c305")).all()


def test_dashboard_ensemble_only_has_no_crps_note(tmp_path):
    ens = make_file("ens", True)
    truth = make_truth()
    truth["n_samples"] = ens["n_samples"]
    both = xr.concat([ens, truth], "source", join="outer", data_vars="minimal")
    html = _run_dashboard(tmp_path, [both])
    assert 'data-metric="CRPS"' in html
    assert "deterministic forecasts" not in html
