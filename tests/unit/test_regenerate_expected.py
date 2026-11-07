"""Cover the expected-metrics generator and the comparison branches it feeds.

Both live under tests/integration and normally only run behind `-m heavytest`
(hours of GPU per config). These tests drive the same code against a synthetic
verification dataset, so the generation flow, its failure paths and both
expected-file formats stay covered by the default suite.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integration"))

import regenerate_expected as rgx  # noqa: E402
import test_configs  # noqa: E402

FORECASTER = "forecaster-aaaa-1111"
OTHER = "forecaster-bbbb-2222"
DIMS = ("source", "region", "season", "init_hour", "step")
REGIONS = ["CH", "EU"]
SEASONS = ["DJF"]
INIT_HOURS = [0, 12]
STEPS = [1, 2]
# One sel row per region/season/init_hour combination.
N_SELS = len(REGIONS) * len(SEASONS) * len(INIT_HOURS)


def make_dataset(run_source, offset=0.0, finite=True):
    """A verification dataset shaped like verif_aggregated_*.nc, holding one
    truth source and one run source."""
    sources = [f"truth-obs/{run_source}", f"{run_source}/deadbeef"]
    shape = (len(sources), len(REGIONS), len(SEASONS), len(INIT_HOURS), len(STEPS))

    def values(base):
        return np.arange(np.prod(shape), dtype=float).reshape(shape) * 0.001 + base

    return xr.Dataset(
        {
            "PMSL.rmse": (
                DIMS,
                values(1.0 + offset) if finite else np.full(shape, np.nan),
            ),
            "T_2M.bias": (
                DIMS,
                values(2.0 + offset) if finite else np.full(shape, np.nan),
            ),
            # Excluded: a per-source statistic, a degenerate score, and a
            # variable that is not resolved per source.
            "PMSL.max": (DIMS, values(9.0)),
            "FBI.frac": (DIMS, np.full(shape, np.inf)),
            "n_obs": (("region",), np.arange(len(REGIONS), dtype=float)),
        },
        coords={
            "source": sources,
            "region": REGIONS,
            "season": SEASONS,
            "init_hour": INIT_HOURS,
            "step": STEPS,
        },
    )


@pytest.fixture
def nc_file(tmp_path):
    path = tmp_path / "verif_aggregated_test.nc"
    make_dataset(FORECASTER).to_netcdf(path)
    return str(path)


def ok(*_args):
    return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")


def touch(path):
    stamp = Path(path).stat().st_mtime + 10
    os.utime(path, (stamp, stamp))


# --- build_expected -------------------------------------------------------


def test_build_expected_keys_filters_and_values(nc_file):
    built = rgx.build_expected([nc_file])

    # The truth source is never recorded, and the key is the part before "/".
    assert set(built) == {FORECASTER}
    entries = built[FORECASTER]
    assert len(entries) == N_SELS
    assert {tuple(sorted(e["sel"])) for e in entries} == {
        ("init_hour", "region", "season")
    }

    # Stat suffixes, non-finite scores and non-source variables are excluded.
    assert set(entries[0]["metrics"]) == {"PMSL.rmse", "T_2M.bias"}

    # Each value is the mean over lead time of that source's series, computed
    # here independently of metric_value() — asserting against that helper would
    # be a tautology, since the generator reads through it too.
    ds = xr.open_dataset(nc_file)
    for entry in entries:
        for metric, written in entry["metrics"].items():
            series = ds[metric].sel(source=f"{FORECASTER}/deadbeef", **entry["sel"])
            assert series.dims == ("step",)
            assert written == round(float(np.mean(series.values)), 6)


def test_build_expected_records_every_source_across_files(tmp_path):
    paths = []
    for i, source in enumerate([FORECASTER, OTHER]):
        path = tmp_path / f"verif_aggregated_{i}.nc"
        make_dataset(source, offset=i).to_netcdf(path)
        paths.append(str(path))

    built = rgx.build_expected(paths)
    assert set(built) == {FORECASTER, OTHER}
    assert built[FORECASTER] != built[OTHER]


def test_build_expected_drops_sources_with_no_finite_metric(tmp_path):
    path = tmp_path / "verif_aggregated_nan.nc"
    make_dataset(FORECASTER, finite=False).to_netcdf(path)
    assert rgx.build_expected([str(path)]) == {}


# --- regenerate() ---------------------------------------------------------


@pytest.fixture
def regen(monkeypatch, tmp_path, nc_file):
    """regenerate() wired to the synthetic dataset and a temp expected dir."""
    monkeypatch.setattr(rgx, "EXPECTED_DIR", tmp_path)
    monkeypatch.setattr(rgx, "find_nc_files", lambda: [nc_file])
    monkeypatch.setattr(rgx, "run_experiment", ok)
    return tmp_path / "synthetic.yaml"


def test_regenerate_writes_the_reference(monkeypatch, regen, nc_file):
    # A real run rewrites its outputs; regeneration keys off exactly that.
    monkeypatch.setattr(rgx, "run_experiment", lambda name: (touch(nc_file), ok())[1])

    rgx.regenerate("synthetic.yaml")

    assert yaml.safe_load(regen.read_text()) == rgx.build_expected([nc_file])


def test_regenerate_aborts_when_the_experiment_fails(monkeypatch, regen, nc_file):
    monkeypatch.setattr(
        rgx,
        "run_experiment",
        lambda name: (
            touch(nc_file),
            subprocess.CompletedProcess([], returncode=1, stdout="", stderr="boom"),
        )[1],
    )

    with pytest.raises(SystemExit, match="evalml experiment failed"):
        rgx.regenerate("synthetic.yaml")
    assert not regen.exists()


def test_regenerate_aborts_when_the_run_rewrote_nothing(regen):
    # Snakemake considered the outputs up to date, so they cannot be attributed
    # to this config — writing from the stale tree would mix configs.
    with pytest.raises(SystemExit, match="up to date"):
        rgx.regenerate("synthetic.yaml")
    assert not regen.exists()


def test_regenerate_aborts_when_no_outputs_exist(monkeypatch, regen):
    monkeypatch.setattr(rgx, "find_nc_files", lambda: [])
    with pytest.raises(SystemExit, match="no verif_aggregated"):
        rgx.regenerate("synthetic.yaml")
    assert not regen.exists()


def test_regenerate_refuses_to_write_an_empty_reference(
    monkeypatch, regen, tmp_path, nc_file
):
    path = tmp_path / "verif_aggregated_nan.nc"
    make_dataset(FORECASTER, finite=False).to_netcdf(path)
    monkeypatch.setattr(rgx, "find_nc_files", lambda: [str(path)])
    monkeypatch.setattr(rgx, "run_experiment", lambda name: (touch(path), ok())[1])

    with pytest.raises(SystemExit, match="empty reference"):
        rgx.regenerate("synthetic.yaml")
    assert not regen.exists()


# --- command line ---------------------------------------------------------


def test_cli_requires_a_config():
    with pytest.raises(SystemExit) as exc:
        rgx.main([])
    assert exc.value.code == 2


@pytest.mark.parametrize("name", ["varda-single", "nonsense.yaml"])
def test_cli_rejects_inexact_names(name):
    # A substring such as "varda-single" would also match a future
    # varda-single-2.0.yaml, at hours of GPU per config.
    with pytest.raises(SystemExit) as exc:
        rgx.main([name])
    assert exc.value.code == 2


def test_cli_all_expands_to_every_config(monkeypatch):
    called = []
    monkeypatch.setattr(rgx, "regenerate", called.append)
    rgx.main(["all"])
    assert called == rgx.CONFIGS


def test_cli_regenerates_only_the_named_configs(monkeypatch):
    called = []
    monkeypatch.setattr(rgx, "regenerate", called.append)
    rgx.main([rgx.CONFIGS[0]])
    assert called == [rgx.CONFIGS[0]]


# --- the comparison side --------------------------------------------------


@pytest.fixture
def compare(monkeypatch, nc_file):
    """test_experiment_metrics wired to the synthetic dataset, with the
    experiment subprocess stubbed out."""
    monkeypatch.setattr(test_configs, "run_experiment", ok)
    monkeypatch.setattr(test_configs, "find_nc_files", lambda: [nc_file])

    def run(expected):
        monkeypatch.setattr(test_configs, "load_expected", lambda name: expected)
        test_configs.test_experiment_metrics("synthetic.yaml")

    return run


def test_comparison_passes_on_matching_reference(compare, nc_file):
    compare(rgx.build_expected([nc_file]))


def test_comparison_reads_the_legacy_flat_list_format(compare, nc_file):
    # varda-single-1.0.yaml is still a bare list of {sel, metrics}.
    compare(rgx.build_expected([nc_file])[FORECASTER])


@pytest.mark.parametrize("form", ["keyed", "legacy"])
def test_comparison_fails_on_drift_beyond_tolerance(compare, nc_file, form):
    expected = rgx.build_expected([nc_file])
    entry = expected[FORECASTER][0]
    entry["metrics"]["PMSL.rmse"] += 10 * test_configs.TOLERANCE

    with pytest.raises(AssertionError, match="PMSL"):
        compare(expected if form == "keyed" else expected[FORECASTER])


def test_comparison_fails_when_an_expected_source_is_missing(compare, nc_file):
    # Guards against a vacuous pass: a run that lost one of its forecasters
    # must fail rather than quietly check less.
    expected = rgx.build_expected([nc_file])
    expected[OTHER] = expected[FORECASTER]

    with pytest.raises(AssertionError, match="but only found"):
        compare(expected)
