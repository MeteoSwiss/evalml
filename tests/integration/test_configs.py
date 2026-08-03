import glob
import math
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

import pytest
import xarray as xr
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
EXPECTED_DIR = Path(__file__).resolve().parent / "expected"

# Loose absolute tolerance. Forecaster metrics drift run-to-run (GPU inference
# nondeterminism, pipeline changes) by far more than any tight tolerance could
# survive: empirically 0 values drift past 0.1 while ~360 drift past 0.01, so a
# stricter bound would make the test flaky without meaningfully catching real
# regressions. (Base #245 independently arrived at the same 1e-1.)
TOLERANCE = 0.1

# Configs to test — add or remove names here to control which runs are exercised.
CONFIGS = [
    "varda-single-1.0.yaml",
    "forecasters-ich1.yaml",
]


# Per-source statistics are not verification metrics and would need a separate
# truth-source selection, so they are never compared or written.
STAT_SUFFIXES = (".max", ".mean", ".min", ".std")


def load_expected(config_name):
    path = EXPECTED_DIR / config_name
    with open(path) as f:
        return yaml.safe_load(f)


def _find_nc_files():
    return glob.glob(
        str(PROJECT_ROOT / "output/data/runs/**/verif_aggregated_*.nc"),
        recursive=True,
    )


def _mtimes(paths):
    return {p: Path(p).stat().st_mtime for p in paths}


def _run_sources(ds):
    """Non-truth sources in a dataset, i.e. the forecaster/baseline runs."""
    return [
        str(s) for s in ds.coords["source"].values if not str(s).startswith("truth")
    ]


def _metric_value(ds, metric, run_source, sel):
    """The single number both comparison and regeneration use, so the reference
    values can never be produced by a different selection than the one checked."""
    return float(ds[metric].sel(source=run_source, **sel).mean("step").values)


def _collect_failures(ds, run_source, entries, label, failures, diffs, failed_vars):
    """Compare one run source's metrics against expected, appending each diff
    to `diffs` and every (diff > TOLERANCE) mismatch to `failures` /
    `failed_vars`. Returns the number of metric values checked. Collecting
    rather than asserting per-metric lets the caller report full stats across
    all sources instead of aborting on the first mismatch."""
    n = 0
    for entry in entries:
        sel = entry["sel"]
        for metric, expected_value in entry["metrics"].items():
            n += 1
            actual = _metric_value(ds, metric, run_source, sel)
            diff = abs(actual - expected_value)
            diffs.append(diff)
            if diff > TOLERANCE:
                failures.append((label, metric, sel, expected_value, actual, diff))
                failed_vars[metric.split(".")[0]] += 1
    return n


def _build_expected(nc_files):
    """Build the expected-metrics mapping from a run's verification output:
    ``{source_hash_prefix: [{"sel": ..., "metrics": ...}, ...]}``, covering every
    region/season/init_hour combination present.

    A config may produce several verif_aggregated_*.nc files (one per
    forecaster/baseline run). Every non-truth source found across them gets its
    own key — the part of the source name before the "/", e.g.
    "forecaster-b30a-4d02" — so all runs are recorded, not just the first one
    glob() happens to return.

    Metrics with NaN or ±inf values are skipped: too few samples, or a
    degenerate score such as FBI when no events are forecast.
    """
    by_source = {}
    for nc_file in nc_files:
        ds = xr.open_dataset(nc_file)
        metrics = [
            v
            for v in sorted(ds.data_vars)
            if "source" in ds[v].dims and not v.endswith(STAT_SUFFIXES)
        ]
        regions = ds.coords["region"].values.tolist()
        seasons = ds.coords["season"].values.tolist()
        init_hours = ds.coords["init_hour"].values.tolist()

        for run_source in _run_sources(ds):
            entries = by_source.setdefault(run_source.split("/")[0], [])
            for region in regions:
                for season in seasons:
                    for init_hour in init_hours:
                        sel = {
                            "region": region,
                            "season": season,
                            "init_hour": int(init_hour),
                        }
                        row_metrics = {}
                        for metric in metrics:
                            try:
                                val = _metric_value(ds, metric, run_source, sel)
                            except Exception:
                                continue
                            if math.isfinite(val):
                                row_metrics[metric] = round(val, 6)
                        if row_metrics:
                            entries.append({"sel": sel, "metrics": row_metrics})
    return by_source


def _regenerate_expected(config_name, nc_files, mtimes_before):
    """Overwrite expected/<config_name> with the values this run produced.

    Only files this run actually (re)wrote are used. `output/data/runs/` is
    shared across configs — the heavytest job runs every config in CONFIGS into
    the same tree — so globbing everything would write one config's references
    into the other's file. If nothing was rewritten, snakemake considered the
    outputs up to date; that is reported rather than silently regenerating from
    a stale tree.
    """
    fresh = [p for p in nc_files if mtimes_before.get(p) != Path(p).stat().st_mtime]
    assert fresh, (
        f"{config_name}: the experiment rewrote no verif_aggregated_*.nc, so the run's own "
        f"outputs cannot be identified (snakemake likely considered them up to date). "
        f"Remove the run's directory under output/data/runs/ (or the whole output/ tree) "
        f"and re-run with --regenerate-expected."
    )

    by_source = _build_expected(fresh)
    assert by_source, (
        f"{config_name}: no finite metrics found in {len(fresh)} freshly written file(s) — "
        f"refusing to write an empty reference."
    )

    out_path = EXPECTED_DIR / config_name
    with open(out_path, "w") as f:
        yaml.dump(
            by_source, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )
    n_entries = sum(len(v) for v in by_source.values())
    print(
        f"\nREGENERATED {out_path} — {n_entries} entries across "
        f"{len(by_source)} source(s): {', '.join(sorted(by_source))}"
    )


@pytest.mark.heavytest
@pytest.mark.parametrize("config_name", CONFIGS)
def test_experiment_metrics(config_name, regenerate_expected):
    # Snapshot before the run so regeneration can tell this config's freshly
    # written outputs from other configs' leftovers in the shared output/ tree.
    mtimes_before = _mtimes(_find_nc_files())

    result = subprocess.run(
        ["evalml", "experiment", str(CONFIGS_DIR / config_name)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"evalml experiment failed for {config_name} (exit {result.returncode}).\n"
        f"stdout:\n{result.stdout[-2000:]}\n"
        f"stderr (first 2000):\n{result.stderr[:2000]}\n"
        f"stderr (last 2000):\n{result.stderr[-2000:]}"
    )

    nc_files = _find_nc_files()
    assert nc_files, (
        f"No verif_aggregated_*.nc found in output/data/runs/ for {config_name}"
    )

    if regenerate_expected:
        _regenerate_expected(config_name, nc_files, mtimes_before)
        return

    expected = load_expected(config_name)

    # Collect every metric mismatch across all run sources (rather than
    # failing on the first), then emit a pass/fail summary with diff stats and
    # per-variable failure counts. Extends the single-source failure reporting
    # from base #245 to the multi-source (one entry set per forecaster) layout.
    failures = []
    diffs = []
    failed_vars = defaultdict(int)
    total = 0

    if isinstance(expected, list):
        # Legacy format: a flat list of {sel, metrics}, checked against the
        # first non-truth source found — correct only when a config has a
        # single run.
        ds = xr.open_dataset(nc_files[0])
        run_source = next(
            str(s) for s in ds.coords["source"].values if not str(s).startswith("truth")
        )
        total += _collect_failures(
            ds,
            run_source,
            expected,
            run_source.split("/")[0],
            failures,
            diffs,
            failed_vars,
        )
    else:
        # Current format: entries keyed by source hash-prefix, so every
        # forecaster/baseline run produced by the config gets checked, not just
        # whichever nc file glob() happens to return first.
        checked_sources = set()
        for nc_file in nc_files:
            ds = xr.open_dataset(nc_file)
            run_sources = [
                str(s)
                for s in ds.coords["source"].values
                if not str(s).startswith("truth")
            ]
            for run_source in run_sources:
                source_key = run_source.split("/")[0]
                if source_key not in expected:
                    continue
                checked_sources.add(source_key)
                total += _collect_failures(
                    ds,
                    run_source,
                    expected[source_key],
                    source_key,
                    failures,
                    diffs,
                    failed_vars,
                )

        assert checked_sources == set(expected), (
            f"{config_name}: expected entries for sources {sorted(expected)} but only "
            f"found {sorted(checked_sources)} among this run's verif_aggregated_*.nc outputs"
        )

    n_fail = len(failures)
    n_pass = total - n_fail
    pct = 100 * n_fail / total if total else 0
    diff_range = f"{min(diffs):.2e} – {max(diffs):.2e}" if diffs else "n/a"
    diff_median = f"{statistics.median(diffs):.2e}" if diffs else "n/a"
    failed_var_summary = (
        ", ".join(f"{v}({c})" for v, c in sorted(failed_vars.items())) or "none"
    )

    summary = (
        f"\n{config_name}: {n_pass}/{total} pass, {n_fail}/{total} fail"
        f" ({pct:.0f}%) | tol={TOLERANCE:.0e}"
        f" | diff range [{diff_range}], median {diff_median}"
        f" | failed vars: {failed_var_summary}"
    )
    print(summary)

    if failures:
        detail_lines = [
            f"  {label} {metric} {sel}: expected={exp}, got={act}, diff={d:.2e}"
            for label, metric, sel, exp, act, d in failures[:20]
        ]
        if len(failures) > 20:
            detail_lines.append(f"  ... and {len(failures) - 20} more")
        raise AssertionError(summary + "\n" + "\n".join(detail_lines))
