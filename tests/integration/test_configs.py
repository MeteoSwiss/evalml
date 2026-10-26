import statistics
from collections import defaultdict

import pytest
import xarray as xr
import yaml

from expected_metrics import (
    CONFIGS,
    EXPECTED_DIR,
    find_nc_files,
    metric_value,
    run_experiment,
    run_sources,
    source_key,
)

# Loose absolute tolerance. Forecaster metrics drift run-to-run (GPU inference
# nondeterminism, pipeline changes) by far more than any tight tolerance could
# survive: empirically 0 values drift past 0.1 while ~360 drift past 0.01, so a
# stricter bound would make the test flaky without meaningfully catching real
# regressions. (Base #245 independently arrived at the same 1e-1.)
TOLERANCE = 0.1


def load_expected(config_name):
    path = EXPECTED_DIR / config_name
    with open(path) as f:
        return yaml.safe_load(f)


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
            actual = metric_value(ds, metric, run_source, sel)
            diff = abs(actual - expected_value)
            diffs.append(diff)
            if diff > TOLERANCE:
                failures.append((label, metric, sel, expected_value, actual, diff))
                failed_vars[metric.split(".")[0]] += 1
    return n


@pytest.mark.heavytest
@pytest.mark.parametrize("config_name", CONFIGS)
def test_experiment_metrics(config_name):
    result = run_experiment(config_name)
    assert result.returncode == 0, (
        f"evalml experiment failed for {config_name} (exit {result.returncode}).\n"
        f"stdout:\n{result.stdout[-2000:]}\n"
        f"stderr (first 2000):\n{result.stderr[:2000]}\n"
        f"stderr (last 2000):\n{result.stderr[-2000:]}"
    )

    nc_files = find_nc_files()
    assert nc_files, (
        f"No verif_aggregated_*.nc found in output/data/runs/ for {config_name}"
    )

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
        run_source = run_sources(ds)[0]
        total += _collect_failures(
            ds,
            run_source,
            expected,
            source_key(run_source),
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
            for run_source in run_sources(ds):
                key = source_key(run_source)
                if key not in expected:
                    continue
                checked_sources.add(key)
                total += _collect_failures(
                    ds,
                    run_source,
                    expected[key],
                    key,
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
