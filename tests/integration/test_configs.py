import glob
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

TOLERANCE = 1e-1

# Configs to test — add or remove names here to control which runs are exercised.
CONFIGS = [
    "varda-single-1.0.yaml",
]


def load_expected(config_name):
    path = EXPECTED_DIR / config_name
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Run from the project root after a reference experiment to regenerate all
# expected/*.yaml files with fresh values covering every region, season,
# init_hour combination present in the output:
#
#   python tests/integration/test_configs.py
#
# Metrics with NaN or ±inf values are silently skipped (too few samples, or
# a degenerate score like FBI when no events are forecast).
# Per-source statistics (.max / .mean / .min / .std) are excluded — they are
# not verification metrics and would need separate truth-source selection.
# ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     nc_files = glob.glob(
#         str(PROJECT_ROOT / "output/data/runs/**/verif_aggregated_*.nc"),
#         recursive=True,
#     )
#     assert nc_files, "No verif_aggregated_*.nc found — run an experiment first."
#     ds = xr.open_dataset(nc_files[0])
#     run_src = next(
#         str(s) for s in ds.coords["source"].values if not str(s).startswith("truth")
#     )
#
#     skip_suffixes = (".max", ".mean", ".min", ".std")
#     metrics = [
#         v for v in sorted(ds.data_vars)
#         if "source" in ds[v].dims and not any(v.endswith(s) for s in skip_suffixes)
#     ]
#     regions   = ds.coords["region"].values.tolist()
#     seasons   = ds.coords["season"].values.tolist()
#     init_hrs  = ds.coords["init_hour"].values.tolist()
#
#     for config_name in CONFIGS:
#         entries = []
#         for region in regions:
#             for season in seasons:
#                 for init_hour in init_hrs:
#                     row_metrics = {}
#                     for metric in metrics:
#                         try:
#                             val = float(
#                                 ds[metric]
#                                 .sel(source=run_src, region=region,
#                                      season=season, init_hour=init_hour)
#                                 .mean("step")
#                                 .values
#                             )
#                         except Exception:
#                             continue
#                         if math.isfinite(val):
#                             row_metrics[metric] = round(val, 6)
#                     if row_metrics:
#                         entries.append({
#                             "sel": {
#                                 "region": region,
#                                 "season": season,
#                                 "init_hour": int(init_hour),
#                             },
#                             "metrics": row_metrics,
#                         })
#         out_path = EXPECTED_DIR / config_name
#         with open(out_path, "w") as f:
#             yaml.dump(entries, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
#         print(f"Updated {out_path} ({len(entries)} entries)")


@pytest.mark.heavytest
@pytest.mark.parametrize("config_name", CONFIGS)
def test_experiment_metrics(config_name):
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

    nc_files = glob.glob(
        str(PROJECT_ROOT / "output/data/runs/**/verif_aggregated_*.nc"),
        recursive=True,
    )
    assert nc_files, (
        f"No verif_aggregated_*.nc found in output/data/runs/ for {config_name}"
    )

    ds = xr.open_dataset(nc_files[0])
    run_source = next(
        str(s) for s in ds.coords["source"].values if not str(s).startswith("truth")
    )

    failures = []
    total = 0
    diffs = []
    failed_vars = defaultdict(int)

    for entry in load_expected(config_name):
        sel = entry["sel"]
        for metric, expected_value in entry["metrics"].items():
            total += 1
            actual = float(ds[metric].sel(source=run_source, **sel).mean("step").values)
            diff = abs(actual - expected_value)
            diffs.append(diff)
            if diff > TOLERANCE:
                failures.append((metric, sel, expected_value, actual, diff))
                failed_vars[metric.split(".")[0]] += 1

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
            f"  {m} {sel}: expected={exp}, got={act}, diff={d:.2e}"
            for m, sel, exp, act, d in failures[:20]
        ]
        if len(failures) > 20:
            detail_lines.append(f"  ... and {len(failures) - 20} more")
        raise AssertionError(summary + "\n" + "\n".join(detail_lines))
