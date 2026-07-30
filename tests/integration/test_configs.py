import glob
import math
import subprocess
from pathlib import Path

import pytest
import xarray as xr
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
EXPECTED_DIR = Path(__file__).resolve().parent / "expected"

# Relative tolerance dominates for any metric with real magnitude; the abs
# floor only matters for metrics that legitimately land on exactly 0.0 (e.g.
# a threshold score with no observed events), where a relative check alone
# would demand a bit-exact match.
RELATIVE_TOLERANCE = 1e-3
ABSOLUTE_TOLERANCE = 1e-6

# Configs to test — add or remove names here to control which runs are exercised.
CONFIGS = [
    "varda-single-1.0.yaml",
    "forecasters-ich1.yaml",
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
#
# A config may produce several verif_aggregated_*.nc files (one per
# forecaster/baseline run). Each non-truth source found across all of them is
# written under its own key, keyed by the run's hash-prefix (the part of the
# source name before the "/", e.g. "forecaster-b30a-4d02") so every run gets
# its own expected entries instead of only the first one glob() happens to
# return.
# ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     nc_files = glob.glob(
#         str(PROJECT_ROOT / "output/data/runs/**/verif_aggregated_*.nc"),
#         recursive=True,
#     )
#     assert nc_files, "No verif_aggregated_*.nc found — run an experiment first."
#
#     skip_suffixes = (".max", ".mean", ".min", ".std")
#
#     for config_name in CONFIGS:
#         by_source = {}
#         for nc_file in nc_files:
#             ds = xr.open_dataset(nc_file)
#             run_sources = [
#                 str(s) for s in ds.coords["source"].values if not str(s).startswith("truth")
#             ]
#             metrics = [
#                 v for v in sorted(ds.data_vars)
#                 if "source" in ds[v].dims and not any(v.endswith(s) for s in skip_suffixes)
#             ]
#             regions   = ds.coords["region"].values.tolist()
#             seasons   = ds.coords["season"].values.tolist()
#             init_hrs  = ds.coords["init_hour"].values.tolist()
#
#             for run_src in run_sources:
#                 entries = by_source.setdefault(run_src.split("/")[0], [])
#                 for region in regions:
#                     for season in seasons:
#                         for init_hour in init_hrs:
#                             row_metrics = {}
#                             for metric in metrics:
#                                 try:
#                                     val = float(
#                                         ds[metric]
#                                         .sel(source=run_src, region=region,
#                                              season=season, init_hour=init_hour)
#                                         .mean("step")
#                                         .values
#                                     )
#                                 except Exception:
#                                     continue
#                                 if math.isfinite(val):
#                                     row_metrics[metric] = round(val, 6)
#                             if row_metrics:
#                                 entries.append({
#                                     "sel": {
#                                         "region": region,
#                                         "season": season,
#                                         "init_hour": int(init_hour),
#                                     },
#                                     "metrics": row_metrics,
#                                 })
#         out_path = EXPECTED_DIR / config_name
#         with open(out_path, "w") as f:
#             yaml.dump(by_source, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
#         n_entries = sum(len(v) for v in by_source.values())
#         print(f"Updated {out_path} ({n_entries} entries across {len(by_source)} source(s))")


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

    expected = load_expected(config_name)

    # Legacy format: a flat list of {sel, metrics}, checked against the first
    # non-truth source found — correct only when a config has a single run.
    if isinstance(expected, list):
        ds = xr.open_dataset(nc_files[0])
        run_source = next(
            str(s) for s in ds.coords["source"].values if not str(s).startswith("truth")
        )
        for entry in expected:
            sel = entry["sel"]
            for metric, expected_value in entry["metrics"].items():
                actual = float(ds[metric].sel(source=run_source, **sel).mean("step").values)
                assert actual == pytest.approx(expected_value, rel=RELATIVE_TOLERANCE, abs=ABSOLUTE_TOLERANCE), (
                    f"{config_name} {metric} {sel}: got {actual}"
                )
        return

    # Current format: entries keyed by source hash-prefix, so every
    # forecaster/baseline run produced by the config gets checked, not just
    # whichever nc file glob() happens to return first.
    checked_sources = set()
    for nc_file in nc_files:
        ds = xr.open_dataset(nc_file)
        run_sources = [
            str(s) for s in ds.coords["source"].values if not str(s).startswith("truth")
        ]
        for run_source in run_sources:
            source_key = run_source.split("/")[0]
            if source_key not in expected:
                continue
            checked_sources.add(source_key)
            for entry in expected[source_key]:
                sel = entry["sel"]
                for metric, expected_value in entry["metrics"].items():
                    actual = float(ds[metric].sel(source=run_source, **sel).mean("step").values)
                    assert actual == pytest.approx(expected_value, rel=RELATIVE_TOLERANCE, abs=ABSOLUTE_TOLERANCE), (
                        f"{config_name} {source_key} {metric} {sel}: got {actual}"
                    )

    assert checked_sources == set(expected), (
        f"{config_name}: expected entries for sources {sorted(expected)} but only "
        f"found {sorted(checked_sources)} among this run's verif_aggregated_*.nc outputs"
    )
