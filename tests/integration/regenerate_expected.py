#!/usr/bin/env python3
"""Regenerate the expected metrics that test_experiment_metrics asserts against.

    python tests/integration/regenerate_expected.py forecasters-ich1.yaml

Runs the config's experiment exactly as the test does, then writes
tests/integration/expected/<config> from its output instead of comparing.
Review the resulting diff before committing it.

Each config costs hours of GPU time and a pipeline change rarely invalidates all
of them, so configs are named explicitly; pass 'all' to regenerate every one.
"""

import argparse
import math
import sys
from pathlib import Path

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

# Per-source statistics are not verification metrics and would need a separate
# truth-source selection, so they are never compared or written.
STAT_SUFFIXES = (".max", ".mean", ".min", ".std")


def fail(message):
    raise SystemExit(f"error: {message}")


def mtimes(paths):
    return {p: Path(p).stat().st_mtime for p in paths}


def build_expected(nc_files):
    """Build the expected-metrics mapping from a run's verification output:
    ``{source_hash_prefix: [{"sel": ..., "metrics": ...}, ...]}``, covering every
    region/season/init_hour combination present.

    A config may produce several verif_aggregated_*.nc files (one per
    forecaster/baseline run). Every non-truth source found across them gets its
    own key, so all runs are recorded, not just the first one glob() returns.

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

        for run_source in run_sources(ds):
            entries = by_source.setdefault(source_key(run_source), [])
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
                                val = metric_value(ds, metric, run_source, sel)
                            except Exception:
                                continue
                            if math.isfinite(val):
                                row_metrics[metric] = round(val, 6)
                        if row_metrics:
                            entries.append({"sel": sel, "metrics": row_metrics})
    # Drop sources that contributed no finite metric at all: setdefault() above
    # creates the key before any row is built, so without this a reference whose
    # every value was NaN would still be written as a dict of empty lists — and
    # the "refusing to write an empty reference" guard below would not fire.
    return {key: entries for key, entries in by_source.items() if entries}


def regenerate(config_name):
    """Run one config's experiment and overwrite its expected file."""
    # Snapshot before the run so the config's freshly written outputs can be
    # told apart from other configs' leftovers in the shared output/ tree.
    before = mtimes(find_nc_files())

    print(f"\n=== {config_name}: running experiment ===", flush=True)
    result = run_experiment(config_name)
    if result.returncode != 0:
        fail(
            f"evalml experiment failed for {config_name} (exit {result.returncode}).\n"
            f"stdout:\n{result.stdout[-2000:]}\n"
            f"stderr (first 2000):\n{result.stderr[:2000]}\n"
            f"stderr (last 2000):\n{result.stderr[-2000:]}"
        )

    nc_files = find_nc_files()
    if not nc_files:
        fail(f"no verif_aggregated_*.nc found in output/data/runs/ for {config_name}")

    # Only files this run actually (re)wrote are used. output/data/runs/ is
    # shared across configs, so globbing everything would write one config's
    # references into another's file. If nothing was rewritten, snakemake
    # considered the outputs up to date; that is reported rather than silently
    # regenerating from a stale tree.
    fresh = [p for p in nc_files if before.get(p) != Path(p).stat().st_mtime]
    if not fresh:
        fail(
            f"{config_name}: the experiment rewrote no verif_aggregated_*.nc, so the "
            f"run's own outputs cannot be identified (snakemake likely considered them "
            f"up to date). Remove the run's directory under output/data/runs/ (or the "
            f"whole output/ tree) and re-run."
        )

    by_source = build_expected(fresh)
    if not by_source:
        fail(
            f"{config_name}: no finite metrics found in {len(fresh)} freshly written "
            f"file(s) — refusing to write an empty reference."
        )

    out_path = EXPECTED_DIR / config_name
    out_path.write_text(
        yaml.dump(
            by_source, default_flow_style=False, sort_keys=False, allow_unicode=True
        )
    )
    n_entries = sum(len(v) for v in by_source.values())
    print(
        f"REGENERATED {out_path} — {n_entries} entries across "
        f"{len(by_source)} source(s): {', '.join(sorted(by_source))}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "configs",
        nargs="+",
        # Exact names only: a substring match such as "varda-single" would also
        # hit a future varda-single-2.0.yaml, silently overwriting a reference
        # nobody meant to touch at hours of GPU time. Invalid names are rejected
        # with the valid ones listed, before any experiment starts.
        choices=CONFIGS + ["all"],
        metavar="CONFIG",
        help=f"config(s) to regenerate: {', '.join(CONFIGS)}, or 'all'",
    )
    args = parser.parse_args(argv)

    selected = CONFIGS if "all" in args.configs else args.configs
    for config_name in selected:
        regenerate(config_name)


if __name__ == "__main__":
    sys.exit(main())
