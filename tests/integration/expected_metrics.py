"""Plumbing shared by the heavytest metric check (``test_configs.py``) and the
reference generator (``regenerate_expected.py``).

Both locate outputs through :func:`find_nc_files`, run the experiment through
:func:`run_experiment` and read values through :func:`metric_value`, so a
reference can never be produced by a different selection than the one it is
later asserted against — the drift that let an earlier generator rot unnoticed.
"""

import glob
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
EXPECTED_DIR = Path(__file__).resolve().parent / "expected"

# Configs exercised by the metric check and accepted by regenerate_expected.py —
# add or remove names here to control both.
CONFIGS = [
    "varda-single-1.0.yaml",
    "forecasters-ich1.yaml",
]


def run_experiment(config_name):
    """Run ``evalml experiment`` for a config, capturing its output."""
    return subprocess.run(
        ["evalml", "experiment", str(CONFIGS_DIR / config_name)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )


def find_nc_files():
    # Sorted because glob returns filesystem order: regeneration writes the
    # source keys in the order the files are visited, so an unsorted glob would
    # reorder the keys of a committed expected file when a multi-run config is
    # regenerated on a different machine, producing a large meaningless diff.
    return sorted(
        glob.glob(
            str(PROJECT_ROOT / "output/data/runs/**/verif_aggregated_*.nc"),
            recursive=True,
        )
    )


def run_sources(ds):
    """Non-truth sources in a dataset, i.e. the forecaster/baseline runs."""
    return [
        str(s) for s in ds.coords["source"].values if not str(s).startswith("truth")
    ]


def source_key(run_source):
    """Expected-file key for a run source: the part before the ``/``, e.g.
    ``forecaster-b30a-4d02``."""
    return run_source.split("/")[0]


def metric_value(ds, metric, run_source, sel):
    """The single number both comparison and regeneration use, so the reference
    values can never be produced by a different selection than the one checked."""
    return float(ds[metric].sel(source=run_source, **sel).mean("step").values)
