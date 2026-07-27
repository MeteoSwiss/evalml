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

TOLERANCE = 1e-6

# Configs to test — add or remove names here to control which runs are exercised.
CONFIGS = [
    "varda-single-1.0.yaml",
]


def load_expected(config_name):
    path = EXPECTED_DIR / config_name
    with open(path) as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------------------------
# run from the project root after a reference experiment to
# rewrite all expected/*.yaml files with fresh numbers:
#   python tests/integration/test_configs.py
# ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     nc_files = glob.glob(
#         str(PROJECT_ROOT / "output/data/runs/**/verif_aggregated_*.nc"),
#         recursive=True,
#     )
#     ds = xr.open_dataset(nc_files[0])
#     run_src = next(
#         str(s) for s in ds.coords["source"].values if not str(s).startswith("truth")
#     )
#     for config_name in CONFIGS:
#         entries = load_expected(config_name)
#         for entry in entries:
#             sel = entry["sel"]
#             for metric in list(entry["metrics"]):
#                 val = float(ds[metric].sel(source=run_src, **sel).mean("step").values)
#                 if math.isnan(val):
#                     del entry["metrics"][metric]
#                 else:
#                     entry["metrics"][metric] = round(val, 6)
#         out_path = EXPECTED_DIR / config_name
#         with open(out_path, "w") as f:
#             yaml.dump(entries, f, default_flow_style=False, sort_keys=False)
#         print(f"Updated {out_path}")


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

    for entry in load_expected(config_name):
        sel = entry["sel"]
        for metric, expected_value in entry["metrics"].items():
            actual = float(ds[metric].sel(source=run_source, **sel).mean("step").values)
            assert actual == pytest.approx(expected_value, abs=TOLERANCE), (
                f"{config_name} {metric} {sel}: got {actual}"
            )
