import glob
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).resolve().parent / "configs" / "meteogram.yaml"

# Parameters the fixture config plots meteograms for (temperature and wind speed).
EXPECTED_PARAMS = ["T_2M", "SP_10M", "TOT_PREC6"]
EXPECTED_STATIONS = ["GVE", "SAE"]


@pytest.mark.longtest
def test_showcase_meteogram():
    """Run the showcase workflow on a minimal config and check meteograms are produced.

    Drives the full ``evalml showcase`` pipeline (inference + plotting) end to end
    and asserts that a meteogram PNG for station GVE is written for each expected
    parameter (temperature and wind speed). Marked ``longtest`` because it needs a
    GPU, MLflow credentials, DWH (jretrievedwh) credentials, and access to the
    /store_new datasets, so it is skipped in ordinary test runs.
    """
    result = subprocess.run(
        ["evalml", "showcase", str(CONFIG)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"evalml showcase failed (exit {result.returncode}).\n"
        f"stdout tail:\n{result.stdout[-2000:]}\n"
        f"stderr tail:\n{result.stderr[-2000:]}"
    )

    for param in EXPECTED_PARAMS:
        for station in EXPECTED_STATIONS:
            pngs = glob.glob(
                str(
                    PROJECT_ROOT
                    / f"output/results/**/202503010000_{param}_{station}.png"
                ),
                recursive=True,
            )
            assert pngs, (
                f"expected meteogram PNG (202503010000_{param}_{station}.png) was not produced"
            )
            assert all(Path(p).stat().st_size > 0 for p in pngs), (
                f"meteogram PNG for {param} in {station} is empty"
            )
