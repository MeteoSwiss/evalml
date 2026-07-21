import glob
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).resolve().parent / "configs" / "meteogram.yaml"


@pytest.mark.longtest
def test_showcase_meteogram():
    """Run the showcase workflow on a minimal config and check a meteogram is produced.

    Drives the full ``evalml showcase`` pipeline (inference + plotting) end to end
    and asserts that the expected meteogram PNG for station GVE / parameter T_2M is
    written. Marked ``longtest`` because it needs a GPU, MLflow credentials, DWH
    (jretrievedwh) credentials, and access to the /store_new datasets, so it is
    skipped in ordinary test runs.
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

    pngs = glob.glob(
        str(PROJECT_ROOT / "output/results/**/202401010000_T_2M_GVE.png"),
        recursive=True,
    )
    assert pngs, "expected meteogram PNG (202401010000_T_2M_GVE.png) was not produced"
    assert all(Path(p).stat().st_size > 0 for p in pngs), "meteogram PNG is empty"
