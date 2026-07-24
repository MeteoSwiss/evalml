import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).resolve().parent / "configs" / "meteogram_small.yaml"


@pytest.mark.longtest
def test_showcase_meteogram():
    """Run the showcase workflow on a minimal config and check meteograms are produced.

    Drives the full ``evalml showcase`` pipeline (inference + plotting) end to end.
    Marked ``longtest`` because it needs a GPU, MLflow credentials, DWH
    (jretrievedwh) credentials, and access to the /store_new datasets, so it is
    skipped in ordinary test runs. PNG existence is not checked explicitly because
    snakemake fails if they are not produced.
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
