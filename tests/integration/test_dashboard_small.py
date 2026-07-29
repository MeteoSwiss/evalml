import subprocess
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).resolve().parent / "configs" / "dashboard_small.yaml"

# The config always replays inference from its `fixture_root` (no GPU / MLflow /
# sandbox build); truth still comes live from the DWH. Read the path from the
# config so the test and config never disagree.

FIXTURE_ROOT = Path(yaml.safe_load(CONFIG.read_text())["fixture_root"])

@pytest.mark.longtest
def test_experiment_dashboard():
    """Run the experiment workflow on a minimal config and check dashboard is produced.

    Drives the ``evalml experiment`` pipeline post-inference.
    Marked ``longtest`` because it needs DWH (jretrievedwh) credentials and
    access to the /store_new datasets, so it is skipped in ordinary test runs.
    PNG/HTML existence is not checked explicitly because snakemake fails if they
    are not produced.
    """
    # A completed capture writes MANIFEST.yaml; keying off that (not mere
    # directory existence) avoids running against an empty/partial fixture.
    if not (FIXTURE_ROOT / "MANIFEST.yaml").exists():
        pytest.skip(
            f"inference fixture not populated at {FIXTURE_ROOT}; create it once with "
            f"`evalml capture-fixture {CONFIG} {FIXTURE_ROOT}` after a real GPU run "
            "(see tests/integration/README.md)."
        )

    result = subprocess.run(
        ["evalml", "experiment", str(CONFIG), "--report"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"evalml experiment failed (exit {result.returncode}).\n"
        f"stdout tail:\n{result.stdout[-2000:]}\n"
        f"stderr tail:\n{result.stderr[-2000:]}"
    )
