import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).resolve().parent / "configs" / "meteogram_small.yaml"

# Replay the frozen inference GRIB instead of running inference (no GPU / MLflow
# / sandbox build). This is a test-only switch passed at invocation via
# ``--config fixture_root=<path>`` rather than committed to the YAML, so the
# config still does a real run by default. Populate the fixture once with
# ``evalml capture-fixture <config> <FIXTURE_ROOT>`` after a real GPU run.
FIXTURE_ROOT = Path("/store_new/mch/msopr/cmerker/evalml_test_fixtures/meteogram-small")


@pytest.mark.longtest
def test_showcase_meteogram():
    """Run the showcase workflow on a minimal config and check meteograms are produced.

    Drives the full ``evalml showcase`` pipeline (plotting + verification) end to
    end, replaying inference from a frozen fixture (passed via ``--config
    fixture_root=…``) so no GPU/MLflow/sandbox build is needed; truth still comes
    live from the DWH. Marked ``longtest`` because it needs DWH (jretrievedwh)
    credentials and ``/store_new`` access. PNG existence is not checked explicitly
    because snakemake fails if they are not produced.
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
        [
            "evalml",
            "showcase",
            str(CONFIG),
            "--",
            "--config",
            f"fixture_root={FIXTURE_ROOT}",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"evalml showcase failed (exit {result.returncode}).\n"
        f"stdout tail:\n{result.stdout[-2000:]}\n"
        f"stderr tail:\n{result.stderr[-2000:]}"
    )
