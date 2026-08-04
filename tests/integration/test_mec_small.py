import glob
import subprocess
from pathlib import Path

import pytest
import xarray as xr
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).resolve().parent / "configs" / "mec_small.yaml"

# The config always replays inference from its `fixture_root` (no GPU / MLflow /
# venv build); the verSYNOP obs are frozen alongside it. Read the paths from the
# config so the test and config never disagree.
_CFG = yaml.safe_load(CONFIG.read_text())
FIXTURE_ROOT = Path(_CFG["fixture_root"])
# output_root is the ephemeral, relative `output/`; resolve it against PROJECT_ROOT
# (the run cwd) so the feedback-file glob below works regardless of the test's cwd.
OUTPUT_ROOT = PROJECT_ROOT / _CFG["locations"]["output_root"]


# Marked longtest: still needs sarus (MEC container). Inference is replayed from
# the frozen fixture and obs come from the frozen ver_synop_root, so no
# GPU/MLflow is needed. Observed runtime on Balfrin: ~5 min.
@pytest.mark.longtest
def test_mec_produces_feedback_files():
    """Run `evalml experiment ... --mec` end to end and check fdbk_files are produced.

    Exercises the full MEC chain: prepare_mec_input -> link_mec_input ->
    generate_mec_namelist -> sarus_pull_mec -> run_mec (see
    workflow/rules/verif_obs.smk). Inference is replayed from the frozen fixture
    (no GPU/MLflow), so this only requires `sarus` and can run on the CSCS
    baremetal-runner-balfrin runner.
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
        ["evalml", "experiment", str(CONFIG), "--mec"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"evalml experiment --mec failed (exit {result.returncode}).\n"
        f"stdout tail:\n{result.stdout[-2000:]}\n"
        f"stderr tail:\n{result.stderr[-2000:]}"
    )

    # mec_all target produces one verSYNOP_{init_time}00.nc per eligible
    # reftime, under data/runs/{run_id}/fdbk_files/ (see rule run_mec).
    fdbk_files = glob.glob(
        str(OUTPUT_ROOT / "data/runs/**/fdbk_files/verSYNOP_*.nc"),
        recursive=True,
    )
    assert fdbk_files, (
        f"No verSYNOP_*.nc feedback files found under {OUTPUT_ROOT}/data/runs/**/fdbk_files/"
    )

    # this checks the file opens and has at least one data
    # variable — no assertion on actual values.
    ds = xr.open_dataset(fdbk_files[0])
    assert len(ds.data_vars) > 0, f"{fdbk_files[0]} opened but has no data variables"
