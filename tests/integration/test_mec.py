import glob
import subprocess
from pathlib import Path

import pytest
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).resolve().parent / "configs" / "mec_small.yaml"


# Marked longtest: needs GPU (inference) + sarus (MEC container) + ekfSYNOP obs
# at ekf_root. Observed runtime on Balfrin: ~5 min (2 dates, steps 0/12/6).
# mon_synop and ver_synop are not used (stubbed in verif_obs.smk).
@pytest.mark.longtest
def test_mec_produces_feedback_files():
    """Run `evalml experiment ... --mec` end to end and check fdbk_files are produced.

    Exercises the full MEC chain: prepare_mec_input -> link_mec_input ->
    generate_mec_namelist -> sarus_pull_mec -> run_mec (see
    workflow/rules/verif_obs.smk). Needs GPU/MLflow credentials for the
    forecaster inference step, plus `sarus` and access to the configured
    ekf_root/mon_synop_root/ver_synop_root observation archives — so this can
    only run on the CSCS baremetal-runner-balfrin runner, never locally or in
    GitHub Actions.
    """
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
    # CAUTION: this glob will find all runs in the output/data/runs/ tree,
    # not just the one run_id from this test.
    fdbk_files = glob.glob(
        str(PROJECT_ROOT / "output/data/runs/**/fdbk_files/verSYNOP_*.nc"),
        recursive=True,
    )
    assert fdbk_files, (
        "No verSYNOP_*.nc feedback files found under output/data/runs/**/fdbk_files/"
    )

    # this checks the file opens and has at least one data
    # variable — no assertion on actual values.
    ds = xr.open_dataset(fdbk_files[0])
    assert len(ds.data_vars) > 0, f"{fdbk_files[0]} opened but has no data variables"
