import glob
import subprocess
from pathlib import Path

import pytest
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).resolve().parent / "configs" / "mec_small.yaml"


# TODO: confirm the right tier. MEC needs a GPU inference run (like longtest),
# PLUS `sarus` + the MEC container image + real observation archives at
# ekf_root/mon_synop_root/ver_synop_root — arguably heavier than the existing
# longtest cases. Start in heavytest (weekly) and promote to longtest (PR-gating)
# once real runtime/cost on balfrin is known.
@pytest.mark.heavytest
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
    fdbk_files = glob.glob(
        str(PROJECT_ROOT / "output/data/runs/**/fdbk_files/verSYNOP_*.nc"),
        recursive=True,
    )
    assert fdbk_files, (
        "No verSYNOP_*.nc feedback files found under output/data/runs/**/fdbk_files/"
    )

    # TODO: this only checks the file opens and has the expected obstype
    # variable — no assertion yet on actual values. Once a reference run has
    # been captured (same approach as EXPECTED in test_configs.py), replace
    # with concrete value checks, e.g.:
    #   ds["veri_data"].sel(varno=...).values == pytest.approx(...)
    ds = xr.open_dataset(fdbk_files[0])
    assert len(ds.data_vars) > 0, f"{fdbk_files[0]} opened but has no data variables"


# TODO: decide whether to add a second test asserting the *absence* of a
# 'mec' block in the config raises click.UsageError before anything is
# submitted (fast, no CSCS needed — candidate for tests/unit/ instead, not
# this file, since it needs no external resources).