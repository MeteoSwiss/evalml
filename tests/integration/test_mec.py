import glob
import shutil
import subprocess
from pathlib import Path

import pytest
import xarray as xr
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).resolve().parent / "configs" / "mec_small.yaml"


def _output_root() -> Path:
    cfg = yaml.safe_load(CONFIG.read_text())
    p = Path(cfg["locations"]["output_root"])
    return p if p.is_absolute() else PROJECT_ROOT / p


# Marked longtest: needs sarus (MEC container) + ekfSYNOP obs at ekf_root.
# Inference is skipped: GRIB fixtures are pre-staged in output_root so Snakemake
# finds them and only runs the MEC chain. mon_synop and ver_synop are not used
# (stubbed in verif_obs.smk). Observed runtime on Balfrin: ~5 min.
@pytest.mark.longtest
def test_mec_produces_feedback_files():
    """Run `evalml experiment ... --mec` end to end and check fdbk_files are produced.

    Exercises the full MEC chain: prepare_mec_input -> link_mec_input ->
    generate_mec_namelist -> sarus_pull_mec -> run_mec (see
    workflow/rules/verif_obs.smk). Inference GRIB is pre-staged in output_root
    so no GPU/MLflow is needed. Still requires `sarus` and ekfSYNOP obs, so
    this can only run on the CSCS baremetal-runner-balfrin runner.
    """
    output_root = _output_root()

    # Remove fdbk_files before running so MEC is forced to re-run on every test,
    # even if a previous run left verSYNOP_*.nc behind.
    for fdbk_dir in output_root.glob("data/runs/**/fdbk_files"):
        shutil.rmtree(fdbk_dir)

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
        str(output_root / "data/runs/**/fdbk_files/verSYNOP_*.nc"),
        recursive=True,
    )
    assert fdbk_files, (
        f"No verSYNOP_*.nc feedback files found under {output_root}/data/runs/**/fdbk_files/"
    )

    # this checks the file opens and has at least one data
    # variable — no assertion on actual values.
    ds = xr.open_dataset(fdbk_files[0])
    assert len(ds.data_vars) > 0, f"{fdbk_files[0]} opened but has no data variables"
