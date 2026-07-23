import glob
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).resolve().parent / "configs" / "scoremaps.yaml"

# The scoremaps block plots one map per (param, score, region, season, init_hour,
# leadtime). The fixture requests both an instantaneous param (T_2M) and a
# de-accumulated one (TOT_PREC6) at BIAS / switzerland / all / all / 6 h, so one
# PNG per parameter is expected.
EXPECTED_PARAMS = ["T_2M", "TOT_PREC6"]


@pytest.mark.longtest
def test_experiment_scoremaps():
    """Run the experiment workflow on the minimal scoremaps config and check that
    a score-map PNG is produced for each configured parameter.

    Baseline-only (ICON-CH2-CTRL), no inference — so no GPU, MLflow, or DWH is
    needed, only access to /store_new (the ICON-CH2-EPS baseline archive and the
    KENDA-CH1 truth zarr). Marked ``longtest`` so it is skipped by default (and on
    GitHub Actions, which runs ``pytest tests/unit`` only) and runs on the CSCS
    balfrin runner, which invokes ``pytest tests/integration -m longtest``
    (see ci/cscs.yml).
    """
    result = subprocess.run(
        ["evalml", "experiment", str(CONFIG)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"evalml experiment failed (exit {result.returncode}).\n"
        f"stdout tail:\n{result.stdout[-2000:]}\n"
        f"stderr tail:\n{result.stderr[-2000:]}"
    )

    for param in EXPECTED_PARAMS:
        pngs = glob.glob(
            str(PROJECT_ROOT / f"output/results/**/scoremaps/**/{param}_*.png"),
            recursive=True,
        )
        assert pngs, (
            f"expected score-map PNG for {param} was not produced under "
            "output/results/**/scoremaps/"
        )
        assert all(Path(p).stat().st_size > 0 for p in pngs), (
            f"score-map PNG for {param} is empty"
        )
