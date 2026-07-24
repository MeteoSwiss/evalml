import glob
import subprocess
from pathlib import Path

import pytest
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = Path(__file__).resolve().parent / "configs"

# Configs to test — add or remove names here to control which runs are exercised.
CONFIGS = [
    "varda-single-1.0.yaml",
]

# Known-good metric values per config, captured from reference runs.
# After running a config for the first time, read values with:
#   ds = xr.open_dataset(glob.glob("output/data/runs/**/verif_aggregated_*.nc", recursive=True)[0])
#   run_src = next(s for s in ds.coords["source"].values if not str(s).startswith("truth"))
#   SEL = {"region": "all", "season": "all", "init_hour": -999}
#   print(float(ds["T_2M.RMSE"].sel(source=run_src, **SEL).mean("step").values))
# Values are step-averaged over the fully aggregated slice (region=all, season=all, init_hour=-999).
EXPECTED = {
    "varda-single-1.0.yaml": {
        "T_2M.RMSE":      pytest.approx(1.613003, abs=1e-6),
        "T_2M.BIAS":      pytest.approx(0.093322, abs=1e-6),
        "T_2M.MAE":       pytest.approx(1.076129, abs=1e-6),
        "T_2M.CORR":      pytest.approx(0.938171, abs=1e-6),
        "TD_2M.RMSE":     pytest.approx(2.146990, abs=1e-6),
        "TD_2M.BIAS":     pytest.approx(0.657801, abs=1e-6),
        "TD_2M.MAE":      pytest.approx(1.379343, abs=1e-6),
        "TD_2M.CORR":     pytest.approx(0.949576, abs=1e-6),
        "SP_10M.RMSE":    pytest.approx(1.697493, abs=1e-6),
        "SP_10M.BIAS":    pytest.approx(-0.404315, abs=1e-6),
        "SP_10M.MAE":     pytest.approx(1.235294, abs=1e-6),
        "SP_10M.CORR":    pytest.approx(0.680784, abs=1e-6),
        "PMSL.RMSE":      pytest.approx(41.761943, abs=1e-6),
        "PMSL.BIAS":      pytest.approx(5.627536, abs=1e-6),
        "PMSL.MAE":       pytest.approx(32.933578, abs=1e-6),
        "PMSL.CORR":      pytest.approx(0.928961, abs=1e-6),
        "TOT_PREC6.RMSE": pytest.approx(0.450449, abs=1e-6),
        "TOT_PREC6.BIAS": pytest.approx(0.077970, abs=1e-6),
        "TOT_PREC6.MAE":  pytest.approx(0.112071, abs=1e-6),
        "TOT_PREC6.CORR": pytest.approx(0.720550, abs=1e-6),
    },
}

SEL = {"region": "all", "season": "all", "init_hour": -999}


@pytest.mark.longtest
@pytest.mark.parametrize("config_name", CONFIGS)
def test_experiment_metrics(config_name):
    expected = EXPECTED[config_name]

    result = subprocess.run(
        ["evalml", "experiment", str(CONFIGS_DIR / config_name)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"evalml experiment failed for {config_name} (exit {result.returncode}).\n"
        f"stdout:\n{result.stdout[-2000:]}\n"
        f"stderr:\n{result.stderr[-2000:]}"
    )

    nc_files = glob.glob(
        str(PROJECT_ROOT / "output/data/runs/**/verif_aggregated_*.nc"),
        recursive=True,
    )
    assert nc_files, f"No verif_aggregated_*.nc found in output/data/runs/ for {config_name}"

    ds = xr.open_dataset(nc_files[0])

    run_source = next(
        str(s) for s in ds.coords["source"].values if not str(s).startswith("truth")
    )
    for metric, expected_value in expected.items():
        actual = float(ds[metric].sel(source=run_source, **SEL).mean("step").values)
        assert actual == expected_value, f"{config_name} {metric}: got {actual}"
