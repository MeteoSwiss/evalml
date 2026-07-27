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

# ---------------------------------------------------------------------------
# Refresh helper — run from the project root after a reference experiment to
# reprint all values in EXPECTED with fresh numbers:
#   python tests/integration/test_configs.py
# ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     import math
#
#     nc_files = glob.glob(
#         str(PROJECT_ROOT / "output/data/runs/**/verif_aggregated_*.nc"),
#         recursive=True,
#     )
#     ds = xr.open_dataset(nc_files[0])
#     run_src = next(
#         str(s) for s in ds.coords["source"].values if not str(s).startswith("truth")
#     )
#     for config_name, entries in EXPECTED.items():
#         print(f'    "{config_name}": [')
#         for entry in entries:
#             sel = entry["sel"]
#             print(f"        {{")
#             print(f'            "sel": {sel!r},')
#             for metric in entry:
#                 if metric == "sel":
#                     continue
#                 val = float(ds[metric].sel(source=run_src, **sel).mean("step").values)
#                 if math.isnan(val):
#                     print(f'            # "{metric}": NaN — skipped')
#                 else:
#                     print(f'            "{metric}": pytest.approx({val:.6f}, abs=1e-6),')
#             print(f"        }},")
#         print(f"    ],")
EXPECTED = {
    "varda-single-1.0.yaml": [
        {
            "sel": {"region": "all", "season": "all", "init_hour": -999},
            "T_2M.RMSE": pytest.approx(1.611747, abs=1e-6),
            "T_2M.BIAS": pytest.approx(0.092753, abs=1e-6),
            "T_2M.MAE": pytest.approx(1.075605, abs=1e-6),
            "T_2M.CORR": pytest.approx(0.938292, abs=1e-6),
            "TD_2M.RMSE": pytest.approx(2.145868, abs=1e-6),
            "TD_2M.BIAS": pytest.approx(0.657530, abs=1e-6),
            "TD_2M.MAE": pytest.approx(1.378000, abs=1e-6),
            "TD_2M.CORR": pytest.approx(0.949596, abs=1e-6),
            "SP_10M.RMSE": pytest.approx(1.696772, abs=1e-6),
            "SP_10M.BIAS": pytest.approx(-0.399737, abs=1e-6),
            "SP_10M.MAE": pytest.approx(1.233097, abs=1e-6),
            "SP_10M.CORR": pytest.approx(0.680623, abs=1e-6),
            "PMSL.RMSE": pytest.approx(41.649109, abs=1e-6),
            "PMSL.BIAS": pytest.approx(5.384452, abs=1e-6),
            "PMSL.MAE": pytest.approx(32.770433, abs=1e-6),
            "PMSL.CORR": pytest.approx(0.929027, abs=1e-6),
            "TOT_PREC6.RMSE": pytest.approx(0.449400, abs=1e-6),
            "TOT_PREC6.BIAS": pytest.approx(0.077549, abs=1e-6),
            "TOT_PREC6.MAE": pytest.approx(0.111717, abs=1e-6),
            "TOT_PREC6.CORR": pytest.approx(0.720677, abs=1e-6),
            # T_2M threshold metrics (gt_288p15 / gt_298p15 are NaN — only 2 dates in March)
            "T_2M.ETS_lt_273p15": pytest.approx(0.736541, abs=1e-6),
            "T_2M.POD_lt_273p15": pytest.approx(0.889075, abs=1e-6),
            "T_2M.FAR_lt_273p15": pytest.approx(0.049425, abs=1e-6),
            "T_2M.FBI_lt_273p15": pytest.approx(0.936333, abs=1e-6),
            # SP_10M threshold metrics (gt_10p0 / FAR_gt_10p0 are zero / NaN)
            "SP_10M.ETS_gt_2p5": pytest.approx(0.387069, abs=1e-6),
            "SP_10M.ETS_gt_5p0": pytest.approx(0.258426, abs=1e-6),
            "SP_10M.ETS_gt_10p0": pytest.approx(0.000000, abs=1e-6),
            "SP_10M.POD_gt_2p5": pytest.approx(0.705256, abs=1e-6),
            "SP_10M.POD_gt_5p0": pytest.approx(0.436008, abs=1e-6),
            "SP_10M.POD_gt_10p0": pytest.approx(0.000000, abs=1e-6),
            "SP_10M.FAR_gt_2p5": pytest.approx(0.206813, abs=1e-6),
            "SP_10M.FAR_gt_5p0": pytest.approx(0.441614, abs=1e-6),
            # SP_10M.FAR_gt_10p0 is NaN — skipped
            "SP_10M.FBI_gt_2p5": pytest.approx(0.896507, abs=1e-6),
            "SP_10M.FBI_gt_5p0": pytest.approx(0.779540, abs=1e-6),
            "SP_10M.FBI_gt_10p0": pytest.approx(0.000000, abs=1e-6),
            # TOT_PREC6 threshold metrics (gt_50p0 is NaN — no events at that threshold)
            "TOT_PREC6.ETS_gt_0p0": pytest.approx(0.156563, abs=1e-6),
            "TOT_PREC6.ETS_gt_0p1": pytest.approx(0.278809, abs=1e-6),
            "TOT_PREC6.ETS_gt_5p0": pytest.approx(0.463557, abs=1e-6),
            "TOT_PREC6.ETS_gt_10p0": pytest.approx(0.488005, abs=1e-6),
            "TOT_PREC6.ETS_gt_20p0": pytest.approx(0.000000, abs=1e-6),
            "TOT_PREC6.POD_gt_0p0": pytest.approx(0.929660, abs=1e-6),
            "TOT_PREC6.POD_gt_0p1": pytest.approx(0.918232, abs=1e-6),
            "TOT_PREC6.POD_gt_5p0": pytest.approx(0.717857, abs=1e-6),
            "TOT_PREC6.POD_gt_10p0": pytest.approx(0.488889, abs=1e-6),
            "TOT_PREC6.POD_gt_20p0": pytest.approx(0.000000, abs=1e-6),
            "TOT_PREC6.FAR_gt_0p0": pytest.approx(0.804066, abs=1e-6),
            "TOT_PREC6.FAR_gt_0p1": pytest.approx(0.680339, abs=1e-6),
            "TOT_PREC6.FAR_gt_5p0": pytest.approx(0.492778, abs=1e-6),
            "TOT_PREC6.FAR_gt_10p0": pytest.approx(0.000000, abs=1e-6),
            # TOT_PREC6.FAR_gt_20p0 / FAR_gt_50p0 are NaN — skipped
            "TOT_PREC6.FBI_gt_0p0": pytest.approx(4.846663, abs=1e-6),
            "TOT_PREC6.FBI_gt_0p1": pytest.approx(3.032086, abs=1e-6),
            "TOT_PREC6.FBI_gt_5p0": pytest.approx(1.703571, abs=1e-6),
            "TOT_PREC6.FBI_gt_10p0": pytest.approx(0.488889, abs=1e-6),
            "TOT_PREC6.FBI_gt_20p0": pytest.approx(0.000000, abs=1e-6),
            # TOT_PREC6.FBI_gt_50p0 is NaN — skipped
        },
        {
            "sel": {"region": "mittelland", "season": "all", "init_hour": -999},
            "T_2M.RMSE": pytest.approx(0.828009, abs=1e-6),
            "T_2M.BIAS": pytest.approx(0.057105, abs=1e-6),
            "T_2M.MAE": pytest.approx(0.628740, abs=1e-6),
            "T_2M.CORR": pytest.approx(0.913702, abs=1e-6),
            "TD_2M.RMSE": pytest.approx(0.989164, abs=1e-6),
            "TD_2M.BIAS": pytest.approx(0.129622, abs=1e-6),
            "TD_2M.MAE": pytest.approx(0.764716, abs=1e-6),
            "TD_2M.CORR": pytest.approx(0.537654, abs=1e-6),
            "SP_10M.RMSE": pytest.approx(1.588304, abs=1e-6),
            "SP_10M.BIAS": pytest.approx(0.269079, abs=1e-6),
            "SP_10M.MAE": pytest.approx(1.266803, abs=1e-6),
            "SP_10M.CORR": pytest.approx(0.529139, abs=1e-6),
            "PMSL.RMSE": pytest.approx(37.365382, abs=1e-6),
            "PMSL.BIAS": pytest.approx(14.113738, abs=1e-6),
            "PMSL.MAE": pytest.approx(30.696915, abs=1e-6),
            "PMSL.CORR": pytest.approx(0.964590, abs=1e-6),
            "TOT_PREC6.RMSE": pytest.approx(0.047306, abs=1e-6),
            "TOT_PREC6.BIAS": pytest.approx(0.010906, abs=1e-6),
            "TOT_PREC6.MAE": pytest.approx(0.010906, abs=1e-6),
            # TOT_PREC6.CORR is NaN for this slice — omitted
        },
        {
            "sel": {"region": "berge", "season": "MAM", "init_hour": -999},
            "T_2M.RMSE": pytest.approx(1.953728, abs=1e-6),
            "T_2M.BIAS": pytest.approx(0.109933, abs=1e-6),
            "T_2M.MAE": pytest.approx(1.344072, abs=1e-6),
            "T_2M.CORR": pytest.approx(0.903028, abs=1e-6),
            "TD_2M.RMSE": pytest.approx(2.715898, abs=1e-6),
            "TD_2M.BIAS": pytest.approx(0.966444, abs=1e-6),
            "TD_2M.MAE": pytest.approx(1.841392, abs=1e-6),
            "TD_2M.CORR": pytest.approx(0.927391, abs=1e-6),
            "SP_10M.RMSE": pytest.approx(1.854596, abs=1e-6),
            "SP_10M.BIAS": pytest.approx(-0.735502, abs=1e-6),
            "SP_10M.MAE": pytest.approx(1.304851, abs=1e-6),
            "SP_10M.CORR": pytest.approx(0.692568, abs=1e-6),
            "PMSL.RMSE": pytest.approx(51.910924, abs=1e-6),
            "PMSL.BIAS": pytest.approx(-19.917067, abs=1e-6),
            "PMSL.MAE": pytest.approx(41.043029, abs=1e-6),
            "PMSL.CORR": pytest.approx(0.683732, abs=1e-6),
            "TOT_PREC6.RMSE": pytest.approx(0.451276, abs=1e-6),
            "TOT_PREC6.BIAS": pytest.approx(0.099134, abs=1e-6),
            "TOT_PREC6.MAE": pytest.approx(0.136898, abs=1e-6),
            "TOT_PREC6.CORR": pytest.approx(0.721876, abs=1e-6),
        },
    ],
}



@pytest.mark.heavytest
@pytest.mark.parametrize("config_name", CONFIGS)
def test_experiment_metrics(config_name):
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
    assert nc_files, (
        f"No verif_aggregated_*.nc found in output/data/runs/ for {config_name}"
    )

    ds = xr.open_dataset(nc_files[0])

    run_source = next(
        str(s) for s in ds.coords["source"].values if not str(s).startswith("truth")
    )
    for entry in EXPECTED[config_name]:
        sel = entry["sel"]
        for metric, expected_value in entry.items():
            if metric == "sel":
                continue
            actual = float(ds[metric].sel(source=run_source, **sel).mean("step").values)
            assert actual == expected_value, (
                f"{config_name} {metric} {sel}: got {actual}"
            )
