import sys
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).parents[2]
DATA = Path(__file__).parent / "data"
sys.path.insert(0, str(ROOT / "workflow" / "scripts"))
from verification_aggregation import aggregate_results  # noqa: E402

N = 4
INITS = np.array(
    ["2024-01-01T00", "2024-01-02T00", "2024-07-01T12"], dtype="datetime64[h]"
)
DIMS = ["forecast_reference_time", "source", "region", "step"]
SHAPE = (3, 1, 2, 2)
COORDS = {
    "forecast_reference_time": INITS,
    "source": ["run"],
    "region": ["a", "b"],
    "step": [6, 12],
}


def _det_dataset(seed=0):
    rng = np.random.default_rng(seed)
    ds = xr.Dataset(
        {
            "T_2M.BIAS": (DIMS, rng.normal(size=SHAPE)),
            "T_2M.MSE": (DIMS, rng.uniform(1, 2, size=SHAPE)),
            "T_2M.CORR": (DIMS, rng.uniform(0.5, 1, size=SHAPE)),
            "T_2M.var": (DIMS, rng.uniform(1, 2, size=SHAPE)),
            "TOT_PREC.contingency_table": (
                DIMS + ["contingency", "TOT_PREC.threshold"],
                rng.integers(0, 50, size=SHAPE + (5, 2)).astype(float),
            ),
        },
        coords={
            **COORDS,
            "contingency": [
                "tp_count",
                "tn_count",
                "fp_count",
                "fn_count",
                "total_count",
            ],
            "TOT_PREC.threshold": ["gt_0p1", "gt_1p0"],
        },
    )
    return ds


def _ens_dataset():
    rng = np.random.default_rng(1)
    ds = _det_dataset()
    # tables of counts per init: dims (..., TOT_PREC.threshold, n_members_yes, obs)
    table = rng.integers(0, 20, size=SHAPE + (2, N + 1, 2)).astype(float)
    ens_var = rng.uniform(0.5, 2, size=SHAPE)
    ens_var[:, :, :, 0] = 0.0  # zero spread at the first step
    ds["T_2M.ENS_VAR"] = (DIMS, ens_var)
    ds["T_2M.CRPS"] = (DIMS, rng.uniform(0.1, 1, size=SHAPE))
    ds["T_2M.MEMBER_MSE"] = (DIMS, rng.uniform(1, 3, size=SHAPE))
    ds["T_2M.rank_hist"] = (
        DIMS + ["rank"],
        rng.integers(0, 30, size=SHAPE + (N + 1,)).astype(float),
    )
    ds["TOT_PREC.prob_table"] = (
        DIMS + ["TOT_PREC.threshold", "n_members_yes", "obs"],
        table,
    )
    ds = ds.assign_coords(
        {"rank": np.arange(N + 1), "n_members_yes": np.arange(N + 1), "obs": [0, 1]}
    )
    ds.attrs["n_members"] = N
    return ds


def test_deterministic_output_unchanged():
    """Deterministic inputs give the output of the code before ensemble support.

    ``data/aggregation_golden_deterministic.nc`` was written by
    ``aggregate_results(_det_dataset())`` before the ensemble scores were added.
    """
    expected = xr.load_dataset(DATA / "aggregation_golden_deterministic.nc")
    result = aggregate_results(_det_dataset())
    xr.testing.assert_identical(result, expected)
    for var in expected.variables:
        if expected[var].dtype.kind in "fiub":
            np.testing.assert_array_equal(result[var].values, expected[var].values)


def test_ensemble_spread_ssr_and_means():
    ds = _ens_dataset()
    out = aggregate_results(ds)
    allsel = dict(season="all", init_hour=-999)
    sel = out.sel(**allsel)
    assert "T_2M.ENS_VAR" not in out
    assert "T_2M.MEMBER_MSE" not in out and "T_2M.MEMBER_RMSE" in out
    assert "T_2M.ENS_std" not in out
    np.testing.assert_allclose(
        sel["T_2M.CRPS"], ds["T_2M.CRPS"].mean("forecast_reference_time")
    )
    exp_member = np.sqrt(ds["T_2M.MEMBER_MSE"].mean("forecast_reference_time"))
    np.testing.assert_allclose(sel["T_2M.MEMBER_RMSE"], exp_member)
    spread = np.sqrt(ds["T_2M.ENS_VAR"].mean("forecast_reference_time"))
    rmse = np.sqrt(ds["T_2M.MSE"].mean("forecast_reference_time"))
    np.testing.assert_allclose(sel["T_2M.SPREAD"], spread)
    exp_ssr = (spread / rmse * np.sqrt((N + 1) / N)).where(spread != 0)
    np.testing.assert_allclose(sel["T_2M.SSR"], exp_ssr)
    # zero spread at first step -> NaN SSR, SPREAD is 0
    assert np.isnan(sel["T_2M.SSR"].isel(step=0)).all()
    assert (sel["T_2M.SPREAD"].isel(step=0) == 0).all()
    assert out.attrs["n_members"] == N


def test_ssr_nan_where_rmse_zero():
    ds = _ens_dataset()
    ds["T_2M.MSE"] = ds["T_2M.MSE"] * 0
    ds["T_2M.BIAS"] = ds["T_2M.BIAS"] * 0
    out = aggregate_results(ds)
    assert np.isnan(out["T_2M.SSR"]).all()


def test_ssr_n_inferred_from_rank():
    ds = _ens_dataset()
    ds.attrs.pop("n_members")
    out = aggregate_results(ds)
    ref = aggregate_results(_ens_dataset())
    xr.testing.assert_allclose(out["T_2M.SSR"], ref["T_2M.SSR"])
    assert out.attrs["n_members"] == N


def test_counts_conversion_with_stratification():
    ds = _ens_dataset()
    out = aggregate_results(ds)
    # rank_hist: sum over inits of the per-init counts, in every stratum
    total = ds["T_2M.rank_hist"].sum("forecast_reference_time")
    np.testing.assert_allclose(
        out.sel(season="all", init_hour=-999)["T_2M.rank_hist"], total
    )
    djf = (
        ds["T_2M.rank_hist"]
        .isel(forecast_reference_time=[0, 1])
        .sum("forecast_reference_time")
    )
    np.testing.assert_allclose(
        out.sel(season="DJF", init_hour=-999)["T_2M.rank_hist"], djf
    )
    jja_12 = ds["T_2M.rank_hist"].isel(forecast_reference_time=2)
    np.testing.assert_allclose(
        out.sel(season="JJA", init_hour=12)["T_2M.rank_hist"], jja_12
    )
    # prob_table split per threshold, dims n_members_yes, obs
    pt = out["TOT_PREC.prob_table_gt_0p1"]
    assert set(pt.dims) >= {"n_members_yes", "obs"}
    exp = (
        ds["TOT_PREC.prob_table"]
        .isel({"TOT_PREC.threshold": 0})
        .sum("forecast_reference_time")
    )
    np.testing.assert_allclose(pt.sel(season="all", init_hour=-999), exp)
    assert "TOT_PREC.prob_table" not in out


def test_brier_and_murphy_decomposition():
    ds = _ens_dataset()
    out = aggregate_results(ds).sel(season="all", init_hour=-999)
    counts = ds["TOT_PREC.prob_table"].sum("forecast_reference_time")
    for i, label in enumerate(["gt_0p1", "gt_1p0"]):
        c = (
            counts.isel({"TOT_PREC.threshold": i})
            .transpose(..., "n_members_yes", "obs")
            .values
        )
        t = c.sum(axis=(-1, -2))
        f = np.arange(N + 1) / N
        n1, n0 = c[..., 1], c[..., 0]
        nk = n1 + n0
        bs = ((n1 * (f - 1) ** 2 + n0 * f**2).sum(-1)) / t
        obar = n1.sum(-1) / t
        ok = n1 / nk
        rel = (nk * (f - ok) ** 2).sum(-1) / t
        res = (nk * (ok - obar[..., None]) ** 2).sum(-1) / t
        unc = obar * (1 - obar)
        got = {
            k: out[f"TOT_PREC.{k}_{label}"].transpose("source", "region", "step").values
            for k in ["BS", "BS_REL", "BS_RES", "BS_UNC"]
        }
        for k, ref in zip(["BS", "BS_REL", "BS_RES", "BS_UNC"], [bs, rel, res, unc]):
            np.testing.assert_allclose(got[k], ref[0] if ref.ndim == 4 else ref)
        np.testing.assert_allclose(
            got["BS"], got["BS_REL"] - got["BS_RES"] + got["BS_UNC"]
        )


def test_brier_hand_example():
    # one init pair, N=2; bins k=0,1,2 with (obs0, obs1) counts
    ds = _ens_dataset().isel(forecast_reference_time=[0], region=[0], step=[0])
    tab = np.zeros((1, 1, 1, 1, 2, 3, 2))
    tab[..., 0, :, :] = [[6, 0], [1, 1], [0, 2]]  # k=0: 6 no,0 yes; k=1: 1,1; k=2: 0,2
    tab[..., 1, :, :] = tab[..., 0, :, :]
    ds = ds.isel(rank=[0, 1, 2], n_members_yes=[0, 1, 2])
    ds["TOT_PREC.prob_table"] = (ds["TOT_PREC.prob_table"].dims, tab)
    ds.attrs["n_members"] = 2
    out = (
        aggregate_results(ds)
        .sel(season="all", init_hour=-999)
        .isel(source=0, region=0, step=0)
    )
    # T=10, f=(0,.5,1): BS = (0.25*(1+1) + 0)/10 = 0.05 ; obar = 3/10
    assert np.isclose(out["TOT_PREC.BS_gt_0p1"], 0.05)
    assert np.isclose(out["TOT_PREC.BS_UNC_gt_0p1"], 0.21)
    # REL = (2*(0.5-0.5)^2 + 0)/10 = 0 ; RES = (6*.09 + 2*.04 + 2*.49)/10 = 0.16
    assert np.isclose(out["TOT_PREC.BS_REL_gt_0p1"], 0.0)
    assert np.isclose(out["TOT_PREC.BS_RES_gt_0p1"], 0.16)
