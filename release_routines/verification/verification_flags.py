"""Verification flags of the release candidate (RC) per baseline, region 'icon', truth KENDA-CH1.

Per variable and baseline, on the lead times 1 h .. baseline max (where both models have values):
  RMSE score = mean over lead times l of sqrt(mean over forecasts of MSE[i, l]);  compared as ratio RC / baseline
  BIAS score = mean over lead times l of |mean over forecasts of BIAS[i, l]|;    compared as difference RC - baseline
95 % range: paired circular block bootstrap (blocks of 3 consecutive forecasts, inits are 42 h apart), B rounds.
Flag (lo = lower bound of the range, "no change" = 1 for the ratio, 0 for the difference):
  OK if lo <= no change; marginal if no change < lo <= tolerance; deficient if lo > tolerance.
Tolerances: verification_tolerances.yaml.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "output/data"
MODELS = {
    "RC": "runs/temporal_downscaler-d642-b38b-on-forecaster-07f6-6e42/14ea",
    "Varda 1.0": "runs/temporal_downscaler-sruc-f5d4-on-forecaster-vard-f25f/e5a5",
    "ICON-CH1": "baselines/baseline-b7ef",
    "ICON-CH2": "baselines/baseline-5ed2",
}
VARS = [
    "T_2M",
    "TD_2M",
    "PMSL",
    "SP_10M",
    "TOT_PREC1",
    "TOT_PREC6",
    "CLCT",
    "CLCL",
    "CLCM",
    "CLCH",
]
SCALE = {"PMSL": 0.01}  # Pa -> hPa (applied to BIAS, squared to MSE)
B, BLOCK, SEED = 10_000, 3, 0
TOL = yaml.safe_load((HERE / "verification_tolerances.yaml").read_text())


def load(sub):
    """-> {score: DataArray(init, step, vname)} with init = directory name (YYYYMMDDHHMM)."""
    fs = sorted((DATA / sub).glob("20*/verif_f791.nc"))
    out = {"MSE": [], "BIAS": []}
    for f in fs:
        with xr.open_dataset(f) as ds:
            ds = ds.sel(region="icon")
            src = next(s for s in ds.source.values if not s.startswith("truth"))
            for sc, arrays in out.items():
                vs = [v for v in VARS if f"{v}.{sc}" in ds]
                a = xr.concat(
                    [
                        ds[f"{v}.{sc}"]
                        .sel(source=src)
                        .squeeze("forecast_reference_time", drop=True)
                        * SCALE.get(v, 1) ** (2 if sc == "MSE" else 1)
                        for v in vs
                    ],
                    pd.Index(vs, name="vname"),
                )
                arrays.append(
                    a.drop_vars(
                        [c for c in a.coords if c not in ("step", "vname")]
                    ).load()
                )
    inits = pd.Index([f.parent.name for f in fs], name="init")
    return {sc: xr.concat(a, inits) for sc, a in out.items()}


def counts(n, rng):
    """(B, n) matrix: how often each forecast is drawn in each bootstrap round (circular overlapping blocks)."""
    idx = (
        (rng.integers(0, n, (B, -(-n // BLOCK)))[:, :, None] + np.arange(BLOCK)) % n
    ).reshape(B, -1)[:, :n]
    return np.stack([np.bincount(r, minlength=n) for r in idx])


def score(C, mse, bias, valid):
    """C (k, n) weights; mse/bias/valid (n, L) -> RMSE and |BIAS| scores, each (k,)."""
    w = C @ valid
    return np.sqrt(C @ mse / w).mean(1), np.abs(C @ bias / w).mean(1)


def flag(lo, none, tol):
    return "OK" if lo <= none else "marginal" if lo <= tol else "deficient"


data = {m: load(s) for m, s in MODELS.items()}
rc = data["RC"]
rng = np.random.default_rng(SEED)
rows = []
for base in ["Varda 1.0", "ICON-CH1", "ICON-CH2"]:
    bd = data[base]
    inits = np.intersect1d(rc["MSE"].init, bd["MSE"].init)  # sorted = chronological
    steps = bd["MSE"].step.values
    steps = steps[steps > np.timedelta64(0)]
    C = counts(len(inits), rng)
    for v in VARS + ["SKT"]:  # SKT: no baseline (ICON T_G is ground temperature) -> NA
        if v not in rc["MSE"].vname.values or v not in bd["MSE"].vname.values:
            rows.append(
                {"baseline": base, "variable": v, "RMSE_flag": "NA", "BIAS_flag": "NA"}
            )
            continue
        m = {
            k: [d[k].sel(init=inits, step=steps, vname=v).values for d in (rc, bd)]
            for k in ("MSE", "BIAS")
        }
        valid = np.isfinite(np.stack(m["MSE"] + m["BIAS"])).all(0)
        keep = valid.any(
            0
        )  # drop lead times without any value (e.g. TOT_PREC6 below +6 h)
        valid = valid[:, keep].astype(float)
        z = {k: [np.where(valid > 0, x[:, keep], 0.0) for x in p] for k, p in m.items()}
        (r_rc, b_rc), (r_bl, b_bl) = [
            [
                s[0]
                for s in score(
                    np.ones((1, len(inits))), z["MSE"][j], z["BIAS"][j], valid
                )
            ]
            for j in (0, 1)
        ]
        (R_rc, A_rc), (R_bl, A_bl) = [
            score(C, z["MSE"][j], z["BIAS"][j], valid) for j in (0, 1)
        ]
        r_lo, r_hi = np.percentile(R_rc / R_bl, [2.5, 97.5])
        b_lo, b_hi = np.percentile(A_rc - A_bl, [2.5, 97.5])
        h = (steps[keep] / np.timedelta64(1, "h")).astype(int)
        rows.append(
            {
                "baseline": base,
                "variable": v,
                "unit": TOL["BIAS"][v]["unit"],
                "n_forecasts": len(inits),
                "lead_h": f"{h[0]}-{h[-1]}",
                "RMSE_RC": r_rc,
                "RMSE_base": r_bl,
                "RMSE_ratio": r_rc / r_bl,
                "RMSE_lo": r_lo,
                "RMSE_hi": r_hi,
                "RMSE_tol": TOL["RMSE_ratio"],
                "RMSE_flag": flag(r_lo, 1, TOL["RMSE_ratio"]),
                "absBIAS_RC": b_rc,
                "absBIAS_base": b_bl,
                "BIAS_diff": b_rc - b_bl,
                "BIAS_lo": b_lo,
                "BIAS_hi": b_hi,
                "BIAS_tol": TOL["BIAS"][v]["tol"],
                "BIAS_flag": flag(b_lo, 0, TOL["BIAS"][v]["tol"]),
            }
        )

df = pd.DataFrame(rows)
out = HERE / f"verification_flags_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv"
df.to_csv(out, index=False, float_format="%.4g")
with pd.option_context(
    "display.width",
    250,
    "display.max_columns",
    30,
    "display.float_format",
    "{:.3g}".format,
):
    print(df.drop(columns=["n_forecasts", "RMSE_tol"]).to_string(index=False))
print(f"-> {out}")
