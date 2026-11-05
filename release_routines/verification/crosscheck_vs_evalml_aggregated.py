"""Recompute the point estimates of verification_flags_*.csv from evalML's own verif_aggregated_f791.nc
(season=all, init_hour=all, region=icon), i.e. what the dashboard shows, and print the relative differences."""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

H = Path(__file__).parent
D = H.parents[1] / "output/data"
M = {
    "RC": "runs/temporal_downscaler-d642-b38b-on-forecaster-07f6-6e42/14ea",
    "Varda 1.0": "runs/temporal_downscaler-sruc-f5d4-on-forecaster-vard-f25f/e5a5",
    "ICON-CH1": "baselines/baseline-b7ef",
    "ICON-CH2": "baselines/baseline-5ed2",
}
agg = {}
for m, s in M.items():
    d = xr.open_dataset(D / s / "verif_aggregated_f791.nc").sel(
        season="all", init_hour=-999, region="icon"
    )
    agg[m] = d.sel(source=next(x for x in d.source.values if not x.startswith("truth")))
csv = pd.read_csv(max(H.glob("verification_flags_*.csv"))).dropna(subset=["lead_h"])
rows = []
for r in csv.itertuples():
    lo, hi = map(int, r.lead_h.split("-"))
    st = slice(np.timedelta64(lo, "h"), np.timedelta64(hi, "h"))
    f = lambda m, sc, v=r.variable, st=st: agg[m][f"{v}.{sc}"].sel(step=st)
    ev = {
        "RMSE_RC": f("RC", "RMSE").mean(),
        "RMSE_base": f(r.baseline, "RMSE").mean(),
        "absBIAS_RC": abs(f("RC", "BIAS")).mean(),
        "absBIAS_base": abs(f(r.baseline, "BIAS")).mean(),
    }
    rows.append(
        dict(
            baseline=r.baseline,
            variable=r.variable,
            **{f"{k}_reldiff": float(v) / getattr(r, k) - 1 for k, v in ev.items()},
        )
    )
df = pd.DataFrame(rows)
with pd.option_context("display.width", 200, "display.float_format", "{:.2e}".format):
    print(df.to_string(index=False))
print("max |rel. diff|:", df.select_dtypes("number").abs().max().max())
