"""Falsification check of the release candidate: overall min / max / mean (region 'icon', lead time
+0 h excluded) per model from the per-forecast verif files, flagged against physical limits."""

from datetime import datetime
from pathlib import Path

import pandas as pd
import xarray as xr
import yaml

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "output/data"
MODELS = {  # first one is the release candidate; KENDA is taken from its files (source 'truth-*')
    "RC": "runs/temporal_downscaler-d642-b38b-on-forecaster-07f6-6e42/14ea",
    "Varda 1.0": "runs/temporal_downscaler-sruc-f5d4-on-forecaster-vard-f25f/e5a5",
    "ICON-CH1": "baselines/baseline-b7ef",
    "ICON-CH2": "baselines/baseline-5ed2",
}
COLUMNS = ["RC", "Varda 1.0", "ICON-CH1", "ICON-CH2", "KENDA"]
cfg = yaml.safe_load((HERE / "falsification_thresholds.yaml").read_text())
VARS = cfg["variables"]


def collect(sub, with_truth):
    files = sorted((DATA / sub).glob("20*/verif_f791.nc"))
    print(f"{sub}: {len(files)} files", flush=True)
    per_fc = []
    for f in files:
        with xr.open_dataset(f) as ds:
            keep = [
                v for v in ds.data_vars if v.split(".")[-1] in ("min", "max", "mean")
            ]
            per_fc.append(ds[keep].sel(region="icon").isel(step=slice(1, None)).load())
    ds = xr.concat(per_fc, dim="forecast_reference_time")
    out = {}
    for src in ds.source.values:
        is_truth = str(src).startswith("truth")
        if is_truth and not with_truth:
            continue
        d = ds.sel(source=src)
        out["KENDA" if is_truth else None] = {
            var: {
                "min": float(d[f"{var}.min"].min()),
                "max": float(d[f"{var}.max"].max()),
                "mean": float(d[f"{var}.mean"].mean()),
                "mean_lo": float(d[f"{var}.mean"].min()),
                "mean_hi": float(d[f"{var}.mean"].max()),
            }
            for var in {v.split(".")[0] for v in d.data_vars}
        }
    return out


raw = {}
for i, (model, sub) in enumerate(MODELS.items()):
    res = collect(sub, with_truth=i == 0)
    raw[model] = res.pop(None)
    raw.update(res)
pd.DataFrame(
    [dict(model=m, variable=v, **s) for m, d in raw.items() for v, s in d.items()]
).to_csv(HERE / "falsification_extremes_raw_no_step0.csv", index=False)


def get(model, var, stat):
    c = VARS[var]
    if (
        model in cfg["not_available"].get(var, [])
        or (s := raw[model].get(c.get("source_name", var))) is None
    ):
        return None
    return s[stat] * c.get("scale", 1) + c.get("offset", 0)


fmt = lambda x: "NA" if x is None else f"{x:.1f}"
rows = []
for cat, stat in [("Maxima", "max"), ("Mean", "mean"), ("Minima", "min")]:
    for var, c in VARS.items():
        u = c["units"]
        if stat == "mean":
            lo, hi = get("RC", var, "mean_lo"), get("RC", var, "mean_hi")
            ok = c["mean"][0] <= lo and hi <= c["mean"][1]
            comment = f"hourly domain means {lo:.1f} to {hi:.1f} {u}, limits {c['mean'][0]} to {c['mean'][1]} {u}"
        else:
            x, lim = get("RC", var, stat), c[stat]
            ok = x <= lim if stat == "max" else x >= lim
            comment = f"limit {'≤' if stat == 'max' else '≥'} {lim} {u}"
        o = (
            cfg.get("overrides", {}).get(cat, {}).get(var, {})
        )  # flags / comments set by hand after inspecting the cases
        comment = f"{comment}. {o['comment']}" if "comment" in o else comment
        rows.append(
            {
                "Category": cat,
                "Variable": var,
                "Unit": u,
                "Flag": o.get("flag", "OK" if ok else "deficient"),
                **{m: fmt(get(m, var, stat)) for m in COLUMNS},
                "Comment": comment,
            }
        )

table = pd.DataFrame(rows)
stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
table.to_csv(HERE / f"falsification_table_{stamp}.csv", index=False)
print(table.to_string(index=False))
