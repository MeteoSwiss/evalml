"""Joint (season, param, init, station) table of band variance and squared error.

Answers the section 12 question: where Varda retains more 3-6 h variance, does
multistep score better or worse? Both sides are measured against REA-L so the
two axes share a truth; the SwissMetNet analogue is a separate run.

Stage 1 writes a tidy long table at full (season, param, init, station)
resolution holding RAW quantities only: the 3-6 h band variance of each source
and the mean squared error of each model against REA-L. No ratios, no
fractions. Section 6 of FINDINGS establishes that single-case retention ratios
are wildly scattered and biased upward, so every ratio is formed in stage 2,
after aggregating over inits.

Stage 2 reads that table and reports, per (season, param) and per station class:

  gap    median over stations of (FRAC_Multi - FRAC_Varda), FRAC against REA-L.
         NOTE the orientation is flipped relative to FINDINGS section 2, so
         that positive means "multistep" on BOTH axes.
  MSSS   Murphy skill score, multistep as candidate, Varda-single as baseline:
         1 - MSE_Multi / MSE_Varda. Positive means multistep is better.

Both are computed from init-averaged quantities, so the ratio rule holds.

`--extra-sources` adds further hourly models (e.g. ICON-CH2-CTRL): their band
variance, errors and covariance are stored alongside, and per-station frac_, r_,
amp_ and pha_ columns are written for them, keyed by `key()` (ICON-CH2-CTRL ->
icon_ch2). gap and msss stay Multistep against Varda-single. They also enter the
matched-sample mask, so pass the same list to every run that is to be compared.

`--scale` (see scales.py) swaps the 3-6 h bandpass for another time-scale
component, on that scale's interior lead times (errors too); the default runs
the original code unchanged.

    uv run --with rdata workflow/scripts/joint_variance_skill.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, detrend
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from gap_robustness import station_classes
from spectra_forecast import (FORECAST_SOURCES, TRUTH_LABEL, load_season)
import scales

SEASONS = ["winter2024", "summer2024"]
PARAMS = ["T_2M", "TD_2M", "SP_10M", "PS"]
NSTEP, GUARD, ORDER, FS = 121, 24, 4, 1.0
BAND = (3, 6)                                # period limits in hours
SCALE, ANOM = "3-6h", False
NBOOT = 2000
OUT = Path("output/joint_variance_skill")
VARDA, MULTI = FORECAST_SOURCES
TABLE = OUT / "table.csv"
KEYS = {VARDA: "varda", MULTI: "multi"}


def key(src: str) -> str:
    return KEYS.get(src) or src.removesuffix("-CTRL").lower().replace("-", "_")


def build(params, seasons, extra=()) -> pd.DataFrame:
    """Stage 1: raw band variance and squared error, per init and station."""
    st = pd.read_csv("resources/smn_grid_points.csv")
    cls = st["nat_abbr"].map(station_classes(
        Path("resources/spaziurat.lf_2.6.1.tar.gz")))
    interior = slice(GUARD, NSTEP - GUARD) if SCALE == "3-6h" else scales.interior(SCALE)
    lo, hi = BAND
    sos = butter(ORDER, [1.0 / hi, 1.0 / lo], btype="band", fs=FS, output="sos")
    rows = []

    for season in seasons:
        for param in params:
            print(f"=== {season} {param} ===", flush=True)
            data, inits = load_season(Path("output/points"),
                                      Path("output/points_truth"), season, param,
                                      tuple(extra))
            if ANOM:
                data = scales.remove_daily_clim(data, inits)
            models = [VARDA, MULTI, *extra]
            srcs = [TRUTH_LABEL, *models]
            missing = [s for s in srcs if s not in data
                       or np.isnan(data[s]).all()]
            if missing:
                print(f"  skipped, no data for {missing}", flush=True)
                continue
            ok = np.ones(len(inits), bool)
            for s in srcs:
                ok &= ~np.isnan(data[s][:, :NSTEP]).any(axis=(1, 2))
            idx = np.flatnonzero(ok)
            print(f"  {len(idx)}/{len(inits)} inits usable", flush=True)

            for i in idx:
                # bandpassed interior series, and its variance, per station
                bpass = ({s: sosfiltfilt(
                              sos, detrend(data[s][i, :NSTEP], axis=0,
                                           type="linear"),
                              axis=0)[interior] for s in srcs} if SCALE == "3-6h" else
                         {s: scales.component(data[s][i, :NSTEP], SCALE) for s in srcs})
                bv = {s: a.var(axis=0) for s, a in bpass.items()}
                # squared error over the SAME interior samples, so the variance
                # and skill sides see identical data
                truth = data[TRUTH_LABEL][i, :NSTEP][interior]
                mse = {s: ((data[s][i, :NSTEP][interior] - truth) ** 2
                           ).mean(axis=0) for s in models}
                # Band-limited error. Full-series MSE is dominated by synoptic
                # scales, so it barely reflects the 3-6 h band the variance
                # axis measures; this puts both axes in the same band. The
                # bandpassed series has ~zero mean, so
                #   MSE_band = (sd_f - sd_t)^2 + 2 sd_f sd_t (1 - r)
                # splitting the error into an AMPLITUDE and a PHASE part. The
                # covariance is carried per case so both terms can be formed
                # after averaging over inits, never per case.
                bt = bpass[TRUTH_LABEL]
                bmse = {s: ((bpass[s] - bt) ** 2).mean(axis=0) for s in models}
                cov = {s: (bpass[s] * bt).mean(axis=0) for s in models}
                rows.append(pd.DataFrame(dict(
                    season=season, param=param, init=inits[i],
                    station=st["nat_abbr"], cls=cls, bvar_truth=bv[TRUTH_LABEL],
                    **{f"{n}_{key(m)}": d[m] for n, d in
                       (("bvar", bv), ("mse", mse), ("bmse", bmse), ("cov", cov))
                       for m in models})))

    df = pd.concat(rows, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE, index=False)
    print(f"\nwrote {TABLE} ({len(df)} rows)")
    return df


def per_station(g: pd.DataFrame) -> pd.DataFrame:
    """Average over inits FIRST, then form the ratios (section 6)."""
    m = g.groupby(["station", "cls"], dropna=False).mean(numeric_only=True)
    sd_t = np.sqrt(m["bvar_truth"])
    keys = [c.removeprefix("bvar_") for c in m.columns
            if c.startswith("bvar_") and c != "bvar_truth"]
    out = dict(
        **{f"frac_{k}": m[f"bvar_{k}"] / m["bvar_truth"] for k in keys},
        # positive = multistep retains more, matching the MSSS orientation
        gap=(m["bvar_multi"] - m["bvar_varda"]) / m["bvar_truth"],
        msss=1.0 - m["mse_multi"] / m["mse_varda"],
        # band-limited skill, same band as the variance axis
        msss_band=1.0 - m["bmse_multi"] / m["bmse_varda"],
    )
    # amplitude/phase split of the band error, normalised by the truth's band
    # variance so the terms are comparable across stations and variables
    for s in keys:
        sd_f = np.sqrt(m[f"bvar_{s}"])
        r = m[f"cov_{s}"] / (sd_f * sd_t)
        out[f"r_{s}"] = r
        out[f"amp_{s}"] = (sd_f - sd_t) ** 2 / m["bvar_truth"]
        out[f"pha_{s}"] = 2 * sd_f * sd_t * (1 - r) / m["bvar_truth"]
    return pd.DataFrame(out).reset_index()


def analyse(df: pd.DataFrame) -> None:
    """Stage 2: the joint cluster, per case and per station class."""
    rng = np.random.default_rng(0)
    case_rows, class_rows = [], []

    for (season, param), g in df.groupby(["season", "param"], sort=False):
        s = per_station(g)
        inits = g["init"].unique()
        boot = np.array([
            per_station(g[g["init"].isin(rng.choice(inits, len(inits)))]
                        )[["gap", "msss"]].median().values
            for _ in range(NBOOT)])
        case_rows.append(dict(
            season=season, param=param, ninit=len(inits), nstation=len(s),
            gap=s["gap"].median(), gap_lo=np.percentile(boot[:, 0], 2.5),
            gap_hi=np.percentile(boot[:, 0], 97.5),
            msss=s["msss"].median(), msss_lo=np.percentile(boot[:, 1], 2.5),
            msss_hi=np.percentile(boot[:, 1], 97.5),
            msss_band=s["msss_band"].median(),
            n_msss_pos=int((s["msss"] > 0).sum()),
            rho_diag=spearmanr(s["gap"], s["msss"]).statistic))
        for c, gc in s.dropna(subset=["cls"]).groupby("cls"):
            class_rows.append(dict(
                season=season, param=param, cls=c, n=len(gc),
                gap=gc["gap"].median(), msss=gc["msss"].median(),
                msss_band=gc["msss_band"].median(),
                r_varda=gc["r_varda"].median(), r_multi=gc["r_multi"].median(),
                amp_varda=gc["amp_varda"].median(),
                amp_multi=gc["amp_multi"].median(),
                pha_varda=gc["pha_varda"].median(),
                pha_multi=gc["pha_multi"].median(),
                rho_diag=spearmanr(gc["gap"], gc["msss"]).statistic))
        s.insert(0, "param", param)
        s.insert(0, "season", season)
        case_rows and s.to_csv(OUT / f"per_station_{season}_{param}.csv",
                               index=False)

    cd = pd.DataFrame(case_rows)
    cl = pd.DataFrame(class_rows)
    cd.to_csv(OUT / "by_case.csv", index=False)
    cl.to_csv(OUT / "by_class.csv", index=False)
    pd.set_option("display.width", 250)

    print("\n########## JOINT VARIANCE AND SKILL, vs REA-L ##########")
    print("gap  = median over stations of (FRAC_Multi - FRAC_Varda).")
    print("       POSITIVE = multistep retains more. This is the OPPOSITE")
    print("       orientation to FINDINGS section 2.")
    print("msss = 1 - MSE_Multi/MSE_Varda, positive = multistep more skilful.")
    print("msss_band = the same on the 3-6 h bandpassed series only.")
    print("rho_diag = Spearman across stations, DIAGNOSTIC ONLY: stations are")
    print("       spatially correlated, so it carries no significance and no")
    print("       p-value is quoted.\n")
    print(cd.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    print("\n########## BY STATION CLASS ##########")
    print("amp/pha = the amplitude and phase parts of the 3-6 h band error,")
    print("       normalised by the truth band variance. amp is what getting")
    print("       the variance wrong costs; pha is what getting the timing")
    print("       wrong costs. pha near 2 means no useful phase information.\n")
    for (season, param), g in cl.groupby(["season", "param"], sort=False):
        print(f"--- {season} {param} ---")
        print(g.set_index("cls")[
            ["n", "gap", "msss", "msss_band", "r_varda", "r_multi",
             "amp_varda", "amp_multi", "pha_varda", "pha_multi", "rho_diag"]
        ].to_string(float_format=lambda x: f"{x:+.3f}"), "\n")
    print(f"wrote {OUT}/")


def main() -> None:
    global OUT, TABLE, SCALE, ANOM
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", nargs="+", default=PARAMS)
    ap.add_argument("--seasons", nargs="+", default=SEASONS)
    ap.add_argument("--reuse", action="store_true",
                    help="skip stage 1 and read the cached table")
    ap.add_argument("--extra-sources", nargs="+", default=[])
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--scale", default="3-6h", choices=scales.SCALES)
    ap.add_argument("--anom", action="store_true",
                    help="remove each source's mean daily cycle first (scales.py)")
    args = ap.parse_args()
    OUT, TABLE, SCALE, ANOM = (args.out_dir, args.out_dir / "table.csv", args.scale,
                               args.anom)
    df = (pd.read_csv(TABLE) if args.reuse and TABLE.exists()
          else build(args.params, args.seasons, args.extra_sources))
    analyse(df)


if __name__ == "__main__":
    main()
