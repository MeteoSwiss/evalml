"""Band edges, bootstrap significance and station-class structure, in one pass.

Covers the three remaining knobs from the FINDINGS section 8 robustness work:

  band edges   3-6 h against 4-6, 3-8 and 4-8 h. The 3 h edge sits next to the
               alias-contaminated bins (section 5), so it is a judgement call.
  bootstrap    paired, resampling INITS with replacement. Inits are 54 h apart
               and are the closest thing to independent replicates here.
               Stations are spatially correlated, so resampling them would
               understate uncertainty badly; they are held fixed.
  classes      the `T2m_based_class` cut, to settle whether multistep really
               leads within classes while trailing overall (section 4).

Estimation is the time-domain bandpass throughout: it needs no taper and no
periodic extension, and section 8 showed it agrees with the Hann periodogram
while being the cleaner of the two.

The comparison is PAIRED by station. Section 8 established that comparing the
two marginal medians can differ from the median per-station difference by a
factor of two, and can vanish entirely (summer T_2M). The headline statistic
here is therefore the median over stations of (FRAC_Varda - FRAC_Multistep).

`--other` replaces Multistep as the model Varda-single is compared against
(the columns keep their _M names), and `--extra-sources` loads further sources
into the matched-sample mask without comparing them, so that several comparisons
share one sample. Each run seeds its own bootstrap, so the Multistep numbers are
reproducible whatever else is compared.

`--scale` (see scales.py) replaces the 3-6 h bandpass by another time-scale
component; the band-edge variants are then skipped and the band column holds
the scale name. The default, 3-6h, runs exactly the original code.

    uv run --with rdata workflow/scripts/gap_robustness.py
"""

import sys, tarfile, tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, detrend
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
from spectra_forecast import (FORECAST_SOURCES, INTERP_LABEL, OBS_LABEL,
                              TRUTH_LABEL, load_season)
from scales import SCALES, component, remove_daily_clim

SEASONS = ["winter2024", "summer2024"]
PARAMS = ["T_2M", "TD_2M", "SP_10M"]
# Observations exist only for T_2M, TD_2M and SP_10M. For anything else the
# comparison is model-to-REA-L only, which is all the paired gap needs.
NSTEP, GUARD, ORDER, FS = 121, 24, 4, 1.0
BANDS = [(3, 6), (4, 6), (3, 8), (4, 8)]     # period limits in hours
NBOOT = 2000
OUT = Path("output/gap_robustness")
VARDA, MULTI = FORECAST_SOURCES


def station_classes(tarball: Path) -> dict:
    import rdata
    member = "spaziurat.lf/data/stat.class.info.rda"
    with tarfile.open(tarball) as t, tempfile.TemporaryDirectory() as tmp:
        t.extract(t.getmember(member), tmp)
        d = rdata.conversion.convert(
            rdata.parser.parse_file(Path(tmp) / member))["stat.class.info"]
    return dict(zip(d["nat_abbr"], d["T2m_based_class"]))


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", nargs="+", default=PARAMS)
    ap.add_argument("--seasons", nargs="+", default=SEASONS)
    ap.add_argument("--tag", default="", help="suffix for the output CSVs")
    ap.add_argument("--other", default=MULTI, help="model compared against Varda")
    ap.add_argument("--extra-sources", nargs="+", default=[])
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--scale", default="3-6h", choices=SCALES)
    ap.add_argument("--anom", action="store_true",
                    help="remove each source's mean daily cycle first (scales.py)")
    args = ap.parse_args()
    other, out = args.other, args.out_dir
    st = pd.read_csv("resources/smn_grid_points.csv")
    cls = st["nat_abbr"].map(station_classes(
        Path("resources/spaziurat.lf_2.6.1.tar.gz")))
    interior = slice(GUARD, NSTEP - GUARD)
    rng = np.random.default_rng(0)
    out.mkdir(parents=True, exist_ok=True)
    band_rows, class_rows, station_rows = [], [], []

    for season in args.seasons:
        for param in args.params:
            print(f"=== {season} {param} ===", flush=True)
            extra = tuple(dict.fromkeys(e for e in (other, *args.extra_sources)
                                        if e not in FORECAST_SOURCES))
            data, inits = load_season(Path("output/points"),
                                      Path("output/points_truth"), season, param, extra)
            if args.anom:
                data = remove_daily_clim(data, inits)
            # observations are optional: only the model/REA-L ratio is needed
            srcs = [s for s in [OBS_LABEL, TRUTH_LABEL, VARDA, MULTI,
                                INTERP_LABEL, *extra]
                    if s in data and not np.isnan(data[s]).all()]
            if OBS_LABEL not in srcs:
                print("  no observations; FRAC (vs REA-L) only", flush=True)
            ok = np.ones(len(inits), bool)
            for s in srcs:
                ok &= ~np.isnan(data[s][:, :NSTEP]).any(axis=(1, 2))
            idx = np.flatnonzero(ok)
            ninit = len(idx)
            print(f"  {ninit}/{len(inits)} inits usable", flush=True)

            for band in (BANDS if args.scale == "3-6h" else [args.scale]):
                if args.scale == "3-6h":
                    lo, hi = band
                    name = f"{lo}-{hi}h"
                    sos = butter(ORDER, [1.0 / hi, 1.0 / lo], btype="band",
                                 fs=FS, output="sos")
                    # per-init, per-station band variance
                    v = {s: np.stack([
                            sosfiltfilt(sos, detrend(data[s][i, :NSTEP], axis=0,
                                                     type="linear"),
                                        axis=0)[interior].var(axis=0)
                            for i in idx]) for s in srcs}
                else:
                    name = band
                    v = {s: np.stack([component(data[s][i, :NSTEP], band).var(axis=0)
                                      for i in idx]) for s in srcs}

                def stat(sel):
                    """median over stations of the paired FRAC difference."""
                    t = v[TRUTH_LABEL][sel].mean(axis=0)
                    return (v[VARDA][sel].mean(axis=0) / t
                            - v[other][sel].mean(axis=0) / t)

                d_all = stat(np.arange(ninit))
                boot = np.array([stat(rng.integers(0, ninit, ninit))
                                 for _ in range(NBOOT)])
                med = np.median(boot, axis=1)
                lo_ci, hi_ci = np.percentile(med, [2.5, 97.5])
                fV = v[VARDA].mean(axis=0) / v[TRUTH_LABEL].mean(axis=0)
                fM = v[other].mean(axis=0) / v[TRUTH_LABEL].mean(axis=0)

                band_rows.append(dict(
                    season=season, param=param, band=name, ninit=ninit,
                    FRAC_V=np.median(fV), FRAC_M=np.median(fM),
                    gap=np.median(d_all), ci_lo=lo_ci, ci_hi=hi_ci,
                    frac_boot_pos=float((med > 0).mean()),
                    n_V_gt_M=int((d_all > 0).sum()),
                    wilcoxon_p=wilcoxon(fV, fM).pvalue))

                if name == args.scale:
                    station_rows.append(pd.DataFrame(dict(
                        season=season, param=param, nat_abbr=st["nat_abbr"],
                        cls=cls, gap=d_all)))
                    for c, g in pd.DataFrame(
                            {"cls": cls, "d": d_all, "fV": fV, "fM": fM}
                            ).dropna(subset=["cls"]).groupby("cls"):
                        # same resamples as the pooled CI, median within class
                        cmed = np.median(boot[:, g.index], axis=1)
                        class_rows.append(dict(
                            season=season, param=param, cls=c, n=len(g),
                            FRAC_V=g["fV"].median(), FRAC_M=g["fM"].median(),
                            gap=g["d"].median(),
                            ci_lo=np.percentile(cmed, 2.5),
                            ci_hi=np.percentile(cmed, 97.5),
                            frac_boot_pos=float((cmed > 0).mean()),
                            n_V_gt_M=int((g["d"] > 0).sum())))

    bd = pd.DataFrame(band_rows)
    bd.to_csv(out / f"band_edges{args.tag}.csv", index=False)
    cd = pd.DataFrame(class_rows)
    cd.to_csv(out / f"by_class{args.tag}.csv", index=False)
    pd.concat(station_rows).to_csv(out / f"per_station{args.tag}.csv",
                                   index=False)
    pd.set_option("display.width", 250)

    print("\n\n########## 2+3. BAND EDGES AND BOOTSTRAP ##########")
    print("gap = median over stations of (FRAC_Varda - FRAC_Multistep).")
    print("positive = Varda retains more. CI = 95 % paired bootstrap over inits.\n")
    for (season, param), g in bd.groupby(["season", "param"], sort=False):
        print(f"--- {season} {param} ---")
        t = g.set_index("band")[["FRAC_V", "FRAC_M", "gap", "ci_lo", "ci_hi",
                                 "frac_boot_pos", "n_V_gt_M", "wilcoxon_p"]]
        print(t.to_string(float_format=lambda x: f"{x:.4g}"), "\n")

    print("\n########## 4. STATION CLASSES (3-6 h) ##########")
    print("gap = median paired difference within the class.\n")
    for (season, param), g in cd.groupby(["season", "param"], sort=False):
        print(f"--- {season} {param} ---")
        print(g.set_index("cls")[["n", "FRAC_V", "FRAC_M", "gap", "ci_lo", "ci_hi",
                                    "n_V_gt_M"]]
              .to_string(float_format=lambda x: f"{x:.4g}"), "\n")

    print("\n--- class summary: sign of the gap, by case ---")
    piv = cd.pivot_table(index="cls", columns=["season", "param"], values="gap")
    print(piv.to_string(float_format=lambda x: f"{x:+.4f}"))
    print("\nnegative = multistep retains more in that class/case")
    print(f"\nwrote {out}/ (Varda-single vs {other})")


if __name__ == "__main__":
    main()
