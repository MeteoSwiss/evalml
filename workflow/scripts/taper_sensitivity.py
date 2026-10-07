"""Sensitivity of the 3-6 h FRAC result to the taper, detrend fixed at lsq.

Companion to detrend_sensitivity.py, which found the detrend choice to be a
null. This varies the other axis. Everything else is spectra_by_station.py:
two non-overlapping 48 h windows, matched-sample masking, average-then-ratio.

Configurations:

  boxcar     no taper. The control: shows what the taper is protecting against.
  tukey0.25  weak taper, only the outer 12.5 % of each end is rolled off.
  hann       what the code currently uses.
  blackman   a stronger single taper, lower sidelobes, more data discarded.
  dpss_nw3   multitaper, NW=3, K=5 orthogonal Slepian tapers averaged. Not a
             single-taper choice: the higher tapers carry weight where the first
             one discards data, so it buys degrees of freedom rather than just
             trading leakage against resolution.

Leakage inflates the model and REA-L band power alike, so it may cancel in the
FRAC ratio even where it badly corrupts the absolute power. Both are reported:
`med_FRAC_*` is what the figure shows, `relsd_*` is the across-case scatter of a
single station's band power, which is what multitaper should improve.

    uv run workflow/scripts/taper_sensitivity.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import periodogram, get_window
from scipy.signal.windows import dpss

sys.path.insert(0, str(Path(__file__).parent))
from spectra_forecast import (FORECAST_SOURCES, INTERP_LABEL, OBS_LABEL,
                              TRUTH_LABEL, WINDOW_H, load_season)

WINDOWS = (0, 48)
SEASONS = ["winter2024", "summer2024"]
PARAMS = ["T_2M", "TD_2M", "SP_10M"]

T = WINDOW_H
_t = np.arange(float(T))
_A = np.vander(_t, 2)          # design matrix for the linear detrend


def detrend_lsq(x: np.ndarray) -> np.ndarray:
    """Unweighted least-squares line, as the production code uses."""
    return x - _A @ np.linalg.lstsq(_A, x, rcond=None)[0]


# taper name -> stack of tapers, (K, T). Single tapers are just K=1.
TAPERS = {
    "boxcar":    get_window("boxcar", T)[None],
    "tukey0.25": get_window(("tukey", 0.25), T)[None],
    "hann":      get_window("hann", T)[None],
    "blackman":  get_window("blackman", T)[None],
    "dpss_nw3":  dpss(T, 3.0, Kmax=5),
}


def psd_multi(x: np.ndarray, tapers: np.ndarray) -> np.ndarray:
    """Average the periodogram over the taper stack. x is (time, station)."""
    acc = None
    for w in tapers:
        _, p = periodogram(x, fs=1.0, window=w, detrend=False,
                           scaling="density", axis=0)
        acc = p if acc is None else acc + p
    return acc / len(tapers)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--points", type=Path, default=Path("output/points"))
    ap.add_argument("--truth", type=Path, default=Path("output/points_truth"))
    ap.add_argument("--stations", type=Path,
                    default=Path("resources/smn_grid_points.csv"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("output/spectra_taper_sens"))
    args = ap.parse_args()

    srcs = [OBS_LABEL, TRUTH_LABEL, *FORECAST_SOURCES, INTERP_LABEL]
    st = pd.read_csv(args.stations)
    f = np.fft.rfftfreq(T, d=1.0)
    period = np.divide(1.0, f, out=np.full_like(f, np.inf), where=f > 0)
    band = (period >= 3) & (period <= 6)
    summary = []

    for season in SEASONS:
        for param in PARAMS:
            print(f"\n=== {season} {param} ===", flush=True)
            data, inits = load_season(args.points, args.truth, season, param)
            if OBS_LABEL not in data:
                print("  no observations, skipped")
                continue

            # per-case band power, so across-case scatter can be measured
            cases = {(c, s): [] for c in TAPERS for s in srcs}
            for w in WINDOWS:
                sl = slice(w, w + T)
                ok = np.ones(len(inits), bool)
                for s in srcs:
                    ok &= ~np.isnan(data[s][:, sl]).any(axis=(1, 2))
                print(f"  window {w}-{w+T} h: {ok.sum()}/{len(inits)} inits",
                      flush=True)
                for i in np.flatnonzero(ok):
                    for s in srcs:
                        xd = detrend_lsq(data[s][i, sl])
                        for c, tap in TAPERS.items():
                            cases[(c, s)].append(psd_multi(xd, tap)[band].sum(axis=0))
            n = len(cases[("hann", srcs[0])])
            print(f"  {n} (init, window) cases per station", flush=True)

            for c in TAPERS:
                arr = {s: np.asarray(cases[(c, s)]) for s in srcs}  # (ncase, nstat)
                out = pd.DataFrame({"nat_abbr": st["nat_abbr"],
                                    "elevation": st["elevation"]})
                for s in srcs:
                    out[s] = arr[s].mean(axis=0)
                for s in srcs[1:]:
                    out[f"frac_{s}"] = out[s] / out[OBS_LABEL]
                for s in srcs[2:]:
                    out[f"FRAC_{s}"] = out[s] / out[TRUTH_LABEL]

                d = args.out_dir / c
                d.mkdir(parents=True, exist_ok=True)
                out.to_csv(d / f"by_station_{season}_{param}.csv", index=False)

                row = {"season": season, "param": param, "taper": c, "n": n}
                for s in FORECAST_SOURCES:
                    row[f"med_FRAC_{s}"] = out[f"FRAC_{s}"].median()
                    # across-case relative scatter, median over stations
                    row[f"relsd_{s}"] = np.median(arr[s].std(axis=0)
                                                  / arr[s].mean(axis=0))
                row["med_abs_Varda"] = out[FORECAST_SOURCES[0]].median()
                summary.append(row)

    sm = pd.DataFrame(summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sm.to_csv(args.out_dir / "summary_medians.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n\n=== median FRAC (3-6 h power / REA-L), 145 stations ===")
    for (season, param), g in sm.groupby(["season", "param"], sort=False):
        t = g.set_index("taper")
        o = pd.DataFrame({
            "Varda": t[f"med_FRAC_{FORECAST_SOURCES[0]}"],
            "Multi": t[f"med_FRAC_{FORECAST_SOURCES[1]}"],
            "gap": t[f"med_FRAC_{FORECAST_SOURCES[0]}"] - t[f"med_FRAC_{FORECAST_SOURCES[1]}"],
            "abs_power_Varda": t["med_abs_Varda"],
            "relsd_Varda": t[f"relsd_{FORECAST_SOURCES[0]}"]})
        print(f"\n--- {season} {param} ---")
        print(o.to_string(float_format=lambda x: f"{x:.4g}"))
    print(f"\nwrote {args.out_dir}/")


if __name__ == "__main__":
    main()
