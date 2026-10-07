"""Sensitivity of the 3-6 h FRAC result to the detrend choice, Hann taper fixed.

Repeats spectra_by_station.py exactly (two non-overlapping 48 h windows,
matched-sample masking, average-then-ratio) but swaps the detrend. The taper is
held at Hann throughout, so this isolates the endpoint/trend handling alone.

Variants, all fitted per station and subtracted before the taper:

  none        no detrend
  mean        subtract the mean only, isolating offset from slope
  lsq         unweighted least-squares line. What the code currently does.
  wls_p2      weighted least-squares line, w = |s|^2, s in [-1,1]
  wls_p6      the same, w = |s|^6, pushing the fit harder toward the ends
  endpoint    the line through the first and last sample; zero wrap seam
  taper2      weighted least squares, w = hann^2, the weighting that minimises
              the energy of the *tapered* residual, i.e. the opposite emphasis
  quad        unweighted least-squares quadratic, so curvature can be removed

Writes one CSV per (variant, season, param) in the layout
plot_frac_boxplots_overview.py expects, so the real plotting script can be run
against each variant directory to reproduce the overview figure.

    uv run workflow/scripts/detrend_sensitivity.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import periodogram, get_window

sys.path.insert(0, str(Path(__file__).parent))
from spectra_forecast import (FORECAST_SOURCES, INTERP_LABEL, OBS_LABEL,
                              TRUTH_LABEL, WINDOW_H, load_season)

WINDOWS = (0, 48)  # non-overlapping, as in spectra_by_station.py
SEASONS = ["winter2024", "summer2024"]
PARAMS = ["T_2M", "TD_2M", "SP_10M"]

T = WINDOW_H
_t = np.arange(float(T))
_s = 2 * _t / (T - 1) - 1.0          # normalised position, -1 .. +1
_hann = get_window("hann", T)


def _wls(x: np.ndarray, deg: int, w: np.ndarray) -> np.ndarray:
    """Subtract a degree-`deg` polynomial fitted with weights `w`.

    x is (time, station); every station is fitted at once.
    """
    A = np.vander(_t, deg + 1)
    r = np.sqrt(w)[:, None]
    b = np.linalg.lstsq(A * r, x * r, rcond=None)[0]
    return x - A @ b


def dt_none(x):     return x
def dt_mean(x):     return x - x.mean(axis=0)
def dt_lsq(x):      return _wls(x, 1, np.ones(T))
def dt_wls_p2(x):   return _wls(x, 1, np.abs(_s) ** 2 + 1e-6)
def dt_wls_p6(x):   return _wls(x, 1, np.abs(_s) ** 6 + 1e-6)
def dt_endpoint(x): return x - (x[0] + np.outer(_t / (T - 1), x[-1] - x[0]))
def dt_taper2(x):   return _wls(x, 1, _hann ** 2 + 1e-6)
def dt_quad(x):     return _wls(x, 2, np.ones(T))


DETRENDS = {"none": dt_none, "mean": dt_mean, "lsq": dt_lsq,
            "wls_p2": dt_wls_p2, "wls_p6": dt_wls_p6, "endpoint": dt_endpoint,
            "taper2": dt_taper2, "quad": dt_quad}


def psd_hann(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hann taper, no scipy detrend: detrending is done by the variant."""
    return periodogram(x, fs=1.0, window="hann", detrend=False,
                       scaling="density", axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--points", type=Path, default=Path("output/points"))
    ap.add_argument("--truth", type=Path, default=Path("output/points_truth"))
    ap.add_argument("--stations", type=Path,
                    default=Path("resources/smn_grid_points.csv"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("output/spectra_detrend_sens"))
    args = ap.parse_args()

    srcs = [OBS_LABEL, TRUTH_LABEL, *FORECAST_SOURCES, INTERP_LABEL]
    st = pd.read_csv(args.stations)
    summary = []

    for season in SEASONS:
        for param in PARAMS:
            print(f"\n=== {season} {param} ===", flush=True)
            data, inits = load_season(args.points, args.truth, season, param)
            if OBS_LABEL not in data:
                print("  no observations, skipped")
                continue

            nstat = data[OBS_LABEL].shape[2]
            acc = {(v, s): np.zeros((T // 2 + 1, nstat))
                   for v in DETRENDS for s in srcs}
            n = 0
            for w in WINDOWS:
                sl = slice(w, w + T)
                ok = np.ones(len(inits), bool)
                for s in srcs:
                    ok &= ~np.isnan(data[s][:, sl]).any(axis=(1, 2))
                print(f"  window {w}-{w+T} h: {ok.sum()}/{len(inits)} inits",
                      flush=True)
                for i in np.flatnonzero(ok):
                    for s in srcs:
                        x = data[s][i, sl]
                        for v, fn in DETRENDS.items():
                            acc[(v, s)] += psd_hann(fn(x))[1]
                    n += 1
            print(f"  {n} (init, window) cases per station", flush=True)

            f = psd_hann(data[OBS_LABEL][0, 0:T])[0]
            period = np.divide(1.0, f, out=np.full_like(f, np.inf), where=f > 0)
            band = (period >= 3) & (period <= 6)

            for v in DETRENDS:
                out = pd.DataFrame({"nat_abbr": st["nat_abbr"],
                                    "elevation": st["elevation"]})
                for s in srcs:
                    out[s] = (acc[(v, s)] / n)[band].sum(axis=0)
                for s in srcs[1:]:
                    out[f"frac_{s}"] = out[s] / out[OBS_LABEL]
                for s in srcs[2:]:
                    out[f"FRAC_{s}"] = out[s] / out[TRUTH_LABEL]

                d = args.out_dir / v
                d.mkdir(parents=True, exist_ok=True)
                out.to_csv(d / f"by_station_{season}_{param}.csv", index=False)

                row = {"season": season, "param": param, "detrend": v, "n": n}
                for s in FORECAST_SOURCES:
                    row[f"med_FRAC_{s}"] = out[f"FRAC_{s}"].median()
                row["med_frac_REA-L"] = out["frac_REA-L"].median()
                summary.append(row)

    sm = pd.DataFrame(summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sm.to_csv(args.out_dir / "summary_medians.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n\n=== median FRAC (3-6 h power / REA-L) across 145 stations ===")
    for (season, param), g in sm.groupby(["season", "param"], sort=False):
        print(f"\n--- {season} {param} ---")
        t = g.set_index("detrend")[[f"med_FRAC_{s}" for s in FORECAST_SOURCES]]
        t.columns = ["Varda", "Multistep"]
        t["Varda-Multi"] = t["Varda"] - t["Multistep"]
        print(t.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nwrote {args.out_dir}/")


if __name__ == "__main__":
    main()
