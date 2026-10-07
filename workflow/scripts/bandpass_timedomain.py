"""Independent check on the taper question: 3-6 h variance by time-domain filtering.

Every configuration in taper_sensitivity.py estimates the band power through a
periodogram, so all of them share the windowing/periodic-extension assumption and
differ only in how they soften it. This script estimates the same quantity by a
route that has no taper and no periodic extension at all:

  zero-phase Butterworth bandpass over 3-6 h, applied to the FULL 121 h record,
  then the variance of the interior after discarding `--guard` hours at each end
  so the filter transients have died out.

Using all 121 h rather than 48 h chunks makes the edge fraction much smaller, and
filtfilt introduces no phase distortion. If this lands near the Hann spectral
answer, the spectral estimate is confirmed by an independent method; if it lands
near the boxcar answer, the spectral route has a problem.

Only ratios are compared. A summed periodogram and a filtered variance differ by
a constant factor (the bin width), which cancels in FRAC.

Matched-sample masking is stricter here than in the spectral scripts: a whole
121 h record must be finite in every source, not just a 48 h window.

`--extra-sources` adds further hourly sources (e.g. ICON-CH2-CTRL); they enter
the matched-sample mask too. Variables without observations (PS) get FRAC only.

`--scale` (see scales.py) replaces the bandpass by another time-scale component;
output then goes to <out-dir>/<scale>/ instead of <out-dir>/order<N>/.

    uv run workflow/scripts/bandpass_timedomain.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, detrend

sys.path.insert(0, str(Path(__file__).parent))
from spectra_forecast import (FORECAST_SOURCES, INTERP_LABEL, OBS_LABEL,
                              TRUTH_LABEL, load_season)
from scales import SCALES, component, remove_daily_clim

SEASONS = ["winter2024", "summer2024"]
PARAMS = ["T_2M", "TD_2M", "SP_10M"]
NSTEP = 121
FS = 1.0                           # one sample per hour
LOW, HIGH = 1.0 / 6.0, 1.0 / 3.0   # cycles per hour: the 3-6 h band
# `fs` must be passed to butter(). Without it Wn is read as a fraction of
# Nyquist (0.5 cyc/h here), which silently turns this into a 6-12 h bandpass.


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--points", type=Path, default=Path("output/points"))
    ap.add_argument("--truth", type=Path, default=Path("output/points_truth"))
    ap.add_argument("--stations", type=Path,
                    default=Path("resources/smn_grid_points.csv"))
    ap.add_argument("--order", type=int, nargs="+", default=[2, 4, 6],
                    help="Butterworth orders to try; the first is the headline")
    ap.add_argument("--guard", type=int, default=24,
                    help="hours discarded at each end before taking the variance")
    ap.add_argument("--out-dir", type=Path, default=Path("output/spectra_bandpass"))
    ap.add_argument("--params", nargs="+", default=PARAMS)
    ap.add_argument("--extra-sources", nargs="+", default=[])
    ap.add_argument("--scale", default="3-6h", choices=SCALES)
    ap.add_argument("--anom", action="store_true",
                    help="remove each source's mean daily cycle first (scales.py)")
    args = ap.parse_args()
    keys = args.order if args.scale == "3-6h" else [args.scale]

    models = [*FORECAST_SOURCES, INTERP_LABEL, *args.extra_sources]
    st = pd.read_csv(args.stations)
    interior = slice(args.guard, NSTEP - args.guard)
    # second-order sections: the transfer-function form is numerically unstable
    # for a bandpass at these orders.
    sos = {o: butter(o, [LOW, HIGH], btype="band", fs=FS, output="sos")
           for o in args.order}
    summary = []

    for season in SEASONS:
        for param in args.params:
            print(f"\n=== {season} {param} ===", flush=True)
            data, inits = load_season(args.points, args.truth, season, param,
                                      tuple(args.extra_sources))
            if args.anom:
                data = remove_daily_clim(data, inits)
            srcs = [OBS_LABEL, TRUTH_LABEL, *models] if OBS_LABEL in data \
                else [TRUTH_LABEL, *models]
            if OBS_LABEL not in data:
                print("  no observations: FRAC (vs REA-L) only")

            ok = np.ones(len(inits), bool)
            for s in srcs:
                ok &= ~np.isnan(data[s][:, :NSTEP]).any(axis=(1, 2))
            n = int(ok.sum())
            print(f"  full 121 h record finite in all sources: {n}/{len(inits)} inits",
                  flush=True)

            for o in keys:
                var = {}
                for s in srcs:
                    acc = []
                    for i in np.flatnonzero(ok):
                        if args.scale != "3-6h":
                            acc.append(component(data[s][i, :NSTEP], o).var(axis=0))
                            continue
                        x = detrend(data[s][i, :NSTEP], axis=0, type="linear")
                        y = sosfiltfilt(sos[o], x, axis=0)
                        acc.append(y[interior].var(axis=0))
                    var[s] = np.mean(acc, axis=0)

                out = pd.DataFrame({"nat_abbr": st["nat_abbr"],
                                    "elevation": st["elevation"]})
                for s in srcs:
                    out[s] = var[s]
                if OBS_LABEL in data:
                    for s in srcs[1:]:
                        out[f"frac_{s}"] = out[s] / out[OBS_LABEL]
                for s in models:
                    out[f"FRAC_{s}"] = out[s] / out[TRUTH_LABEL]

                d = args.out_dir / (f"order{o}" if args.scale == "3-6h" else o)
                d.mkdir(parents=True, exist_ok=True)
                out.to_csv(d / f"by_station_{season}_{param}.csv", index=False)

                row = {"season": season, "param": param, "order": o, "n": n}
                for s in (*FORECAST_SOURCES, *args.extra_sources):
                    row[f"med_FRAC_{s}"] = out[f"FRAC_{s}"].median()
                summary.append(row)

    sm = pd.DataFrame(summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sm.to_csv(args.out_dir / "summary_medians.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n\n=== median FRAC (3-6 h variance / REA-L), time-domain bandpass ===")
    for (season, param), g in sm.groupby(["season", "param"], sort=False):
        t = g.set_index("order")
        o = pd.DataFrame({"Varda": t[f"med_FRAC_{FORECAST_SOURCES[0]}"],
                          "Multi": t[f"med_FRAC_{FORECAST_SOURCES[1]}"]})
        o["gap"] = o["Varda"] - o["Multi"]
        print(f"\n--- {season} {param} ---")
        print(o.to_string(float_format=lambda x: f"{x:.4g}"))
    print(f"\nwrote {args.out_dir}/")


if __name__ == "__main__":
    main()
