"""Window length: does the 48 h choice matter, and is any effect real or artefact?

Window length confounds two things in the periodogram route: the quality of the
estimate (a shorter window gives coarser bins, fewer bins inside the 3-6 h band,
and worse leakage) and the sampling (which lead times are covered). The
time-domain bandpass separates them, because the filter runs once over the whole
121 h record and only the segmentation changes. So:

  spectral   Hann taper + least-squares detrend, one periodogram per segment
  bandpass   filter the full record once, then the variance of each segment

Any L-dependence the bandpass shows is a real property of the forecasts. Extra
L-dependence in the spectral route is an artefact of the estimator.

L = 24, 48 and 96 h all tile lead [0,96) exactly, so time coverage is identical
across L and the samples are matched. L = 48 reproduces the production setting.

Both methods use the same inits: the whole 121 h record must be finite in every
source, since the bandpass needs the full record to filter. That is stricter than
spectra_by_station.py, so the spectral numbers here differ slightly from
section 8 of FINDINGS.md.

    uv run workflow/scripts/window_length_sensitivity.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import periodogram, butter, sosfiltfilt, detrend

sys.path.insert(0, str(Path(__file__).parent))
from spectra_forecast import (FORECAST_SOURCES, INTERP_LABEL, OBS_LABEL,
                              TRUTH_LABEL, load_season)

SEASONS = ["winter2024", "summer2024"]
PARAMS = ["T_2M", "TD_2M", "SP_10M"]
NSTEP = 121
COVER = 96                          # lead [0,96), tiled by every L
LENGTHS = [24, 48, 96]
FS = 1.0
LOW, HIGH = 1.0 / 6.0, 1.0 / 3.0    # 3-6 h; fs must be passed to butter()
ORDER = 4


def band_spectral(x: np.ndarray) -> np.ndarray:
    """Hann periodogram of one segment, summed over the 3-6 h bins."""
    xd = x - np.vander(np.arange(len(x), dtype=float), 2) @ np.linalg.lstsq(
        np.vander(np.arange(len(x), dtype=float), 2), x, rcond=None)[0]
    f, p = periodogram(xd, fs=FS, window="hann", detrend=False,
                       scaling="density", axis=0)
    per = np.divide(1.0, f, out=np.full_like(f, np.inf), where=f > 0)
    return p[(per >= 3) & (per <= 6)].sum(axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--points", type=Path, default=Path("output/points"))
    ap.add_argument("--truth", type=Path, default=Path("output/points_truth"))
    ap.add_argument("--stations", type=Path,
                    default=Path("resources/smn_grid_points.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("output/spectra_winlen"))
    args = ap.parse_args()

    srcs = [OBS_LABEL, TRUTH_LABEL, *FORECAST_SOURCES, INTERP_LABEL]
    st = pd.read_csv(args.stations)
    sos = butter(ORDER, [LOW, HIGH], btype="band", fs=FS, output="sos")
    summary = []

    for season in SEASONS:
        for param in PARAMS:
            print(f"\n=== {season} {param} ===", flush=True)
            data, inits = load_season(args.points, args.truth, season, param)
            if OBS_LABEL not in data:
                print("  no observations, skipped")
                continue

            ok = np.ones(len(inits), bool)
            for s in srcs:
                ok &= ~np.isnan(data[s][:, :NSTEP]).any(axis=(1, 2))
            idx = np.flatnonzero(ok)
            print(f"  {len(idx)}/{len(inits)} inits finite over the full 121 h",
                  flush=True)

            # filter the whole record once per init and source
            filt = {s: np.stack([sosfiltfilt(sos, detrend(data[s][i, :NSTEP],
                                                          axis=0, type="linear"),
                                             axis=0) for i in idx])
                    for s in srcs}

            for L in LENGTHS:
                starts = range(0, COVER, L)
                acc = {("spectral", s): [] for s in srcs}
                acc.update({("bandpass", s): [] for s in srcs})
                for k, i in enumerate(idx):
                    for w in starts:
                        sl = slice(w, w + L)
                        for s in srcs:
                            acc[("spectral", s)].append(band_spectral(data[s][i, sl]))
                            acc[("bandpass", s)].append(filt[s][k, sl].var(axis=0))
                ncase = len(idx) * len(list(starts))

                for meth in ("spectral", "bandpass"):
                    out = pd.DataFrame({"nat_abbr": st["nat_abbr"],
                                        "elevation": st["elevation"]})
                    for s in srcs:
                        out[s] = np.mean(acc[(meth, s)], axis=0)
                    for s in srcs[1:]:
                        out[f"frac_{s}"] = out[s] / out[OBS_LABEL]
                    for s in srcs[2:]:
                        out[f"FRAC_{s}"] = out[s] / out[TRUTH_LABEL]

                    d = args.out_dir / f"{meth}_L{L}"
                    d.mkdir(parents=True, exist_ok=True)
                    out.to_csv(d / f"by_station_{season}_{param}.csv", index=False)

                    row = {"season": season, "param": param, "method": meth,
                           "L": L, "ncase": ncase}
                    for s in FORECAST_SOURCES:
                        row[f"med_FRAC_{s}"] = out[f"FRAC_{s}"].median()
                    summary.append(row)

    sm = pd.DataFrame(summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sm.to_csv(args.out_dir / "summary_medians.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n\n=== median FRAC (3-6 h / REA-L). L=48 spectral is production ===")
    for (season, param), g in sm.groupby(["season", "param"], sort=False):
        print(f"\n--- {season} {param} ---")
        t = g.pivot_table(index="L", columns="method",
                          values=[f"med_FRAC_{s}" for s in FORECAST_SOURCES])
        t.columns = [f"{m[:4]}_{'V' if 'Varda' in a else 'M'}"
                     for a, m in t.columns]
        t["gap_band"] = t["band_V"] - t["band_M"]
        t["gap_spec"] = t["spec_V"] - t["spec_M"]
        print(t.to_string(float_format=lambda x: f"{x:.4g}"))
    print(f"\nwrote {args.out_dir}/")


if __name__ == "__main__":
    main()
