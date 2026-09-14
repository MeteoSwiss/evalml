"""Does REA-L-CH1 contain a daily discontinuity from concatenated forecast segments?

REA-L is assembled from 24-hourly forecast runs (latent heat nudging at 00 UTC
only), so consecutive days come from different model runs. Where two segments
meet, the field can jump. A jump repeating every 24 h is not a narrow-band
feature: it injects power at 24 h and every harmonic, including 6 h and 4 h,
which sit inside the band the temporal-spectrum diagnostic of exp_plan_01.md
Step 1 is meant to measure. It would also contaminate the 6 h bin that was
proposed as the clean test for Varda's block seam.

Detection: the root-mean-square of hourly increments as a function of hour of
day. A join shows up as a one-hour spike on top of the smooth diurnal shape.

T_2M alone cannot settle this, because its increments have a large genuine
diurnal cycle (fastest change near sunrise and sunset), so a spike could hide in
real structure. PMSL is the discriminating variable: it has only a weak
semi-diurnal tide, so its increment curve is nearly flat and a join stands out.

Run:
    uv run python workflow/scripts/join_detect.py
    uv run python workflow/scripts/join_detect.py --days 2      # quick check

Writes join_detect.png (per-variable curves), join_detect_summary.png (the 00 UTC
spike index per variable, which is the actual result) and join_detect.npz.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import zarr

LOG = logging.getLogger("join_detect")

REAL_ZARR = "/store_new/mch/msopr/ml/datasets/mch-realch1-fdb-1km-2005-2025-1h-pl13-v1.0.zarr"

# Every surface field Varda outputs (checkpoint surface variables 2t, 2d, 10u,
# 10v, msl, sp, tp), plus upper-air levels to discriminate the mechanism: a soil
# or snow analysis should hit the near-surface fields and barely touch 850/500
# hPa, whereas latent heat nudging acts on latent heating in the column and
# should show in humidity, precipitation and mid-level temperature.
SURFACE = ("T_2M", "TD_2M", "U_10M", "V_10M", "PMSL", "PS", "TOT_PREC_1H")
UPPER = ("T_850", "QV_850", "FI_500")
PARAMS = SURFACE + UPPER

# Hourly precipitation is already a rate, so differencing it is the wrong
# diagnostic. For these we look at the mean amount by hour of day instead: a
# nudging spike at 00 UTC shows up directly as anomalous precipitation.
RATE_PARAMS = {"TOT_PREC_1H"}

# contiguous stretches, one per season, avoiding month edges
STARTS = {"winter": np.datetime64("2024-01-10T00"), "summer": np.datetime64("2024-07-01T00")}


def spike_index(rms: np.ndarray) -> np.ndarray:
    """Each hour's RMS increment divided by the mean of its two neighbours.

    A smooth diurnal cycle gives ~1 everywhere. A single-hour discontinuity
    gives a value clearly above 1 at the join hour only. This is what separates
    a join from real diurnal structure.
    """
    return rms / (0.5 * (np.roll(rms, 1) + np.roll(rms, -1)))


def analyse(z, start: np.datetime64, days: int, pts: np.ndarray, var_idx: list[int]):
    """Return (rms_by_hour, spike_by_hour) with shape (nvars, 24)."""
    dates = z["dates"][:]
    s = int(np.searchsorted(dates, start))
    n = days * 24 + 1  # +1 so every hour of day gets the same number of increments
    LOG.info("reading %d hourly steps from %s", n, dates[s])

    series = np.empty((n, len(var_idx), len(pts)), dtype=np.float32)
    for h in range(n):
        series[h] = z["data"].oindex[s + h, var_idx, 0, :][:, pts]
        if h % 48 == 0:
            LOG.info("  %d/%d", h, n)

    diff = np.diff(series, axis=0)  # increment ending at step h+1
    hours = (dates[s + 1 : s + n].astype("datetime64[h]").astype(int)) % 24
    hours_abs = (dates[s : s + n].astype("datetime64[h]").astype(int)) % 24

    stat = np.empty((len(PARAMS), 24))
    for i, p in enumerate(PARAMS):
        for hh in range(24):
            if p in RATE_PARAMS:  # mean amount, not increment
                stat[i, hh] = series[hours_abs == hh, i].mean()
            else:
                stat[i, hh] = np.sqrt((diff[hours == hh, i] ** 2).mean())
    return stat, np.stack([spike_index(r) for r in stat])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=21, help="contiguous days per season")
    ap.add_argument("--n-points", type=int, default=200)
    ap.add_argument("--out-dir", type=Path, default=Path("output/join_detect"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    z = zarr.open(REAL_ZARR, mode="r")
    var_idx = [list(z.attrs["variables"]).index(p) for p in PARAMS]
    pts = np.random.default_rng(0).choice(z["latitudes"].shape[0], args.n_points, replace=False)

    res = {}
    for season, start in STARTS.items():
        res[season] = analyse(z, start, args.days, pts, var_idx)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / "join_detect.npz",
             **{f"{s}_{k}": v for s, (rms, sp) in res.items()
                for k, v in (("rms", rms), ("spike", sp))},
             params=np.array(PARAMS), days=args.days, n_points=args.n_points)

    print(f"\n=== spike index at 00 UTC and worst hour ({args.days} d, {args.n_points} pts) ===")
    worst = {}
    for season, (rms, sp) in res.items():
        for i, p in enumerate(PARAMS):
            h = int(np.argmax(sp[i]))
            worst[(season, p)] = (h, sp[i][h])
            print(f"{season:6s} {p:5s}: max spike index {sp[i][h]:.2f} at {h:02d} UTC "
                  f"(stat {rms[i][h]:.4g}); median spike {np.median(sp[i]):.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Summary: the 00 UTC spike index per variable. This is the actual result;
    # the per-variable curves below are the supporting detail.
    figs, axs = plt.subplots(figsize=(9, 4.5))
    w, x = 0.38, np.arange(len(PARAMS))
    for k, season in enumerate(STARTS):
        sp = res[season][1]
        axs.bar(x + (k - 0.5) * w, [sp[i][0] for i in range(len(PARAMS))], w, label=season)
    axs.axhline(1.0, color="0.3", lw=1)
    axs.axhline(1.25, color="C3", lw=0.8, ls="--", label="flag threshold")
    axs.set_xticks(x)
    axs.set_xticklabels(PARAMS, rotation=45, ha="right", fontsize=8)
    axs.set_ylabel("spike index at 00 UTC")
    axs.set_title("REA-L-CH1: size of the 00 UTC artefact by variable\n"
                  "1.0 = no discontinuity; surface vs upper air discriminates the mechanism",
                  fontsize=10)
    axs.legend(fontsize=8)
    axs.grid(axis="y", alpha=0.3)
    figs.tight_layout()
    figs.savefig(args.out_dir / "join_detect_summary.png", dpi=150)

    n = len(PARAMS)
    fig, axes = plt.subplots(n, 2, figsize=(11, 1.9 * n), sharex=True, squeeze=False)
    for col, season in enumerate(STARTS):
        rms, sp = res[season]
        for row, p in enumerate(PARAMS):
            ax, hh = axes[row, col], np.arange(24)
            lbl = "mean amount" if p in RATE_PARAMS else "RMS hourly increment"
            ax.plot(hh, rms[row], "o-", color="C0", ms=3, label=lbl)
            ax.set_title(f"{season} 2024, {p}", fontsize=9)
            ax.set_xticks(range(0, 24, 3))
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=7)
            ax2 = ax.twinx()
            ax2.plot(hh, sp[row], "s--", color="C3", ms=2, lw=0.7)
            ax2.axhline(1.0, color="C3", lw=0.5, ls=":")
            ax2.tick_params(axis="y", labelcolor="C3", labelsize=6)
            h, v = worst[(season, p)]
            if v > 1.25:
                ax.axvline(h, color="0.5", lw=1.0)
            if row == n - 1:
                ax.set_xlabel("hour of day (UTC), end of increment")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("REA-L-CH1: one-hour spike (red, right axis) above the smooth diurnal shape "
                 "indicates a daily artefact", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(args.out_dir / "join_detect.png", dpi=150)
    print(f"\nwrote {args.out_dir}/join_detect{{,_summary}}.png and join_detect.npz")


if __name__ == "__main__":
    main()
