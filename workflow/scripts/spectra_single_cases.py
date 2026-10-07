"""Plot individual (station, init) periodograms, unaveraged, to show their noise.

A single 48 h periodogram has about 2 degrees of freedom per frequency bin, so
each point scatters by roughly its own size. This script exists to make that
visible: the smooth curves in spectra_forecast.py are means over ~40 inits x 145
stations, and nothing about an individual case should be read off them.

    uv run workflow/scripts/spectra_single_cases.py --season winter2024
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

import sys
sys.path.insert(0, str(Path(__file__).parent))
from spectra_forecast import (OBS_LABEL, TRUTH_LABEL, WINDOW_H, load_season, psd)

SHOW = (TRUTH_LABEL, OBS_LABEL, "Varda-single-stage-C", "Multistep-stage-C")
STYLE = {TRUTH_LABEL: ("k", 1.4), OBS_LABEL: ("0.45", 1.2),
         "Varda-single-stage-C": ("C0", 1.0), "Multistep-stage-C": ("C3", 1.0)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default="winter2024")
    ap.add_argument("--param", default="T_2M")
    ap.add_argument("--window", type=int, default=0, help="lead-time window start")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--points", type=Path, default=Path("output/points"))
    ap.add_argument("--truth", type=Path, default=Path("output/points_truth"))
    ap.add_argument("--out", type=Path,
                    default=Path("output/spectra/single_cases.png"))
    args = ap.parse_args()

    data, inits = load_season(args.points, args.truth, args.season, args.param)
    sl = slice(args.window, args.window + WINDOW_H)

    # only cases where every shown source is finite
    ok = np.ones(data[TRUTH_LABEL][:, sl].shape[::2], bool)  # (n_init, n_station)
    for src in SHOW:
        ok &= ~np.isnan(data[src][:, sl]).any(axis=1)
    ii, jj = np.nonzero(ok)
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(ii), size=min(args.n, len(ii)), replace=False)

    stations = xr.open_dataset(
        sorted((args.points / args.season / "Multistep-stage-C").glob("*.nc"))[0]
    )["station"].values

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nc = 5
    nr = int(np.ceil(len(pick) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(3.0 * nc, 2.3 * nr),
                             sharex=True, sharey=True)
    for ax, k in zip(axes.ravel(), pick):
        i, j = ii[k], jj[k]
        for src in SHOW:
            f, p = psd(data[src][i, sl, j][:, None])
            per = np.divide(1.0, f, out=np.full_like(f, np.inf), where=f > 0)
            c, lw = STYLE[src]
            ax.loglog(per[f > 0], p[f > 0, 0], color=c, lw=lw)
        ax.axvspan(3, 6, color="0.85", zorder=0)
        ax.tick_params(labelsize=7)
        ax.set_title(f"{stations[j]}  {inits[i]}", fontsize=7)
        ax.grid(alpha=0.3, which="both")
    for ax in axes.ravel()[len(pick):]:
        ax.axis("off")
    # sharex=True: set this once, not per panel, or 20 inversions cancel out
    axes.ravel()[0].invert_xaxis()
    axes.ravel()[0].set_xticks([24, 12, 6, 3, 2])
    axes.ravel()[0].set_xticklabels(["24", "12", "6", "3", "2"], fontsize=7)
    axes.ravel()[0].set_xticks([], minor=True)
    handles = [plt.Line2D([], [], color=STYLE[s][0], lw=1.4, label=s) for s in SHOW]
    fig.legend(handles=handles, fontsize=8, loc="lower center", ncol=4)
    fig.suptitle(f"{args.season} {args.param}, lead {args.window}-{args.window + WINDOW_H} h: "
                 f"{len(pick)} single station/init periodograms, no averaging", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
