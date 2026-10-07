"""MSE skill score of Multistep over Varda-single, by station class.

    SS = 1 - MSE(Multistep-stage-C) / MSE(Varda-single-stage-C)

both scored against REA-L. SS > 0 means Multistep is the better of the two,
SS = 0 means they tie. Truth is REA-L, so this asks which model reproduces the
training target better, not which is closer to reality.

One skill score per (init, station) cell, from the RMSE table written by
rmse_by_station_init.py --reference REA-L. Cells are then grouped by the
T2m_based_class of the station and shown as boxplots on one axes per
season/parameter, in the fixed CLASS_ORDER so the files can be compared.

Two summaries are printed, and they answer different questions. The median of
the per-cell SS says what a typical run looks like. The pooled SS, formed from
the summed MSE of each class, says what the class contributes overall; it is
dominated by the high-error cells, since a ratio of means is not a mean of
ratios. Where the two disagree, the disagreement is the result.

Station classes come from the spaziurat.lf R package, same source as
spectra_by_station.py. PFA is absent from that table and is dropped.

    uv run workflow/scripts/skill_by_station_class.py --season winter2024
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from spectra_forecast import FORECAST_SOURCES, TRUTH_LABEL
from spectra_by_station import station_classes

BASELINE, CANDIDATE = FORECAST_SOURCES[0], FORECAST_SOURCES[1]
# Fixed, so the same class sits at the same x position in every file. Sorting
# by median instead would reshuffle the axis between seasons and variables and
# make the six plots impossible to compare side by side.
CLASS_ORDER = ("flat", "peaks", "support", "local", "strange")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default="winter2024")
    ap.add_argument("--param", default="T_2M", choices=["T_2M", "TD_2M", "SP_10M"])
    ap.add_argument("--rmse-dir", type=Path, default=Path("output/rmse"))
    ap.add_argument("--classes", type=Path,
                    default=Path("resources/spaziurat.lf_2.6.1.tar.gz"),
                    help="R package holding stat.class.info")
    ap.add_argument("--out-dir", type=Path, default=Path("output/skill"))
    args = ap.parse_args()

    tag = f"{args.season}_{args.param}"
    csv = args.rmse_dir / f"rmse_by_station_init_{tag}_vs-{TRUTH_LABEL}.csv"
    if not csv.exists():
        raise SystemExit(f"{csv} missing: run rmse_by_station_init.py "
                         f"--reference {TRUTH_LABEL} first")
    d = pd.read_csv(csv, dtype={"init": str})

    # One row per (init, station) with both models side by side. An inner join
    # keeps only cells where both were scored, so the pair is always comparable.
    w = d.pivot_table(index=["init", "station"], columns="source",
                      values="rmse").dropna(subset=[BASELINE, CANDIDATE])
    w["ss"] = 1.0 - (w[CANDIDATE] ** 2) / (w[BASELINE] ** 2)
    w = w.reset_index()

    cls = station_classes(args.classes)
    w["class"] = w["station"].map(cls)
    missing = sorted(set(w.loc[w["class"].isna(), "station"]))
    if missing:
        print(f"dropped {len(missing)} station(s) with no class: {missing}")
    w = w.dropna(subset=["class"])

    # Pooled SS per class: ratio of summed MSE, not the mean of the ratios.
    g = w.groupby("class")
    tab = pd.DataFrame({
        "stations": g["station"].nunique(),
        "cells": g.size(),
        "median_SS": g["ss"].median(),
        "mean_SS": g["ss"].mean(),
        "pooled_SS": 1.0 - g.apply(
            lambda x: (x[CANDIDATE] ** 2).sum() / (x[BASELINE] ** 2).sum(),
            include_groups=False),
        "frac_SS>0": g["ss"].apply(lambda s: (s > 0).mean()),
    })
    if unknown := [c for c in tab.index if c not in CLASS_ORDER]:
        raise SystemExit(f"class(es) {unknown} not in CLASS_ORDER")
    tab = tab.reindex([c for c in CLASS_ORDER if c in tab.index])

    print(f"\n=== {tag}: SS = 1 - MSE({CANDIDATE}) / MSE({BASELINE}), "
          f"truth {TRUTH_LABEL} ===")
    print("positive = Multistep better\n")
    print(tab.to_string(float_format=lambda v: f"{v:+.3f}"))
    allss = w["ss"]
    print(f"\nall classes pooled: median {allss.median():+.3f}   "
          f"pooled {1 - (w[CANDIDATE] ** 2).sum() / (w[BASELINE] ** 2).sum():+.3f}   "
          f"frac SS>0 {(allss > 0).mean():.1%}   n={len(allss)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tab.to_csv(args.out_dir / f"skill_by_class_{tag}.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = list(tab.index)
    groups = [w.loc[w["class"] == c, "ss"].values for c in order]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bp = ax.boxplot(groups, tick_labels=order, showfliers=False, widths=0.6,
                    medianprops=dict(color="C3", lw=2))
    ax.axhline(0, color="0.3", lw=1.4, ls="--")
    for i, c in enumerate(order, start=1):
        ax.plot(i, tab.loc[c, "pooled_SS"], "D", color="C0", ms=6,
                zorder=3, label="pooled SS" if i == 1 else None)
        ax.annotate(f"n={tab.loc[c, 'cells']}", (i, 1), xycoords=("data", "axes fraction"),
                    xytext=(0, -12), textcoords="offset points",
                    ha="center", fontsize=7, color="0.4")
    ax.set_ylabel(f"SS = 1 - MSE({CANDIDATE}) / MSE({BASELINE})")
    ax.set_xlabel("T2m-based station class")
    # Fixed across all files so the six plots share a scale. Asymmetric because
    # SS is capped at 1 above but unbounded below, so the losing tail is longer.
    ax.set_ylim(-1.5, 1)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"{args.season} {args.param}: Multistep vs Varda-single, truth "
                 f"{TRUTH_LABEL}\npositive = Multistep better; boxes are "
                 f"per-(init, station) cells, whiskers 1.5 IQR, outliers hidden",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = args.out_dir / f"skill_by_class_{tag}.png"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out} and the matching .csv")


if __name__ == "__main__":
    main()
