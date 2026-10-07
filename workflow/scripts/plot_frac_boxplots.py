"""Boxplots of the per-station 3-6 h variance fractions, grouped by station class.

One figure per fraction column of output/spectra/by_station_<season>_<param>.csv:

  frac_*  fraction of the SwissMetNet (observed) variance -> output/frac_var_bxplt_by_stat_class/
  FRAC_*  fraction of the REA-L variance                  -> output/FRAC_var_bxplt_by_stat_class/

Each box is one T2m_based_class. Box width is proportional to the number of
stations in that class, so the eye is not drawn to a class of 11 as strongly as
to one of 60. Stations without a class (PFA) are dropped.

    uv run workflow/scripts/plot_frac_boxplots.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ORDER = ["flat", "local", "support", "peaks", "strange"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path,
                    default=Path("output/spectra/by_station_winter2024_T_2M.csv"))
    ap.add_argument("--frac-dir", type=Path,
                    default=Path("output/frac_var_bxplt_by_stat_class"))
    ap.add_argument("--FRAC-dir", dest="FRAC_dir", type=Path,
                    default=Path("output/FRAC_var_bxplt_by_stat_class"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = pd.read_csv(args.csv)
    d = d[d["T2m_based_class"].notna()]
    tag = args.csv.stem.replace("by_station_", "")

    classes = [c for c in ORDER if c in set(d["T2m_based_class"])]
    groups = [d.loc[d["T2m_based_class"] == c] for c in classes]
    counts = np.array([len(g) for g in groups])
    widths = 0.25 + 0.55 * counts / counts.max()  # width encodes sample size

    cols = [c for c in d.columns if c.startswith(("frac_", "FRAC_"))]
    for col in cols:
        vals = [g[col].to_numpy() for g in groups]
        is_upper = col.startswith("FRAC_")
        ref = "REA-L" if is_upper else "SwissMetNet"
        outdir = args.FRAC_dir if is_upper else args.frac_dir
        outdir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        bp = ax.boxplot(vals, widths=widths, patch_artist=True, showfliers=True,
                        medianprops=dict(color="k", lw=1.6),
                        flierprops=dict(marker=".", ms=4, mfc="0.4", mec="none"))
        for patch in bp["boxes"]:
            patch.set(facecolor="C0", alpha=0.45, edgecolor="0.3")
        # individual stations behind the boxes, so small classes stay honest
        for i, v in enumerate(vals, start=1):
            ax.scatter(np.full(len(v), i) + np.random.default_rng(0).uniform(-0.06, 0.06, len(v)),
                       v, s=8, color="0.25", alpha=0.45, zorder=3)
        ax.axhline(1.0, color="C3", lw=0.8, ls="--")
        ax.set_xticks(range(1, len(classes) + 1))
        ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(classes, counts)], fontsize=9)
        ax.set_ylabel(f"3-6 h power / {ref} power")
        ax.set_title(f"{col}\n{tag}: 3-6 h variance relative to {ref}, by station class",
                     fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out = outdir / f"{tag}_{col}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
