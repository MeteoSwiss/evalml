"""Like plot_frac_boxplots.py, but both models in one figure.

Per station class the two models get a box each, side by side, so the
model-to-model difference is read within a class rather than across figures.
Box width still encodes the number of stations in the class.

One figure per fraction flavour of output/spectra/by_station_<season>_<param>.csv:

  frac_  fraction of the SwissMetNet (observed) variance -> output/frac_var_bxplt_by_stat_class_combined/
  FRAC_  fraction of the REA-L variance                  -> output/FRAC_var_bxplt_by_stat_class_combined/

    uv run workflow/scripts/plot_frac_boxplots_combined.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ORDER = ["flat", "local", "support", "peaks", "strange"]
MODELS = ["Varda-single-stage-C", "Multistep-stage-C"]
COLORS = ["C0", "C1"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path,
                    default=Path("output/spectra/by_station_winter2024_T_2M.csv"))
    ap.add_argument("--frac-dir", type=Path,
                    default=Path("output/frac_var_bxplt_by_stat_class_combined"))
    ap.add_argument("--FRAC-dir", dest="FRAC_dir", type=Path,
                    default=Path("output/FRAC_var_bxplt_by_stat_class_combined"))
    ap.add_argument("--models", nargs="+", default=MODELS)
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
    offset = 0.21  # half the gap between the two boxes of a class

    for prefix, ref, outdir in (("frac_", "SwissMetNet", args.frac_dir),
                                ("FRAC_", "REA-L", args.FRAC_dir)):
        models = [m for m in args.models if prefix + m in d.columns]
        if not models:
            continue
        outdir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        rng = np.random.default_rng(0)
        for k, (model, color) in enumerate(zip(models, COLORS)):
            vals = [g[prefix + model].to_numpy() for g in groups]
            pos = np.arange(1, len(classes) + 1) + (k - (len(models) - 1) / 2) * 2 * offset
            bp = ax.boxplot(vals, positions=pos, widths=widths * 0.45,
                            patch_artist=True, showfliers=True, manage_ticks=False,
                            medianprops=dict(color="k", lw=1.6),
                            flierprops=dict(marker=".", ms=4, mfc="0.4", mec="none"))
            for patch in bp["boxes"]:
                patch.set(facecolor=color, alpha=0.45, edgecolor="0.3")
            # individual stations behind the boxes, so small classes stay honest
            for p, v in zip(pos, vals):
                ax.scatter(p + rng.uniform(-0.05, 0.05, len(v)), v,
                           s=8, color="0.25", alpha=0.45, zorder=3)
            bp["boxes"][0].set_label(model)

        ax.axhline(1.0, color="C3", lw=0.8, ls="--")
        ax.set_xticks(range(1, len(classes) + 1))
        ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(classes, counts)], fontsize=9)
        ax.set_xlim(0.5, len(classes) + 0.5)
        ax.set_ylabel(f"3-6 h power / {ref} power")
        ax.set_title(f"{tag}: 3-6 h variance relative to {ref}, by station class",
                     fontsize=10)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out = outdir / f"{tag}_{prefix}models.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
