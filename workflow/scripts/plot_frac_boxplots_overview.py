"""All stations pooled: both models, every variable and season in one figure.

Drops the station-class stratification of plot_frac_boxplots_combined.py and
puts one group of two boxes per (season, variable) instead, so the overall
model-to-model difference can be read across the whole experiment at once.

Two figures, into output/{frac,FRAC}_var_bxplt_overview/:

  frac_  fraction of the SwissMetNet (observed) variance
  FRAC_  fraction of the REA-L variance

    uv run workflow/scripts/plot_frac_boxplots_overview.py
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ["Varda-single-stage-C", "Multistep-stage-C"]
COLORS = ["C0", "C1", "C2", "C4"]
SEASONS = ["winter2024", "summer2024"]
PARAMS = ["T_2M", "TD_2M", "SP_10M", "PS"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spectra-dir", type=Path, default=Path("output/spectra"))
    ap.add_argument("--frac-dir", type=Path,
                    default=Path("output/frac_var_bxplt_overview"))
    ap.add_argument("--FRAC-dir", dest="FRAC_dir", type=Path,
                    default=Path("output/FRAC_var_bxplt_overview"))
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--note", default="", help="extra line for the title")
    ap.add_argument("--scale-label", default="3-6 h")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # keep a stable season/variable order, but only what actually exists
    cases = []
    for season in SEASONS:
        for param in PARAMS:
            f = args.spectra_dir / f"by_station_{season}_{param}.csv"
            if f.exists():
                d = pd.read_csv(f)
                cases.append((season, param, d))
    for f in sorted(args.spectra_dir.glob("by_station_*.csv")):
        m = re.fullmatch(r"by_station_(\w+?)_(\w+)", f.stem)
        if m and (m[1], m[2]) not in [(s, p) for s, p, _ in cases]:
            cases.append((m[1], m[2], pd.read_csv(f)))
    if not cases:
        raise SystemExit(f"no by_station_*.csv in {args.spectra_dir}")

    all_cases = cases
    for prefix, ref, outdir in (("frac_", "SwissMetNet", args.frac_dir),
                                ("FRAC_", "REA-L", args.FRAC_dir)):
        # PS has no observations, so it only enters the REA-L figure
        cases = [c for c in all_cases if all(prefix + m in c[2].columns for m in args.models)]
        models = args.models
        if not cases:
            continue
        outdir.mkdir(parents=True, exist_ok=True)
        offset = 0.42 / len(models)  # half the gap between neighbouring boxes

        fig, ax = plt.subplots(figsize=(max(9.0, 1.5 * len(cases)), 4.6))
        rng = np.random.default_rng(0)
        for k, (model, color) in enumerate(zip(models, COLORS)):
            vals = [d[prefix + model].dropna().to_numpy() for _, _, d in cases]
            pos = np.arange(1, len(cases) + 1) + (k - (len(models) - 1) / 2) * 2 * offset
            bp = ax.boxplot(vals, positions=pos, widths=0.72 / len(models),
                            patch_artist=True, showfliers=True, manage_ticks=False,
                            medianprops=dict(color="k", lw=1.6),
                            flierprops=dict(marker=".", ms=4, mfc="0.4", mec="none"))
            for patch in bp["boxes"]:
                patch.set(facecolor=color, alpha=0.45, edgecolor="0.3")
            # individual stations behind the boxes
            for p, v in zip(pos, vals):
                ax.scatter(p + rng.uniform(-0.05, 0.05, len(v)), v,
                           s=6, color="0.25", alpha=0.35, zorder=3)
            bp["boxes"][0].set_label(model)

        ax.axhline(1.0, color="C3", lw=0.8, ls="--")
        ax.set_xticks(range(1, len(cases) + 1))
        ax.set_xticklabels([f"{p}\n{s}\n(n={len(d)})" for s, p, d in cases], fontsize=9)
        ax.set_xlim(0.5, len(cases) + 0.5)
        ax.set_ylabel(f"{args.scale_label} power / {ref} power")
        ax.set_title(f"{args.scale_label} variance relative to {ref}, all stations pooled"
                     + (f"\n{args.note}" if args.note else ""), fontsize=11)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out = outdir / f"all_seasons_params_{prefix}models.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
