"""How the comparison changes with time scale: one figure across all scales.

Reads the per-scale outputs written by output/icon_test/make_scale.sh under
output/with_icon/scales/<scale>/ and shows, per variable (columns) and season
(line style), against time scale (x):
  row 1  median retained variance (FRAC vs REA-L) for Varda, Multistep, ICON-CH2
  row 2  median timing correlation r with REA-L, same models
  row 3  paired gap Varda minus Multistep (median over stations) with its 95 %
         init-bootstrap interval; positive = Varda retains more
Also writes the numbers to scales_summary.csv. `--anom` uses the sets with each
source's mean daily cycle removed (<scale>_anom/) and writes *_anom.{png,csv}.

    uv run workflow/scripts/plot_scales_summary.py [--anom]
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--anom", action="store_true")
args = ap.parse_args()
D = Path("output/with_icon/scales")
SUF = "_anom" if args.anom else ""
SCALES = ["3-6h", "6-12h", "diurnal", "synoptic"]
XT = ["3-6 h", "6-12 h", "daily\n(24+12 h)", "synoptic\n(> ~2 d)"]
PARAMS = ["T_2M", "TD_2M", "SP_10M", "PS"]
SEASONS = {"winter2024": "-", "summer2024": "--"}
MODELS = [("varda", "Varda-single", "C0"), ("multi", "Multistep", "C1"),
          ("icon_ch2", "ICON-CH2", "C2")]

rows = []
for sc in SCALES:
    band = "3-6h" if sc == "3-6h" else sc
    for s in SEASONS:
        for p in PARAMS:
            d = pd.read_csv(D / (sc + SUF) / f"joint_variance_skill/per_station_{s}_{p}.csv")
            tag = "_PS" if p == "PS" else ""
            g = pd.read_csv(D / (sc + SUF) / f"gap_robustness/band_edges{tag}.csv")
            g = g[(g.season == s) & (g.param == p) & (g.band == band)].iloc[0]
            rows.append(dict(scale=sc, season=s, param=p,
                             **{f"frac_{k}": d[f"frac_{k}"].median() for k, _, _ in MODELS},
                             **{f"r_{k}": d[f"r_{k}"].median() for k, _, _ in MODELS},
                             gap=g.gap, ci_lo=g.ci_lo, ci_hi=g.ci_hi))
T = pd.DataFrame(rows)
T.to_csv(D / f"scales_summary{SUF}.csv", index=False)

x = np.arange(len(SCALES))
fig, axes = plt.subplots(3, len(PARAMS), figsize=(15, 10), sharex=True, sharey="row")
for j, p in enumerate(PARAMS):
    for s, ls in SEASONS.items():
        t = T[(T.param == p) & (T.season == s)].set_index("scale").loc[SCALES]
        for k, lab, c in MODELS:
            axes[0, j].plot(x, t[f"frac_{k}"], ls=ls, marker="o", ms=5, color=c, lw=1.6)
            axes[1, j].plot(x, t[f"r_{k}"], ls=ls, marker="o", ms=5, color=c, lw=1.6)
        off = -0.06 if s == "winter2024" else 0.06
        axes[2, j].errorbar(x + off, t["gap"], yerr=[t["gap"] - t["ci_lo"], t["ci_hi"] - t["gap"]],
                            ls=ls, marker="o", ms=5, color="0.25", lw=1.4, capsize=3)
    axes[0, j].set_title(p, fontsize=11)
    axes[2, j].axhline(0, color="k", lw=0.9)
    axes[1, j].axhline(0, color="0.5", lw=0.8)
    axes[0, j].axhline(1, color="0.5", lw=0.8, ls=":")
    for ax in axes[:, j]:
        ax.grid(alpha=0.3)
axes[0, 0].set_yscale("log")
axes[0, 0].set_ylabel("retained variance\n(median FRAC vs REA-L, log)")
axes[1, 0].set_ylabel("timing correlation r\n(median over stations)")
axes[2, 0].set_ylabel("FRAC(Varda) - FRAC(Multistep)\nmedian + 95 % CI")
for ax in axes[-1]:
    ax.set_xticks(x); ax.set_xticklabels(XT, fontsize=9)
h = [plt.Line2D([], [], color=c, lw=2, label=lab) for _, lab, c in MODELS]
h += [plt.Line2D([], [], color="0.3", ls=ls, marker="o", label=s[:-4]) for s, ls in SEASONS.items()]
fig.legend(handles=h, loc="lower center", ncol=5, fontsize=10)
fig.suptitle("How the comparison depends on time scale: 145 stations, against REA-L, lead ~24-96 h"
             + (", each source's mean daily cycle removed" if args.anom else "") + "\n"
             "ICON: control run; REA-L is ICON-CH1-based, which flatters ICON. "
             "Row 3: positive = Varda retains more", fontsize=11)
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
fig.savefig(D / f"scales_summary{SUF}.png", dpi=150)
print(f"wrote {D}/scales_summary{SUF}.png and .csv")
