"""Paired per-station gap by station class, all variables and comparisons in one figure.

Rows: Varda-single minus Multistep, and Varda-single minus ICON-CH2 (own y axis).
Columns: variables. Within a panel, per class a winter and a hatched summer box
of the per-station FRAC differences (3-6 h, vs REA-L), dots = stations, orange
bar = 95 % init-bootstrap CI on the class median. Box colour: blue = Varda
retains more (CI excludes 0), red = the other model retains more, grey = CI
includes 0. Class 'strange' (n=11) is left out as too small. Reads the CSVs of
gap_robustness.py in output/with_icon/gap_robustness/; nothing is recomputed.

    uv run workflow/scripts/plot_paired_gap_class_overview.py
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

D = Path("output/with_icon/gap_robustness")
OUT = Path("output/with_icon/gap_robustness_by_stat_class/paired_gap_by_station_class.png")
ROWS = [("Multistep", ""), ("ICON-CH2", "_ICON-CH2")]
PARAMS = ["T_2M", "TD_2M", "SP_10M", "PS"]
CLASSES = ["flat", "local", "peaks", "support"]
SEASONS = ["winter2024", "summer2024"]

fig, axes = plt.subplots(len(ROWS), len(PARAMS), figsize=(16, 9), sharey="row")
rng = np.random.default_rng(0)
for (label, tag), row in zip(ROWS, axes):
    load = lambda f: pd.concat([pd.read_csv(D / f"{f}{tag}.csv"), pd.read_csv(D / f"{f}{tag}_PS.csv")])
    ST, CI = load("per_station"), load("by_class")
    for ax, p in zip(row, PARAMS):
        for k, c in enumerate(CLASSES):
            for m, s in enumerate(SEASONS):
                x = k + (m - 0.5) * 0.38
                v = ST[(ST.season == s) & (ST.param == p) & (ST.cls == c)].gap.to_numpy()
                r = CI[(CI.season == s) & (CI.param == p) & (CI.cls == c)].iloc[0]
                b = ax.boxplot([v], positions=[x], widths=0.3, patch_artist=True, showfliers=False,
                               manage_ticks=False, medianprops=dict(color="k", lw=1.5))["boxes"][0]
                b.set(facecolor="C0" if r.ci_lo > 0 else ("C3" if r.ci_hi < 0 else "0.7"),
                      alpha=0.45, edgecolor="0.3", hatch="//" if m else None)
                ax.scatter(x + rng.uniform(-0.06, 0.06, len(v)), v, s=5, color="0.25", alpha=0.4, zorder=3)
                ax.plot([x + 0.18] * 2, [r.ci_lo, r.ci_hi], color="C1", lw=2.5, zorder=4)
        n = CI[(CI.season == SEASONS[0]) & (CI.param == p)].set_index("cls").n
        ax.set_xticks(range(len(CLASSES)), [f"{c}\n(n={n[c]})" for c in CLASSES], fontsize=9)
        ax.axhline(0, color="k", lw=0.9, ls="--")
        ax.set_xlim(-0.6, len(CLASSES) - 0.4)
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(f"{p}: Varda-single - {label}", fontsize=10)
    row[0].set_ylabel(f"FRAC(Varda-single) - FRAC({label}), per station")
fig.legend(handles=[Patch(facecolor="C0", alpha=0.45, label="Varda retains more, CI excludes 0"),
                    Patch(facecolor="C3", alpha=0.45, label="other model retains more, CI excludes 0"),
                    Patch(facecolor="0.7", alpha=0.45, label="CI includes 0"),
                    Patch(facecolor="w", edgecolor="0.3", label="winter"),
                    Patch(facecolor="w", edgecolor="0.3", hatch="//", label="summer"),
                    plt.Line2D([], [], color="C1", lw=2.5, label="95 % CI of class median")],
           loc="lower center", ncol=6, fontsize=9)
fig.suptitle("Paired per-station difference in 3-6 h variance retention (vs REA-L, lead 24-96 h), by station "
             "class\npositive = Varda retains more. CIs are optimistic (stations held fixed). Class "
             "'strange' (n=11) omitted. ICON: control run; REA-L is ICON-CH1-based", fontsize=11)
fig.tight_layout(rect=(0, 0.05, 1, 0.93))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}")
