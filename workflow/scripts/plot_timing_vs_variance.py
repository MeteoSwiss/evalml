"""Distance from a perfect model: retained 3-6 h variance against timing.

One dot per station and model. x is FRAC, the 3-6 h band variance as a fraction
of REA-L's; y is r, the band timing correlation with REA-L, pooled over inits.
Both from joint_variance_skill.py, lead times 24-97 h. A perfect model sits at
(1, 1). Axes are fixed, equal-aspect and shared, so the panels show how far both
models are from that corner, not the small differences between them.

Writes the 8-panel overview and one PNG per (season, param). `--dir` and
`--models KEY:LABEL ...` plot other runs of joint_variance_skill.py, e.g. one
with the ICON baselines added (keys as in its per_station CSVs).

    uv run workflow/scripts/plot_timing_vs_variance.py
"""
import argparse
import textwrap
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--dir", type=Path, default=Path("output/joint_variance_skill"))
ap.add_argument("--models", nargs="+", default=["varda:Varda-single", "multi:Multistep"])
ap.add_argument("--note", default="", help="extra line for the subtitle")
ap.add_argument("--scale-label", default="3-6 h")
ap.add_argument("--lead", default="24-97 h", help="lead-time window for the subtitle")
args = ap.parse_args()
D = args.dir
N = pd.read_csv(D / "by_case.csv").set_index(["season", "param"])["ninit"]
SEASONS, PARAMS = ("winter2024", "summer2024"), ("T_2M", "TD_2M", "SP_10M", "PS")
CASES = [(s, p) for s in SEASONS for p in PARAMS]
MODELS = [(*m.split(":", 1), f"C{i}") for i, m in enumerate(args.models)]
DATA = {c: pd.read_csv(D / f"per_station_{c[0]}_{c[1]}.csv") for c in CASES}
YMIN = np.floor((min(d[[f"r_{k}" for k, _, _ in MODELS]].min().min()
                     for d in DATA.values()) - 0.02) / 0.05) * 0.05
# the ICON baseline can retain more than REA-L's variance, so let x grow past 1
XMAX = max(1.05, np.ceil((max(d[[f"frac_{k}" for k, _, _ in MODELS]].max().max()
                              for d in DATA.values()) + 0.02) / 0.05) * 0.05)
TITLE = (f"Both models retain little {args.scale_label} variability, and what they retain is "
         "poorly timed") if args.scale_label == "3-6 h" else (
         f"{args.scale_label}: retained variance against timing, per station")
SUB = f"one dot per station (145), against REA-L, lead time {args.lead}" + (
    f"; {args.note}" if args.note else "")


def draw(ax, season, param):
    d = DATA[season, param]
    for key, label, c in MODELS:
        x, y = d[f"frac_{key}"], d[f"r_{key}"]
        ax.scatter(x, y, s=7, color=c, alpha=0.4, lw=0, label=label)
        ax.plot(x.median(), y.median(), "o", ms=8, mfc=c, mec="k", mew=1.0, zorder=4)
    ax.plot(1, 1, "*", ms=14, color="k", zorder=5)
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_title(f"{param}, {season[:-4]} ({N[season, param]} inits)", fontsize=10)
    ax.set_xlim(0, XMAX); ax.set_ylim(YMIN, 1.05); ax.set_aspect("equal")
    ax.grid(alpha=0.3)


def legend(fig, ax, ncol):
    h, l = ax.get_legend_handles_labels()
    h += [plt.Line2D([], [], ls="", marker="*", ms=12, color="k"),
          plt.Line2D([], [], ls="", marker="o", ms=8, mfc="w", mec="k")]
    l += ["perfect model", "median over stations"]
    fig.legend(h, l, loc="lower center", ncol=ncol, fontsize=9, markerscale=1.5)


XL, YL = f"retained {args.scale_label} variance (fraction of REA-L)", "timing correlation r with REA-L"
fig, axes = plt.subplots(2, 4, figsize=(13, 7.6), sharex=True, sharey=True)
for ax, c in zip(axes.flat, CASES):
    draw(ax, *c)
for ax in axes[-1]:
    ax.set_xlabel(XL)
for ax in axes[:, 0]:
    ax.set_ylabel(YL)
legend(fig, axes[0, 0], len(MODELS) + 2)
fig.suptitle(f"{TITLE}\n{SUB}", fontsize=11)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(D / "timing_vs_variance.png", dpi=150)
plt.close(fig)
print(f"wrote {D}/timing_vs_variance.png")

out = D / "timing_vs_variance"; out.mkdir(exist_ok=True)
for c in CASES:
    fig, ax = plt.subplots(figsize=(min(10.0, 5.2 * XMAX / 1.05), 6.8))
    draw(ax, *c)
    ax.set_xlabel(XL); ax.set_ylabel(YL)
    legend(fig, ax, 2)
    fig.suptitle(textwrap.fill(TITLE, 50) + "\n" + textwrap.fill(SUB, 60), fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / f"{c[0]}_{c[1]}.png", dpi=150)
    plt.close(fig)
print(f"wrote {out}/ ({len(CASES)} files), y from {YMIN:.2f}")
