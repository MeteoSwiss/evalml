"""Paired per-station difference, the comparison the overview boxplot cannot show.

all_seasons_params_FRAC_models.png puts the two models' marginal distributions
side by side, so the eye compares marginal medians. The data are paired by
station, and the two summaries can disagree badly: for summer T_2M under the
bandpass the marginal medians coincide exactly while the median paired
difference is +0.019 with 90 of 145 stations favouring Varda.

This plots FRAC_Varda - FRAC_Multistep per station, with zero marked and the
95 % paired bootstrap interval over inits overlaid. Both come from
gap_robustness.py (full 121 h record), so the boxes and intervals match.

`--models LABEL:TAG ...` draws one row per model Varda-single is compared
against, reading per_station{TAG}.csv, per_station{TAG}_PS.csv and the matching
band_edges files from `--dir` (default: Multistep only, as before).

    uv run workflow/scripts/plot_paired_gap.py
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--dir", type=Path, default=Path("output/gap_robustness"))
ap.add_argument("--models", nargs="+", default=["Multistep:"])
ap.add_argument("--note", default="", help="extra line for the title")
ap.add_argument("--band", default="3-6h", help="band/scale name in band_edges")
ap.add_argument("--scale-label", default="3-6 h")
args = ap.parse_args()
D = args.dir
CASES = [(s, p) for s in ("winter2024", "summer2024")
         for p in ("T_2M", "TD_2M", "SP_10M", "PS")]
MODELS = [m.split(":", 1) for m in args.models]

fig, axes = plt.subplots(len(MODELS), 1, figsize=(10.5, 4.8 if len(MODELS) == 1
                                                   else 3.6 * len(MODELS) + 0.8),
                         squeeze=False)  # own y axis per row: the gaps differ by 10x
rng = np.random.default_rng(0)
any_neg = False
for ax, (label, tag) in zip(axes[:, 0], MODELS):
    ST = pd.concat([pd.read_csv(D / f"per_station{tag}.csv"),
                    pd.read_csv(D / f"per_station{tag}_PS.csv")])
    CI = pd.concat([pd.read_csv(D / f"band_edges{tag}.csv"),
                    pd.read_csv(D / f"band_edges{tag}_PS.csv")]).query("band == @args.band")
    vals, labs, cis = [], [], []
    for season, param in CASES:
        d = ST[(ST.season == season) & (ST.param == param)].gap.to_numpy()
        vals.append(d)
        r = CI[(CI.season == season) & (CI.param == param)].iloc[0]
        cis.append((r.ci_lo, r.ci_hi, r.frac_boot_pos))
        labs.append(f"{param}\n{season}\n(n={len(d)})")

    pos = np.arange(1, len(vals) + 1)
    bp = ax.boxplot(vals, positions=pos, widths=0.5, patch_artist=True,
                    showfliers=True, manage_ticks=False,
                    medianprops=dict(color="k", lw=1.6),
                    flierprops=dict(marker=".", ms=4, mfc="0.4", mec="none"))
    for patch, (lo, hi, fp) in zip(bp["boxes"], cis):
        patch.set(facecolor="C0" if lo > 0 else ("C3" if hi < 0 else "0.7"),
                  alpha=0.45, edgecolor="0.3")
        any_neg |= hi < 0
    for p, v, (lo, hi, fp) in zip(pos, vals, cis):
        ax.scatter(p + rng.uniform(-0.07, 0.07, len(v)), v, s=6, color="0.25",
                   alpha=0.35, zorder=3)
        ax.plot([p + 0.33] * 2, [lo, hi], color="C1", lw=3, solid_capstyle="butt",
                zorder=4)
        ax.text(p, ax.get_ylim()[1], "", ha="center")

    ax.axhline(0.0, color="k", lw=1.0, ls="--")
    ax.axvline(4.5, color="0.5", lw=0.8)
    ax.set_xticks(pos); ax.set_xticklabels(labs, fontsize=9)
    ax.set_xlim(0.5, len(vals) + 0.5)
    ax.set_ylabel(f"FRAC(Varda-single) - FRAC({label}), per station",
                  fontsize=None if len(MODELS) == 1 else 9)
    ax.grid(axis="y", alpha=0.3)
    print(f"--- Varda-single - {label}")
    for (s, p), (lo, hi, fp), v in zip(CASES, cis, vals):
        print(f"  {s:11s} {p:7s} median {np.median(v):+.4f}  CI [{lo:+.4f},{hi:+.4f}]"
              f"  P(gap>0)={fp:.3f}")

axes[0, 0].set_title(f"Paired per-station difference in {args.scale_label} variance retention\n"
                     "positive = Varda retains more. Orange bar: 95 % bootstrap CI on "
                     "the median, over inits" + (f"\n{args.note}" if args.note else ""),
                     fontsize=10)
handles = [Patch(facecolor="C0", alpha=0.45, label="Varda retains more, CI excludes 0"),
           Patch(facecolor="0.7", alpha=0.45, label="CI includes 0")]
if any_neg:
    handles.insert(1, Patch(facecolor="C3", alpha=0.45,
                            label="other model retains more, CI excludes 0"))
axes[0, 0].legend(handles=handles, fontsize=8, loc="upper left")
fig.tight_layout()
D.mkdir(parents=True, exist_ok=True)
fig.savefig(D / "paired_gap_overview.png", dpi=150)
print(f"wrote {D}/paired_gap_overview.png")
