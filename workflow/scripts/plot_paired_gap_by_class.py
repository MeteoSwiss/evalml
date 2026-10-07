"""plot_paired_gap.py split by station class, one figure per class.

Per-station FRAC_Varda - FRAC_Multistep (3-6 h, vs REA-L) from
gap_robustness.py, with the class-level 95 % bootstrap CI over inits. Y limits
are shared across classes so the figures can be compared side by side. The
'strange' class (n=11) is skipped as too small to support anything.

Stations are held fixed in the bootstrap and are spatially correlated, so the
CIs are optimistic, more so for small classes.

    uv run workflow/scripts/plot_paired_gap_by_class.py
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

D = Path("output/gap_robustness")
ST = pd.concat([pd.read_csv(D / "per_station.csv"),
                pd.read_csv(D / "per_station_PS.csv")]).dropna(subset=["cls"])
ST = ST[ST.cls != "strange"]
CI = pd.concat([pd.read_csv(D / "by_class.csv"), pd.read_csv(D / "by_class_PS.csv")])
SEASONS = ("winter2024", "summer2024")
CASES = [(s, p) for s in SEASONS for p in ("T_2M", "TD_2M", "SP_10M", "PS")]
lo, hi = ST.gap.min(), ST.gap.max()
pad = 0.05 * (hi - lo)

rng = np.random.default_rng(0)
for cls, sc in ST.groupby("cls"):
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for p, (season, param) in enumerate(CASES, start=1):
        v = sc[(sc.season == season) & (sc.param == param)].gap.to_numpy()
        r = CI[(CI.season == season) & (CI.param == param)
               & (CI.cls == cls)].iloc[0]
        b = ax.boxplot([v], positions=[p], widths=0.5, patch_artist=True,
                       showfliers=False, manage_ticks=False,
                       medianprops=dict(color="k", lw=1.6))["boxes"][0]
        b.set(facecolor="C0" if r.ci_lo > 0 else ("C3" if r.ci_hi < 0 else "0.7"),
              alpha=0.45, edgecolor="0.3")
        ax.scatter(p + rng.uniform(-0.07, 0.07, len(v)), v, s=8, color="0.25",
                   alpha=0.5, zorder=3)
        ax.plot([p + 0.33] * 2, [r.ci_lo, r.ci_hi], color="C1", lw=3,
                solid_capstyle="butt", zorder=4)
    ax.axhline(0.0, color="k", lw=1.0, ls="--")
    ax.set_xticks(range(1, len(CASES) + 1))
    ax.set_xticklabels([f"{p}\n{s}" for s, p in CASES], fontsize=9)
    ax.set_xlim(0.5, len(CASES) + 0.5)
    ax.set_ylim(lo - pad, hi + pad)
    ax.axvline(4.5, color="0.5", lw=0.8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylabel("FRAC(Varda-single) - FRAC(Multistep), per station")
    ax.legend(handles=[
        Patch(facecolor="C0", alpha=0.45, label="Varda retains more, CI excludes 0"),
        Patch(facecolor="C3", alpha=0.45, label="Multistep retains more, CI excludes 0"),
        Patch(facecolor="0.7", alpha=0.45, label="CI includes 0")],
        fontsize=8, loc="upper left")
    fig.suptitle(f"Station class '{cls}' (n={sc.nat_abbr.nunique()}): paired 3-6 h "
                 "variance retention difference\npositive = Varda retains more. "
                 "Orange bar: 95 % bootstrap CI on the class median, over inits",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(D / f"paired_gap_class_{cls}.png", dpi=150)
    plt.close(fig)
    print(f"wrote {D}/paired_gap_class_{cls}.png")
