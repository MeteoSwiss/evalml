"""One figure for the whole Multistep-vs-Varda-single comparison.

Same skill score as skill_by_station_class.py,

    SS = 1 - MSE(Multistep-stage-C) / MSE(Varda-single-stage-C)

against REA-L, but with the station classes collapsed: one box per
(season, variable), both seasons side by side for each variable. This is the
summary view, showing the season effect that the per-class figures spread over
six files.

Because classes are not used here, PFA is kept (it has no entry in the
spaziurat.lf class table), so this figure covers all 145 stations while the
per-class figures cover 144.

`--candidates` scores several models against the same Varda-single baseline
(e.g. Multistep plus the ICON baselines); boxes are then coloured by model and
hatched for summer, and only (init, station) cells valid for every candidate
are used. `--suffix` picks a lead-time-restricted RMSE set such as `_0-33h`.
`--from-table` scores the time-scale component instead of the full series, from
a joint_variance_skill.py table.csv (its bmse_* columns): SS = 1 - band MSE ratio.
That MSE favours a smoother model when timing is poor (FINDINGS section 3).

    uv run workflow/scripts/skill_overview.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from spectra_forecast import FORECAST_SOURCES, TRUTH_LABEL

BASELINE, CANDIDATE = FORECAST_SOURCES[0], FORECAST_SOURCES[1]
SEASONS = ("winter2024", "summer2024")
PARAMS = ("T_2M", "TD_2M", "SP_10M")
COLOR = {"winter2024": "C0", "summer2024": "C1"}


def cell_skill(rmse_dir: Path, season: str, param: str, cands=(CANDIDATE,),
               suffix: str = "") -> pd.DataFrame:
    """Per-(init, station) skill score per candidate, from the vs-REA-L RMSE table."""
    csv = rmse_dir / f"rmse_by_station_init_{season}_{param}_vs-{TRUTH_LABEL}{suffix}.csv"
    if not csv.exists():
        raise SystemExit(f"{csv} missing: run rmse_by_station_init.py "
                         f"--reference {TRUTH_LABEL} first")
    d = pd.read_csv(csv, dtype={"init": str})
    w = d.pivot_table(index=["init", "station"], columns="source",
                      values="rmse").dropna(subset=[BASELINE, *cands])
    return pd.DataFrame({c: 1.0 - (w[c] ** 2) / (w[BASELINE] ** 2) for c in cands})


def table_skill(table: pd.DataFrame, season: str, param: str, cands) -> pd.DataFrame:
    """Per-(init, station) skill on the time-scale component, from table.csv."""
    from joint_variance_skill import key
    g = table[(table.season == season) & (table.param == param)]
    return pd.DataFrame({c: 1.0 - g[f"bmse_{key(c)}"].to_numpy()
                         / g[f"bmse_{key(BASELINE)}"].to_numpy() for c in cands})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rmse-dir", type=Path, default=Path("output/rmse"))
    ap.add_argument("--out-dir", type=Path, default=Path("output/skill"))
    ap.add_argument("--candidates", nargs="+", default=[CANDIDATE])
    ap.add_argument("--params", nargs="+", default=list(PARAMS))
    ap.add_argument("--suffix", default="", help="RMSE file suffix, e.g. _0-33h")
    ap.add_argument("--note", default="", help="extra line for the title")
    ap.add_argument("--from-table", type=Path, help="joint_variance_skill table.csv")
    args = ap.parse_args()
    cands, params = args.candidates, args.params
    multi = len(cands) > 1

    if args.from_table:
        tbl = pd.read_csv(args.from_table)
        ss = {(s, p): table_skill(tbl, s, p, cands) for s in SEASONS for p in params}
    else:
        ss = {(s, p): cell_skill(args.rmse_dir, s, p, cands, args.suffix)
              for s in SEASONS for p in params}

    rows = []
    for (s, p), df in ss.items():
        for c in cands:
            v = df[c]
            rows.append({"season": s, "param": p, **({"candidate": c} if multi else {}),
                         "cells": len(v),
                         "median_SS": v.median(), "mean_SS": v.mean(),
                         "q25": v.quantile(.25), "q75": v.quantile(.75),
                         "frac_SS>0": (v > 0).mean()})
    tab = pd.DataFrame(rows)
    print(f"\n=== SS = 1 - MSE({'/'.join(cands)}) / MSE({BASELINE}), truth {TRUTH_LABEL} ===")
    print("positive = candidate better; all 145 stations\n")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tab.to_csv(args.out_dir / f"skill_overview{args.suffix}.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(9, 2.2 * len(params) * len(cands) / 2 + 3), 5.5))
    # One box per (candidate, season) per variable, offset around the variable's tick.
    groups = [(c, s) for c in cands for s in SEASONS]
    step = 0.68 / len(groups)
    for k, (c, season) in enumerate(groups):
        pos = np.arange(len(params)) + (k - (len(groups) - 1) / 2) * step
        bp = ax.boxplot([ss[(season, p)][c].values for p in params],
                        positions=pos, widths=0.6 / len(groups), showfliers=False,
                        patch_artist=True, medianprops=dict(color="k", lw=1.8))
        for box in bp["boxes"]:
            if multi:
                box.set(facecolor=f"C{cands.index(c)}", alpha=0.55,
                        hatch="//" if season == SEASONS[1] else None)
            else:
                box.set(facecolor=COLOR[season], alpha=0.55)
        bp["boxes"][0].set_label(f"{c}, {season}" if multi else season)
        # a median below the fixed y range would vanish silently: say where it is
        for x, p in zip(pos, params):
            m = ss[(season, p)][c].median()
            if m < -1.5:
                ax.annotate(f"{m:.0f}", (x, -1.5), xytext=(x, -1.38), ha="center",
                            fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.8))

    ax.axhline(0, color="0.3", lw=1.4, ls="--")
    ax.set_xticks(np.arange(len(params)), params)
    ax.set_xlim(-0.6, len(params) - 0.4)
    ax.set_ylim(-1.5, 1)  # same scale as the per-class figures
    ax.set_ylabel(f"SS = 1 - MSE({'model' if multi else CANDIDATE}) / MSE({BASELINE})")
    ax.set_xlabel("variable")
    ax.legend(fontsize=8 if multi else 9, loc="lower left", ncol=len(cands) if multi else 1)
    ax.grid(alpha=0.3, axis="y")
    who = "Models" if multi else "Multistep"
    fig.suptitle(f"{who} vs Varda-single, truth {TRUTH_LABEL}, station "
                 f"classes pooled\npositive = {'model' if multi else 'Multistep'} better; boxes are "
                 f"per-(init, station) cells, whiskers 1.5 IQR, outliers hidden"
                 + (f"\n{args.note}" if args.note else ""),
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92 if not args.note else 0.88))
    out = args.out_dir / f"skill_overview{args.suffix}.png"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out} and the matching .csv")


if __name__ == "__main__":
    main()
