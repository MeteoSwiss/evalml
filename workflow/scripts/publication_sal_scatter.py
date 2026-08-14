"""Publication SAL Structure–Amplitude scatter figure.

One stacked panel per participant (candidate first, then baselines), showing the
per-initialisation SAL (Structure vs Amplitude), coloured by season, with marker
size proportional to the Location term. Supports the paper claim that warm-season
(convective) precipitation is forecast worse than cool-season (synoptic) cases.

Data are the per-init SAL CSVs written by the SAL verification
(``verification_sal.py``): one CSV per (participant, param, lead time), one row
per initialisation, columns ``reftime,S,A,L,fcst_mean,truth_mean`` with a
``#``-commented metadata header. For each participant the lead-time CSVs are
combined and reduced to one point per init: the median S/A/L over the wet windows
(a wet-case filter drops inits whose summed truth precipitation is below a
threshold, so "well forecast" never means "dry and trivially right").

Outputs (into ``--output``):
  publication_sal_scatter.{png,pdf}          the scatter figure (single column)
  publication_sal_scatter_stats_{split}.txt  Mann–Whitney U + Cliff's delta table
  publication_sal_scatter.html               report index
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

_script_dir = Path(__file__).resolve().parent
sys.path.append(str(_script_dir))
plt.style.use(_script_dir / "publication.mplstyle")

# Named season splits: group label -> months. JJA vs Nov–Dec gives the sharpest
# convective/synoptic contrast (shoulder months excluded).
SPLITS = {
    "jja-novdec": {"JJA": (6, 7, 8), "Nov-Dec": (11, 12)},
}
GROUP_COLORS = ["#e07b39", "#3d65a5"]  # warm-ish orange, cool blue
MARKER_ALPHA = 0.8


def load_participant_medians(
    label: str, csv_paths: list[str], leadtimes: list[int], min_truth_mm: float
) -> pd.DataFrame:
    """Reduce a participant's per-lead-time SAL CSVs to one row per wet init.

    Reads each ``{param}_{leadtime}`` CSV (rows = inits), concatenates them, and
    groups by initialisation to the median S/A/L over the wet (non-NaN) windows.
    ``reftime`` is read back as an int64 by ``pandas`` and cast to a string before
    the month is parsed. An init is kept only if the truth precipitation summed
    over the lead times is at least ``min_truth_mm``.
    """
    if len(csv_paths) != len(leadtimes):
        raise ValueError(
            f"{label}: got {len(csv_paths)} CSVs for {len(leadtimes)} lead times."
        )
    frames = []
    for lt, path in zip(leadtimes, csv_paths):
        df = pd.read_csv(path, comment="#")
        df["reftime"] = df["reftime"].astype(str)
        df["lt"] = lt
        frames.append(df)
    long = pd.concat(frames, ignore_index=True)

    rows = []
    for reftime, g in long.groupby("reftime"):
        truth_mm = float(g["truth_mean"].sum())
        valid = g.dropna(subset=["S", "A", "L"])
        if valid.empty:
            continue
        s, a, ll = valid["S"].median(), valid["A"].median(), valid["L"].median()
        rows.append(
            {
                "label": label,
                "reftime": reftime,
                "month": int(reftime[4:6]),
                "S": s,
                "A": a,
                "L": ll,
                "rho": float(np.sqrt(s**2 + a**2 + ll**2)),
                "truth_mm": truth_mm,
                "n_lt": len(valid),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out[out["truth_mm"] >= min_truth_mm].reset_index(drop=True)


def assign_group(df: pd.DataFrame, split: dict) -> pd.DataFrame:
    month_to_group = {m: g for g, months in split.items() for m in months}
    return df.assign(group=df["month"].map(month_to_group)).dropna(subset=["group"])


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta effect size: P(x>y) - P(x<y) over all pairs."""
    diff = np.subtract.outer(np.asarray(x, float), np.asarray(y, float))
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


def stats_table(df: pd.DataFrame, labels: list[str], split: dict) -> pd.DataFrame:
    g1, g2 = list(split)
    rows = []
    for label in labels:
        for metric in ("S", "A", "L", "rho"):
            x = df.loc[(df["label"] == label) & (df["group"] == g1), metric].values
            y = df.loc[(df["label"] == label) & (df["group"] == g2), metric].values
            if len(x) < 3 or len(y) < 3:
                continue
            u, p = mannwhitneyu(x, y, alternative="two-sided")
            rows.append(
                {
                    "metric": metric,
                    "participant": label,
                    f"n {g1}": len(x),
                    f"n {g2}": len(y),
                    f"med {g1}": np.median(x),
                    f"med {g2}": np.median(y),
                    "MWU": u,
                    "p": p,
                    "cliffs_delta": cliffs_delta(x, y),
                }
            )
    return pd.DataFrame(rows).round(4)


def plot_scatter(
    df: pd.DataFrame,
    labels: list[str],
    split: dict,
    param: str,
    leadtimes: list[int],
    annotate: dict,
    out_paths: list[Path],
):
    groups = list(split)
    candidate = labels[0]
    fig, axes = plt.subplots(
        len(labels), 1, figsize=(4.7, 4.3 * len(labels)), sharex=True, sharey=True
    )
    if len(labels) == 1:
        axes = [axes]

    for k, (ax, label) in enumerate(zip(axes, labels)):
        for g, c in zip(groups, GROUP_COLORS):
            sub = df[(df["label"] == label) & (df["group"] == g)]
            ax.scatter(
                sub["S"],
                sub["A"],
                s=20 + 900 * sub["L"],
                color=c,
                alpha=MARKER_ALPHA,
                edgecolor="black",
                linewidth=0.4,
            )
            if not sub.empty:  # season centroid (median S, median A)
                ax.scatter(
                    sub["S"].median(),
                    sub["A"].median(),
                    marker="D",
                    s=55,
                    color=c,
                    edgecolor="black",
                    linewidth=0.65,
                    zorder=6,
                )
        # per-season mean±sd in one framed box, one coloured line per group
        stat_lines = []
        for g, c in zip(groups, GROUP_COLORS):
            sub = df[(df["label"] == label) & (df["group"] == g)]
            if sub.empty:
                continue
            txt = (
                f"{g}:  "
                f"S={sub['S'].mean():+.2f}$\\pm${sub['S'].std():.2f}  "
                f"A={sub['A'].mean():+.2f}$\\pm${sub['A'].std():.2f}  "
                f"L={sub['L'].mean():.2f}$\\pm${sub['L'].std():.2f}"
            )
            stat_lines.append(TextArea(txt, textprops={"color": c, "fontsize": 6.8}))
        if stat_lines:
            box = AnchoredOffsetbox(
                loc="lower left",
                bbox_to_anchor=(0.03, 0.03),
                bbox_transform=ax.transAxes,
                borderpad=0.0,
                frameon=True,
                child=VPacker(children=stat_lines, pad=1.5, sep=2.5),
            )
            box.patch.set(facecolor="white", edgecolor="black", linewidth=0.5)
            box.set_zorder(7)
            ax.add_artist(box)
        # annotate candidate panel with the selected cases
        if label == candidate:
            for init, txt in annotate.items():
                sub = df[(df["label"] == label) & (df["reftime"] == init)]
                if not sub.empty:
                    ax.annotate(
                        txt,
                        (sub["S"].iloc[0], sub["A"].iloc[0]),
                        textcoords="offset points",
                        xytext=(6, 4),
                        ha="left",
                        fontsize=7.5,
                        zorder=8,
                    )
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.axvline(0, color="grey", lw=0.8, ls="--")
        ax.set_xlim(-2.1, 2.1)
        ax.set_ylim(-2.1, 2.1)
        ax.set_ylabel("Amplitude A")
        ax.set_title(f"({chr(97 + k)}) {label}", fontsize=11)
        ax.set_aspect("equal")
    axes[-1].set_xlabel("Structure S")

    # single combined legend (seasons + centroid + L size scale) in the first panel
    for g, c in zip(groups, GROUP_COLORS):
        n = int(((df["group"] == g) & (df["label"] == candidate)).sum())
        axes[0].scatter(
            [],
            [],
            s=70,
            color=c,
            alpha=MARKER_ALPHA,
            edgecolor="black",
            linewidth=0.4,
            label=f"{g} (n={n})",
        )
    axes[0].scatter(
        [],
        [],
        marker="D",
        s=35,
        color="grey",
        edgecolor="black",
        linewidth=0.6,
        label="season median",
    )
    for lval in (0.05, 0.15, 0.3):
        axes[0].scatter(
            [],
            [],
            s=20 + 900 * lval,
            color="grey",
            alpha=MARKER_ALPHA,
            edgecolor="black",
            linewidth=0.4,
            label=f"L = {lval}",
        )
    axes[0].legend(
        fontsize=7, loc="upper right", framealpha=0.9, borderpad=0.6, labelspacing=0.55
    )

    lt_lo, lt_hi = min(leadtimes), max(leadtimes)
    fig.suptitle(
        f"SAL per init ({param}, median over lead times {lt_lo}–{lt_hi} h)\n"
        "rainy cases, marker size = Location L",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    for path in out_paths:
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)


def _parse_annotate(spec: str | None) -> dict:
    if not spec:
        return {}
    out = {}
    for entry in spec.split(","):
        if "=" in entry:
            init, label = entry.split("=", 1)
            out[init.strip()] = label.strip()
    return out


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--participant",
        nargs=2,
        action="append",
        metavar=("LABEL", "CSVS"),
        required=True,
        help="Participant label and comma-separated SAL CSVs (ordered like "
        "--leadtimes). Repeat; the first participant is the candidate panel.",
    )
    parser.add_argument(
        "--leadtimes", required=True, help="Comma-separated lead times (hours)."
    )
    parser.add_argument("--param", default="TOT_PREC6")
    parser.add_argument("--season-split", dest="season_split", default="jja-novdec")
    parser.add_argument("--min-truth-mm", dest="min_truth_mm", type=float, default=2.0)
    parser.add_argument("--annotate", default=None, help="init=label,init=label ...")
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    args = parser.parse_args()

    if args.season_split not in SPLITS:
        raise ValueError(
            f"Unknown season split {args.season_split!r}; available: {list(SPLITS)}."
        )
    split = SPLITS[args.season_split]
    leadtimes = [int(x) for x in args.leadtimes.split(",")]
    annotate = _parse_annotate(args.annotate)

    labels = [label for label, _ in args.participant]
    frames = []
    for label, csvs in args.participant:
        med = load_participant_medians(
            label, csvs.split(","), leadtimes, args.min_truth_mm
        )
        if med.empty:
            print(f"WARNING: no wet inits for {label}; panel will be empty.")
        frames.append(med)
    base = pd.concat(frames, ignore_index=True)
    df = assign_group(base, split)
    n_inits = df[df["label"] == labels[0]]["reftime"].nunique()
    print(f"{n_inits} candidate inits in split {args.season_split!r}; {len(df)} rows.")

    args.output.mkdir(parents=True, exist_ok=True)

    table = stats_table(df, labels, split)
    stats_path = args.output / f"publication_sal_scatter_stats_{args.season_split}.txt"
    with stats_path.open("w") as fh:
        g1, g2 = list(split)
        fh.write(
            f"Seasonally stratified SAL statistics - split '{args.season_split}'\n"
            f"Groups: " + "; ".join(f"{g}: months {m}" for g, m in split.items()) + "\n"
            "Sampling unit: per-init median over wet lead-time windows; rainy inits "
            f"only (truth summed over lead times >= {args.min_truth_mm} mm).\n"
            "Test: two-sided Mann-Whitney U; effect size: Cliff's delta "
            "(P(g1>g2) - P(g1<g2); |0.33|/|0.47| ~ medium/large).\n\n"
        )
        fh.write(table.to_string(index=False))
        fh.write("\n")
    print(f"Wrote {stats_path}")

    out_png = args.output / "publication_sal_scatter.png"
    out_pdf = args.output / "publication_sal_scatter.pdf"
    plot_scatter(df, labels, split, args.param, leadtimes, annotate, [out_pdf, out_png])

    (args.output / "publication_sal_scatter.html").write_text(
        f'<!doctype html><html><body><img src="{out_png.name}" '
        'style="max-width:100%"></body></html>'
    )


if __name__ == "__main__":
    main()
