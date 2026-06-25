from pathlib import Path

import earthkit.plots  # noqa: F401  applies the paper styling (Roboto, soft grid, no top/right spines) via rcParams
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D

DATA_DIR = Path("/scratch/mch/apennino/evalml/output/paper/data")

# Lines overlaid in every panel. Listed in legend order; `z` sets draw order.
# Each series gets a distinct marker shape so the figure reads in greyscale and
# for colourblind readers (Nature/Science guidance: encode series by shape AND
# colour, not colour alone).
SERIES = [
    {
        "nc": DATA_DIR / "runs/forecaster-c304-a7a3/verif_aggregated.nc",
        "source": "stage_E_icon_1km_cutoff_edges_subgrid_horography",
        "label": "Varda",
        "color": "#CC3311",  # Okabe-Ito vermillion
        "marker": "o",
        "lw": 1.5,
        "z": 5,
    },
    {
        "nc": DATA_DIR / "baselines/ICON-CH1-EPS/verif_aggregated.nc",
        "source": "ICON-CH1-ctrl",
        "label": "ICON-CH1",
        "color": "#0077BB",  # Okabe-Ito blue
        "marker": "o",
        "lw": 1.2,
        "z": 4,
    },
    {
        "nc": DATA_DIR / "baselines/ICON-CH2-EPS/verif_aggregated.nc",
        "source": "ICON-CH2-ctrl",
        "label": "ICON-CH2",
        "color": "#000000",  # black
        "marker": "o",
        "lw": 1.2,
        "z": 3,
    },
]

REGION = "all"  # whole domain
SEASON = "all"  # all seasons
INIT_HOUR = -999  # all initialisation hours pooled

# Columns of the figure (left -> right). Column header = full name (wrapped) + unit.
VARIABLES = ["T_2M", "U_10M", "PMSL", "TOT_PREC"]
COL_NAME = {
    "T_2M": "2 m\ntemperature",
    "TD_2M": "2 m\ndewpoint",
    "U_10M": "10 m\nwind (U)",
    "V_10M": "10 m wind (V)",
    "PMSL": "Mean sea-level\npressure",
    "PS": "Surface\npressure",
    "TOT_PREC": "Total\nprecipitation",
}
UNITS = {
    "T_2M": "K",
    "TD_2M": "K",
    "U_10M": "m s$^{-1}$",
    "V_10M": "m s$^{-1}$",
    "PMSL": "hPa",
    "PS": "hPa",
    "TOT_PREC": "mm",
}
SCALE = {
    "PMSL": 1e-2,
    "PS": 1e-2,
    "TOT_PREC": 1e3,
}

# Rows of the figure (top -> bottom): metric and its row label.
METRICS = ["RMSE", "STDE", "BIAS"]
METRIC_TITLE = {"RMSE": "RMSE", "BIAS": "Bias", "STDE": "Standard\nderror"}

START_LEAD = 6 
LEAD_TICKS = [6, 36, 66, 96, 120]


FIG_W, FIG_H = 10.0, 5.0
FS_TICK, FS_COL, FS_ROW, FS_LEG, FS_AXIS = 10, 11, 12, 11, 12

# Markers: one per 6-hourly sample.
MS, MARKEVERY = 3.0, 1


def select_line(ds: xr.Dataset, key: str, source: str) -> np.ndarray:
    """Return the metric vs lead-time line for the configured slice."""
    da = ds[key]
    sel = {"region": REGION, "season": SEASON, "init_hour": INIT_HOUR, "source": source}
    if "eps" in da.dims:
        sel["eps"] = da["eps"].values[0]
    return da.sel(**sel).values


def style_panel(ax) -> None:
    """Apply the sleek shared look to one panel."""
    ax.grid(True, color="0.90", linewidth=0.7, zorder=0)
    ax.margins(y=0.12)  # vertical breathing room inside each panel
    ax.set_xticks(LEAD_TICKS)
    ax.set_xlim(START_LEAD - 5, LEAD_TICKS[-1] + 5)  # let the 6 h / 120 h points clear the edges
    ax.tick_params(length=3, labelsize=FS_TICK)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("0.4")
        ax.spines[spine].set_linewidth(0.8)


def plot_line(ax, lead_h, values, s) -> None:
    """Draw one metric series as a coloured line with shape-coded markers."""
    ax.plot(
        lead_h, values, color=s["color"], linewidth=s["lw"], zorder=s["z"],
        marker=s["marker"], markersize=MS, markevery=MARKEVERY,
        markerfacecolor=s["color"], markeredgecolor=s["color"],
    )


def main(outstem: Path) -> None:
    outstem.parent.mkdir(parents=True, exist_ok=True)

    series = []
    for s in SERIES:
        ds = xr.open_dataset(s["nc"])
        lead_h = (ds["lead_time"].values / np.timedelta64(1, "h")).astype(float)
        keep = lead_h >= START_LEAD  # each series keeps its own lead times
        series.append({**s, "ds": ds, "lead_h": lead_h[keep], "keep": keep})

    nrows, ncols = len(METRICS), len(VARIABLES)
    fig, axes = plt.subplots(nrows, ncols, figsize=(FIG_W, FIG_H), constrained_layout=True)
    fig.get_layout_engine().set(h_pad=0.03, w_pad=0.00, hspace=0.04, wspace=0.02)

    for j, var in enumerate(VARIABLES):
        scale = SCALE.get(var, 1.0)
        for i, metric in enumerate(METRICS):
            ax = axes[i, j]
            if metric == "BIAS":
                ax.axhline(0.0, color="0.6", linestyle=":", linewidth=1.0, zorder=1)

            for s in series:  # RMSE / STDE / Bias all share the variable's units
                values = select_line(s["ds"], f"{var}.{metric}", s["source"])[s["keep"]] * scale
                plot_line(ax, s["lead_h"], values, s)
            style_panel(ax)
            if metric == "BIAS":  # keep the zero reference line in view
                lo, hi = ax.get_ylim()
                ax.set_ylim(min(lo, 0.0), max(hi, 0.0))

            if i == 0:  # column header: full variable name + unit
                ax.set_title(f"{COL_NAME[var]}\n[{UNITS[var]}]", fontsize=FS_COL, pad=6)
            if i != nrows - 1:
                ax.tick_params(labelbottom=False)
            if j == 0:  # row label: metric
                ax.set_ylabel(METRIC_TITLE[metric], fontsize=FS_ROW,) # fontweight="bold")

    fig.supxlabel("Lead time [h]", fontsize=FS_AXIS)

    handles = [
        Line2D([0], [0], color=s["color"], linewidth=s["lw"], marker=s["marker"],
               markersize=MS, markerfacecolor=s["color"], markeredgecolor=s["color"],
               label=s["label"])
        for s in SERIES
    ]
    fig.legend(
        handles=handles, loc="outside upper center", ncol=len(SERIES),
        fontsize=FS_LEG, frameon=False, handlelength=2.4,
        handletextpad=0.6, columnspacing=2.0,
    )

    for ext, dpi in (("png", 300), ("pdf", None)):
        out = outstem.with_suffix(f".{ext}")
        fig.savefig(out, dpi=dpi)
        print(f"saved: {out}")


if __name__ == "__main__":
    main(Path("figures/verif_skill_leadtime"))
