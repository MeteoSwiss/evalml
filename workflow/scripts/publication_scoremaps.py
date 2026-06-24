"""Publication-quality 2×2 relative skill score map panel.

Plots MSE skill and the bias² contribution to MSE skill for a 2×2 grid of
(param × metric) combinations.  Positive skill = candidate is better.

Rows:    parameters      (default T_2M, SP_10M)
Columns: derived metrics (default MSE_SKILL, BIAS_CONTRIB)

Metrics
-------
MSE_SKILL    : 1 − RMSE²_cand / RMSE²_base
BIAS_CONTRIB : (BIAS²_base − BIAS²_cand) / RMSE²_base
               Additive complement: BIAS_CONTRIB + STDE²_contrib = MSE_SKILL
"""

import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

import earthkit.plots as ekp
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr
from cartopy.mpl.gridliner import Gridliner
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex

_script_dir = Path(__file__).resolve().parent
sys.path.append(str(_script_dir))
sys.path.append(str(_script_dir.parent.parent / "src"))

plt.style.use(_script_dir / "publication.mplstyle")

from plotting import DOMAINS, StatePlotter  # noqa: E402
from publication_style import (  # noqa: E402
    COLOR_SKILL_BASELINE_BETTER,
    COLOR_SKILL_MODEL_BETTER,
    PARAM_LABELS,
    SCORE_LABELS,
    SKILL_CMAP,
    SKILL_GREY,
    SKILL_LEVELS,
)

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Tighter geographic crop for publication: roughly equal visual margins around Switzerland.
_PUB_EXTENTS = {
    "switzerland": [5.6, 10.8, 45.6, 48.0],
}


def _darken(hex_color: str, factor: float = 0.6) -> str:
    """Return a darkened hex color by scaling RGB channels."""
    r, g, b = mcolors.to_rgb(hex_color)
    return mcolors.to_hex((r * factor, g * factor, b * factor))


# Sentinel boundary that covers all realistic skill values (SP_10M can reach ~1e9).
_SENTINEL = 1e15


def _build_skill_artifacts():
    """Return (ekp_style, mpl_cmap, mpl_norm) for the discrete skill colormap.

    Inner bins (within ±0.35) use SKILL_CMAP-sampled colors with a white neutral band.
    Outer bins (|skill| > 0.35) use a darkened extreme color both in the map (via
    sentinel levels in the earthkit Style) and in the colorbar triangular tips (via
    cmap.set_under / set_over).
    """
    n_side = (len(SKILL_LEVELS) - 2) // 2  # bins per side excluding the neutral bin
    reds = [to_hex(SKILL_CMAP(i / (2 * n_side))) for i in range(n_side)]
    blues = [to_hex(SKILL_CMAP((n_side + 1 + i) / (2 * n_side))) for i in range(n_side)]
    inner_colors = reds + [SKILL_GREY] + blues
    darker_red = _darken(reds[0])
    darker_blue = _darken(blues[-1])

    # Earthkit Style: sentinel outer levels ensure values beyond ±0.3 fall into the
    # dedicated darker outer bins rather than being clipped to the adjacent bin color.
    sentinel_levels = [-_SENTINEL] + list(SKILL_LEVELS) + [_SENTINEL]
    all_colors = [darker_red] + inner_colors + [darker_blue]
    style = ekp.styles.Style(
        levels=sentinel_levels, colors=all_colors, extend="neither", units="skill"
    )

    # Matplotlib colorbar: inner levels only, with darker under/over for the tips.
    cmap = mcolors.ListedColormap(inner_colors)
    cmap.set_under(darker_red)
    cmap.set_over(darker_blue)
    norm = mcolors.BoundaryNorm(SKILL_LEVELS, ncolors=len(inner_colors))

    return style, cmap, norm


def _load_raw(
    nc_file: Path, param: str, score: str, season: str, init_hour: int
) -> np.ndarray:
    """Load one raw score variable and return as a flat 1-D array."""
    ds = xr.open_dataset(nc_file)
    var = f"{param}.{score}"
    if var not in ds:
        raise KeyError(
            f"Variable {var!r} not found in {nc_file}. Available: {list(ds.data_vars)}"
        )
    return ds[var].sel(season=season, init_hour=init_hour).values.ravel()


def _compute_panel(
    metric: str,
    cand_file: Path,
    base_file: Path,
    param: str,
    season: str,
    init_hour: int,
) -> np.ndarray:
    """Return flat skill values for one panel.

    MSE_SKILL    : 1 − RMSE²_cand / RMSE²_base
    BIAS_CONTRIB : (BIAS²_base − BIAS²_cand) / RMSE²_base
    Fallback     : 1 − score_cand / score_base  (raw ratio, for backwards compat)
    """
    kw = dict(param=param, season=season, init_hour=init_hour)
    with np.errstate(invalid="ignore", divide="ignore"):
        if metric == "MSE_SKILL":
            rmse_c = _load_raw(cand_file, score="RMSE", **kw)
            rmse_b = _load_raw(base_file, score="RMSE", **kw)
            return 1.0 - rmse_c**2 / rmse_b**2
        if metric == "BIAS_CONTRIB":
            bias_c = _load_raw(cand_file, score="BIAS", **kw)
            bias_b = _load_raw(base_file, score="BIAS", **kw)
            rmse_b = _load_raw(base_file, score="RMSE", **kw)
            return (bias_b**2 - bias_c**2) / rmse_b**2
        # Fallback: plain ratio skill for any raw score name
        cand = _load_raw(cand_file, score=metric, **kw)
        base = _load_raw(base_file, score=metric, **kw)
        return 1.0 - cand / base


def _remove_latlon_labels(ax) -> None:
    """Remove all lat/lon label artifacts from a cartopy GeoAxes.

    Cartopy 0.25+ stores Gridliner objects in ax._children (not ax._gridliners).
    """
    # Suppress gridliner labels (earthkit schema default: draw_labels=['bottom','left'])
    for child in getattr(ax, "_children", []) + getattr(ax, "_gridliners", []):
        if not isinstance(child, Gridliner):
            continue
        try:
            child.left_labels = child.right_labels = False
            child.top_labels = child.bottom_labels = False
        except AttributeError:
            try:
                child.xlabels_top = child.xlabels_bottom = False
                child.ylabels_left = child.ylabels_right = False
            except AttributeError:
                pass
    # Clear any xlabel/ylabel set by earthkit
    ax.set_xlabel("")
    ax.set_ylabel("")


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate_files",
        nargs="+",
        required=True,
        help="Scoremap NC files for the candidate, one per param (same order as --params).",
    )
    parser.add_argument(
        "--baseline_files",
        nargs="+",
        required=True,
        help="Scoremap NC files for the baseline, one per param (same order as --params).",
    )
    parser.add_argument(
        "--params",
        default="T_2M,SP_10M",
        help="Comma-separated parameter names matching the order of the NC files.",
    )
    parser.add_argument(
        "--scores",
        default="MSE_SKILL,BIAS_CONTRIB",
        help="Comma-separated metric names (columns of the 2×2 panel). "
        "Supports MSE_SKILL, BIAS_CONTRIB, or any raw score (RMSE, STDE, …).",
    )
    parser.add_argument("--candidate_label", default="Varda-Single")
    parser.add_argument("--baseline_label", default="ICON-CH1-CTRL")
    parser.add_argument(
        "--leadtime", type=int, default=24, help="Lead time in hours (title only)."
    )
    parser.add_argument("--season", default="all")
    parser.add_argument("--region", default="switzerland")
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    args = parser.parse_args()

    params = [p.strip() for p in args.params.split(",")]
    scores = [s.strip() for s in args.scores.split(",")]
    candidate_files = [Path(f) for f in args.candidate_files]
    baseline_files = [Path(f) for f in args.baseline_files]
    init_hour = -999  # "all" sentinel

    if len(candidate_files) != len(params):
        parser.error(
            f"Got {len(candidate_files)} candidate files but {len(params)} params."
        )
    if len(baseline_files) != len(params):
        parser.error(
            f"Got {len(baseline_files)} baseline files but {len(params)} params."
        )

    nrows = len(params)
    ncols = len(scores)

    ds0 = xr.open_dataset(candidate_files[0])
    lons = ds0["longitude"].values
    lats = ds0["latitude"].values
    LOG.info(
        "Grid: %d points, lon [%.2f, %.2f], lat [%.2f, %.2f]",
        len(lons),
        lons.min(),
        lons.max(),
        lats.min(),
        lats.max(),
    )

    args.output.mkdir(parents=True, exist_ok=True)
    plotter = StatePlotter(lons, lats, args.output)

    domain = DOMAINS.get(args.region, DOMAINS["switzerland"])
    if args.region in _PUB_EXTENTS:
        domain = {**domain, "extent": _PUB_EXTENTS[args.region]}

    fig = plotter.init_geoaxes(
        projection=domain["projection"],
        bbox=domain["extent"],
        nrows=nrows,
        ncols=ncols,
        name=args.region,
        size=(6 * ncols, 5 * nrows),
    )

    style, skill_cmap, skill_norm = _build_skill_artifacts()
    mpl_axes = []

    for row, (param, cand_file, base_file) in enumerate(
        zip(params, candidate_files, baseline_files)
    ):
        for col, score in enumerate(scores):
            skill_vals = _compute_panel(
                score, cand_file, base_file, param, args.season, init_hour
            )

            LOG.info(
                "%s %s: skill min=%.3f  max=%.3f  n_nan=%d / %d",
                param,
                score,
                np.nanmin(skill_vals),
                np.nanmax(skill_vals),
                int(np.isnan(skill_vals).sum()),
                skill_vals.size,
            )

            subplot = fig.add_map(row=row, column=col)

            if np.all(np.isnan(skill_vals)):
                LOG.warning("All-NaN for %s %s — plotting empty panel.", param, score)
                subplot.ax.set_facecolor("#cccccc")
                subplot.standard_layers()
            else:
                plotter.plot_field(subplot, skill_vals, style=style, colorbar=False)

            subplot.coastlines(edgecolor="black", linewidth=1.0, zorder=5)
            subplot.borders(edgecolor="black", linewidth=0.5, zorder=5)

            _remove_latlon_labels(subplot.ax)
            mpl_axes.append(subplot.ax)

            param_lbl = PARAM_LABELS.get(param, param)
            score_lbl = SCORE_LABELS.get(score, score)
            subplot.title(f"{param_lbl} — {score_lbl}")

    # Single shared horizontal colorbar below all panels
    mpl_fig = fig.fig
    sm = plt.cm.ScalarMappable(cmap=skill_cmap, norm=skill_norm)
    sm.set_array([])
    cbar = mpl_fig.colorbar(
        sm,
        ax=mpl_axes,
        orientation="horizontal",
        location="bottom",
        fraction=0.04,
        pad=0.05,
        aspect=50,
        extend="both",
    )
    cbar.set_ticks(SKILL_LEVELS)
    cbar.set_ticklabels([f"{v:g}" for v in SKILL_LEVELS])
    cbar.set_label("Skill  (1 − model / baseline)", labelpad=4)

    # Draw the figure so text positions are finalised, then query the label position
    # to align the coloured "better/worse" texts on the same horizontal line.
    mpl_fig.canvas.draw()
    renderer = mpl_fig.canvas.get_renderer()
    label_bbox = cbar.ax.xaxis.label.get_window_extent(renderer)
    fig_height_px = mpl_fig.get_figheight() * mpl_fig.dpi
    y_fig = label_bbox.y0 / fig_height_px  # figure fraction at top of label text

    cb_pos = cbar.ax.get_position()
    mpl_fig.text(
        cb_pos.x0,
        y_fig,
        f"{args.baseline_label} better",
        ha="left",
        va="top",
        color=COLOR_SKILL_BASELINE_BETTER,
        fontsize=plt.rcParams["font.size"],
    )
    mpl_fig.text(
        cb_pos.x1,
        y_fig,
        f"{args.candidate_label} better",
        ha="right",
        va="top",
        color=COLOR_SKILL_MODEL_BETTER,
        fontsize=plt.rcParams["font.size"],
    )

    out_png = args.output / "publication_scoremaps.png"
    out_pdf = args.output / "publication_scoremaps.pdf"
    fig.save(out_pdf, bbox_inches="tight", dpi=200)
    fig.save(out_png, bbox_inches="tight", dpi=200)
    (args.output / "publication_scoremaps.html").write_text(
        "<!doctype html><html><body>"
        '<img src="publication_scoremaps.png" style="max-width:100%">'
        "</body></html>"
    )
    LOG.info("Saved %s", out_png)


if __name__ == "__main__":
    main()
