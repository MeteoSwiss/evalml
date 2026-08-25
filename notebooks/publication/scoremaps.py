import marimo

__generated_with = "0.23.3"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path

    # Repo root: cwd when run from repo root, else walk up (nbconvert runs with
    # cwd = the notebook's own directory). __file__ is undefined in a kernel.
    PROJECT_ROOT = Path.cwd().resolve()
    if not (PROJECT_ROOT / "workflow").is_dir():
        for _p in [PROJECT_ROOT] + list(PROJECT_ROOT.parents):
            if (_p / "workflow").is_dir() and (_p / "src").is_dir():
                PROJECT_ROOT = _p
                break
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    import logging

    import earthkit.plots as ekp
    import matplotlib.colors as mcolors
    import numpy as np
    import xarray as xr
    from cartopy.mpl.gridliner import Gridliner
    from matplotlib import pyplot as plt
    from matplotlib.colors import to_hex

    from evalml.publication import style as _style

    plt.style.use(_style.mplstyle_path())

    sys.path.insert(0, str(PROJECT_ROOT / "workflow" / "scripts"))
    from plotting import DOMAINS, StatePlotter  # noqa: E402

    from evalml.publication.style import (  # noqa: E402
        COLOR_SKILL_BASELINE_BETTER,
        COLOR_SKILL_MODEL_BETTER,
        PARAM_LABELS,
        SCORE_LABELS,
        SKILL_CMAP,
        SKILL_GREY,
        SKILL_LEVELS,
    )

    LOG = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    # Force the standard map furniture (drawn by `subplot.standard_layers()` inside
    # StatePlotter.plot_field) to high-resolution Natural Earth geometry. Otherwise
    # it defaults to medium (50m), leaving a fuzzy low-res border.
    # We also darken/thicken the country borders slightly for publication legibility.
    ekp.schema.borders["resolution"] = "high"
    ekp.schema.borders["edgecolor"] = "black"
    ekp.schema.borders["linewidth"] = 1.0
    ekp.schema.coastlines["resolution"] = "high"

    SCALING = 0.85 # scale figure size to control relative size of labels
    return (
        COLOR_SKILL_BASELINE_BETTER,
        COLOR_SKILL_MODEL_BETTER,
        DOMAINS,
        Gridliner,
        LOG,
        PARAM_LABELS,
        PROJECT_ROOT,
        Path,
        SCALING,
        SCORE_LABELS,
        SKILL_CMAP,
        SKILL_GREY,
        SKILL_LEVELS,
        StatePlotter,
        ekp,
        mcolors,
        np,
        plt,
        to_hex,
        xr,
    )


@app.cell
def _(PROJECT_ROOT):
    from evalml.publication.manifest import load_manifest, figures_dir

    m = load_manifest(PROJECT_ROOT / "output/manifests/manifest_varda-single_paper_forecaster_scoremaps.json")
    _sm = m.publication.get("scoremaps") or {}

    # Configured case; override in-cell to retarget.
    params = _sm.get("params", ["T_2M", "SP_10M"])
    scores = _sm.get("scores", ["MSE_SKILL", "BIAS_CONTRIB"])
    leadtimes = [
        int(s) for s in (_sm.get("steps") or [6])
    ]  # Lead times from publication.scoremaps.steps (default [24]). Set 'steps' in the config's publication.scoremaps so these match the files publication_all builds.
    baseline_label = _sm.get("baseline_label", "ICON-CH1-CTRL")
    region = _sm.get("region", "switzerland")
    season = _sm.get("season", "all")
    candidate_label = m.get_candidate().label
    output = figures_dir(PROJECT_ROOT / "output", m.truth["label"]) / "scoremaps"

    cand = m.get_candidate()
    base = m.resolve_baseline(baseline_label)
    for _lt in leadtimes:
        m.validate_request("scoremaps", baseline=baseline_label, leadtime=_lt)

    # Leadtime-major ordering (all params for leadtimes[0], then leadtimes[1], ...).
    candidate_files = [
        PROJECT_ROOT / m.scoremap_path(cand, p, lt) for lt in leadtimes for p in params
    ]
    baseline_files = [
        PROJECT_ROOT / m.scoremap_path(base, p, lt) for lt in leadtimes for p in params
    ]
    n_params = len(params)
    return (
        baseline_files,
        baseline_label,
        candidate_files,
        candidate_label,
        leadtimes,
        n_params,
        output,
        params,
        region,
        scores,
        season,
    )


@app.cell
def _(
    COLOR_SKILL_BASELINE_BETTER,
    COLOR_SKILL_MODEL_BETTER,
    DOMAINS,
    Gridliner,
    LOG,
    PARAM_LABELS,
    Path,
    SCALING,
    SCORE_LABELS,
    SKILL_CMAP,
    SKILL_GREY,
    SKILL_LEVELS,
    StatePlotter,
    baseline_files,
    baseline_label,
    candidate_files,
    candidate_label,
    ekp,
    leadtimes,
    mcolors,
    n_params,
    np,
    output,
    params,
    plt,
    region,
    scores,
    season,
    to_hex,
    xr,
):
    # Tighter geographic crop for publication: roughly equal visual margins around Switzerland.
    PUB_EXTENTS = {
        "switzerland": [5.6, 10.8, 45.6, 48.0],
    }

    # Sentinel boundary that covers all realistic skill values (SP_10M can reach ~1e9).
    SENTINEL = 1e15

    def _build_skill_artifacts():
        """Return (ekp_style, mpl_cmap, mpl_norm) for the discrete skill colormap."""
        n_side = (len(SKILL_LEVELS) - 2) // 2
        reds = [to_hex(SKILL_CMAP(i / (2 * n_side))) for i in range(n_side)]
        blues = [
            to_hex(SKILL_CMAP((n_side + 1 + i) / (2 * n_side))) for i in range(n_side)
        ]
        inner_colors = reds + [SKILL_GREY] + blues
        outer_red = COLOR_SKILL_BASELINE_BETTER
        outer_blue = COLOR_SKILL_MODEL_BETTER
        sentinel_levels = [-SENTINEL] + list(SKILL_LEVELS) + [SENTINEL]
        all_colors = [outer_red] + inner_colors + [outer_blue]
        style = ekp.styles.Style(
            levels=sentinel_levels, colors=all_colors, extend="neither", units="skill"
        )
        cmap = mcolors.ListedColormap(inner_colors)
        cmap.set_under(outer_red)
        cmap.set_over(outer_blue)
        norm = mcolors.BoundaryNorm(SKILL_LEVELS, ncolors=len(inner_colors))
        return style, cmap, norm

    def _load_raw(
        nc_file: Path, param: str, score: str, season: str, init_hour: int
    ) -> np.ndarray:
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
            cand_v = _load_raw(cand_file, score=metric, **kw)
            base_v = _load_raw(base_file, score=metric, **kw)
            return 1.0 - cand_v / base_v

    def _remove_latlon_labels(ax) -> None:
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
        ax.set_xlabel("")
        ax.set_ylabel("")

    _LABEL_W = 0.8   # inches left of panels for row labels
    _TITLE_H = 0.4   # inches above panels for column titles
    _LEGEND_H = 0.8  # inches below panels for colorbar + labels

    def _make_map_figure(
        skill_arrays,
        row_labels,
        col_labels,
        plotter,
        domain,
        region,
        style,
        skill_cmap,
        skill_norm,
        candidate_label,
        baseline_label,
    ):
        """Plot a grid of skill score maps.

        skill_arrays : List[List[np.ndarray]], indexed [row][col].
        row_labels / col_labels : matching label strings for margin annotations.
        """
        nrows = len(row_labels)
        ncols = len(col_labels)
        fig = plotter.init_geoaxes(
            projection=domain["projection"],
            bbox=domain["extent"],
            nrows=nrows,
            ncols=ncols,
            name=region,
            size=(6 * SCALING * ncols + _LABEL_W, 4.4 * SCALING * nrows + _TITLE_H + _LEGEND_H),
        )
        mpl_axes = []
        row_axes = {}
        panel_idx = 0
        for row, row_label in enumerate(row_labels):
            for col, col_label in enumerate(col_labels):
                skill_vals = skill_arrays[row][col]
                LOG.info(
                    "(%s | %s) skill min=%.3f  max=%.3f  n_nan=%d / %d",
                    row_label,
                    col_label,
                    np.nanmin(skill_vals),
                    np.nanmax(skill_vals),
                    int(np.isnan(skill_vals).sum()),
                    skill_vals.size,
                )
                subplot = fig.add_map(row=row, column=col)
                if np.all(np.isnan(skill_vals)):
                    LOG.warning("All-NaN (%s | %s) — plotting empty panel.", row_label, col_label)
                    subplot.ax.set_facecolor("#cccccc")
                    subplot.standard_layers()
                else:
                    plotter.plot_field(subplot, skill_vals, style=style, colorbar=False)
                _remove_latlon_labels(subplot.ax)
                if col == 0:
                    row_axes[row] = subplot.ax
                mpl_axes.append(subplot.ax)
                subplot.ax.text(
                    0.03,
                    0.97,
                    f"({chr(ord('a') + panel_idx)})",
                    transform=subplot.ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=plt.rcParams["axes.titlesize"],
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=2),
                )
                panel_idx += 1
                if row == 0:
                    subplot.ax.text(
                        0.5,
                        1.03,
                        col_label,
                        transform=subplot.ax.transAxes,
                        ha="center",
                        va="bottom",
                        fontsize=plt.rcParams["axes.titlesize"],
                        clip_on=False,
                    )
        mpl_fig = fig.fig
        for row, row_label in enumerate(row_labels):
            row_axes[row].text(
                -0.05,
                0.5,
                row_label,
                transform=row_axes[row].transAxes,
                ha="right",
                va="center",
                fontsize=plt.rcParams["axes.titlesize"],
                rotation=90,
                clip_on=False,
            )
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
        mpl_fig.canvas.draw()
        renderer = mpl_fig.canvas.get_renderer()
        label_bbox = cbar.ax.xaxis.label.get_window_extent(renderer)
        fig_height_px = mpl_fig.get_figheight() * mpl_fig.dpi
        y_fig = label_bbox.y0 / fig_height_px
        axes_x0 = min(ax.get_position().x0 for ax in mpl_axes)
        axes_x1 = max(ax.get_position().x1 for ax in mpl_axes)
        mpl_fig.text(
            axes_x0,
            y_fig,
            f"{baseline_label} better",
            ha="left",
            va="top",
            color=COLOR_SKILL_BASELINE_BETTER,
            fontsize=plt.rcParams["font.size"],
        )
        mpl_fig.text(
            axes_x1,
            y_fig,
            f"{candidate_label} better",
            ha="right",
            va="top",
            color=COLOR_SKILL_MODEL_BETTER,
            fontsize=plt.rcParams["font.size"],
        )
        return fig

    assert len(candidate_files) == n_params * len(leadtimes)
    assert len(baseline_files) == n_params * len(leadtimes)

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

    output.mkdir(parents=True, exist_ok=True)
    plotter = StatePlotter(lons, lats, output)

    domain = DOMAINS.get(region, DOMAINS["switzerland"])
    if region in PUB_EXTENTS:
        domain = {**domain, "extent": PUB_EXTENTS[region]}

    style, skill_cmap, skill_norm = _build_skill_artifacts()

    _season_labels = {
        "DJF": "Winter (DJF)", "MAM": "Spring (MAM)",
        "JJA": "Summer (JJA)", "SON": "Autumn (SON)",
    }
    _seasons = ("DJF", "MAM", "JJA", "SON")
    _common = dict(
        plotter=plotter, domain=domain, region=region,
        style=style, skill_cmap=skill_cmap, skill_norm=skill_norm,
        candidate_label=candidate_label, baseline_label=baseline_label,
    )

    out_pngs = []
    for i, lt in enumerate(leadtimes):
        cand_files_lt = candidate_files[i * n_params : (i + 1) * n_params]
        base_files_lt = baseline_files[i * n_params : (i + 1) * n_params]

        skill_arrays = [
            [_compute_panel(score, cf, bf, param, season, -999) for score in scores]
            for param, cf, bf in zip(params, cand_files_lt, base_files_lt)
        ]
        fig = _make_map_figure(
            skill_arrays=skill_arrays,
            row_labels=[PARAM_LABELS.get(p, p) for p in params],
            col_labels=[SCORE_LABELS.get(s, s) for s in scores],
            **_common,
        )

        out_png = output / f"publication_scoremaps_{lt}h.png"
        out_pdf = output / f"publication_scoremaps_{lt}h.pdf"
        fig.save(out_pdf, bbox_inches="tight", dpi=200)
        fig.save(out_png, bbox_inches="tight", dpi=200)
        out_pngs.append(out_png)
        LOG.info("Saved %s", out_png)

    out_seasonal_pngs = []
    for i, lt in enumerate(leadtimes):
        cand_files_lt = candidate_files[i * n_params : (i + 1) * n_params]
        base_files_lt = baseline_files[i * n_params : (i + 1) * n_params]

        skill_arrays = [
            [_compute_panel("MSE_SKILL", cf, bf, param, seas, -999)
             for param, cf, bf in zip(params, cand_files_lt, base_files_lt)]
            for seas in _seasons
        ]
        fig_seas = _make_map_figure(
            skill_arrays=skill_arrays,
            row_labels=[_season_labels[s] for s in _seasons],
            col_labels=[PARAM_LABELS.get(p, p) for p in params],
            **_common,
        )

        out_png = output / f"publication_scoremaps_seasonal_{lt}h.png"
        out_pdf = output / f"publication_scoremaps_seasonal_{lt}h.pdf"
        fig_seas.save(out_pdf, bbox_inches="tight", dpi=200)
        fig_seas.save(out_png, bbox_inches="tight", dpi=200)
        out_seasonal_pngs.append(out_png)
        LOG.info("Saved %s", out_png)

    img_tags = "".join(
        f'<img src="{p.name}" style="max-width:100%"><br>' for p in out_pngs + out_seasonal_pngs
    )
    (output / "publication_scoremaps.html").write_text(
        f"<!doctype html><html><body>{img_tags}</body></html>"
    )
    LOG.info("Saved HTML index")

    plt.show()
    return


if __name__ == "__main__":
    app.run()
