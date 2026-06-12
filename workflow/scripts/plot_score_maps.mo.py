import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import logging
    from argparse import ArgumentParser
    from pathlib import Path

    import earthkit.plots as ekp
    import numpy as np
    import xarray as xr

    from plotting import DOMAINS
    from plotting import StatePlotter
    from plotting.colormap_defaults import CMAP_DEFAULTS

    return (
        ArgumentParser,
        CMAP_DEFAULTS,
        DOMAINS,
        Path,
        StatePlotter,
        ekp,
        logging,
        np,
        xr,
    )


@app.cell
def _(logging):
    LOG = logging.getLogger(__name__)
    LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)
    return (LOG,)


@app.cell
def _(ArgumentParser, Path, np):
    parser = ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Directory to .nc data containing the error fields",
    )
    parser.add_argument("--outfn", type=str, help="output filename")
    parser.add_argument("--leadtime", type=str, help="leadtime")
    parser.add_argument("--param", type=str, help="parameter")
    parser.add_argument("--region", type=str, help="name of region")
    parser.add_argument(
        "--score",
        type=str,
        help="Evaluation Score. So far Bias, RMSE, MAE or STDE are implemented.",
    )
    parser.add_argument("--season", type=str, default="all", help="season filter")
    parser.add_argument(
        "--init_hour", type=str, default="all", help="initialization hour filter"
    )

    args = parser.parse_args()
    verif_file = Path(args.input)
    outfn = Path(args.outfn)
    lead_time = args.leadtime
    param = args.param
    region = args.region
    season = args.season
    init_hour = args.init_hour
    score = args.score

    if isinstance(init_hour, str):
        if init_hour == "all":
            init_hour = -999
        else:
            try:
                init_hour = int(init_hour)
            except ValueError as exc:
                raise ValueError("init_hour must be 'all' or an integer hour") from exc

    lead_time = np.timedelta64(lead_time, "h")
    return (
        init_hour,
        lead_time,
        outfn,
        param,
        region,
        score,
        season,
        verif_file,
    )


@app.cell
def _(LOG, init_hour, param, score, season, verif_file, xr):
    ds = xr.open_dataset(verif_file)
    LOG.info("Opened dataset: %s", ds)
    var = f"{param}.{score}"
    LOG.info(
        "Selecting variable '%s' for season '%s', init_hour=%s", var, season, init_hour
    )
    ds = ds[var].sel(season=season, init_hour=init_hour)
    LOG.info(
        "Selected DataArray: dims=%s, shape=%s, dtype=%s", ds.dims, ds.shape, ds.dtype
    )
    LOG.info(
        "Value range: min=%.4g, max=%.4g, n_nan=%d",
        float(ds.min()),
        float(ds.max()),
        int(ds.isnull().sum()),
    )
    return (ds,)


@app.cell
def _(CMAP_DEFAULTS, ekp):
    def get_style(param, score, units_override=None):
        """Get style and colormap settings for the plot.
        Needed because cmap/norm does not work in Style(colors=cmap),
        still needs to be passed as arguments to tripcolor()/tricontourf().
        """
        score_key = f"{param}.{score}.map"
        cfg = (
            CMAP_DEFAULTS[score_key]
            if score_key in CMAP_DEFAULTS
            else CMAP_DEFAULTS.get(param, {})
        )
        units = units_override if units_override is not None else cfg.get("units", "")
        return {
            "style": ekp.styles.Style(
                levels=cfg.get("bounds", cfg.get("levels", None)),
                extend="both",
                units=units,
                colors=cfg.get("colors", None),
            ),
            "norm": cfg.get("norm", None),
            "cmap": cfg.get("cmap", None),
            "levels": cfg.get("levels", None),
            "vmin": cfg.get("vmin", None),
            "vmax": cfg.get("vmax", None),
            "colors": cfg.get("colors", None),
        }

    return (get_style,)


@app.cell
def _(
    DOMAINS,
    LOG,
    StatePlotter,
    ds,
    get_style,
    init_hour,
    lead_time,
    np,
    outfn,
    param,
    region,
    score,
    season,
):
    # plot individual fields

    plotter = StatePlotter(
        ds["lon"].values.ravel(),
        ds["lat"].values.ravel(),
        outfn.parent,
    )
    fig = plotter.init_geoaxes(
        nrows=1,
        ncols=1,
        projection=DOMAINS[region]["projection"],
        bbox=DOMAINS[region]["extent"],
        name=region,
        size=(6, 6),
    )
    subplot = fig.add_map(row=0, column=0)

    plot_vals = ds.values.ravel()

    style_kwargs = get_style(param, score)
    LOG.info("style_kwargs: %s", style_kwargs)

    if np.all(np.isnan(plot_vals)):
        LOG.warning(
            "All values are NaN for %s %s season=%s — plotting empty map.",
            param,
            score,
            season,
        )
        import matplotlib.patches as mpatches

        subplot.ax.set_facecolor("#cccccc")
        subplot.standard_layers()
        grey_patch = mpatches.Patch(color="#cccccc", label="No data")
        subplot.ax.legend(handles=[grey_patch], loc="lower left", fontsize=8)
    else:
        plotter.plot_field(subplot, plot_vals, **style_kwargs)

    # black coast lines and country borders for better visibility
    # grey is hardly visible, especially when the shading colours are intense.
    subplot.coastlines(edgecolor="black", linewidth=1.0, zorder=5)
    subplot.borders(edgecolor="black", linewidth=0.5, zorder=5)

    init_hour_lbl = "all" if init_hour == -999 else f"{init_hour:02d}"
    fig.title(
        f"{score} of {param}, Season: {season}, "
        f"Init hour: {init_hour_lbl}, Lead Time: {lead_time}"
    )

    fig.save(outfn, bbox_inches="tight", dpi=200)
    LOG.info(f"saved: {outfn}")
    return


if __name__ == "__main__":
    app.run()
