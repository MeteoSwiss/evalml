import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    from argparse import ArgumentParser
    import logging
    import numpy as np
    from pathlib import Path

    import earthkit.plots as ekp

    from src.plotting import StatePlotter
    from src.compat import load_state_from_raw
    from src.colormap_defaults import CMAP_DEFAULTS
    return (
        ArgumentParser,
        CMAP_DEFAULTS,
        Path,
        StatePlotter,
        ekp,
        load_state_from_raw,
        logging,
        np,
    )


@app.cell
def _(logging):
    LOG = logging.getLogger(__name__)
    LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)
    return (LOG,)


@app.cell
def _(ArgumentParser, Path):
    ROOT = Path(__file__).parent

    parser = ArgumentParser()

    parser.add_argument("--input", type=str, default=None, help="Directory to raw data")
    parser.add_argument("--date", type=str, default=None, help="reference datetime")
    parser.add_argument("--outfn", type=str, help="output filename")
    parser.add_argument("--leadtime", type=str, help="leadtime")
    parser.add_argument("--param", type=str, help="parameter")
    parser.add_argument("--projection", type=str, help="projection")
    parser.add_argument("--region", type=str, help="region")

    args = parser.parse_args()
    raw_dir = Path(args.input)
    outfn = Path(args.outfn)
    leadtime = int(args.leadtime)
    param = args.param
    region = args.region
    projection = args.projection
    return args, leadtime, outfn, param, projection, raw_dir, region


@app.cell
def _(raw_dir):
    # get all input files
    raw_files = sorted(raw_dir.glob(f"*.npz"))
    raw_files
    return (raw_files,)


@app.cell
def _(leadtime, load_state_from_raw, raw_files):
    # TODO: do not hardcode leadtimes
    leadtimes = list(range(0, 126, 6))
    file_index = leadtimes.index(leadtime)
    state = load_state_from_raw(raw_files[file_index])
    return (state,)


@app.cell
def _(CMAP_DEFAULTS, ekp):
    def get_style(param):
        """"Get style and colormap settings for the plot.
        Needed because cmap/norm does not work in Style(colors=cmap), still needs
        to be passed as arguments to tripcolor()/tricontourf().
        """
        cfg = CMAP_DEFAULTS[param]
        return {
            "style": ekp.styles.Style(
                levels=cfg.get("bounds", None),
                extend="both",
                units=cfg.get("units", ""),
            ),
            "cmap": cfg["cmap"],
            "norm": cfg.get("norm", None),
        }
    return (get_style,)


@app.cell
def _(
    LOG,
    StatePlotter,
    args,
    get_style,
    np,
    outfn,
    param,
    projection,
    region,
    state,
):
    # plot individual fields
    plotter = StatePlotter(
            state["longitudes"],
            state["latitudes"],
            outfn.parent,
    )
    fig = plotter.init_geoaxes(
        nrows=1, ncols=1, projection=projection, region=region, size=(8,8),
    )
    subplot = fig.add_map(row=0, column=0)

    if param == "uv":
        field = np.sqrt(
            state["fields"]["10u"] ** 2 + state["fields"]["10v"] ** 2
        )
    elif param == "2t":
        field = state["fields"][param] - 273.15
    else:
        field = state["fields"][param]

    plotter.plot_field(
        subplot,
        field,
        **get_style(args.param)
    )

    validtime = state["valid_time"].strftime("%Y%m%d%H%M")
    # leadtime = int(state["lead_time"].total_seconds() // 3600)

    fig.title(f"{param}, time: {validtime}")

    fig.save(outfn, bbox_inches="tight", dpi=400)
    LOG.info(f"saved: {outfn}")
    return


if __name__ == "__main__":
    app.run()
