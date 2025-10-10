import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import logging
    from argparse import ArgumentParser
    from pathlib import Path

    import earthkit.plots as ekp
    import numpy as np
    from src.colormap_defaults import CMAP_DEFAULTS
    from src.compat import load_state_from_grib
    from src.plotting import StatePlotter

    return (
        ArgumentParser,
        CMAP_DEFAULTS,
        Path,
        StatePlotter,
        ekp,
        load_state_from_grib,
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

    parser.add_argument("--input", type=str, default=None, help="Directory to grib data")
    parser.add_argument("--date", type=str, default=None, help="reference datetime")
    parser.add_argument("--outfn", type=str, help="output filename")
    parser.add_argument("--leadtime", type=str, help="leadtime")
    parser.add_argument("--param", type=str, help="parameter")
    parser.add_argument("--projection", type=str, help="projection")
    parser.add_argument("--region", type=str, default="none", help="region (or 'none')")
    parser.add_argument(
        "--with_global", type=str, choices=["true", "false"], default="true", help="include global data (true/false)"
    )

    args = parser.parse_args()
    raw_dir = Path(args.input)
    outfn = Path(args.outfn)
    leadtime = int(args.leadtime)
    param = args.param
    region = None if (args.region is None or str(args.region).lower() in {"none", "", "null"}) else args.region
    projection = args.projection
    with_global = args.with_global.lower() == "true"
    return args, leadtime, outfn, param, projection, raw_dir, region, with_global


@app.cell
def _(raw_dir):
    # get all input files
    grib_files = sorted(raw_dir.glob("*.grib"))
    return (grib_files,)


@app.cell
def _(leadtime, load_state_from_grib, grib_files, param, with_global):
    # TODO: do not hardcode leadtimes
    leadtimes = list(range(0, 126, 6))
    file_index = leadtimes.index(leadtime)
    state = load_state_from_grib(grib_files[file_index], paramlist=[param], with_global=with_global)
    return (state,)


@app.cell
def _(CMAP_DEFAULTS, ekp):
    def get_style(param, units_override=None):
        """Get style and colormap settings for the plot.
        Needed because cmap/norm does not work in Style(colors=cmap),
        still needs to be passed as arguments to tripcolor()/tricontourf().
        """
        cfg = CMAP_DEFAULTS[param]
        units = units_override if units_override is not None else cfg.get("units", "")
        return {
            "style": ekp.styles.Style(
                levels=cfg.get("bounds", None),
                extend="both",
                units=units,
            ),
            "cmap": cfg["cmap"],
            "norm": cfg.get("norm", None),
        }

    return (get_style,)


@app.cell
def _(LOG, np):
    """Preprocess fields with pint-based unit conversion and derived quantities."""
    try:
        import pint  # type: ignore

        _ureg = pint.UnitRegistry()

        def _k_to_c(arr):
            # robust conversion with pint, fallback if dtype unsupported
            try:
                return (_ureg.Quantity(arr, _ureg.kelvin).to(_ureg.degC)).magnitude
            except Exception:
                return arr - 273.15

    except Exception:
        LOG.warning("pint not available; falling back to K->C by subtracting 273.15")

        def _k_to_c(arr):
            return arr - 273.15

    def preprocess_field(param: str, state: dict):
        """
        - Temperatures (2t, 2d, t, d): K -> °C
        - Wind speed at 10m (10sp): sqrt(10u^2 + 10v^2)
        - Wind speed (sp): sqrt(u^2 + v^2)
        Returns: (field_array, units_override or None)
        """
        fields = state["fields"]
        # temperature variables
        if param in ("2t", "2d", "t", "d"):
            return _k_to_c(fields[param]), "°C"
        # 10m wind speed (allow legacy 'uv' alias)
        if param == "10sp":
            u = fields.get("10u")
            v = fields.get("10v")
            if u is None or v is None:
                raise KeyError("Required components 10u/10v not in state['fields']")
            return np.sqrt(u**2 + v**2), "m s$^{-1}$"
        # wind speed from standard-level components
        if param == "sp":
            u = fields.get("u")
            v = fields.get("v")
            if u is None or v is None:
                raise KeyError("Required components u/v not in state['fields']")
            return np.sqrt(u**2 + v**2), "m s$^{-1}$"
        # default: passthrough
        return fields[param], None

    return (preprocess_field,)


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
    preprocess_field,
):
    # plot individual fields
    plotter = StatePlotter(
        state["longitudes"],
        state["latitudes"],
        outfn.parent,
    )
    fig = plotter.init_geoaxes(
        nrows=1,
        ncols=1,
        projection=projection,
        region=region,
        size=(8, 8),
    )
    subplot = fig.add_map(row=0, column=0)

    # preprocess field (unit conversion, derived quantities)
    field, units_override = preprocess_field(param, state)

    plotter.plot_field(subplot, field, **get_style(args.param, units_override))

    validtime = state["valid_time"].strftime("%Y%m%d%H%M")
    # leadtime = int(state["lead_time"].total_seconds() // 3600)

    fig.title(f"{param}, time: {validtime}")

    fig.save(outfn, bbox_inches="tight", dpi=400)
    LOG.info(f"saved: {outfn}")
    return


if __name__ == "__main__":
    app.run()
