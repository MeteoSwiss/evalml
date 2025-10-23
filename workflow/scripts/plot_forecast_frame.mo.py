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

    from plotting import REGIONS, StatePlotter
    from plotting.colormap_defaults import CMAP_DEFAULTS
    from plotting.compat import load_state_from_grib

    return (
        ArgumentParser,
        CMAP_DEFAULTS,
        Path,
        StatePlotter,
        ekp,
        load_state_from_grib,
        logging,
        np,
        REGIONS,
    )


@app.cell
def _(logging):
    LOG = logging.getLogger(__name__)
    LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)
    return (LOG,)


@app.cell
def _(ArgumentParser, Path):
    parser = ArgumentParser()

    parser.add_argument(
        "--input", type=str, default=None, help="Directory to grib data"
    )
    parser.add_argument("--date", type=str, default=None, help="reference datetime")
    parser.add_argument("--outfn", type=str, help="output filename")
    parser.add_argument("--leadtime", type=str, help="leadtime")
    parser.add_argument("--param", type=str, help="parameter")
    parser.add_argument("--region", type=str, help="name of region")

    args = parser.parse_args()
    grib_dir = Path(args.input)
    init_time = args.date
    outfn = Path(args.outfn)
    lead_time = args.leadtime
    param = args.param
    region = args.region
    return (
        args,
        grib_dir,
        init_time,
        lead_time,
        outfn,
        param,
        region,
    )


@app.cell
def _(grib_dir, init_time, lead_time, load_state_from_grib, param):
    # load grib file
    grib_file = grib_dir / f"{init_time}_{lead_time}.grib"
    if param == "10sp":
        paramlist = ["10u", "10v"]
    elif param == "sp":
        paramlist = ["u", "v"]
    else:
        paramlist = [param]
    state = load_state_from_grib(grib_file, paramlist=paramlist)
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

        def _ms_to_knots(arr):
            # robust conversion with pint, fallback if dtype unsupported
            try:
                return (
                    _ureg.Quantity(arr, _ureg.meter / _ureg.second).to(_ureg.knot)
                ).magnitude
            except Exception:
                return arr * 1.943844

    except Exception:
        LOG.warning("pint not available; falling back hardcoded conversions")

        def _k_to_c(arr):
            return arr - 273.15

        def _ms_to_knots(arr):
            return arr * 1.943844

    def preprocess_field(param: str, state: dict):
        """
        - Temperatures (2t, 2d, t, d): K -> °C
        - Wind speed at 10m (10sp): m/s -> kn, sqrt(10u^2 + 10v^2)
        - Wind speed (sp): m/s -> kn, sqrt(u^2 + v^2)
        Returns: (field_array, units_override or None)
        """
        fields = state["fields"]
        # temperature variables
        if param in ("2t", "2d", "t", "d"):
            return _k_to_c(fields[param]), "°C"
        # 10m wind speed (allow legacy 'uv' alias)
        if param == "10sp":
            u = fields["10u"]
            v = fields["10v"]
            return np.sqrt(u**2 + v**2), "m/s"
        # wind speed from standard-level components
        if param == "sp":
            u = fields["u"]
            v = fields["v"]
            return np.sqrt(u**2 + v**2), "m/s"
        # default: passthrough
        return fields[param], None

    return (preprocess_field,)


@app.cell
def _(
    LOG,
    StatePlotter,
    args,
    get_style,
    outfn,
    param,
    preprocess_field,
    region,
    state,
    REGIONS,
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
        projection=REGIONS[region]["projection"],
        bbox=REGIONS[region]["extent"],
        name=region,
        size=(6, 6),
    )
    subplot = fig.add_map(row=0, column=0)

    # preprocess field (unit conversion, derived quantities)
    field, units_override = preprocess_field(param, state)

    plotter.plot_field(subplot, field, **get_style(args.param, units_override))

    validtime = state["valid_time"].strftime("%Y%m%d%H%M")
    # leadtime = int(state["lead_time"].total_seconds() // 3600)

    fig.title(f"{param}, time: {validtime}")

    fig.save(outfn, bbox_inches="tight", dpi=200)
    LOG.info(f"saved: {outfn}")
    return


if __name__ == "__main__":
    app.run()
