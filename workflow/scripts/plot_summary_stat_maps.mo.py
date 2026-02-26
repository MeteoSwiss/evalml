import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():

    # this sure stays the same.
    import logging
    from argparse import ArgumentParser
    from pathlib import Path

    # this sure stays the same.
    import cartopy.crs as ccrs
    import earthkit.plots as ekp
    import numpy as np
    import xarray as xr

    # this stays the same as well.
    from plotting import DOMAINS

    # no changes to StatePlotter required according to ChatGPT.
    from plotting import StatePlotter

    # Added some new colour maps for the Bias / MAE / RMSE map plots. 
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
        "--input", type=str, default=None, help="Directory to .nc data containing the error fields"
    )
    parser.add_argument("--outfn", type=str, help="output filename")
    parser.add_argument("--leadtime", type=str, help="leadtime")
    parser.add_argument("--param", type=str, help="parameter")
    parser.add_argument("--region", type=str, help="name of region")
    parser.add_argument("--metric", type=str, help = "Evaluation Metric. So far Bias, RMSE or MAE are implemented.")
    parser.add_argument("--season", type=str, default="all", help="season filter")
    parser.add_argument("--init_hour", type=str, default="all", help="initialization hour filter")

    args = parser.parse_args()
    verif_file = Path(args.input)
    outfn = Path(args.outfn)
    lead_time = args.leadtime
    param = args.param
    region = args.region
    season = args.season
    init_hour = args.init_hour
    metric = args.metric

    if isinstance(init_hour, str):
        if init_hour == "all":
            init_hour = -999
        else:
            raise ValueError("init_hour must be 'all' or an integer hour")

    lead_time = np.timedelta64(lead_time, 'h')
    return (
        init_hour,
        lead_time,
        metric,
        outfn,
        param,
        region,
        season,
        verif_file,
    )


@app.cell
def _(init_hour, lead_time, metric, param, season, verif_file, xr):
    ds = xr.open_dataset(verif_file)
    var = f"{param}.{metric}.spatial"
    ds = ds[var].sel(init_hour=init_hour, lead_time=lead_time, season=season)
    ds
    return ds, var

@app.cell
def _(CMAP_DEFAULTS, ekp):
    def get_style(param, metric, units_override=None):
        """Get style and colormap settings for the plot.
        Needed because cmap/norm does not work in Style(colors=cmap),
        still needs to be passed as arguments to tripcolor()/tricontourf().
        """
        metric_key = f"{param}.{metric}.spatial"
        cfg = CMAP_DEFAULTS[metric_key] if metric_key in CMAP_DEFAULTS else CMAP_DEFAULTS[param]
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

# @app.cell
# def _(LOG, np):
#     """Preprocess fields with pint-based unit conversion and derived quantities."""
#     try:
#         import pint  # type: ignore

#         _ureg = pint.UnitRegistry()

#         def _k_to_c(arr):
#             # robust conversion with pint, fallback if dtype unsupported
#             try:
#                 return (_ureg.Quantity(arr, _ureg.kelvin).to(_ureg.degC)).magnitude
#             except Exception:
#                 return arr - 273.15

#         def _ms_to_knots(arr):
#             # robust conversion with pint, fallback if dtype unsupported
#             try:
#                 return (
#                     _ureg.Quantity(arr, _ureg.meter / _ureg.second).to(_ureg.knot)
#                 ).magnitude
#             except Exception:
#                 return arr * 1.943844

#         def _m_to_mm(arr):
#             # robust conversion with pint, fallback if dtype unsupported
#             try:
#                 return (_ureg.Quantity(arr, _ureg.meter).to(_ureg.millimeter)).magnitude
#             except Exception:
#                 return arr * 1000

#     except Exception:
#         LOG.warning("pint not available; falling back hardcoded conversions")

#         def _k_to_c(arr):
#             return arr - 273.15

#         def _ms_to_knots(arr):
#             return arr * 1.943844

#         def _m_to_mm(arr):
#             return arr * 1000

#     def preprocess_field(param: str, state: dict):
#         """
#         - Temperatures: K -> °C
#         - Wind speed: sqrt(u^2 + v^2)
#         - Precipitation: m -> mm
#         Returns: (field_array, units_override or None)
#         """
#         fields = state["fields"]
#         # temperature variables
#         if param in ("T_2M", "TD_2M", "T", "TD"):
#             return _k_to_c(fields[param]), "°C"
#         # 10m wind speed (allow legacy 'uv' alias)
#         if param == "SP_10M":
#             u = fields["U_10M"]
#             v = fields["V_10M"]
#             return np.sqrt(u**2 + v**2), "m/s"
#         # wind speed from standard-level components
#         if param == "SP":
#             u = fields["U"]
#             v = fields["V"]
#             return np.sqrt(u**2 + v**2), "m/s"
#         if param == "TOT_PREC":
#             return _m_to_mm(fields[param]), "mm"
#         # default: passthrough
#         return fields[param], None

#     return (preprocess_field,)

@app.cell
def _(
    DOMAINS,
    LOG,
    StatePlotter,
    ds,
    get_style,
    lead_time,
    metric,
    outfn,
    param,
    region, 
    season, 
    var,
):
    # plot individual fields
    import matplotlib.pyplot as plt

    plotter = StatePlotter(
        ds["longitude"].values.ravel(),
        ds["latitude"].values.ravel(),
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

    # # preprocess field (unit conversion, derived quantities)
    # field, units_override = preprocess_field(param, state)

    # Quick fix for precipitation (might have to use preprocess_field in the end (see above))
    if param == "TOT_PREC":
        plot_vals = ds.values.ravel()*1000
    else:
        plot_vals = ds.values.ravel()

    plotter.plot_field(subplot, plot_vals, **get_style(var, metric))
    # subplot.ax.add_geometries(
    #     state["lam_envelope"],
    #     edgecolor="black",
    #     facecolor="none",
    #     crs=ccrs.PlateCarree(),
    # )

    # validtime = state["valid_time"].strftime("%Y%m%d%H%M")
    # # leadtime = int(state["lead_time"].total_seconds() // 3600)

    fig.title(f"{metric} of {param}, Season: {season}, Lead Time: {lead_time}")

    fig.save(outfn, bbox_inches="tight", dpi=200)
    LOG.info(f"saved: {outfn}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
