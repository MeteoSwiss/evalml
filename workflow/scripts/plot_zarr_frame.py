"""Plot a single forecast frame from a zarr source (truth or baseline).

Analogous to plot_forecast_frame.mo.py but reads zarr instead of GRIB.
TOT_PREC disaggregation is handled by the data_input loading functions,
so no accumulation arithmetic is needed here.

Usage
-----
    python plot_zarr_frame.py \\
        --zarr /path/to/data.zarr \\
        --source_type analysis \\   # or 'baseline'
        --date 202503270600 \\
        --leadtime 006 \\
        --param T_2M \\
        --region switzerland \\
        --outfn /path/to/frame.png \\
        [--extent LON_MIN LON_MAX LAT_MIN LAT_MAX] \\
        [--projection orthographic] \\
        [--accu 1]
"""

import logging
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import cartopy.crs as ccrs
import numpy as np

from plotting import DOMAINS, StatePlotter, get_projection
from plotting.colormap_defaults import CMAP_DEFAULTS
from plotting.compat import load_state_from_zarr

LOG = logging.getLogger(__name__)
LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)


def get_style(param, units_override=None, accu=1):
    lookup = f"{param}_{accu}H" if param == "TOT_PREC" else param
    cfg = CMAP_DEFAULTS[lookup]
    import earthkit.plots as ekp

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


def preprocess_field(param, state):
    try:
        import pint

        _ureg = pint.UnitRegistry()

        def _k_to_c(arr):
            try:
                return (_ureg.Quantity(arr, _ureg.kelvin).to(_ureg.degC)).magnitude
            except Exception:
                return arr - 273.15

        def _ms_to_knots(arr):
            try:
                return (
                    _ureg.Quantity(arr, _ureg.meter / _ureg.second).to(_ureg.knot)
                ).magnitude
            except Exception:
                return arr * 1.943844

        def _m_to_mm(arr):
            try:
                return (_ureg.Quantity(arr, _ureg.meter).to(_ureg.millimeter)).magnitude
            except Exception:
                return arr * 1000

    except Exception:
        LOG.warning("pint not available; using hardcoded conversions")

        def _k_to_c(arr):
            return arr - 273.15

        def _ms_to_knots(arr):
            return arr * 1.943844

        def _m_to_mm(arr):
            return arr * 1000

    fields = state["fields"]
    if param in ("T_2M", "TD_2M", "T", "TD"):
        return _k_to_c(fields[param]), "°C"
    if param == "SP_10M":
        return np.sqrt(fields["U_10M"] ** 2 + fields["V_10M"] ** 2), "m/s"
    if param == "SP":
        return np.sqrt(fields["U"] ** 2 + fields["V"] ** 2), "m/s"
    if param == "TOT_PREC":
        return np.maximum(_m_to_mm(fields[param]), 0), "mm"
    return fields[param], None


def main():
    parser = ArgumentParser()
    parser.add_argument("--zarr", type=str, required=True, help="Path to zarr dataset")
    parser.add_argument(
        "--source_type",
        type=str,
        default="analysis",
        choices=["analysis", "baseline"],
        help="Zarr source type",
    )
    parser.add_argument(
        "--date", type=str, required=True, help="Reference datetime (YYYYmmddHHMM)"
    )
    parser.add_argument("--outfn", type=str, required=True, help="Output filename")
    parser.add_argument(
        "--leadtime", type=str, required=True, help="Lead time (hours, zero-padded)"
    )
    parser.add_argument("--param", type=str, required=True, help="Parameter name")
    parser.add_argument("--region", type=str, required=True, help="Region name")
    parser.add_argument(
        "--extent",
        type=float,
        nargs=4,
        default=None,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
    )
    parser.add_argument("--projection", type=str, default=None)
    parser.add_argument(
        "--accu", type=int, default=1, help="Accumulation period in hours"
    )
    args = parser.parse_args()

    reftime = datetime.strptime(args.date, "%Y%m%d%H%M")
    lead_time_hours = int(args.leadtime)
    outfn = Path(args.outfn)
    param = args.param

    if param == "SP_10M":
        paramlist = ["U_10M", "V_10M"]
    elif param == "SP":
        paramlist = ["U", "V"]
    else:
        paramlist = [param]

    state = load_state_from_zarr(
        zarr_root=Path(args.zarr),
        reftime=reftime,
        lead_time_hours=lead_time_hours,
        params=paramlist,
        source_type=args.source_type,
    )

    plotter = StatePlotter(state["longitudes"], state["latitudes"], outfn.parent)

    if args.extent is not None:
        projection = get_projection(args.projection or "orthographic")
        extent = args.extent
    else:
        projection = DOMAINS[args.region]["projection"]
        extent = DOMAINS[args.region]["extent"]

    fig = plotter.init_geoaxes(
        nrows=1,
        ncols=1,
        projection=projection,
        bbox=extent,
        name=args.region,
        size=(6, 6),
    )
    subplot = fig.add_map(row=0, column=0)

    field, units_override = preprocess_field(param, state)
    plotter.plot_field(
        subplot, field, **get_style(param, units_override, accu=args.accu)
    )
    subplot.ax.add_geometries(
        state["lam_envelope"],
        edgecolor="black",
        facecolor="none",
        crs=ccrs.PlateCarree(),
    )

    validtime = state["valid_time"].strftime("%Y%m%d%H%M")
    fig.title(f"{param}, time: {validtime}")
    fig.save(outfn, bbox_inches="tight", dpi=200)
    LOG.info(f"saved: {outfn}")


if __name__ == "__main__":
    main()
