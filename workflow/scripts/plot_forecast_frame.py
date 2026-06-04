import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import cartopy.crs as ccrs
from earthkit.meteo.utils.convert import kelvin_to_celsius
import earthkit.plots as ekp
import numpy as np

from plotting import DOMAINS
from plotting import get_projection
from plotting import StatePlotter
from plotting.colormap_defaults import CMAP_DEFAULTS
from plotting.compat import load_state_from_grib
from data.derived import wind_speed

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def get_style(param, units_override=None, accu=1):
    """Get style and colormap settings for the plot."""
    lookup = f"{param}_{accu}H" if param == "TOT_PREC" else param
    cfg = CMAP_DEFAULTS[lookup]
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


def preprocess_field(param: str, state: dict):
    """
    - Temperatures: K -> °C
    - Wind speed: sqrt(u^2 + v^2)
    - Precipitation: m -> mm
    Returns: (field_array, units_override or None)
    """
    fields = state["fields"]
    if param in ("T_2M", "TD_2M", "T", "TD"):
        return kelvin_to_celsius(fields[param]), "°C"
    if param == "SP_10M":
        return wind_speed(fields["U_10M"], fields["V_10M"]), "m/s"
    if param == "SP":
        return wind_speed(fields["U"], fields["V"]), "m/s"
    if param == "TOT_PREC":
        return np.maximum(fields[param], 0), "mm"
    return fields[param], None


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input", type=str, default=None, help="Directory to grib data"
    )
    parser.add_argument("--date", type=str, default=None, help="reference datetime")
    parser.add_argument("--leadtime", type=str, help="leadtime")
    parser.add_argument("--param", type=str, help="parameter")
    parser.add_argument(
        "--regions_json",
        type=str,
        help="JSON dict mapping region name -> {extent, projection}",
    )
    parser.add_argument("--outdir", type=str, help="output directory")
    parser.add_argument(
        "--accu", type=int, default=1, help="accumulation period in hours"
    )

    args = parser.parse_args()
    grib_dir = Path(args.input)
    init_time = args.date
    lead_time = args.leadtime
    param = args.param
    regions = json.loads(args.regions_json)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    accu = args.accu

    LOG.info(
        "Plotting forecast frame: param=%s, init_time=%s, lead_time=%s, regions=%s",
        param,
        init_time,
        lead_time,
        list(regions.keys()),
    )

    if param == "SP_10M":
        paramlist = ["U_10M", "V_10M"]
    elif param == "SP":
        paramlist = ["U", "V"]
    else:
        paramlist = [param]

    # Load grib once — shared across all region plots
    # TODO: fix file pattern & globbing
    grib_file = list(grib_dir.glob(f"2*_{lead_time}.grib"))
    if not grib_file:
        grib_file = list(grib_dir.glob(f"2*_{lead_time:03}.grib"))
    grib_file = Path(grib_file[0])
    LOG.info("Loading grib file %s", grib_file)
    state = load_state_from_grib(grib_file, paramlist=paramlist)

    # tp is accumulated from start of forecast; de-accumulate to get period [lt-accu, lt]
    if param == "TOT_PREC":
        prev_lt = int(lead_time) - accu
        if prev_lt > 0:
            prev_grib_file = grib_dir / f"{init_time}_{prev_lt:03d}.grib"
            LOG.info(
                "De-accumulating TOT_PREC: loading previous grib file %s",
                prev_grib_file,
            )
            prev_state = load_state_from_grib(prev_grib_file, paramlist=paramlist)
            state["fields"]["TOT_PREC"] = (
                state["fields"]["TOT_PREC"]
                - prev_state["fields"]["TOT_PREC"][: len(state["fields"]["TOT_PREC"])]
            )

    # Preprocess field once — shared across all region plots
    field, units_override = preprocess_field(param, state)
    validtime = state["valid_time"].strftime("%Y%m%d%H%M")

    for region_name, region_cfg in regions.items():
        LOG.info("Plotting region %s", region_name)
        plotter = StatePlotter(state["longitudes"], state["latitudes"], outdir)
        if region_cfg.get("extent") is not None:
            projection = get_projection(region_cfg.get("projection") or "orthographic")
            extent = region_cfg["extent"]
        else:
            projection = DOMAINS[region_name]["projection"]
            extent = DOMAINS[region_name]["extent"]
        fig = plotter.init_geoaxes(
            nrows=1,
            ncols=1,
            projection=projection,
            bbox=extent,
            name=region_name,
            size=(6, 6),
        )
        subplot = fig.add_map(row=0, column=0)

        plotter.plot_field(
            subplot, field, **get_style(param, units_override, accu=accu)
        )
        subplot.ax.add_geometries(
            state["lam_envelope"],
            edgecolor="black",
            facecolor="none",
            crs=ccrs.PlateCarree(),
        )
        fig.title(f"{param}, time: {validtime}")

        outfn = outdir / f"frame_{lead_time}_{param}_{region_name}.png"
        fig.save(outfn, bbox_inches="tight", dpi=200)
        LOG.info("saved: %s", outfn)


if __name__ == "__main__":
    main()
