import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import cartopy.crs as ccrs
from earthkit.meteo.utils.convert import kelvin_to_celsius
import earthkit.meteo.wind as ekm_wind
import earthkit.plots as ekp
from matplotlib.colors import Colormap
import numpy as np

from plotting import DOMAINS
from plotting import get_projection
from plotting import StatePlotter
from plotting.colormap_defaults import CMAP_DEFAULTS
from plotting.compat import load_state_from_grib

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

    bounds = cfg.get("bounds", cfg.get("levels", None))
    prebuilt_cmap = cfg.get("cmap", None)

    # When the config provides a pre-built matplotlib Colormap (e.g. a
    # ListedColormap), we must use the earthkit Style's vmin/vmax path, which
    # handles isinstance(colors, Colormap) correctly.  The levels path calls
    # cmap_and_norm → len(Colormap) → TypeError.
    #
    # earthkit's configure_style intercepts _STYLE_KWARGS (cmap/colors/levels/vmin/vmax)
    # from tricontourf kwargs before matplotlib sees them.  To work around this:
    #  - embed the Colormap directly in the Style via colors=
    #  - inject bounds as 'levels' into style._kwargs so they survive to matplotlib
    #  - pass only norm= as a kwarg (not in _STYLE_KWARGS, so not intercepted)
    extend = cfg.get("extend", "both")

    if isinstance(prebuilt_cmap, Colormap) and bounds is not None:
        style = ekp.styles.Style(
            colors=prebuilt_cmap,
            vmin=bounds[0],
            vmax=bounds[-1],
            extend=extend,
            units=units,
        )
        style._kwargs["levels"] = list(bounds)
        return {
            "style": style,
            "norm": cfg.get("norm", None),
        }

    return {
        "style": ekp.styles.Style(
            levels=bounds,
            extend=extend,
            units=units,
            colors=cfg.get("colors", None),
        ),
        "norm": cfg.get("norm", None),
        "cmap": prebuilt_cmap,
        "vmin": cfg.get("vmin", None),
        "vmax": cfg.get("vmax", None),
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
        return ekm_wind.speed(fields["U_10M"], fields["V_10M"]), "m/s"
    if param == "SP":
        return ekm_wind.speed(fields["U"], fields["V"]), "m/s"
    if param == "TOT_PREC":
        return np.maximum(fields[param], 0), "mm"
    if param in ("CLCT", "CLCL"):
        # Avoid exact 0/1 plateaus breaking tricontourf on orthographic
        # projections (tmp/reproduce_clct_bug.py). Pair with extend="neither".
        # Any new bounded field with silent-blank or GeometryCollection-crash
        # globe frames likely needs the same clip-away-from-boundary fix.
        return np.clip(fields[param], 1e-6, 1 - 1e-6), None
    if param == "SSRD":
        # Same issue, bottom boundary only (night-side plateau).
        return np.maximum(fields[param], 1e-6), None
    return fields[param], None


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input", type=str, default=None, help="Directory to grib data"
    )
    parser.add_argument("--date", type=str, default=None, help="reference datetime")
    parser.add_argument("--leadtime", type=int, help="leadtime")
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
    if not grib_file:
        LOG.warning(
            "No GRIB file found for lead_time=%s in %s (model may not write initial state)."
            " Creating empty placeholder frames.",
            lead_time,
            grib_dir,
        )
        for region_name in regions:
            outfn = outdir / f"frame_{lead_time}_{param}_{region_name}.png"
            outfn.touch()
        return
    grib_file = Path(grib_file[0])
    LOG.info("Loading grib file %s", grib_file)
    state = load_state_from_grib(grib_file, paramlist=paramlist)

    # tp is accumulated from start of forecast; de-accumulate to get period [lt-accu, lt]
    if param == "TOT_PREC":
        prev_lt = lead_time - accu
        if prev_lt > 0:
            prev_grib_files = list(grib_dir.glob(f"2*_{prev_lt}.grib"))
            if not prev_grib_files:
                prev_grib_files = list(grib_dir.glob(f"2*_{prev_lt:03d}.grib"))
            prev_grib_file = Path(prev_grib_files[0])
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
        outfn = outdir / f"frame_{lead_time}_{param}_{region_name}.png"
        plotter = StatePlotter(state["longitudes"], state["latitudes"], outdir)
        if region_cfg.get("extent") is not None:
            projection = get_projection(region_cfg.get("projection") or "orthographic")
            extent = region_cfg["extent"]
        elif region_cfg.get("rotate"):
            base = DOMAINS[region_name]["projection"].proj4_params
            central_longitude = (
                base["lon_0"]
                + 360.0 * lead_time / region_cfg["hours_per_revolution"]
            ) % 360
            #  central_longitude=central_longitude, central_latitude=0.0 for zero-centered
            projection = ccrs.Orthographic(
                central_longitude=central_longitude, central_latitude=base["lat_0"]
            )
            extent = DOMAINS[region_name]["extent"]
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
            subplot,
            field,
            title=f"{param}, time: {validtime}",
            gridline_labels=not region_cfg.get("rotate", False),
            **get_style(param, units_override, accu=accu),
        )
        if len(state["lam_envelope"]) > 0:
            subplot.ax.add_geometries(
                state["lam_envelope"],
                edgecolor="black",
                facecolor="none",
                crs=ccrs.PlateCarree(),
            )

        # earthkit.plots' Figure.save() defaults bbox_inches to "tight", which
        # crops to each frame's own content extent — that extent varies with the
        # rotating globe's gridline labels, making frames jump around when
        # stitched into a GIF. Pass bbox_inches=None explicitly to override that
        # default and always save the fixed full canvas.
        fig.save(outfn, dpi=200, bbox_inches=None)
        LOG.info("saved: %s", outfn)


if __name__ == "__main__":
    main()
