"""Areal-mean meteogram of a parameter over a region polygon.

Loads the configured truth (SwissMetNet observations via jretrieve, or a gridded
analysis zarr), averages it over the points inside the region shapefile at each
valid time, and plots the series in the shared publication figure style. For
TOT_PREC this is the areal-mean hourly precipitation over the region.

Written for the Valais precipitation case-study figure of the paper, but works
for any region shapefile and parameter.
"""

import logging
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data_input import parse_steps, load_truth_data
from verification import ShapefileSpatialAggregationMasks

from evalml.publication.style import line_style, param_label, mplstyle_path, figure_width

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

UNITS = {"TOT_PREC": "mm/h"}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--truth",
        required=True,
        help="Truth root: jretrieve marker (e.g. jretrievedwh:1,2) or analysis .zarr.",
    )
    parser.add_argument("--truth_label", default="truth", help="Legend/title label.")
    parser.add_argument(
        "--shapefile", required=True, help="Region polygon shapefile (EPSG:2056)."
    )
    parser.add_argument(
        "--date", required=True, help="Reference datetime YYYYmmddHHMM."
    )
    parser.add_argument(
        "--steps",
        type=parse_steps,
        default="0/120/1",
        help="Valid-time window as 'start/stop/step' in hours (default 0/120/1).",
    )
    parser.add_argument(
        "--param", default="TOT_PREC", help="Parameter (default TOT_PREC)."
    )
    parser.add_argument("--outfn", required=True, help="Output PNG path.")
    args = parser.parse_args()

    init_time = datetime.strptime(args.date, "%Y%m%d%H%M")
    param = args.param
    region_name = Path(args.shapefile).stem

    LOG.info(
        "Region meteogram: param=%s, region=%s, init=%s, truth=%s",
        param,
        region_name,
        init_time,
        args.truth,
    )

    # Truth has a 'time' dim plus spatial dim(s) carrying lat/lon: a flat 'values'
    # dim for stations and the (1D) analysis grid, or 'y'/'x' for a 2D grid.
    truth = load_truth_data(Path(args.truth), init_time, args.steps, [param])

    # Mask the points that fall inside the region polygon. The mask class takes a
    # list of region specs; build one shapefile spec named after the shapefile.
    masks = ShapefileSpatialAggregationMasks(
        [{"name": region_name, "type": "shp", "path": args.shapefile}]
    ).get_masks(truth["latitude"], truth["longitude"])
    if region_name not in masks["region"].values:
        raise ValueError(
            f"Region {region_name!r} not in masks {list(masks['region'].values)}"
        )
    region_mask = masks.sel(region=region_name)
    n_points = int(region_mask.sum())
    LOG.info("%d points fall inside %s", n_points, region_name)
    if n_points == 0:
        raise ValueError(f"No points inside region {region_name!r}.")

    # Areal mean over the in-region points at each valid time. Average over the
    # spatial dim(s) — 'values', or 'y'/'x' for a 2D grid — leaving 'time'.
    spatial_dims = [d for d in truth[param].dims if d != "time"]
    series = truth[param].where(region_mask).mean(dim=spatial_dims, skipna=True)

    times = np.asarray(truth["time"].values)

    # Single-column publication figure: fixed print width, shared style, no
    # on-figure title (region + period belong in the filename / caption).
    plt.style.use(mplstyle_path())
    width = figure_width(1)  # single column, 3.35 in
    fig, ax = plt.subplots(figsize=(width, width * 0.6))
    _style = {**line_style(args.truth_label), "color": "black"}
    ax.plot(times, np.asarray(series.values, dtype=float), label=args.truth_label, **_style)

    # Nicer y-label (hourly precipitation for TOT_PREC); else fall back to the
    # shared human-readable name.
    _PRETTY = {
        "TOT_PREC1": ("Hourly precipitation", "mm"),
        "TOT_PREC6": ("6-hourly precipitation", "mm"),
    }
    _name, _unit = _PRETTY.get(param, (param_label(param), UNITS.get(param, "")))
    ax.set_ylabel(f"{_name}\n({_unit})" if _unit else _name)

    # Valid-time x-axis: one labelled tick per day (00 UTC), minor every 6 h.
    import matplotlib.dates as mdates

    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
    ax.grid(True, axis="x", which="major", color="0.75", linewidth=0.5, linestyle="solid")
    ax.grid(True, axis="x", which="minor", color="0.85", linewidth=0.4, linestyle="solid")
    ax.set_xlabel("Valid time")
    ax.set_xlim(times[0], times[-1])
    ax.legend()

    outfn = Path(args.outfn)
    outfn.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outfn, dpi=200, bbox_inches="tight")
    fig.savefig(outfn.with_suffix(".pdf"), bbox_inches="tight")  # vector for the paper
    plt.close(fig)
    LOG.info("saved: %s (+ .pdf)", outfn)


if __name__ == "__main__":
    main()
