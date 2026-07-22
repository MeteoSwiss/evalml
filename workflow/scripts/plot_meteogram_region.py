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
import matplotlib.ticker as mticker
import numpy as np

from data_input import parse_steps, load_truth_data
from verification import ShapefileSpatialAggregationMasks

# Shared look for the paper figures (sibling module, on sys.path when run directly).
from publication_style import line_style, param_label

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

    # Mask the points that fall inside the region polygon.
    masks = ShapefileSpatialAggregationMasks(args.shapefile).get_masks(
        truth["latitude"], truth["longitude"]
    )
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
    # Lead time in hours since init.
    lead_hours = (times - np.datetime64(init_time)) / np.timedelta64(1, "h")

    plt.style.use(Path(__file__).resolve().parent / "publication.mplstyle")
    fig, ax = plt.subplots(figsize=(8, 3.0))
    ax.plot(
        lead_hours,
        np.asarray(series.values, dtype=float),
        label=args.truth_label,
        **line_style(args.truth_label),
    )
    ax.set_ylabel(UNITS.get(param, ""))
    ax.text(0.01, 0.97, param_label(param), transform=ax.transAxes, ha="left", va="top")
    # Lead-time x-axis: major gridline every 24 h, minor every 6 h.
    ax.xaxis.set_major_locator(mticker.MultipleLocator(24))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(6))
    ax.grid(True, axis="x", which="major", color="0.6", linewidth=0.8, linestyle="--")
    ax.grid(True, axis="x", which="minor", color="0.8", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Lead time (h)")
    ax.set_xlim(left=0)
    ax.legend()
    fig.suptitle(f"{region_name.capitalize()} — Init time {init_time:%Y-%m-%d %H:%M}")

    outfn = Path(args.outfn)
    outfn.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outfn, dpi=200, bbox_inches="tight")
    plt.close(fig)
    LOG.info("saved: %s", outfn)


if __name__ == "__main__":
    main()
