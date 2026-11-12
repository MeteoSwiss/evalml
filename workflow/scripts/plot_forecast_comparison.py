"""Plot side-by-side maps of the truth, ML runs and baselines at a common valid time."""

import json
import logging
import math
from argparse import ArgumentParser
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import cartopy.crs as ccrs
import earthkit.plots as ekp
import numpy as np

from data_input import load_forecast_data
from data_input import load_truth_data
from data_input import parse_aggregated_param
from plotting import resolve_region
from plotting import StatePlotter
from plotting.styles import get_style
from plotting.styles import prepare_values

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


# What the loaders raise when a baseline or the truth does not cover the lead
# time: missing ICON archive file, INCA step outside the file, valid time
# outside the truth zarr. Anything else is a real failure.
NOT_AVAILABLE = (FileNotFoundError, LookupError)


def _flat(ds, param):
    base_param, _ = parse_aggregated_param(param)
    values, _ = prepare_values(base_param, np.asarray(ds[param].values).ravel())
    lon = np.asarray(ds["longitude"].values).ravel()
    lat = np.asarray(ds["latitude"].values).ravel()
    return lon, lat, values


def load_source(root, init_time, lead_time, param):
    """Return (lon, lat, values) as flat arrays."""
    ds = load_forecast_data(root, init_time, [lead_time], [param]).squeeze()
    return _flat(ds, param)


def load_truth(root, init_time, lead_time, param):
    """Return (lon, lat, values) of the truth at the valid time as flat arrays."""
    ds = load_truth_data(Path(root), init_time, [lead_time], [param])
    valid_time = np.datetime64(init_time + timedelta(hours=lead_time))
    return _flat(ds.sel(time=valid_time).squeeze(), param)


def max_in_view(subplot, lon, lat, values):
    """Max of the values at points inside the subplot's visible map extent."""
    xyz = subplot._crs.transform_points(ccrs.PlateCarree(), lon, lat)
    x, y = xyz[:, 0], xyz[:, 1]
    x0, x1, y0, y1 = subplot.ax.get_extent()
    # Points on the far side of an orthographic globe project to non-finite x/y.
    inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1) & np.isfinite(values)
    return np.max(values[inside]) if inside.any() else np.nan


def parse_source(text):
    """'label=root' -> (label, root); the label may not contain '='."""
    label, sep, root = text.partition("=")
    if not sep or not label or not root:
        raise ValueError(f"Expected 'label=root', got {text!r}")
    return label, root


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        type=parse_source,
        help="ML run panel as 'label=grib dir', repeatable. Must load.",
    )
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        type=parse_source,
        help="Baseline panel as 'label=root', repeatable. "
        "Left empty if it has no data at the lead time.",
    )
    parser.add_argument("--truth", type=str, default=None, help="Truth root (zarr)")
    parser.add_argument("--truth_label", type=str, default="Truth")
    parser.add_argument("--date", type=str, help="reference datetime")
    parser.add_argument("--leadtime", type=int, help="leadtime")
    parser.add_argument("--param", type=str, help="parameter")
    parser.add_argument(
        "--regions_json",
        type=str,
        help="JSON dict mapping region name -> {extent, projection}",
    )
    parser.add_argument("--outdir", type=str, help="output directory")
    args = parser.parse_args()

    init_time = datetime.strptime(args.date, "%Y%m%d%H%M")
    lead_time = args.leadtime
    param = args.param
    regions = json.loads(args.regions_json)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    valid_time = init_time + timedelta(hours=lead_time)

    panels = [(label, root, load_source, False) for label, root in args.source]
    panels += [(label, root, load_source, True) for label, root in args.baseline]
    if args.truth:
        panels.insert(0, (args.truth_label, args.truth, load_truth, True))
    if not panels:
        parser.error("no panels to plot")
    data = {}
    for label, root, loader, optional in panels:
        LOG.info("Loading '%s' from %s", label, root)
        try:
            lon, lat, values = loader(root, init_time, lead_time, param)
        except NOT_AVAILABLE:
            if not optional:
                raise
            LOG.warning(
                "'%s' has no data at +%sh, plotting an empty panel",
                label,
                lead_time,
                exc_info=True,
            )
            continue
        # Triangulate each grid once, shared by all regions.
        data[label] = (StatePlotter(lon, lat, outdir), values)
    if not data:
        raise RuntimeError(f"No source has data at +{lead_time}h")

    base_param, accu = parse_aggregated_param(param)
    _, units = prepare_values(base_param, np.empty(0))
    style = get_style(base_param, units, accu=accu or 1)

    for region_name, region_cfg in regions.items():
        LOG.info("Plotting region %s", region_name)
        projection, extent = resolve_region(region_name, region_cfg, lead_time)
        columns = min(len(panels), 4)
        rows = math.ceil(len(panels) / columns)
        fig = ekp.Figure(
            crs=projection,
            domain=extent,
            rows=rows,
            columns=columns,
            figsize=(5 * columns, 5 * rows),
        )
        for i, (label, *_) in enumerate(panels):
            subplot = fig.add_map(row=i // columns, column=i % columns)
            if label not in data:
                subplot.standard_layers()
                subplot.title(f"{label} (not available)")
                continue
            plotter, values = data[label]
            vmax = max_in_view(subplot, plotter.lon, plotter.lat, values)
            plotter.plot_field(
                subplot,
                values,
                title=f"{label}  (max {vmax:.1f} {units})",
                colorbar=False,
                **style,
            )
        if data:
            # One colorbar shared by all panels, which use identical levels.
            fig.legend(location="bottom")
        # Figure.title() passes an explicit y, which constrained layout ignores
        # when reserving space, so the suptitle can overlap the panel titles.
        fig.fig.suptitle(
            f"{param}  valid: {valid_time:%Y-%m-%d %H:%M} UTC  "
            f"(init: {init_time:%Y-%m-%d %H:%M} UTC, +{lead_time}h)"
        )
        outfn = outdir / f"{args.date}_{lead_time}_{param}_{region_name}.png"
        fig.save(outfn, dpi=200)
        LOG.info("saved: %s", outfn)


if __name__ == "__main__":
    main()
