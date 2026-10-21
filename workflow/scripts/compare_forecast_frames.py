"""Plot side-by-side maps of the truth, ML runs and baselines at a common valid time."""

import json
import logging
import math
from argparse import ArgumentParser
from datetime import datetime
from datetime import timedelta
from pathlib import Path

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
        help="Panel as 'label=root' (ML run grib dir or baseline root), repeatable.",
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

    panels = [(label, root, load_source) for label, root in args.source]
    if args.truth:
        panels.insert(0, (args.truth_label, args.truth, load_truth))
    data = {}
    for label, root, loader in panels:
        LOG.info("Loading '%s' from %s", label, root)
        try:
            lon, lat, values = loader(root, init_time, lead_time, param)
        except Exception:
            # Loaders fail in different ways for lead times a source does not
            # provide (outside its steps, no accumulation window, ...).
            LOG.warning(
                "Could not load '%s' at +%sh, plotting an empty panel",
                label,
                lead_time,
                exc_info=True,
            )
            continue
        # Triangulate each grid once, shared by all regions.
        data[label] = (StatePlotter(lon, lat, outdir), values)

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
            size=(5 * columns, 5 * rows),
        )
        for i, (label, _, _) in enumerate(panels):
            subplot = fig.add_map(row=i // columns, column=i % columns)
            if label not in data:
                subplot.standard_layers()
                subplot.title(f"{label} (not available)")
                continue
            plotter, values = data[label]
            plotter.plot_field(
                subplot,
                values,
                title=f"{label}  (max {np.nanmax(values):.1f} {units})",
                colorbar=False,
                **style,
            )
        if data:
            # One colorbar shared by all panels, which use identical levels.
            fig.legend(location="bottom")
        fig.title(
            f"{param}  valid: {valid_time:%Y-%m-%d %H:%M} UTC  "
            f"(init: {init_time:%Y-%m-%d %H:%M} UTC, +{lead_time}h)"
        )
        outfn = outdir / f"{args.date}_{lead_time}_{param}_{region_name}.png"
        fig.save(outfn, dpi=200)
        LOG.info("saved: %s", outfn)


if __name__ == "__main__":
    main()
