"""Plot side-by-side maps of an ML run and a baseline at a common valid time."""

import json
import logging
from argparse import ArgumentParser
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import earthkit.plots as ekp
import numpy as np

from data_input import load_forecast_data
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


def load_source(root, init_time, lead_time, param):
    """Return (lon, lat, values) as flat arrays."""
    ds = load_forecast_data(root, init_time, [lead_time], [param]).squeeze()
    base_param, _ = parse_aggregated_param(param)
    values, _ = prepare_values(base_param, np.asarray(ds[param].values).ravel())
    lon = np.asarray(ds["longitude"].values).ravel()
    lat = np.asarray(ds["latitude"].values).ravel()
    return lon, lat, values


def main():
    parser = ArgumentParser()
    parser.add_argument("--forecast", type=str, help="Directory to ML run grib data")
    parser.add_argument("--forecast_label", type=str, default="Forecast")
    parser.add_argument("--baseline", type=str, help="Baseline root")
    parser.add_argument("--baseline_label", type=str, default="Baseline")
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

    sources = [
        (args.forecast_label, args.forecast),
        (args.baseline_label, args.baseline),
    ]
    data = {}
    for label, root in sources:
        LOG.info("Loading '%s' from %s", label, root)
        try:
            lon, lat, values = load_source(root, init_time, lead_time, param)
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
        fig = ekp.Figure(
            crs=projection,
            domain=extent,
            rows=1,
            columns=2,
            size=(10, 5),
        )
        for col, (label, _) in enumerate(sources):
            subplot = fig.add_map(row=0, column=col)
            if label not in data:
                subplot.standard_layers()
                subplot.title(f"{label} (not available)")
                continue
            plotter, values = data[label]
            plotter.plot_field(
                subplot,
                values,
                title=label,
                colorbar=False,
                **style,
            )
        if data:
            # One colorbar shared by both panels, which use identical levels.
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
