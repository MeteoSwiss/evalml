import logging
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr

from data_input import (
    parse_steps,
    load_forecast_data,
    load_truth_data,
)
from verification import apply_lapse_rate_correction_inplace
from verification.spatial import map_forecast_to_truth

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--forecast", type=str, default=None, help="Directory to forecast grib data"
    )
    parser.add_argument(
        "--forecast_steps",
        type=parse_steps,
        default="0/120/6",
        help="Forecast steps in the format 'start/stop/step' (default: 0/120/6).",
    )
    parser.add_argument(
        "--forecast_label",
        type=str,
        default="forecast",
        help="Label for forecast line in plot legend.",
    )
    parser.add_argument(
        "--baseline",
        action="append",
        type=str,
        default=[],
        help="Path to baseline data root (repeatable).",
    )
    parser.add_argument(
        "--baseline_steps",
        action="append",
        type=parse_steps,
        default=[],
        help=(
            "Forecast steps in the format 'start/stop/step' for each baseline "
            "(repeatable, must match --baseline count)."
        ),
    )
    parser.add_argument(
        "--baseline_label",
        action="append",
        type=str,
        default=[],
        help="Label for each baseline line in plot legend (repeatable).",
    )
    parser.add_argument(
        "--analysis", type=str, default=None, help="Path to analysis data root"
    )
    parser.add_argument(
        "--analysis_label",
        type=str,
        default="truth",
        help="Label for analysis line in plot legend.",
    )
    parser.add_argument("--date", type=str, default=None, help="reference datetime")
    parser.add_argument("--outdir", type=str, help="output directory")
    parser.add_argument("--param", type=str, help="parameter")
    parser.add_argument("--stations", nargs="+", type=str, help="station IDs")
    parser.add_argument(
        "--lapse_rate_correction",
        action="store_true",
        default=False,
        help="Apply standard-atmosphere lapse-rate correction to T_2M.",
    )

    args = parser.parse_args()

    forecast_grib_dir = Path(args.forecast)
    forecast_steps = args.forecast_steps
    forecast_label = args.forecast_label
    analysis_root = Path(args.analysis)
    analysis_label = args.analysis_label
    baseline_roots = [Path(path) for path in args.baseline]
    baseline_steps = args.baseline_steps
    baseline_labels = args.baseline_label
    if len(baseline_roots) != len(baseline_steps):
        raise ValueError(
            "Mismatched baseline arguments: --baseline and --baseline_steps "
            "must be provided the same number of times."
        )
    if len(baseline_labels) != len(baseline_roots):
        raise ValueError(
            "Mismatched baseline arguments: --baseline and --baseline_label "
            "must be provided the same number of times."
        )
    init_time = datetime.strptime(args.date, "%Y%m%d%H%M")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    param = args.param
    stations = args.stations

    LOG.info(
        "Plotting meteogram: param=%s, stations=%s, init_time=%s",
        param,
        stations,
        init_time,
    )

    LOG.info("Loading analysis data from %s", analysis_root)
    analysis_ds = load_truth_data(
        analysis_root, init_time, forecast_steps, [param]
    ).squeeze()

    # Build station coordinate lookup from the loaded analysis dataset so that
    # any station with data for the plotted parameter is found (a fixed
    # parameter like rre150h0 would exclude stations like JUN).
    catalog_lookup = {
        str(sta): (float(lat), float(lon), float(elev))
        for sta, lat, lon, elev in zip(
            analysis_ds["values"].values,
            analysis_ds["latitude"].values,
            analysis_ds["longitude"].values,
            analysis_ds["elevation"].values,
        )
    }

    # Load gridded data once — shared across all station plots
    LOG.info("Loading forecast data from %s", forecast_grib_dir)
    forecast_ds = load_forecast_data(
        forecast_grib_dir, init_time, forecast_steps, [param]
    ).squeeze()

    baseline_ds_list = []
    for root, step, label in zip(baseline_roots, baseline_steps, baseline_labels):
        LOG.info("Loading baseline '%s' from %s", label, root)
        baseline_ds_list.append(
            load_forecast_data(root, init_time, step, [param]).squeeze()
        )

    param2plot = forecast_ds[param].attrs.get("parameter", {})
    short = param2plot.get("shortName", "")
    units = param2plot.get("units", "")
    name = param2plot.get("name", "")

    # Loop over stations — data is loaded once, mapping is per station
    for station in stations:
        LOG.info(
            "Plotting station %s (%d/%d)",
            station,
            stations.index(station) + 1,
            len(stations),
        )
        if station not in catalog_lookup:
            LOG.warning(
                "Station %r has no observations for parameter %s — writing placeholder.",
                station,
                param,
            )
            outfn = outdir / f"{init_time.strftime('%Y%m%d%H%M')}_{param}_{station}.png"
            fig, ax = plt.subplots()
            ax.text(
                0.5,
                0.5,
                f"No observations for {param}\nat station {station}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            plt.savefig(outfn)
            plt.close(fig)
            LOG.info("saved placeholder: %s", outfn)
            continue
        lat, lon, elev = catalog_lookup[station]
        station_ds = xr.Dataset(
            coords={
                "values": [station],
                "latitude": ("values", [lat]),
                "longitude": ("values", [lon]),
                "elevation": ("values", [elev]),
            }
        )

        forecast_station_ds = map_forecast_to_truth(forecast_ds, station_ds)
        analysis_station_ds = map_forecast_to_truth(analysis_ds, station_ds)
        baseline_station_ds_list = [
            map_forecast_to_truth(ds, station_ds) for ds in baseline_ds_list
        ]

        if args.lapse_rate_correction:
            apply_lapse_rate_correction_inplace(forecast_station_ds, station_ds, param)
            for ds in baseline_station_ds_list:
                apply_lapse_rate_correction_inplace(ds, station_ds, param)

        fig, ax = plt.subplots()

        ax.plot(
            analysis_station_ds["time"].values,
            analysis_station_ds[param].values,
            color="k",
            ls="--",
            label=analysis_label,
        )
        for i, (baseline_label, baseline_station_ds) in enumerate(
            zip(baseline_labels, baseline_station_ds_list), start=1
        ):
            ax.plot(
                baseline_station_ds["valid_time"].values,
                baseline_station_ds[param].values,
                color=f"C{i}",
                label=baseline_label,
            )
        ax.plot(
            forecast_station_ds["valid_time"].values,
            forecast_station_ds[param].values,
            color="C0",
            label=forecast_label,
        )

        ax.legend()
        ax.set_ylabel(f"{short} ({units})" if short or units else "")
        ax.set_title(f"{init_time} {name} at {station}")

        outfn = outdir / f"{init_time.strftime('%Y%m%d%H%M')}_{param}_{station}.png"
        plt.savefig(outfn)
        plt.close(fig)
        LOG.info(f"saved: {outfn}")


if __name__ == "__main__":
    main()
