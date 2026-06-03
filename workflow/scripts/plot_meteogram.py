import logging
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from peakweather import PeakWeatherDataset

from data import (
    parse_steps,
    load_forecast_data,
    load_truth_data,
)
from verification.spatial import map_forecast_to_truth

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def preprocess_ds(ds, param: str):
    ds = ds.copy()
    if param == "SP_10M":
        ds[param] = (ds.U_10M**2 + ds.V_10M**2) ** 0.5
        try:
            units = ds["U_10M"].attrs["parameter"]["units"]
        except KeyError:
            units = None
        ds[param].attrs["parameter"] = {
            "shortName": "SP_10M",
            "units": units,
            "name": "10m wind speed",
        }
        ds = ds.drop_vars(["U_10M", "V_10M"])
    if param == "SP":
        ds[param] = (ds.U**2 + ds.V**2) ** 0.5
        units = ds.U.attrs["parameter"]["units"]
        ds[param].attrs["parameter"] = {
            "shortName": "SP",
            "units": units,
            "name": "Wind speed",
        }
        ds = ds.drop_vars(["U", "V"])
    return ds.squeeze()


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
    parser.add_argument(
        "--peakweather", type=str, default=None, help="Path to PeakWeather dataset"
    )
    parser.add_argument("--date", type=str, default=None, help="reference datetime")
    parser.add_argument("--outdir", type=str, help="output directory")
    parser.add_argument("--param", type=str, help="parameter")
    parser.add_argument("--stations", nargs="+", type=str, help="station IDs")

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
    peakweather_dir = Path(args.peakweather)
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

    if param == "SP_10M":
        paramlist = ["U_10M", "V_10M"]
    elif param == "SP":
        paramlist = ["U", "V"]
    else:
        paramlist = [param]

    # Load gridded data once — shared across all station plots
    LOG.info("Loading forecast data from %s", forecast_grib_dir)
    forecast_ds = load_forecast_data(
        forecast_grib_dir, init_time, forecast_steps, paramlist
    )
    forecast_ds = preprocess_ds(forecast_ds, param)

    steps = [int(s) for s in forecast_ds["step"].dt.total_seconds().values / 3600]
    LOG.info("Loading analysis data from %s", analysis_root)
    analysis_ds = load_truth_data(analysis_root, init_time, steps, paramlist)
    analysis_ds = preprocess_ds(analysis_ds, param)

    baseline_ds_list = []
    for root, step, label in zip(baseline_roots, baseline_steps, baseline_labels):
        LOG.info("Loading baseline '%s' from %s", label, root)
        baseline_ds_list.append(
            preprocess_ds(load_forecast_data(root, init_time, step, paramlist), param)
        )

    # Load station metadata once
    LOG.info("Loading station metadata from %s", peakweather_dir)
    peakweather = PeakWeatherDataset(root=peakweather_dir)
    stations_table = peakweather.stations_table
    stations_table.index.names = ["values"]

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
        station_ds = stations_table.to_xarray().sel(values=[station])
        station_ds = station_ds.set_coords(("latitude", "longitude", "station_name"))
        station_ds = station_ds.drop_vars(list(station_ds.data_vars))

        forecast_station_ds = map_forecast_to_truth(forecast_ds, station_ds)
        analysis_station_ds = map_forecast_to_truth(analysis_ds, station_ds)
        baseline_station_ds_list = [
            map_forecast_to_truth(ds, station_ds) for ds in baseline_ds_list
        ]

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
