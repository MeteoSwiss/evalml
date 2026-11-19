import logging
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from data_input import (
    load_forecast_data,
    load_truth_data,
    parse_steps,
)
from verification import CIRCULAR_PARAMS, apply_lapse_rate_correction_inplace
from verification.spatial import map_forecast_to_truth

LOG = logging.getLogger(__name__)

# Colours as in the meteograms of the Varda-single paper (Figs. 8 and 9).
OBS_COLOR = "mediumseagreen"
# One colour per forecast run, in config order (first run red as Varda-single in the paper).
FORECAST_COLORS = ["tab:red", "black", "tab:orange", "tab:purple", "tab:brown"]
BASELINE_COLORS = {
    "ICON-CH1": "deepskyblue",
    "ICON-CH2": "navy",
}


KELVIN_PARAMS = {"T_2M", "TD_2M"}
ZERO_CELSIUS_IN_KELVIN = 273.15


def _lead_hours(times, init_time: datetime) -> np.ndarray:
    """Hours between each valid time and the forecast initialisation time."""
    return (np.asarray(times, dtype="datetime64[ns]") - np.datetime64(init_time)) / (
        np.timedelta64(1, "h")
    )


def _baseline_color(label: str, default: str) -> str:
    """Paper colour for ICON-CH1/CH2 baselines (matched on the label), else default."""
    for key, color in BASELINE_COLORS.items():
        if key in label.upper():
            return color
    return default


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--forecast",
        action="append",
        type=str,
        default=[],
        help="Directory with forecast grib data (repeatable, one per run).",
    )
    parser.add_argument(
        "--forecast_steps",
        action="append",
        type=parse_steps,
        default=[],
        help=(
            "Forecast steps in the format 'start/stop/step' for each forecast "
            "(repeatable, must match --forecast count)."
        ),
    )
    parser.add_argument(
        "--forecast_label",
        action="append",
        type=str,
        default=[],
        help="Label for each forecast line in plot legend (repeatable).",
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
        help="Apply lapse-rate correction to T_2M and PS.",
    )

    args = parser.parse_args()

    forecast_grib_dirs = [Path(path) for path in args.forecast]
    forecast_steps_list = args.forecast_steps
    forecast_labels = args.forecast_label
    if not forecast_grib_dirs:
        raise ValueError("At least one --forecast is required.")
    if not (
        len(forecast_grib_dirs) == len(forecast_steps_list) == len(forecast_labels)
    ):
        raise ValueError(
            "Mismatched forecast arguments: --forecast, --forecast_steps and "
            "--forecast_label must be provided the same number of times."
        )
    # Observations at every valid time any forecast or baseline covers (e.g. hourly
    # when ICON is hourly and the ML runs 6-hourly).
    obs_steps = sorted(set().union(*forecast_steps_list, *args.baseline_steps))
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
        analysis_root, init_time, obs_steps, [param]
    ).squeeze()

    # Build station coordinate lookup from the loaded analysis dataset so that
    # any station with data for the plotted parameter is found (a fixed
    # parameter like rre150h0 would exclude stations like JUN).
    # A gridded truth on a (y, x) grid (e.g. INCA) has no station axis.
    if "values" in analysis_ds.dims:
        catalog_lookup = {
            str(sta): (float(lat), float(lon), float(elev))
            for sta, lat, lon, elev in zip(
                analysis_ds["values"].values,
                analysis_ds["latitude"].values,
                analysis_ds["longitude"].values,
                analysis_ds["elevation"].values,
            )
        }
    else:
        LOG.warning(
            "Truth %s has no station axis; meteograms require station observations.",
            analysis_root,
        )
        catalog_lookup = {}

    # Load gridded data once — shared across all station plots
    forecast_ds_list = []
    for grib_dir, steps, label in zip(
        forecast_grib_dirs, forecast_steps_list, forecast_labels
    ):
        LOG.info("Loading forecast '%s' from %s", label, grib_dir)
        forecast_ds_list.append(
            load_forecast_data(grib_dir, init_time, steps, [param]).squeeze()
        )
    forecast_ds = forecast_ds_list[0]

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

        forecast_station_ds_list = [
            map_forecast_to_truth(ds, station_ds) for ds in forecast_ds_list
        ]
        analysis_station_ds = map_forecast_to_truth(analysis_ds, station_ds)
        baseline_station_ds_list = [
            map_forecast_to_truth(ds, station_ds) for ds in baseline_ds_list
        ]

        if args.lapse_rate_correction:
            for ds in forecast_station_ds_list + baseline_station_ds_list:
                apply_lapse_rate_correction_inplace(ds, station_ds, [param])

        fig, ax = plt.subplots()

        # Wind direction wraps at 0/360 deg, so lines would jump across the plot:
        # draw it as dots instead.
        line_style = (
            {"ls": "none", "marker": "o", "ms": 3} if param in CIRCULAR_PARAMS else {}
        )

        # Temperatures in degrees Celsius, as in the paper.
        offset = ZERO_CELSIUS_IN_KELVIN if param in KELVIN_PARAMS else 0.0
        plot_units = "°C" if param in KELVIN_PARAMS else units

        ax.plot(
            _lead_hours(analysis_station_ds["time"].values, init_time),
            analysis_station_ds[param].values - offset,
            color=OBS_COLOR,
            label=analysis_label,
            **line_style,
        )
        for i, (baseline_label, baseline_station_ds) in enumerate(
            zip(baseline_labels, baseline_station_ds_list), start=1
        ):
            ax.plot(
                _lead_hours(baseline_station_ds["valid_time"].values, init_time),
                baseline_station_ds[param].values - offset,
                color=_baseline_color(baseline_label, default=f"C{i}"),
                label=baseline_label,
                # Baselines dashed, as the ICON lines in the paper's Fig. 8.
                **(line_style or {"ls": "--"}),
            )
        for i, (forecast_label, forecast_station_ds) in enumerate(
            zip(forecast_labels, forecast_station_ds_list)
        ):
            ax.plot(
                _lead_hours(forecast_station_ds["valid_time"].values, init_time),
                forecast_station_ds[param].values - offset,
                color=FORECAST_COLORS[i % len(FORECAST_COLORS)],
                label=forecast_label,
                **line_style,
            )

        ax.legend()
        ylabel = short or param
        ax.set_ylabel(f"{ylabel} ({plot_units})" if plot_units else ylabel)
        ax.set_xlabel(f"Lead time from {init_time:%Y-%m-%d %H:%M} (h)")
        ax.grid(alpha=0.3)
        ax.set_title(f"{name or param} at {station}")

        outfn = outdir / f"{init_time.strftime('%Y%m%d%H%M')}_{param}_{station}.png"
        plt.savefig(outfn)
        plt.close(fig)
        LOG.info(f"saved: {outfn}")


if __name__ == "__main__":
    main()
