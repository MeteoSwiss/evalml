import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    from argparse import ArgumentParser
    from datetime import datetime
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    from peakweather import PeakWeatherDataset

    from data_input import (
        parse_steps,
        load_forecast_data,
        load_truth_data,
    )
    from verification.spatial import map_forecast_to_truth

    return (
        ArgumentParser,
        Path,
        PeakWeatherDataset,
        datetime,
        load_forecast_data,
        load_truth_data,
        map_forecast_to_truth,
        np,
        parse_steps,
        plt,
    )


@app.cell
def _(ArgumentParser, Path, datetime, parse_steps):
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
        help="Path to baseline zarr data (repeatable).",
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
        "--analysis", type=str, default=None, help="Path to analysis zarr data"
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
    parser.add_argument("--outfn", type=str, help="output filename")
    parser.add_argument("--param", type=str, help="parameter")
    parser.add_argument("--station", type=str, help="station")

    args = parser.parse_args()
    forecast_grib_dir = Path(args.forecast)
    forecast_steps = args.forecast_steps
    forecast_label = args.forecast_label
    analysis_zarr = Path(args.analysis)
    analysis_label = args.analysis_label
    baseline_zarrs = [Path(path) for path in args.baseline]
    baseline_steps = args.baseline_steps
    baseline_labels = args.baseline_label
    if len(baseline_zarrs) != len(baseline_steps):
        raise ValueError(
            "Mismatched baseline arguments: --baseline and --baseline_steps "
            "must be provided the same number of times."
        )
    if len(baseline_labels) != len(baseline_zarrs):
        raise ValueError(
            "Mismatched baseline arguments: --baseline and --baseline_label "
            "must be provided the same number of times."
        )
    peakweather_dir = Path(args.peakweather)
    init_time = datetime.strptime(args.date, "%Y%m%d%H%M")
    outfn = Path(args.outfn)
    station = args.station
    param = args.param
    return (
        analysis_label,
        analysis_zarr,
        baseline_labels,
        baseline_steps,
        baseline_zarrs,
        forecast_label,
        forecast_steps,
        forecast_grib_dir,
        init_time,
        outfn,
        param,
        peakweather_dir,
        station,
    )


@app.cell
def _(np):
    def preprocess_ds(ds, param: str):
        ds = ds.copy()
        # 10m wind speed
        if param == "SP_10M":
            ds[param] = np.sqrt(ds.U_10M**2 + ds.V_10M**2)
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
        # wind speed from standard-level components
        if param == "SP":
            ds[param] = np.sqrt(ds.U**2 + ds.V**2)
            units = ds.U.attrs["parameter"]["units"]
            ds[param].attrs["parameter"] = {
                "shortName": "SP",
                "units": units,
                "name": '"Wind speed',
            }
            ds = ds.drop_vars(["U", "V"])
        return ds.squeeze()

    return (preprocess_ds,)


@app.cell
def load_data(
    analysis_zarr,
    baseline_steps,
    baseline_zarrs,
    forecast_steps,
    forecast_grib_dir,
    init_time,
    load_forecast_data,
    load_truth_data,
    param,
    preprocess_ds,
):
    if param == "SP_10M":
        paramlist = ["U_10M", "V_10M"]
    elif param == "SP":
        paramlist = ["U", "V"]
    else:
        paramlist = [param]

    forecast_ds = load_forecast_data(
        forecast_grib_dir, init_time, forecast_steps, paramlist
    )
    forecast_ds = preprocess_ds(forecast_ds, param)

    steps = forecast_ds.lead_time.dt.total_seconds().values / 3600
    analysis_ds = load_truth_data(analysis_zarr, init_time, steps, paramlist)
    analysis_ds = preprocess_ds(analysis_ds, param)

    baseline_ds_list = [
        preprocess_ds(
            load_forecast_data(zarr, init_time, step, paramlist),
            param,
        )
        for zarr, step in zip(baseline_zarrs, baseline_steps)
    ]

    return analysis_ds, baseline_ds_list, forecast_ds


@app.cell
def _(PeakWeatherDataset, peakweather_dir, station):
    peakweather = PeakWeatherDataset(root=peakweather_dir)
    stations = peakweather.stations_table
    stations.index.names = ["values"]
    station_ds = stations.to_xarray().sel(values=[station])  # keep singleton dim
    station_ds = station_ds.rename({"latitude": "lat", "longitude": "lon"})
    station_ds = station_ds.set_coords(("lat", "lon", "station_name"))
    station_ds = station_ds.drop_vars(list(station_ds.data_vars))
    station_ds
    return (station_ds,)


@app.cell
def _(analysis_ds, baseline_ds_list, forecast_ds, station_ds, map_forecast_to_truth):
    forecast_station_ds = map_forecast_to_truth(forecast_ds, station_ds)
    analysis_station_ds = map_forecast_to_truth(analysis_ds, station_ds)
    baseline_station_ds_list = [
        map_forecast_to_truth(ds, station_ds) for ds in baseline_ds_list
    ]
    return analysis_station_ds, baseline_station_ds_list, forecast_station_ds


@app.cell
def _(
    analysis_label,
    baseline_labels,
    analysis_station_ds,
    baseline_station_ds_list,
    forecast_label,
    forecast_ds,
    forecast_station_ds,
    init_time,
    outfn,
    param,
    plt,
    station,
):
    fig, ax = plt.subplots()

    # truth
    ax.plot(
        analysis_station_ds["time"].values,
        analysis_station_ds[param].values,
        color="k",
        ls="--",
        label=analysis_label,
    )
    # baselines
    for i, (baseline_label, baseline_station_ds) in enumerate(
        zip(baseline_labels, baseline_station_ds_list), start=1
    ):
        ax.plot(
            baseline_station_ds["time"].values,
            baseline_station_ds[param].values,
            color=f"C{i}",
            label=f"{baseline_label}",
        )
    # forecast
    ax.plot(
        forecast_station_ds["time"].values,
        forecast_station_ds[param].values,
        color="C0",
        label=forecast_label,
    )

    ax.legend()

    param2plot = forecast_ds[param].attrs.get("parameter", {})
    short = param2plot.get("shortName", "")
    units = param2plot.get("units", "")
    name = param2plot.get("name", "")

    ax.set_ylabel(f"{short} ({units})" if short or units else "")
    ax.set_title(f"{init_time} {name} at {station}")

    plt.savefig(outfn)
    print(f"saved: {outfn}")
    return


if __name__ == "__main__":
    app.run()
