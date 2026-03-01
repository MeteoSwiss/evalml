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
        "--steps",
        type=parse_steps,
        default="0/120/6",
        help="Forecast steps in the format 'start/stop/step' (default: 0/120/6).",
    )
    parser.add_argument(
        "--baseline", type=str, default=None, help="Path to baseline zarr data"
    )
    parser.add_argument(
        "--baseline_steps",
        type=parse_steps,
        default="0/120/6",
        help="Forecast steps in the format 'start/stop/step' (default: 0/120/6).",
    )
    parser.add_argument(
        "--analysis", type=str, default=None, help="Path to analysis zarr data"
    )
    parser.add_argument(
        "--peakweather", type=str, default=None, help="Path to PeakWeather dataset"
    )
    parser.add_argument("--date", type=str, default=None, help="reference datetime")
    parser.add_argument("--outfn", type=str, help="output filename")
    parser.add_argument("--param", type=str, help="parameter")
    parser.add_argument("--station", type=str, help="station")

    args = parser.parse_args()
    grib_dir = Path(args.forecast)
    forecast_steps = args.steps
    zarr_dir_ana = Path(args.analysis)
    zarr_dir_base = Path(args.baseline)
    baseline_steps = args.baseline_steps
    peakweather_dir = Path(args.peakweather)
    init_time = datetime.strptime(args.date, "%Y%m%d%H%M")
    outfn = Path(args.outfn)
    station = args.station
    param = args.param
    return (
        baseline_steps,
        forecast_steps,
        grib_dir,
        init_time,
        outfn,
        param,
        peakweather_dir,
        station,
        zarr_dir_ana,
        zarr_dir_base,
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
    baseline_steps,
    forecast_steps,
    grib_dir,
    init_time,
    load_forecast_data,
    load_truth_data,
    param,
    peakweather_dir,
    preprocess_ds,
    zarr_dir_ana,
    zarr_dir_base,
):
    if param == "SP_10M":
        paramlist = ["U_10M", "V_10M"]
    elif param == "SP":
        paramlist = ["U", "V"]
    else:
        paramlist = [param]

    ds_fct = load_forecast_data(grib_dir, init_time, forecast_steps, paramlist)
    ds_fct = preprocess_ds(ds_fct, param)

    steps = ds_fct.lead_time.dt.total_seconds().values / 3600
    ds_ana = load_truth_data(zarr_dir_ana, init_time, steps, paramlist)
    ds_ana = preprocess_ds(ds_ana, param)

    ds_base = load_forecast_data(zarr_dir_base, init_time, baseline_steps, paramlist)
    ds_base = preprocess_ds(ds_base, param)

    ds_obs = load_truth_data(peakweather_dir, init_time, steps, paramlist)
    ds_obs = preprocess_ds(ds_obs, param)
    return ds_ana, ds_base, ds_fct


@app.cell
def _(PeakWeatherDataset, peakweather_dir, station):
    peakweather = PeakWeatherDataset(root=peakweather_dir)
    stations = peakweather.stations_table
    stations.index.names = ["values"]
    ds_sta = stations.to_xarray().sel(values=[station])  # keep singleton dim
    ds_sta = ds_sta.rename({"latitude": "lat", "longitude": "lon"})
    ds_sta = ds_sta.set_coords(("lat", "lon", "station_name"))
    ds_sta = ds_sta.drop_vars(list(ds_sta.data_vars))
    ds_sta
    return (ds_sta,)


@app.cell
def _(ds_ana, ds_base, ds_fct, ds_sta, map_forecast_to_truth):
    ds_fct_sta = map_forecast_to_truth(ds_fct, ds_sta)
    ds_ana_sta = map_forecast_to_truth(ds_ana, ds_sta)
    ds_base_sta = map_forecast_to_truth(ds_base, ds_sta)
    return ds_ana_sta, ds_base_sta, ds_fct_sta


@app.cell
def _(
    ds_ana_sta,
    ds_base_sta,
    ds_fct,
    ds_fct_sta,
    init_time,
    outfn,
    param,
    plt,
    station,
):
    fig, ax = plt.subplots()

    # truth
    ax.plot(
        ds_ana_sta["time"].values,
        ds_ana_sta[param].values,
        color="k",
        ls="--",
        label="truth",
    )
    # baseline
    ax.plot(
        ds_base_sta["time"].values,
        ds_base_sta[param].values,
        color="C1",
        label="baseline",
    )
    # forecast
    ax.plot(
        ds_fct_sta["time"].values,
        ds_fct_sta[param].values,
        color="C0",
        label="forecast",
    )

    ax.legend()

    param2plot = ds_fct[param].attrs.get("parameter", {})
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
