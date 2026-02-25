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

    return (
        ArgumentParser,
        Path,
        PeakWeatherDataset,
        datetime,
        load_forecast_data,
        load_truth_data,
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
        return ds

    return (preprocess_ds,)


@app.cell
def load_grib_data(
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
    da_fct = ds_fct[param].squeeze()

    steps = da_fct.lead_time.dt.total_seconds().values / 3600
    ds_ana = load_truth_data(zarr_dir_ana, init_time, steps, paramlist)
    ds_ana = preprocess_ds(ds_ana, param)
    da_ana = ds_ana[param].squeeze()

    ds_base = load_forecast_data(zarr_dir_base, init_time, baseline_steps, paramlist)
    ds_base = preprocess_ds(ds_base, param)
    da_base = ds_base[param].squeeze()

    ds_obs = load_truth_data(peakweather_dir, init_time, steps, paramlist)
    ds_obs = preprocess_ds(ds_obs, param)
    da_obs = ds_obs[param].squeeze()
    return da_ana, da_base, da_fct, da_obs


@app.cell
def _(PeakWeatherDataset, peakweather_dir):
    peakweather = PeakWeatherDataset(root=peakweather_dir, freq="1h")
    stations = peakweather.stations_table
    stations.index.names = ["station"]
    stations
    return (stations,)


@app.cell
def _(da_ana, da_base, da_fct, np, stations):
    def nearest_indexers_euclid(ds, lat_s, lon_s):
        """
        Return a dict of indexers usable as: ds.isel(**indexers)

        Examples:
          - 2D structured grid -> {"y": y_idx, "x": x_idx}
          - 1D unstructured grid -> {"point": i_idx} (or {"cell": i_idx}, etc.)
        """
        lat = ds["lat"]
        lon = ds["lon"]
        dist = (lat - lat_s) ** 2 + (lon - lon_s) ** 2
        arr = dist.values

        flat_idx = int(np.nanargmin(arr))

        if dist.ndim == 1:
            return {dist.dims[0]: flat_idx}

        unr = np.unravel_index(flat_idx, dist.shape)
        return {dim: int(i) for dim, i in zip(dist.dims, unr)}

    def get_idx_row(row, da):
        return nearest_indexers_euclid(da, row["latitude"], row["longitude"])

    # store dicts (indexers) in columns
    sta_idxs = stations.copy()
    sta_idxs["fct_isel"] = sta_idxs.apply(lambda r: get_idx_row(r, da_fct), axis=1)
    sta_idxs["ana_isel"] = sta_idxs.apply(lambda r: get_idx_row(r, da_ana), axis=1)
    sta_idxs["base_isel"] = sta_idxs.apply(lambda r: get_idx_row(r, da_base), axis=1)
    sta_idxs
    return (sta_idxs,)


@app.cell
def _(
    da_ana,
    da_base,
    da_fct,
    da_obs,
    init_time,
    outfn,
    plt,
    sta_idxs,
    station,
):
    # station indices
    row = sta_idxs.loc[station]
    fct_isel = row.fct_isel
    ana_isel = row.ana_isel
    base_isel = row.base_isel

    fig, ax = plt.subplots()

    # station
    obs2plot = da_obs.sel(values=station)
    ax.plot(
        obs2plot["time"].values,
        obs2plot.values,
        color="k",
        ls="--",
        label=station,
    )

    # analysis
    ana2plot = da_ana.isel(**ana_isel)
    ax.plot(
        ana2plot["time"].values,
        ana2plot.values,
        color="k",
        ls="-",
        label="analysis",
    )

    # baseline
    base2plot = da_base.isel(**base_isel)
    ax.plot(
        base2plot["time"].values,
        base2plot.values,
        color="C1",
        label="baseline",
    )

    # forecast
    fct2plot = da_fct.isel(**fct_isel)
    ax.plot(
        fct2plot["time"].values,
        fct2plot.values,
        color="C0",
        label="forecast",
    )

    ax.legend()

    param2plot = da_fct.attrs.get("parameter", {})
    short = param2plot.get("shortName", "")
    units = param2plot.get("units", "")
    name = param2plot.get("name", "")

    ax.set_ylabel(f"{short} ({units})" if short or units else "")
    ax.set_title(f"{init_time} {name} at {station}")

    plt.savefig(outfn)
    return


if __name__ == "__main__":
    app.run()
