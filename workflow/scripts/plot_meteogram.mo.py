import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    from argparse import ArgumentParser
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import xarray as xr
    from meteodatalab import data_source, grib_decoder
    from peakweather import PeakWeatherDataset

    from data_input import load_analysis_data_from_zarr, load_baseline_from_zarr
    return (
        ArgumentParser,
        Path,
        PeakWeatherDataset,
        data_source,
        grib_decoder,
        load_analysis_data_from_zarr,
        load_baseline_from_zarr,
        np,
        plt,
        xr,
    )


@app.cell
def _(ArgumentParser, Path):
    parser = ArgumentParser()

    parser.add_argument(
        "--forecast", type=str, default=None, help="Directory to forecast grib data"
    )
    parser.add_argument(
        "--analysis", type=str, default=None, help="Path to analysis zarr data"
    )
    parser.add_argument(
        "--baseline", type=str, default=None, help="Path to baseline zarr data"
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
    zarr_dir_ana = Path(args.analysis)
    zarr_dir_base = Path(args.baseline)
    peakweather_dir = Path(args.peakweather)
    init_time = args.date
    outfn = Path(args.outfn)
    station = args.station
    param = args.param
    return (
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
    data_source,
    grib_decoder,
    grib_dir,
    init_time,
    load_analysis_data_from_zarr,
    load_baseline_from_zarr,
    param,
    preprocess_ds,
    xr,
    zarr_dir_ana,
    zarr_dir_base,
):
    if param == "SP_10M":
        paramlist = ["U_10M", "V_10M"]
    elif param == "SP":
        paramlist = ["U", "V"]
    else:
        paramlist = [param]

    grib_files = sorted(grib_dir.glob(f"{init_time}*.grib"))
    fds = data_source.FileDataSource(datafiles=grib_files)
    ds_fct = xr.Dataset(grib_decoder.load(fds, {"param": paramlist}))
    ds_fct = preprocess_ds(ds_fct, param)
    da_fct = ds_fct[param].squeeze()

    ds_ana = load_analysis_data_from_zarr(zarr_dir_ana, da_fct.valid_time, paramlist)
    ds_ana = preprocess_ds(ds_ana, param)
    da_ana = ds_ana[param].squeeze()

    steps = list(
        range(da_fct.sizes["lead_time"])
    )  # FIX: this will fail if lead_time is not 0,1,2,...
    ds_base = load_baseline_from_zarr(zarr_dir_base, da_fct.ref_time, steps, paramlist)
    ds_base = preprocess_ds(ds_base, param)
    da_base = ds_base[param].squeeze()
    return da_ana, da_base, da_fct


@app.cell
def _(PeakWeatherDataset, da_fct, np, param, peakweather_dir, station):
    if param == "T_2M":
        parameter = "temperature"
        offset = 273.15  # K to C
    elif param == "SP_10M":
        parameter = "wind_speed"
        offset = 0
    elif param == "TOT_PREC":
        parameter = "precipitation"
        offset = 0
    else:
        raise NotImplementedError(
            f"The mapping for {param=} to PeakWeather is not implemented"
        )

    peakweather = PeakWeatherDataset(root=peakweather_dir, freq="1h")
    obs, mask = peakweather.get_observations(
        parameters=[parameter],
        stations=station,
        first_date=np.datetime_as_string(da_fct.valid_time.values[0]),
        last_date=np.datetime_as_string(da_fct.valid_time.values[-1]),
        return_mask=True,
    )
    obs = obs.loc[:, mask.iloc[0]].droplevel("name", axis=1)
    obs
    return obs, offset, peakweather


@app.cell
def _(peakweather):
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
        try:
            lat = ds["lat"]
            lon = ds["lon"]
        except KeyError:
            lat = ds["latitude"]
            lon = ds["longitude"]

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
    sta_idxs["fct_isel"]  = sta_idxs.apply(lambda r: get_idx_row(r, da_fct),  axis=1)
    sta_idxs["ana_isel"]  = sta_idxs.apply(lambda r: get_idx_row(r, da_ana),  axis=1)
    sta_idxs["base_isel"] = sta_idxs.apply(lambda r: get_idx_row(r, da_base), axis=1)
    sta_idxs 
    return (sta_idxs,)


@app.cell
def _(
    da_ana,
    da_base,
    da_fct,
    init_time,
    obs,
    offset,
    outfn,
    plt,
    sta_idxs,
    station,
):
    # station indices
    row = sta_idxs.loc[station]
    fct_isel  = row.fct_isel
    ana_isel  = row.ana_isel
    base_isel = row.base_isel

    fig, ax = plt.subplots()

    # station
    ax.plot(
        obs.index.to_pydatetime(),
        obs.to_numpy() + offset,
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
        fct2plot["valid_time"].values,
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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
