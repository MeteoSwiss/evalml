import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    from argparse import ArgumentParser
    from datetime import datetime, timedelta
    from pathlib import Path

    import earthkit.data as ekd
    import numpy as np
    import pandas as pd
    import xarray as xr
    from peakweather.dataset import PeakWeatherDataset
    from scipy.spatial import cKDTree

    xr.set_options(keep_attrs=True)
    return (
        ArgumentParser,
        Path,
        PeakWeatherDataset,
        cKDTree,
        datetime,
        ekd,
        np,
        pd,
        timedelta,
        xr,
    )


@app.cell
def _(ArgumentParser, Path, datetime):
    parser = ArgumentParser()

    parser.add_argument(
        "--forecast", type=str, default=None, help="Path to forecast grib data"
    )
    parser.add_argument(
        "--peakweather", type=str, default=None, help="Path to PeakWeather dataset"
    )

    args = parser.parse_args()
    grib_file = Path(args.forecast)
    peakweather_dir = Path(args.peakweather)
    ref_time = datetime.strptime(grib_file.name.split('_')[0], '%Y%m%d%H%M')
    return grib_file, peakweather_dir, ref_time


@app.cell
def _(ekd, grib_file):
    print(f"read: {grib_file}")
    src = ekd.from_source("file", grib_file)
    ds_full = src.to_xarray(add_earthkit_attrs=True, flatten_values=True)
    ds_full
    return (ds_full,)


@app.cell
def _(PeakWeatherDataset, pd, peakweather_dir, ref_time, timedelta):
    peakweather = PeakWeatherDataset(root=peakweather_dir, freq="1h")
    obs, mask = peakweather.get_observations(
        parameters=["temperature", "humidity", "wind_u", "wind_v"],
        first_date=f"{ref_time - timedelta(hours=1):%Y-%m-%d %H:%M}",
        last_date=f"{ref_time:%Y-%m-%d %H:%M}",
        return_mask=True,
    )
    obs = obs.loc[:, mask.iloc[0]]
    obs = obs.iloc[0]
    obs.index = obs.index.set_names(["station", "parameter"])
    obs = obs.unstack("parameter").sort_index().sort_index(axis=1)
    peakweather.stations_table.index.names = ["station"]
    stations = pd.concat([obs, peakweather.stations_table], axis=1)
    stations
    return (stations,)


@app.cell
def _(cKDTree, np):
    def idw_points(xy_obs, v_obs, xy_tgt, k=5, power=2):
        """IDW from scattered obs -> scattered targets (all in 2D)."""
        tree = cKDTree(np.asarray(xy_obs, float))
        dist, idx = tree.query(np.asarray(xy_tgt, float), k=k)

        if k == 1:
            dist = dist[:, None]
            idx = idx[:, None]

        dist = dist + 1e-12
        w = 1.0 / (dist ** power)
        v_obs = np.asarray(v_obs, float)

        return (w * v_obs[idx]).sum(axis=1) / w.sum(axis=1)


    def interpolation_of_residuals(
        background,         # (ny, nx) background field
        grid_lat, grid_lon, # (ny, nx) 2D arrays
        st_lat, st_lon,     # (n,) station coords
        st_obs,             # (n,) station values in same units as background
        k=5,
        power=2,
        max_dist=1,
    ):
        background = np.asarray(background, float)
        lat = np.asarray(grid_lat, float)
        lon = np.asarray(grid_lon, float)
        st_obs = np.asarray(st_obs, float)
        st_lon = np.asarray(st_lon, float)
        st_lat = np.asarray(st_lat, float)

        # ny, nx = B.shape

        # scale longitude by cos(lat)
        lat0 = np.deg2rad(np.nanmean(lat))

        # 1) represent grid as points
        grid_xy = np.c_[lon.ravel() * np.cos(lat0), lat.ravel()]  # (ny*nx, 2)


        # 2) represent stations as points
        st_xy = np.c_[st_lon * np.cos(lat0), st_lat]

        # drop NaN stations
        ok = np.isfinite(st_obs) & np.isfinite(st_xy).all(axis=1)
        st_xy = st_xy[ok]
        st_obs = st_obs[ok]

        # 3) background at station locations (nearest grid point)
        grid_tree = cKDTree(grid_xy)
        _, gi = grid_tree.query(st_xy, k=1)
        b_at_st = background.ravel()[gi]

        # 4) interpolate residuals
        r_at_st = b_at_st - st_obs
        res = idw_points(st_xy, r_at_st, grid_xy, k=k, power=power) #.reshape(ny, nx)

        # 5) apply distance cutoff
        tree = cKDTree(st_xy)
        dmin, _ = tree.query(grid_xy, k=1)  # nearest-station distance for every grid cell
        w = 1.0 - np.clip(dmin / max_dist, 0.0, 1.0)  # 1 at station -> 0 at max_dist
        w = w #.reshape(ny, nx)
        res = w * res

        return background - res
    return (interpolation_of_residuals,)


@app.cell
def _(ds_full, interpolation_of_residuals, stations, xr):
    def nudger(param):
        if param == "T_2M":
            pw_param = "temperature"
            offset = 273.15  # K to C
        elif param == "U_10M":
            pw_param = "wind_u"
            offset = 0
        elif param == "V_10M":
            pw_param = "wind_v"
            offset = 0
        elif param == "TOT_PREC":
            pw_param = "precipitation"
            offset = 0
        else:
            raise NotImplementedError(
                f"The mapping for {param=} to PeakWeather is not implemented"
            )

        # grid (xarray Dataset)
        lat2d = ds_full["latitude"].values
        lon2d = ds_full["longitude"].values
        B = ds_full[param].values
    
        # stations (DataFrame)
        st_lat = stations["latitude"].to_numpy()
        st_lon = stations["longitude"].to_numpy()
        st_obs = stations[pw_param].to_numpy() + offset

        A = interpolation_of_residuals(B, lat2d, lon2d, st_lat, st_lon, st_obs, k=3, power=4, max_dist=0.5)
        A_da = xr.DataArray(
            A,
            dims=ds_full[param].dims,
            coords=ds_full[param].coords,
            name=param,
            attrs=ds_full[param].attrs,  # keep units, standard_name, etc.
        )
        ds_full[param] = A_da
        print(f"Nuding of {param=} done!")
    return (nudger,)


@app.cell
def _(ds_full, nudger):
    for param in ["T_2M", "U_10M", "V_10M"]:
        nudger(param)
    ds_full
    return


@app.cell
def _(ds_full, grib_file):
    out_grib = grib_file.parent / (grib_file.stem + "_corrected" + grib_file.suffix)
    ds_full.earthkit.to_target("file", out_grib)
    ds_full.close()
    out_grib.rename(grib_file)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
