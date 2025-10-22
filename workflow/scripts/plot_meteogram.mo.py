import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import logging
    from argparse import ArgumentParser
    from pathlib import Path

    import earthkit.plots as ekp
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import xarray as xr

    from meteodatalab import data_source, grib_decoder
    return (
        ArgumentParser,
        Path,
        data_source,
        grib_decoder,
        logging,
        np,
        pd,
        plt,
        xr,
    )


@app.cell
def _(logging):
    LOG = logging.getLogger(__name__)
    LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)
    return


@app.cell
def _(ArgumentParser, Path):
    ROOT = Path(__file__).parent

    parser = ArgumentParser()

    parser.add_argument(
        "--forecast", type=str, default=None, help="Directory to forecast grib data"
    )
    parser.add_argument(
        "--analysis", type=str, default=None, help="Path to analysis zarr data"
    )
    parser.add_argument("--date", type=str, default=None, help="reference datetime")
    parser.add_argument("--outfn", type=str, help="output filename")
    parser.add_argument("--param", type=str, help="parameter")
    parser.add_argument("--station", type=str, help="station")

    args = parser.parse_args()
    grib_dir = Path(args.forecast)
    zarr_dir = Path(args.analysis)
    init_time = args.date
    outfn = Path(args.outfn)
    station = args.station
    param = args.param
    return grib_dir, init_time, outfn, param, station, zarr_dir


@app.cell
def _(pd):
    stations = pd.DataFrame(
        [
            ("BAS", "1_75", 1, 75, "BAS", "Basel / Binningen", 1, "MeteoSchweiz", 7.583, 47.541, 316.14, 12.0, 0.0),
            ("LUG", "1_47", 1, 47, "LUG", "Lugano", 1, "MeteoSchweiz", 8.960, 46.004, 272.56, 10.0, 27.34),
            ("GVE", "1_58", 1, 58, "GVE", "Gen\u00e8ve / Cointrin", 1, "MeteoSchweiz", 6.122, 46.248, 415.53, 10.0, 0.0),
            ("GUT", "1_79", 1, 79, "GUT", "G\u00fcttingen", 1, "MeteoSchweiz", 9.279, 47.602, 439.78, 12.0, 0.0),
            ("KLO", "1_59", 1, 59, "KLO", "Z\u00fcrich / Kloten", 1, "MeteoSchweiz", 8.536, 47.48, 435.92, 10.5, 0.0),
            ("SCU", "1_30", 1, 30, "SCU", "Scuol", 1, "MeteoSchweiz", 10.283, 46.793, 1304.42, 10.0, 0.0),
            ("LUZ", "1_68", 1, 68, "LUZ", "Luzern", 1, "MeteoSchweiz", 8.301, 47.036, 454.0, 8.41, 32.51),
            ("DIS", "1_54", 1, 54, "DIS", "Disentis", 1, "MeteoSchweiz", 8.853, 46.707, 1198.03, 10.0, 0.0),
            ("PMA", "1_862", 1, 862, "PMA", "Piz Martegnas", 1, "MeteoSchweiz", 9.529, 46.577, 2668.34, 10.0, 0.0),
            ("CEV", "1_843", 1, 843, "CEV", "Cevio", 1, "MeteoSchweiz", 8.603, 46.32, 420.0, 10.0, 6.85),
            ("MLS", "1_38", 1, 38, "MLS", "Le Mol\u00e9son", 1, "MeteoSchweiz", 7.018, 46.546, 1977.0, 10.0, 13.31),
            ("PAY", "1_32", 1, 32, "PAY", "Payerne", 1, "MeteoSchweiz", 6.942, 46.811, 489.17, 10.0, 0.0),
            ("NAP", "1_48", 1, 48, "NAP", "Napf", 1, "MeteoSchweiz", 7.94, 47.005, 1404.03, 15.0, 0.0),

        ],
        columns=[
            "station",
            "name",
            "type_id",
            "point_id",
            "nat_abbr",
            "fullname",
            "owner_id",
            "owner_name",
            "longitude",
            "latitude",
            "height_masl",
            "pole_height",
            "roof_height",
        ],
    )
    return (stations,)


@app.cell
def load_grib_data(data_source, grib_decoder, grib_dir, init_time, param):
    if param == "10sp":
        paramlist = ["10u", "10v"]
    elif param == "sp":
        paramlist = ["u", "v"]
    else:
        paramlist = [param]

    grib_files = sorted(grib_dir.glob(f"{init_time}*.grib"))
    fds = data_source.FileDataSource(datafiles=grib_files)
    ds_grib = grib_decoder.load(fds, {"param": paramlist})
    ds_grib = ds_grib[param].squeeze()
    ds_grib
    return ds_grib, paramlist


@app.cell
def load_zarr_data(ds_grib, np, param, paramlist, xr, zarr_dir):
    ds_zarr = xr.open_zarr(zarr_dir, consolidated=False)
    ds_zarr = ds_zarr.set_index(time="dates")
    ds_zarr = ds_zarr.sel(time=ds_grib.valid_time.values)
    ds_zarr = ds_zarr.assign_coords({"variable": ds_zarr.attrs["variables"]})
    ds_zarr = ds_zarr.sel(variable=[p for p in paramlist]).squeeze("ensemble", drop=True)

    # recover original 2D shape
    ny, nx = ds_zarr.attrs["field_shape"]
    y_idx, x_idx = np.unravel_index(np.arange(ny * nx), shape=(ny, nx))
    ds_zarr = ds_zarr.assign_coords({"y": ("cell", y_idx), "x": ("cell", x_idx)})
    ds_zarr = ds_zarr.set_index(cell=("y", "x"))
    ds_zarr = ds_zarr.unstack("cell")

    # set lat lon as coords
    ds_zarr = ds_zarr.rename({"latitudes": "lat", "longitudes": "lon"})
    ds_zarr = ds_zarr.set_coords(["lat", "lon"])
    ds_zarr = ds_zarr.sel(variable=param).squeeze()["data"]
    ds_zarr
    return (ds_zarr,)


@app.cell
def _(ds_grib, ds_zarr, np, pd, stations):
    def nearest_yx_euclid(ds_grib, lat_s, lon_s):
        """
        Return (y_idx, x_idx) of the grid point nearest to (lat_s, lon_s)
        using Euclidean distance in degrees.
        """
        lat2d = ds_grib['lat']  # (y, x)
        lon2d = ds_grib['lon']  # (y, x)

        dist2 = (lat2d - lat_s)**2 + (lon2d - lon_s)**2
        flat_idx = np.nanargmin(dist2.values)
        y_idx, x_idx = np.unravel_index(flat_idx, dist2.shape)
        return int(y_idx), int(x_idx)

    def get_grib_idx_row(row):
        return nearest_yx_euclid(ds_grib, row['latitude'], row['longitude'])
    idxs_grib = stations.apply(get_grib_idx_row, axis=1, result_type='expand')
    idxs_grib.columns = ['grib_y_idx', 'grib_x_idx']

    def get_zarr_idx_row(row):
        return nearest_yx_euclid(ds_zarr, row['latitude'], row['longitude'])
    idxs_zarr = stations.apply(get_zarr_idx_row, axis=1, result_type='expand')
    idxs_zarr.columns = ['zarr_y_idx', 'zarr_x_idx']

    sta_idxs = pd.concat([stations, idxs_grib, idxs_zarr], axis=1).set_index("station")
    sta_idxs
    return (sta_idxs,)


@app.cell
def _(ds_grib, ds_zarr, outfn, plt, sta_idxs, station):
    grib_x_idx, grib_y_idx = sta_idxs.loc[station].grib_x_idx, sta_idxs.loc[station].grib_y_idx
    zarr_x_idx, zarr_y_idx = sta_idxs.loc[station].zarr_x_idx, sta_idxs.loc[station].zarr_y_idx

    # analysis
    ds_zarr.isel(x=zarr_x_idx, y=zarr_y_idx).plot(x="time", label="analysis", color="k", ls="--")

    # TODO: plot actual forecaster data...
    ds_grib.isel(x=grib_x_idx, y=grib_y_idx).isel(lead_time=list(range(0, 126, 6))).plot(x="valid_time", marker="o", linestyle='None', color="k", label="forecaster")

    # interpolator
    ds_grib.isel(x=grib_x_idx, y=grib_y_idx).plot(x="valid_time", label="interpolator")

    plt.legend()
    plt.ylabel(f"{ds_grib.attrs["parameter"]["shortName"]} ({ds_grib.attrs["parameter"]["units"]})")
    plt.title(f"{init_time} {ds_grib.attrs["parameter"]["name"]} at {station}")
    plt.savefig(outfn)
    return


if __name__ == "__main__":
    app.run()
