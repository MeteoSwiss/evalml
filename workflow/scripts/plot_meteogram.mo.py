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

    from data_input import load_analysis_data_from_zarr, load_baseline_from_zarr

    return (
        ArgumentParser,
        Path,
        data_source,
        grib_decoder,
        load_analysis_data_from_zarr,
        load_baseline_from_zarr,
        np,
        pd,
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
    parser.add_argument("--date", type=str, default=None, help="reference datetime")
    parser.add_argument("--outfn", type=str, help="output filename")
    parser.add_argument("--param", type=str, help="parameter")
    parser.add_argument("--station", type=str, help="station")

    args = parser.parse_args()
    grib_dir = Path(args.forecast)
    zarr_dir_ana = Path(args.analysis)
    zarr_dir_base = Path(args.baseline)
    init_time = args.date
    outfn = Path(args.outfn)
    station = args.station
    param = args.param
    return (
        grib_dir,
        init_time,
        outfn,
        param,
        station,
        zarr_dir_ana,
        zarr_dir_base,
    )


@app.cell
def _(pd):
    stations = pd.DataFrame(
        [
            (
                "BAS",
                "1_75",
                1,
                75,
                "BAS",
                "Basel / Binningen",
                1,
                "MeteoSchweiz",
                7.583,
                47.541,
                316.14,
                12.0,
                0.0,
            ),
            (
                "LUG",
                "1_47",
                1,
                47,
                "LUG",
                "Lugano",
                1,
                "MeteoSchweiz",
                8.960,
                46.004,
                272.56,
                10.0,
                27.34,
            ),
            (
                "GVE",
                "1_58",
                1,
                58,
                "GVE",
                "Gen\u00e8ve / Cointrin",
                1,
                "MeteoSchweiz",
                6.122,
                46.248,
                415.53,
                10.0,
                0.0,
            ),
            (
                "GUT",
                "1_79",
                1,
                79,
                "GUT",
                "G\u00fcttingen",
                1,
                "MeteoSchweiz",
                9.279,
                47.602,
                439.78,
                12.0,
                0.0,
            ),
            (
                "KLO",
                "1_59",
                1,
                59,
                "KLO",
                "Z\u00fcrich / Kloten",
                1,
                "MeteoSchweiz",
                8.536,
                47.48,
                435.92,
                10.5,
                0.0,
            ),
            (
                "SCU",
                "1_30",
                1,
                30,
                "SCU",
                "Scuol",
                1,
                "MeteoSchweiz",
                10.283,
                46.793,
                1304.42,
                10.0,
                0.0,
            ),
            (
                "LUZ",
                "1_68",
                1,
                68,
                "LUZ",
                "Luzern",
                1,
                "MeteoSchweiz",
                8.301,
                47.036,
                454.0,
                8.41,
                32.51,
            ),
            (
                "DIS",
                "1_54",
                1,
                54,
                "DIS",
                "Disentis",
                1,
                "MeteoSchweiz",
                8.853,
                46.707,
                1198.03,
                10.0,
                0.0,
            ),
            (
                "PMA",
                "1_862",
                1,
                862,
                "PMA",
                "Piz Martegnas",
                1,
                "MeteoSchweiz",
                9.529,
                46.577,
                2668.34,
                10.0,
                0.0,
            ),
            (
                "CEV",
                "1_843",
                1,
                843,
                "CEV",
                "Cevio",
                1,
                "MeteoSchweiz",
                8.603,
                46.32,
                420.0,
                10.0,
                6.85,
            ),
            (
                "MLS",
                "1_38",
                1,
                38,
                "MLS",
                "Le Mol\u00e9son",
                1,
                "MeteoSchweiz",
                7.018,
                46.546,
                1977.0,
                10.0,
                13.31,
            ),
            (
                "PAY",
                "1_32",
                1,
                32,
                "PAY",
                "Payerne",
                1,
                "MeteoSchweiz",
                6.942,
                46.811,
                489.17,
                10.0,
                0.0,
            ),
            (
                "NAP",
                "1_48",
                1,
                48,
                "NAP",
                "Napf",
                1,
                "MeteoSchweiz",
                7.94,
                47.005,
                1404.03,
                15.0,
                0.0,
            ),
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
def _(da_ana, da_base, da_fct, np, pd, stations):
    def nearest_yx_euclid(ds, lat_s, lon_s):
        """
        Return (y_idx, x_idx) of the grid point nearest to (lat_s, lon_s)
        using Euclidean distance in degrees.
        """
        try:
            lat2d = ds["lat"]  # (y, x)
            lon2d = ds["lon"]  # (y, x)
        except KeyError:
            lat2d = ds["latitude"]  # (y, x)
            lon2d = ds["longitude"]  # (y, x)

        dist2 = (lat2d - lat_s) ** 2 + (lon2d - lon_s) ** 2
        flat_idx = np.nanargmin(dist2.values)
        y_idx, x_idx = np.unravel_index(flat_idx, dist2.shape)
        return int(y_idx), int(x_idx)

    def get_fct_idx_row(row):
        return nearest_yx_euclid(da_fct, row["latitude"], row["longitude"])

    idxs_fct = stations.apply(get_fct_idx_row, axis=1, result_type="expand")
    idxs_fct.columns = ["fct_y_idx", "fct_x_idx"]

    def get_ana_idx_row(row):
        return nearest_yx_euclid(da_ana, row["latitude"], row["longitude"])

    idxs_ana = stations.apply(get_ana_idx_row, axis=1, result_type="expand")
    idxs_ana.columns = ["ana_y_idx", "ana_x_idx"]

    def get_base_idx_row(row):
        return nearest_yx_euclid(da_base, row["latitude"], row["longitude"])

    idxs_base = stations.apply(get_base_idx_row, axis=1, result_type="expand")
    idxs_base.columns = ["base_y_idx", "base_x_idx"]

    sta_idxs = pd.concat([stations, idxs_fct, idxs_ana, idxs_base], axis=1).set_index(
        "station"
    )
    sta_idxs
    return (sta_idxs,)


@app.cell
def _(da_ana, da_base, da_fct, init_time, outfn, plt, sta_idxs, station):
    fct_x_idx, fct_y_idx = (
        sta_idxs.loc[station].fct_x_idx,
        sta_idxs.loc[station].fct_y_idx,
    )
    ana_x_idx, ana_y_idx = (
        sta_idxs.loc[station].ana_x_idx,
        sta_idxs.loc[station].ana_y_idx,
    )
    base_x_idx, base_y_idx = (
        sta_idxs.loc[station].base_x_idx,
        sta_idxs.loc[station].base_y_idx,
    )

    # analysis
    da_ana.isel(x=ana_x_idx, y=ana_y_idx).plot(
        x="time", label="analysis", color="k", ls="--"
    )

    # baseline
    da_base.isel(x=base_x_idx, y=base_y_idx).plot(
        x="time", label="baseline", color="C1"
    )

    # forecast
    da_fct.isel(x=fct_x_idx, y=fct_y_idx).plot(
        x="valid_time", label="forecast", color="C0"
    )

    plt.legend()
    plt.ylabel(
        f"{da_fct.attrs['parameter']['shortName']} ({da_fct.attrs['parameter']['units']})"
    )
    plt.title(f"{init_time} {da_fct.attrs['parameter']['name']} at {station}")
    plt.savefig(outfn)
    return


if __name__ == "__main__":
    app.run()
