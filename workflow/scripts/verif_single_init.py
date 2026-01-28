import logging
from argparse import ArgumentParser
from argparse import Namespace
from datetime import datetime, timedelta
from pathlib import Path


import numpy as np
import pandas as pd
from peakweather.dataset import PeakWeatherDataset
from scipy.spatial import cKDTree


from verification import verify  # noqa: E402
from data_input import (
    load_baseline_from_zarr,
    load_analysis_data_from_zarr,
    load_fct_data_from_grib,
)  # noqa: E402

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def _parse_steps(steps: str) -> int:
    # check that steps is in the format "start/stop/step"
    if "/" not in steps:
        raise ValueError(f"Expected steps in format 'start/stop/step', got '{steps}'")
    if len(steps.split("/")) != 3:
        raise ValueError(f"Expected steps in format 'start/stop/step', got '{steps}'")
    start, end, step = map(int, steps.split("/"))
    return list(range(start, end + 1, step))


class ScriptConfig(Namespace):
    """Configuration for the script to verify baseline forecast data."""

    archive_root: Path = None
    analysis_zarr: Path = None
    baseline_zarr: Path = None
    reftime: datetime = None
    params: list[str] = ["T_2M", "TD_2M", "U_10M", "V_10M"]
    steps: list[int] = _parse_steps("0/120/6")


def program_summary_log(args):
    """Log a welcome message with the script information."""
    LOG.info("=" * 80)
    LOG.info("Running verification of baseline forecast data")
    LOG.info("=" * 80)
    LOG.info("baseline zarr dataset: %s", args.baseline_zarr)
    LOG.info("Zarr dataset for analysis: %s", args.analysis_zarr)
    LOG.info("Reference time: %s", args.reftime)
    LOG.info("Parameters to verify: %s", args.params)
    LOG.info("Lead time: %s", args.lead_time)
    LOG.info("Output file: %s", args.output)
    LOG.info("=" * 80)


def main(args: ScriptConfig):
    """Main function to verify baseline forecast data."""

    # get baseline forecast data

    now = datetime.now()

    # try to open the baselin as a zarr, and if it fails load from grib
    if not args.forecast:
        raise ValueError("--forecast must be provided.")

    if any(args.forecast.glob("*.grib")):
        LOG.info("Loading forecasts from GRIB files...")
        fcst = load_fct_data_from_grib(
            grib_output_dir=args.forecast,
            reftime=args.reftime,
            steps=args.steps,
            params=args.params,
        )
        fcst = fcst.rename({"lat": "latitude", "lon": "longitude"})
    else:
        LOG.info("Loading baseline forecasts from zarr dataset...")
        fcst = load_baseline_from_zarr(
            zarr_path=args.forecast,
            reftime=args.reftime,
            steps=args.steps,
            params=args.params,
        )

    LOG.info(
        "Loaded forecast data in %s seconds: \n%s",
        (datetime.now() - now).total_seconds(),
        fcst,
    )

    # get truth data (aka analysis data)
    now = datetime.now()
    if "peakweather" in str(args.groundtruth_zarr).lower():
        years = list(map(int,list(set(fcst.time.dt.year.values))))
        groundtruth = PeakWeatherDataset(root=args.groundtruth_zarr, years=years, freq="1h")
    
    else:
        groundtruth = (
            load_analysis_data_from_zarr(
                analysis_zarr=args.analysis_zarr,
                times=fcst.time,
                params=args.params,
            )
            .compute()
            .chunk(
                {"y": -1, "x": -1}
                if "y" in fcst.dims and "x" in fcst.dims
                else {"values": -1}
            )
        )

    LOG.info(
        "Loaded ground truth data in %s seconds: \n%s",
        (datetime.now() - now).total_seconds(),
        args.groundtruth_zarr,
    )

    if isinstance(groundtruth, PeakWeatherDataset):
        fcst = fcst.stack(station=("y", "x"))

        obs, mask = groundtruth.get_observations(
            parameters=["temperature", "wind_u", "wind_v"],
            first_date=f"{pd.to_datetime(fcst.time.values.min()):%Y-%m-%d %H:%M}",
            last_date=f"{pd.to_datetime(fcst.time.values.max()) + timedelta(hours=1):%Y-%m-%d %H:%M}",
            return_mask=True,
        )
        obs = obs.loc[:, mask.iloc[0]]
        obs = obs.stack(["nat_abbr","name"]).to_xarray().to_dataset(dim="name")
        obs = obs.rename({"datetime": "time", "nat_abbr":"station"})
        obs = obs.rename({"temperature": "T_2M", "wind_u": "U_10M", "wind_v": "V_10M"})
        obs = obs.assign_coords(time=obs.indexes["time"].tz_convert("UTC").tz_localize(None))
        obs = obs.sel(time=fcst["time"])

        obs["T_2M"] = obs["T_2M"] + 273.15  # convert to Kelvin

        gridlat = fcst["latitude"].values
        gridlon = fcst["longitude"].values
        gridlat0 = np.deg2rad(np.nanmean(gridlat))
        grid_xy = np.c_[gridlon.ravel() * np.cos(gridlat0), gridlat.ravel()]
        st_lat = groundtruth.stations_table["latitude"].to_numpy()
        st_lon = groundtruth.stations_table["longitude"].to_numpy()
        st_xy = np.c_[st_lon * np.cos(gridlat0), st_lat]
        grid_tree = cKDTree(grid_xy)
        _, gi = grid_tree.query(st_xy, k=1)
        fcst = fcst.isel(station=gi)
        fcst = fcst.drop_vars(['station', 'y', 'x', 'latitude', 'longitude'])
        fcst = fcst.assign_coords(station=groundtruth.stations_table.index.tolist())
        fcst = fcst.assign_coords(longitude=("station", groundtruth.stations_table["longitude"]))
        fcst = fcst.assign_coords(latitude=("station", groundtruth.stations_table["latitude"]))

        obs = obs.assign_coords(longitude=("station", fcst.longitude.sel(station=obs.station).data))
        obs = obs.assign_coords(latitude=("station", fcst.latitude.sel(station=obs.station).data))
        
        dim = ["station"]
    else:
        obs = groundtruth
        dim = ["x", "y"] if "x" in fcst.dims and "y" in fcst.dims else ["cell"]

    # compute metrics and statistics
    results = verify(fcst, obs, args.label, args.groundtruth_label, args.regions, dim=dim)
    print(results)

    # save results to NetCDF
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_netcdf(args.output)
    LOG.info("Saved verification results to %s", args.output)

    LOG.info("Program completed successfully.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Verify forecast or baseline data.")

    parser.add_argument(
        "--forecast",
        type=Path,
        required=True,
        default="/store_new/mch/msopr/ml/COSMO-E/FCST20.zarr",
        help="Path to the directory containing the grib forecast or to the zarr dataset containing baseline data.",
    )
    parser.add_argument(
        "--groundtruth_zarr",
        type=Path,
        required=True,
        default="/scratch/mch/fzanetta/data/anemoi/datasets/mch-co2-an-archive-0p02-2015-2020-6h-v3-pl13.zarr",
        help="Path to the ground truth data.",
    )
    parser.add_argument(
        "--reftime",
        type=lambda s: datetime.strptime(s, "%Y%m%d%H%M"),
        default="202010010000",
        help="Valid time for the data in ISO format.",
    )
    parser.add_argument(
        "--params",
        type=lambda x: x.split(","),
        default=["T_2M", "U_10M", "V_10M"],
    )
    parser.add_argument(
        "--steps",
        type=_parse_steps,
        default="0/120/6",
        help="Forecast steps in the format 'start/stop/step' (default: 0/120/6).",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="COSMO-E",
        help="Label for the forecast or baseline data (default: COSMO-E).",
    )
    parser.add_argument(
        "--groundtruth_label",
        type=str,
        default="COSMO KENDA",
        help="Label for the ground truth data (default: COSMO KENDA).",
    )
    parser.add_argument(
        "--regions",
        type=lambda x: x.split(","),
        help="Comma-separated list of shapefile paths defining regions for stratification.",
        default="",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="verif.nc",
        help="Output file to save the verification results (default: verif.nc).",
    )
    args = parser.parse_args()

    main(args)
