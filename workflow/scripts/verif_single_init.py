import logging
from argparse import ArgumentParser
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

from verification import verify  # noqa: E402
from data_input import (
    load_baseline_from_zarr,
    load_analysis_data_from_zarr,
    load_fct_data_from_grib,
    load_obs_data_from_peakweather,
    parse_steps,
)  # noqa: E402

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class ScriptConfig(Namespace):
    """Configuration for the script to verify baseline forecast data."""

    archive_root: Path = None
    truth: Path = None
    baseline_zarr: Path = None
    reftime: datetime = None
    params: list[str] = ["T_2M", "TD_2M", "U_10M", "V_10M"]
    steps: list[int] = parse_steps("0/120/6")


def program_summary_log(args):
    """Log a welcome message with the script information."""
    LOG.info("=" * 80)
    LOG.info("Running verification of baseline forecast data")
    LOG.info("=" * 80)
    LOG.info("Baseline dataset: %s", args.baseline_zarr)
    LOG.info("Truth dataset: %s", args.truth)
    LOG.info("Reference time: %s", args.reftime)
    LOG.info("Parameters to verify: %s", args.params)
    LOG.info("Lead time: %s", args.lead_time)
    LOG.info("Output file: %s", args.output)
    LOG.info("=" * 80)


def _map_fcst_to_truth(
    fcst: xr.Dataset, truth: xr.Dataset
) -> tuple[xr.Dataset, xr.Dataset]:
    """Map forecasts to the truth grid or station locations via nearest-neighbor lookup."""

    truth = truth.sel(time=fcst.time)  # swap time dimension to lead_time

    if "y" in fcst.dims and "x" in fcst.dims:
        fcst = fcst.stack(values=("y", "x"))
    fcst_lat = fcst["lat"].values.ravel()
    fcst_lon = fcst["lon"].values.ravel()

    if "y" in truth.dims and "x" in truth.dims:
        truth = truth.stack(values=("y", "x"))
    truth_lat = truth["lat"].values.ravel()
    truth_lon = truth["lon"].values.ravel()

    # TODO: Project to a metric CRS for a proper distance metric
    fcst_lat_rad = np.deg2rad(fcst_lat)
    fcst_lon_rad = np.deg2rad(fcst_lon)
    truth_lat_rad = np.deg2rad(truth_lat)
    truth_lon_rad = np.deg2rad(truth_lon)

    fcst_xyz = np.c_[
        np.cos(fcst_lat_rad) * np.cos(fcst_lon_rad),
        np.cos(fcst_lat_rad) * np.sin(fcst_lon_rad),
        np.sin(fcst_lat_rad),
    ]
    truth_xyz = np.c_[
        np.cos(truth_lat_rad) * np.cos(truth_lon_rad),
        np.cos(truth_lat_rad) * np.sin(truth_lon_rad),
        np.sin(truth_lat_rad),
    ]

    fcst_tree = cKDTree(fcst_xyz)
    _, fi = fcst_tree.query(truth_xyz, k=1)
    fi = np.asarray(fi)
    fcst = fcst.isel(values=fi)
    fcst = fcst.drop_vars(["x", "y", "values"], errors="ignore")
    fcst = fcst.assign_coords(lon=("values", truth.lon.data))
    fcst = fcst.assign_coords(lat=("values", truth.lat.data))
    fcst = fcst.assign_coords(values=truth["values"])

    return fcst, truth


def _load_forecast(args: ScriptConfig) -> xr.Dataset:
    """Load forecast data from GRIB files or a baseline Zarr dataset."""

    if any(args.forecast.glob("*.grib")):
        LOG.info("Loading forecasts from GRIB files...")
        fcst = load_fct_data_from_grib(
            root=args.forecast,
            reftime=args.reftime,
            steps=args.steps,
            params=args.params,
        )
    else:
        LOG.info("Loading baseline forecasts from zarr dataset...")
        fcst = load_baseline_from_zarr(
            root=args.forecast,
            reftime=args.reftime,
            steps=args.steps,
            params=args.params,
        )

    return fcst


def _load_truth(args: ScriptConfig) -> xr.Dataset:
    """Load truth data from analysis Zarr or PeakWeather observations."""
    LOG.info("Loading ground truth from an analysis zarr dataset...")
    if args.truth.suffix == ".zarr":
        truth = load_analysis_data_from_zarr(
            root=args.truth,
            reftime=args.reftime,
            steps=args.steps,
            params=args.params,
        )
        truth = truth.compute().chunk(
            {"y": -1, "x": -1}
            if "y" in truth.dims and "x" in truth.dims
            else {"values": -1}
        )
    elif "peakweather" in str(args.truth):
        LOG.info("Loading ground truth from PeakWeather observations...")
        # TODO: replace with OGD data
        truth = load_obs_data_from_peakweather(
            root=args.truth,
            reftime=args.reftime,
            steps=args.steps,
            params=args.params,
        )
    else:
        raise ValueError(f"Unsupported truth root: {args.truth}")
    return truth


def main(args: ScriptConfig):
    """Main function to verify baseline forecast data."""

    # get baseline forecast data
    now = datetime.now()

    fcst = _load_forecast(args)

    LOG.info(
        "Loaded forecast data in %s seconds: \n%s",
        (datetime.now() - now).total_seconds(),
        fcst,
    )

    # get truth data
    now = datetime.now()
    truth = _load_truth(args)
    LOG.info(
        "Loaded truth data in %s seconds: \n%s",
        (datetime.now() - now).total_seconds(),
        truth,
    )

    fcst, truth = _map_fcst_to_truth(fcst, truth)

    # compute metrics and statistics
    results = verify(fcst, truth, args.label, args.truth_label, args.regions)

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
        "--truth",
        type=Path,
        required=True,
        help="Path to the truth data.",
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
        default=["T_2M", "TD_2M", "U_10M", "V_10M", "PS", "PMSL", "TOT_PREC"],
    )
    parser.add_argument(
        "--steps",
        type=parse_steps,
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
        "--truth_label",
        type=str,
        default="COSMO KENDA",
        help="Label for the truth data (default: COSMO KENDA).",
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
