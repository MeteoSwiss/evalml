import logging
import os
import sys
from argparse import ArgumentParser
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Iterable

eccodes_definition_path = Path(sys.prefix) / "share/eccodes-cosmo-resources/definitions"
os.environ["ECCODES_DEFINITION_PATH"] = str(eccodes_definition_path)

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from verification import verify  # noqa: E402

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def load_analysis_data_from_zarr(
    analysis_zarr: Path, times: Iterable[datetime], params: list[str]
) -> xr.Dataset:
    """Load analysis data from an anemoi-generated Zarr dataset

    This function loads analysis data from a Zarr dataset, processing it to make it more
    xarray-friendly. It renames variables, sets the time index, and pivots the dataset.
    """
    PARAMS_MAP_COSMO2 = {
        "T_2M": "2t",
        "TD_2M": "2d",
        "U_10M": "10u",
        "V_10M": "10v",
        "PS": "sp",
        "PMSL": "msl",
        "TOT_PREC": "tp",
    }
    PARAMS_MAP_COSMO1 = {
        v: v.replace("TOT_PREC", "TOT_PREC_6H") for v in PARAMS_MAP_COSMO2.keys()
    }
    PARAMS_MAP = PARAMS_MAP_COSMO2 if "co2" in analysis_zarr.name else PARAMS_MAP_COSMO1

    ds = xr.open_zarr(analysis_zarr, consolidated=False)

    # rename "dates" to "time" and set it as index
    ds = ds.set_index(time="dates")

    # set 'variables' attr as dimension coordinate
    ds = ds.assign_coords({"variable": ds.attrs["variables"]})

    # select variables and valid time, squeeze ensemble dimension
    ds = ds.sel(variable=[PARAMS_MAP[p] for p in params]).squeeze("ensemble", drop=True)

    # recover original 2D shape
    if len(ds.attrs["field_shape"]) == 2:
        ny, nx = ds.attrs["field_shape"]
        y_idx, x_idx = np.unravel_index(np.arange(ny * nx), shape=(ny, nx))
        ds = ds.assign_coords({"y": ("cell", y_idx), "x": ("cell", x_idx)})
        ds = ds.set_index(cell=("y", "x"))
        ds = ds.unstack("cell")

    # set lat lon as coords (optional)
    if "latitudes" in ds and "longitudes" in ds:
        ds = ds.rename({"latitudes": "latitude", "longitudes": "longitude"})
    ds = ds.set_coords(["latitude", "longitude"])
    ds = (
        ds["data"]
        .to_dataset("variable")
        .rename({v: k for k, v in PARAMS_MAP.items() if v in ds["variable"].values})
    )

    # select valid times
    # (handle special case where some valid times are not in the dataset, e.g. at the end)
    times_included = times.isin(ds.time.values).values
    if all(times_included):
        ds = ds.sel(time=times)
    elif np.sum(times_included) < len(times_included):
        LOG.warning(
            "Some valid times are not included in the dataset: \n%s",
            times[~times_included].values,
        )
        ds = ds.sel(time=times[times_included])
    else:
        raise ValueError(
            "Valid times are not included in the dataset. "
            "Please check the valid times and the dataset."
        )

    return ds


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
    baseline = xr.open_zarr(
        args.baseline_zarr, consolidated=True, decode_timedelta=True
    )
    baseline = baseline.rename(
        {"forecast_reference_time": "ref_time", "step": "lead_time"}
    ).sortby("lead_time")
    if "TOT_PREC" in baseline.data_vars:
        if baseline.TOT_PREC.units == "kg m-2":
            baseline = baseline.assign(TOT_PREC=lambda x: x.TOT_PREC / 1000)
            baseline.TOT_PREC.attrs["units"] = "m"
        ## disaggregate precipitation
        baseline = baseline.assign(
            TOT_PREC=lambda x: (
                x.TOT_PREC.fillna(0)
                .diff("lead_time")
                .pad(lead_time=(1, 0), constant_value=None)
                .clip(min=0.0)
            )
        )
    baseline = baseline[args.params].sel(
        ref_time=args.reftime,
        lead_time=np.array(args.steps, dtype="timedelta64[h]"),
        method="nearest",
    )
    baseline = baseline.assign_coords(time=baseline.ref_time + baseline.lead_time)
    LOG.info(
        "Loaded baseline forecast data in %s seconds: \n%s",
        (datetime.now() - now).total_seconds(),
        baseline,
    )

    # get truth data (aka analysis data)
    now = datetime.now()
    if args.analysis_zarr:
        analysis = (
            load_analysis_data_from_zarr(
                analysis_zarr=args.analysis_zarr,
                times=baseline.time,
                params=args.params,
            )
            .compute()
            .chunk({"y": -1, "x": -1})
        )
    else:
        raise ValueError("--analysis_zarr must be provided.")
    LOG.info(
        "Loaded analysis data in %s seconds: \n%s",
        (datetime.now() - now).total_seconds(),
        analysis,
    )

    # compute metrics and statistics

    results = verify(baseline, analysis, args.baseline_label, args.analysis_label)

    # save results to NetCDF
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_netcdf(args.output)
    LOG.info("Saved verification results to %s", args.output)

    LOG.info("Program completed successfully.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Verify baseline forecast data.")

    parser.add_argument(
        "--baseline_zarr",
        type=Path,
        required=True,
        help="Path to the zarr dataset containing baseline forecast data.",
    )
    parser.add_argument(
        "--analysis_zarr",
        type=Path,
        required=True,
        help="Path to the zarr dataset containing analysis data.",
    )
    parser.add_argument(
        "--reftime",
        type=lambda s: datetime.strptime(s, "%Y%m%d%H%M"),
        help="Valid time for the data in ISO format (default: 6 hours ago).",
    )
    parser.add_argument(
        "--params",
        type=lambda x: x.split(","),
        default=["T_2M", "TD_2M", "U_10M", "V_10M", "PS", "PMSL", "TOT_PREC"],
    )
    parser.add_argument(
        "--steps",
        type=_parse_steps,
        default="0/120/6",
        help="Forecast steps in the format 'start/stop/step' (default: 0/120/6).",
    )
    parser.add_argument(
        "--baseline_label",
        type=str,
        default="COSMO-E",
        help="Label for the baseline forecast data (default: COSMO-E).",
    )
    parser.add_argument(
        "--analysis_label",
        type=str,
        default="COSMO KENDA",
        help="Label for the analysis data (default: COSMO KENDA).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="verif.nc",
        help="Output file to save the verification results (default: verif.nc).",
    )
    args = parser.parse_args()

    main(args)
