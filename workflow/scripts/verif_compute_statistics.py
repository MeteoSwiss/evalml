from pathlib import Path
import logging
import time
import os

import earthkit.data as ekd
import xarray as xr

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def _parse_lead_time(lead_time: str) -> int:
    # check that lead_time is in the format "start/stop/step"
    if "/" not in lead_time:
        raise ValueError(
            f"Expected lead_time in format 'start/stop/step', got '{lead_time}'"
        )
    if len(lead_time.split("/")) != 3:
        raise ValueError(
            f"Expected lead_time in format 'start/stop/step', got '{lead_time}'"
        )

    return list(range(*map(int, lead_time.split("/"))))


def program_summary_log(args):
    """Log a welcome message with the script and template information."""
    LOG.info("=" * 80)
    LOG.info("Computing statistics from forecast data in GRIB format")
    LOG.info("=" * 80)
    LOG.info("GRIB output directory: %s", args.grib_output_dir)
    LOG.info("Output file: %s", args.output)
    LOG.info("=" * 80)


def main(args):
    if os.environ.get("ECCODES_DEFINITION_PATH") is None:
        raise EnvironmentError(
            "Environment variable ECCODES_DEFINITION_PATH is not set. "
            "Please set it to the path of your ecCodes definitions directory."
        )
    program_summary_log(args)

    # read data from GRIB files using earthkit.data
    fds = ekd.from_source("file", args.grib_output_dir / "*.grib")
    LOG.info("Fieldlist loaded from GRIB files: \n%s", fds.ls())
    now = time.time()
    out = xr.Dataset()
    for _ds, split in zip(
        *fds.to_xarray(split_dims="levelist", ensure_dims="forecast_reference_time")
    ):
        rename_dict = {"forecast_reference_time": "ref_time", "step": "lead_time"}
        if _ds.attrs["levtype"] == "pl":
            rename_dict |= {var: f"{var}_{split['levelist']}" for var in _ds.data_vars}
        _ds = _ds.rename(rename_dict)
        out = xr.merge([out, _ds])
    LOG.info(
        "Loaded data from GRIB files to xarray objects in %.2f seconds",
        time.time() - now,
    )

    # compute statistics for each parameter
    now = time.time()
    out = out.to_dataarray("param").chunk("auto")
    out = xr.Dataset(
        {
            "mean": out.mean(dim=["y", "x"]),
            "std": out.std(dim=["y", "x"]),
            "min": out.min(dim=["y", "x"]),
            "max": out.max(dim=["y", "x"]),
            "quantiles": out.quantile([0.1, 0.25, 0.5, 0.75, 0.9], dim=["y", "x"]),
        }
    )
    out = out.compute(num_workers=4, scheduler="threads")
    LOG.info("Computed statistics in %.2f seconds", time.time() - now)
    LOG.info("Statistics dataset: \n%s", out)

    # save the output to a netCDF file
    out.to_netcdf(args.output, mode="w", format="NETCDF4")
    LOG.info("Saved statistics to %s", args.output)

    LOG.info("Program completed successfully.")


if __name__ == "__main__":
    import argparse

    PARAMS = ["T_2M", "TD_2M", "U_10M", "V_10M", "PS", "PMSL", "TOT_PREC", "T_G"]
    PARAMS += ["T", "U", "V", "QV", "FI"]

    parser = argparse.ArgumentParser(
        description="Compute statistics from forecast data in GRIB format."
    )
    parser.add_argument(
        "--grib_output_dir",
        type=Path,
        help="Directory containing GRIB files.",
        required=True,
    )

    known_args = parser.parse_known_args()[0]
    parser.add_argument(
        "--output",
        type=Path,
        default=known_args.grib_output_dir.parent / "statistics.nc",
        help="Output file to save computed statistics.",
    )
    args = parser.parse_args()

    if not args.grib_output_dir.exists():
        raise FileNotFoundError(
            f"GRIB output directory {args.grib_output_dir} does not exist."
        )

    main(args)


"""
Example usage:

python workflow/scripts/verif_compute_statistics.py \
    --grib_output_dir /users/fzanetta/projects/evalml/output/data/runs/2f962c89ff644ca7940072fa9cd088ec/202001031800/grib
"""
