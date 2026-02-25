import logging
from argparse import ArgumentParser
from argparse import Namespace
from datetime import datetime
from pathlib import Path


from verification import verify  # noqa: E402
from verification.spatial import map_forecast_to_truth  # noqa: E402
from data_input import (
    parse_steps,
    load_forecast_data,
    load_truth_data,
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


def main(args: ScriptConfig):
    """Main function to verify baseline forecast data."""

    # get baseline forecast data
    now = datetime.now()

    fcst = load_forecast_data(args.root, args.reftime, args.steps, args.params)

    LOG.info(
        "Loaded forecast data in %s seconds: \n%s",
        (datetime.now() - now).total_seconds(),
        fcst,
    )

    # get truth data
    now = datetime.now()
    truth = load_truth_data(args.root, args.reftime, args.steps, args.params)
    LOG.info(
        "Loaded truth data in %s seconds: \n%s",
        (datetime.now() - now).total_seconds(),
        truth,
    )

    # align forecast and truth data spatially and temporally
    fcst = map_forecast_to_truth(fcst, truth)
    truth = truth.sel(time=fcst.time)

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
