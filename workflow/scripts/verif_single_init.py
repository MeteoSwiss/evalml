import logging
from argparse import ArgumentParser
from argparse import Namespace
from datetime import datetime
from pathlib import Path


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
    if args.analysis_zarr:
        analysis = (
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
    else:
        raise ValueError("--analysis_zarr must be provided.")
    LOG.info(
        "Loaded analysis data in %s seconds: \n%s",
        (datetime.now() - now).total_seconds(),
        analysis,
    )

    # compute metrics and statistics

    results = verify(fcst, analysis, args.label, args.analysis_label, args.regions)
    LOG.info("Verification results:\n%s", results)

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
        "--analysis_zarr",
        type=Path,
        required=True,
        default="/scratch/mch/fzanetta/data/anemoi/datasets/mch-co2-an-archive-0p02-2015-2020-6h-v3-pl13.zarr",
        help="Path to the zarr dataset containing analysis data.",
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
        "--analysis_label",
        type=str,
        default="COSMO KENDA",
        help="Label for the analysis data (default: COSMO KENDA).",
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
