import json
import logging
from argparse import ArgumentParser
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import numpy as np

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


def compute_holdout_stations(all_stations: list, cv_cfg: dict) -> list[str]:
    """Return nat_abbr list of holdout stations derived from the truth dataset's station list.

    Mirrors the selection logic in nudging.py so the evaluation partition matches
    what was actually withheld from nudging, using the experiment-level seed/fraction.
    """
    exclude_stations = cv_cfg.get("exclude_stations")
    holdout_fraction = cv_cfg.get("holdout_fraction")
    holdout_seed = cv_cfg.get("holdout_seed", 42)

    if exclude_stations is not None:
        return [s for s in exclude_stations if s in all_stations]
    if holdout_fraction is not None and 0.0 < float(holdout_fraction) < 1.0:
        n_holdout = round(len(all_stations) * float(holdout_fraction))
        rng = np.random.default_rng(holdout_seed)
        return list(rng.choice(all_stations, size=n_holdout, replace=False))
    return []


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
    LOG.info("Thresholds to verify: %s", args.threshold_dict)
    LOG.info("Output file: %s", args.output)
    LOG.info("=" * 80)


def main(args: ScriptConfig):
    """Main function to verify baseline forecast data."""

    # get baseline forecast data
    now = datetime.now()

    fcst = load_forecast_data(
        args.forecast, args.reftime, args.steps, args.params, member=args.member
    )

    LOG.info(
        "Loaded forecast data in %s seconds: \n%s",
        (datetime.now() - now).total_seconds(),
        fcst,
    )

    # get truth data
    now = datetime.now()
    truth = load_truth_data(args.truth, args.reftime, args.steps, args.params)
    LOG.info(
        "Loaded truth data in %s seconds: \n%s",
        (datetime.now() - now).total_seconds(),
        truth,
    )

    # align forecast and truth data spatially and temporally
    now = datetime.now()
    fcst = map_forecast_to_truth(fcst, truth)
    # map_forecast_to_truth uses fancy indexing which collapses the spatial
    # dimension into one monolithic dask chunk; rechunk by step so that
    # verify() can parallelise over time steps rather than materialising the
    # full (regions × steps × values) array at once.
    fcst = fcst.chunk({"step": 1})
    truth = truth.sel(time=fcst["valid_time"])
    LOG.info(
        "Aligned forecast and truth in %s seconds",
        (datetime.now() - now).total_seconds(),
    )

    # determine holdout stations for cross-validation station stratification
    # holdout stations are derived from the truth dataset's station list so the
    # same partition is used consistently across all models and baselines.
    holdout_stations = None
    cv_cfg = args.cross_validation_cfg
    if cv_cfg and "values" in truth.dims:
        all_stations = list(truth["values"].values)
        holdout_stations = compute_holdout_stations(all_stations, cv_cfg)
        if holdout_stations:
            LOG.info(
                "Cross-validation holdout: %d / %d stations withheld",
                len(holdout_stations),
                len(all_stations),
            )
        else:
            LOG.warning("cross_validation_cfg set but no holdout stations selected (check holdout_fraction / exclude_stations).")
    elif cv_cfg:
        LOG.warning("cross_validation_cfg set but truth dataset has no 'values' dimension; station stratification skipped.")

    # compute metrics and statistics
    now = datetime.now()
    results = verify(
        fcst,
        truth,
        args.label,
        args.truth_label,
        args.regions,
        threshold_dict=args.threshold_dict,
        holdout_stations=holdout_stations,
    )
    LOG.info(
        "Computed verification metrics in %s seconds",
        (datetime.now() - now).total_seconds(),
    )

    # save results to NetCDF
    now = datetime.now()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.earthkit.to_netcdf(args.output)
    LOG.info(
        "Saved verification results to %s in %s seconds",
        args.output,
        (datetime.now() - now).total_seconds(),
    )

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
        type=lambda x: [r for r in x.split(",") if r],
        help="Comma-separated list of shapefile paths defining regions for stratification.",
        default="",
    )
    parser.add_argument(
        "--threshold_dict",
        type=lambda x: eval(x),
        help="Dictionary of thresholds for each parameter in the format '{param: [threshold1, threshold2, ...]}' (default: None).",
        default=None,
    )
    parser.add_argument(
        "--cross_validation_cfg",
        type=lambda s: json.loads(s) if s else None,
        default=None,
        help=(
            "Cross-validation config as a JSON dict with keys: holdout_fraction, holdout_seed, "
            "exclude_stations. When set, adds station_group stratification (all/holdout/holdin) "
            "to all models and baselines using the truth dataset's station list."
        ),
    )
    parser.add_argument(
        "--member",
        type=str,
        default="000",
        help="Ensemble member to load: '000' for control, 'median' for the pre-computed median, 'mean' to average all members, or any 3-digit member ID.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="verif.nc",
        help="Output file to save the verification results (default: verif.nc).",
    )
    args = parser.parse_args()

    main(args)
