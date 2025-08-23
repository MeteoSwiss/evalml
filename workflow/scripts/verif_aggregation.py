from pathlib import Path
import itertools
from argparse import ArgumentParser, Namespace
import logging

import xarray as xr
from xarray.groupers import UniqueGrouper
import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def get_season(time):
    month = time.month
    if month in [12, 1, 2]:
        return "DJF"
    elif month in [3, 4, 5]:
        return "MAM"
    elif month in [6, 7, 8]:
        return "JJA"
    else:
        return "SON"


def aggregate_results(ds: xr.Dataset) -> xr.Dataset:
    """Compute mean metric values aggregated by all combinations of hour, season, init_hour."""

    # ds = ds.assign_coords(
    #     hour = lambda ds: ds.time.dt.hour,
    #     init_hour = lambda ds: ds.ref_time.dt.hour,
    #     season = lambda ds: ds.time.dt.season
    # )
    # ds_grouped = ds.groupby(
    #     hour = UniqueGrouper(labels = np.unique(ds.hour)),
    #     init_hour = UniqueGrouper(labels = np.unique(ds.init_hour)),
    #     season = UniqueGrouper(labels = np.unique(ds.season))
    # )
    # out = ds_grouped.mean()
    aggregated = ds.mean(dim="ref_time", skipna=True)

    return aggregated


def main(args: Namespace) -> None:
    """Main function to verify results from KENDA-1 data."""

    # grouping by all combinations of hour, season, init_hour may not be supported directly
    # TODO: implement grouping

    LOG.info("Reading %d verification files", len(args.verif_files))
    ds = xr.open_mfdataset(args.verif_files, combine="by_coords")

    LOG.info("Concatenated Dataset: \n %s", ds)

    # indexing into multi-dimensional time coordinate is not supported
    # instead index into lead_time
    # TODO: implement filtering based on valid time

    if args.valid_every:
        LOG.info("Filtering data based on lead time")
        max_lead_time = ds.lead_time.max().values.astype("timedelta64[h]").astype(int)
        valid_every = np.arange(
            0, max_lead_time + 1, args.valid_every, dtype="timedelta64[h]"
        )
        ds = ds.sel(lead_time=valid_every)
        if ds.lead_time.size == 0:
            raise ValueError(
                f"No data found with lead time every {args.valid_every} hours."
            )

    LOG.info("Aggregating results")
    results = aggregate_results(ds)

    LOG.info("Aggregated results: \n %s", results)

    # Save results to NetCDF
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_netcdf(args.output)

    LOG.info("Results saved to %s", args.output)


if __name__ == "__main__":
    parser = ArgumentParser(description="Verify results from KENDA-1 data.")
    parser.add_argument(
        "verif_files", type=Path, nargs="+", help="Paths to verification files."
    )
    parser.add_argument(
        "--valid_every",
        type=int,
        default=None,
        help="Only include data where the hour of the day of the valid time is a multiple of this number of hours.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="verif_results.csv",
        help="Path to save the aggregated results.",
    )
    args = parser.parse_args()
    main(args)
    # example usage:
    # uv run workflow/scripts/verif_results.py /users/fzanetta/projects/mch-anemoi-evaluation/output/7c58e59d24e949c9ade3df635bbd37e2/*/verif.csv
