from pathlib import Path
from argparse import ArgumentParser, Namespace
import logging
import time

import xarray as xr
import numpy as np

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
    """Compute mean metric values aggregated by season"""
    LOG.info("Aggregation results")
    start = time.time()

    # for simplicity we group by season based on ref_time (as this is a dimension of the dataset)
    ds = ds.assign_coords(
        season=lambda ds: ds.ref_time.dt.season,
    )
    ds_mean = (
        ds.mean(dim="ref_time")
        .assign_coords({"season": "all"})
        .compute(num_workers=4, scheduler="threads")
    )
    ds_grouped_mean = (
        ds.groupby("season")
        .mean(dim="ref_time")
        .compute(num_workers=4, scheduler="threads")
    )
    out = xr.concat([ds_mean, ds_grouped_mean], dim="season")

    var_transform = {
        d: d.replace("VAR", "STDE").replace("var", "std").replace("MSE", "RMSE")
        for d in out.data_vars
        if "VAR" in d or "var" in d or "MSE" in d
    }
    for var in var_transform:
        out[var] = np.sqrt(out[var])
    out = out.rename(var_transform)

    LOG.info("Computed aggregation in %.2f seconds", time.time() - start)
    LOG.info("Aggregated results: \n %s", out)

    return out


def main(args: Namespace) -> None:
    """Main function to verify results from KENDA-1 data."""

    # grouping by all combinations of hour, season, init_hour may not be supported directly
    # TODO: implement grouping

    LOG.info("Reading %d verification files", len(args.verif_files))
    ds = xr.open_mfdataset(
        args.verif_files,
        combine="by_coords",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        chunks="auto",
    )

    LOG.info("Concatenated Dataset: \n %s", ds)

    # indexing into multi-dimensional time coordinate is not supported
    # instead index into lead_time
    # TODO: implement filtering based on valid time

    # if args.valid_every:
    #     LOG.info("Filtering data based on lead time")
    #     max_lead_time = ds.lead_time.max().values.astype("timedelta64[h]").astype(int)
    #     valid_every = np.arange(
    #         0, max_lead_time + 1, args.valid_every, dtype="timedelta64[h]"
    #     )
    #     ds = ds.sel(lead_time=valid_every)
    #     if ds.lead_time.size == 0:
    #         raise ValueError(
    #             f"No data found with lead time every {args.valid_every} hours."
    #         )

    results = aggregate_results(ds)

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
