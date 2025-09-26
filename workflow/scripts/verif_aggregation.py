import logging
import time
from argparse import ArgumentParser
from argparse import Namespace
from pathlib import Path

import numpy as np
import xarray as xr

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


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
        init_hour=lambda ds: ds.ref_time.dt.hour,
    ).drop_vars(["time"])

    # compute mean with grouping by all permutations of season and init_hour
    ds_mean = []
    for group in [[], "season", "init_hour", ["season", "init_hour"]]:
        if group == []:
            ds_grouped = ds
        else:
            ds_grouped = ds.groupby(group)
        ds_grouped = ds_grouped.mean(dim="ref_time").compute(num_workers=4, scheduler="threads")
        if "init_hour" not in group:
            ds_grouped = ds_grouped.expand_dims({"init_hour": [-999]})
        if "season" not in group:
            ds_grouped = ds_grouped.expand_dims({"season": ["all"]})
        LOG.info("Aggregated by %s: \n %s", group, ds_grouped)
        ds_mean.append(ds_grouped)
    out = xr.merge(ds_mean, compat="no_conflicts", join="outer")

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
        engine="netcdf4",
        parallel=False,  # netcdf4 engine fails silently with parallel=True
    )

    LOG.info("Concatenated Dataset: \n %s", ds)

    results = aggregate_results(ds)

    # Save results to NetCDF
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_netcdf(args.output)

    LOG.info("Results saved to %s", args.output)


if __name__ == "__main__":
    parser = ArgumentParser(description="Verify results from KENDA-1 data.")
    parser.add_argument("verif_files", type=Path, nargs="+", help="Paths to verification files.")
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
    # uv run workflow/scripts/verif_results.py /users/fzanetta/projects/mch-anemoi-evaluation/output/7c58e59d24e949c9ade3df635bbd37e2/*/verif.csv
