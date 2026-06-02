import logging
import time
from argparse import ArgumentParser
from argparse import Namespace
from pathlib import Path

import scores
import numpy as np
import xarray as xr

CATEGORICAL_METRICS = {
    "ETS": lambda m: m.equitable_threat_score(),
    "FBI": lambda m: m.frequency_bias(),
    "POD": lambda m: m.probability_of_detection(),
    "FAR": lambda m: m.false_alarm_ratio(),
}


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
        season=lambda ds: ds["forecast_reference_time"].dt.season,
        init_hour=lambda ds: ds["forecast_reference_time"].dt.hour,
    ).drop_vars(["time"], errors="ignore")

    # Counter used to track the number of ref_time samples per stratum so that
    # aggregated results can be correctly re-aggregated later (weighted mean).
    n_counter = xr.DataArray(
        np.ones(len(ds.ref_time), dtype=np.int64),
        coords={c: ds[c] for c in ["ref_time", "season", "init_hour"]},
        dims=["ref_time"],
    )

    # compute mean with grouping by all permutations of season and init_hour
    ds_mean = []
    for group in [[], "season", "init_hour", ["season", "init_hour"]]:
        if group == []:
            ds_grouped = ds
            n_grouped = n_counter.sum().compute()
        else:
            ds_grouped = ds.groupby(group)
            n_grouped = n_counter.groupby(group).sum().compute()
        ds_result = ds_grouped.mean(dim="forecast_reference_time").compute(
            num_workers=4, scheduler="threads"
        )
        ds_result["n_samples"] = n_grouped
        if "init_hour" not in group:
            ds_result = ds_result.expand_dims({"init_hour": [-999]})
        if "season" not in group:
            ds_result = ds_result.expand_dims({"season": ["all"]})
        LOG.info("Aggregated by %s: \n %s", group, ds_result)
        ds_mean.append(ds_result)
    out = xr.merge(ds_mean, compat="no_conflicts", join="outer")

    for var in out.data_vars:
        if "contingency_table" in var:
            # convert contingency table means back to per-stratum counts
            out[var] = out[var] * out["n_samples"]
            contingency_manager = scores.categorical.BasicContingencyManager(
                {
                    "tp_count": out[var].sel(contingency="tp_count"),
                    "tn_count": out[var].sel(contingency="tn_count"),
                    "fp_count": out[var].sel(contingency="fp_count"),
                    "fn_count": out[var].sel(contingency="fn_count"),
                    "total_count": out[var].sel(contingency="total_count"),
                }
            )
            for metric, fn in CATEGORICAL_METRICS.items():
                out[var.replace("contingency_table", metric)] = fn(contingency_manager)
            # split by threshold dimension into data variables

            out = out.drop_vars(var)

    # second loop to split along thresholds dimension, could be moved to plotting/dashboards scripts
    for var in out.data_vars:
        dim = list(filter(lambda x: "threshold" in x, out[var].dims))
        if dim:
            rename_dict = {d: f"{var}_{d}" for d in out.coords[dim[0]].values}
            out = xr.merge(
                [
                    out[var].to_dataset(dim=dim[0]).rename(rename_dict),
                    out.drop_vars(var),
                ]
            )

    # Derive STDE and R2 from aggregated component metrics
    for var in list(out.data_vars):
        if var.endswith(".MSE"):
            prefix = var[: -len("MSE")]
            bias_var = f"{prefix}BIAS"
            if bias_var in out.data_vars:
                out[f"{prefix}STDE"] = np.sqrt(
                    np.maximum(out[var] - out[bias_var] ** 2, 0)
                )
        if var.endswith(".CORR"):
            prefix = var[: -len("CORR")]
            out[f"{prefix}R2"] = out[var] ** 2

    # Square-root transform parameters: sqrt MSE -> RMSE; sqrt var -> std
    var_transform = {
        d: d.replace("var", "std").replace("MSE", "RMSE")
        for d in out.data_vars
        if "MSE" in d or "var" in d
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
        sorted(args.verif_files),
        combine="nested",
        concat_dim="forecast_reference_time",
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
    parser.add_argument(
        "verif_files", type=Path, nargs="+", help="Paths to verification files."
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
