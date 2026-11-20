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


def _brier_from_prob_table(table: xr.DataArray) -> xr.Dataset:
    """Brier score and Murphy decomposition from a prob_table of counts.

    ``table`` has dims ``n_members_yes`` (0..N) and ``obs`` (0, 1) besides the others.
    The forecast probability of bin k is k/N. BS = REL - RES + UNC holds exactly as
    the forecast probability takes one value per bin.
    """
    n_bins = table.sizes["n_members_yes"]
    n_members = n_bins - 1
    f = xr.DataArray(
        np.arange(n_bins) / n_members,
        dims="n_members_yes",
        coords={"n_members_yes": table["n_members_yes"]},
    )
    n_obs1 = table.isel(obs=1, drop=True)
    n_obs0 = table.isel(obs=0, drop=True)
    n_k = n_obs0 + n_obs1
    total = n_k.sum("n_members_yes", skipna=False)
    total = total.where(total > 0)
    o_k = (n_obs1 / n_k.where(n_k > 0)).fillna(0.0)
    o_bar = n_obs1.sum("n_members_yes", skipna=False) / total
    bs = (n_obs1 * (f - 1) ** 2 + n_obs0 * f**2).sum("n_members_yes", skipna=False)
    rel = (n_k * (f - o_k) ** 2).sum("n_members_yes", skipna=False)
    res = (n_k * (o_k - o_bar) ** 2).sum("n_members_yes", skipna=False)
    return xr.Dataset(
        {
            "BS": bs / total,
            "BS_REL": rel / total,
            "BS_RES": res / total,
            "BS_UNC": o_bar * (1 - o_bar),
        }
    )


def aggregate_results(ds: xr.Dataset) -> xr.Dataset:
    """Compute mean metric values aggregated by season"""
    LOG.info("Aggregation results")
    start = time.time()

    # for simplicity we group by season based on forecast_reference_time (as this is a dimension of the dataset)
    ds = ds.assign_coords(
        season=lambda ds: ds["forecast_reference_time"].dt.season,
        init_hour=lambda ds: ds["forecast_reference_time"].dt.hour,
    ).drop_vars(["time", "valid_time"], errors="ignore")

    # Counter used to track the number of forecast_reference_time samples per stratum so that
    # aggregated results can be correctly re-aggregated later (weighted mean).
    n_counter = xr.DataArray(
        np.ones(len(ds["forecast_reference_time"]), dtype=np.int64),
        coords={c: ds[c] for c in ["forecast_reference_time", "season", "init_hour"]},
        dims=["forecast_reference_time"],
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

    n_members_attr = ds.attrs.get("n_members")
    for var in list(out.data_vars):
        if "rank_hist" in var:
            # ensemble rank histogram: back to per-stratum counts
            out[var] = out[var] * out["n_samples"]
            if n_members_attr is None:
                n_members_attr = out.sizes["rank"] - 1
        elif "prob_table" in var:
            # forecast-probability/observation counts: back to per-stratum counts,
            # then Brier score and its decomposition per threshold
            out[var] = out[var] * out["n_samples"]
            brier = _brier_from_prob_table(out[var])
            for name, da in brier.data_vars.items():
                out[var.replace("prob_table", name)] = da
            if n_members_attr is None:
                n_members_attr = out.sizes["n_members_yes"] - 1
        elif "contingency_table" in var:
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

    # Ensemble spread and spread-skill ratio (after the sqrt/rename above: needs RMSE)
    for var in [v for v in out.data_vars if v.endswith(".ENS_VAR")]:
        prefix = var[: -len("ENS_VAR")]
        spread = np.sqrt(out[var])
        out[f"{prefix}SPREAD"] = spread
        rmse_var = f"{prefix}RMSE"
        if rmse_var in out.data_vars and n_members_attr is not None:
            n = float(n_members_attr)
            rmse = out[rmse_var]
            ssr = spread / rmse.where(rmse != 0) * np.sqrt((n + 1) / n)
            out[f"{prefix}SSR"] = ssr.where(spread != 0)
        out = out.drop_vars(var)
    if n_members_attr is not None and any(
        v.endswith(".SPREAD") or "rank_hist" in v or "prob_table" in v
        for v in out.data_vars
    ):
        out.attrs["n_members"] = int(n_members_attr)

    LOG.info("Computed aggregation in %.2f seconds", time.time() - start)
    LOG.info("Aggregated results: \n %s", out)

    return out


def main(args: Namespace) -> None:
    """Main function to verify results from KENDA-1 data."""

    # grouping by all combinations of hour, season, init_hour may not be supported directly
    # TODO: implement grouping

    LOG.info("Reading %d verification files", len(args.verif_files))
    source_sets = [
        frozenset(xr.open_dataset(f, engine="netcdf4")["source"].values)
        for f in args.verif_files
    ]
    if len(set(source_sets)) > 1:
        raise ValueError(
            f"Inconsistent source values across verif files — a label may have "
            f"changed between runs without triggering a rerun: {set(source_sets)}"
        )
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
