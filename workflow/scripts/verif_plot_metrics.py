import itertools
import logging
from argparse import ArgumentParser
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def _ensure_unique_lead_time(ds: xr.Dataset) -> xr.Dataset:
    """Drop duplicate lead_time entries within a Dataset (keep first occurrence)."""
    try:
        idx = ds.get_index("lead_time")
    except Exception:
        idx = pd.Index(ds["lead_time"].values)
    if getattr(idx, "has_duplicates", False):
        keep = ~idx.duplicated(keep="first")
        ds = ds.isel(lead_time=keep)
    return ds


def _select_best_sources(dfs: list[xr.Dataset]) -> list[xr.Dataset]:
    """
    If the same 'source' exists in multiple datasets, keep it only from the dataset
    that has the largest number of unique lead_time entries. Drop it from others.
    """
    # Compute unique sources per dataset
    src_sets = [set(d.source.values.tolist()) for d in dfs]
    all_sources = set().union(*src_sets)

    # Decide best provider (dataset index) for each source
    best = {}
    for s in all_sources:
        candidates = []
        for i, d in enumerate(dfs):
            if s in d.source.values:
                di = d.sel(source=s)
                try:
                    n = pd.Index(di["lead_time"].values).unique().size
                except Exception:
                    n = len(pd.unique(di["lead_time"].values))
                candidates.append((i, n))
        if candidates:
            best_idx, _ = max(candidates, key=lambda t: t[1])
            best[s] = best_idx

    # Drop non-best occurrences
    out = []
    for i, d in enumerate(dfs):
        drop_src = [s for s, b in best.items() if b != i and s in d.source.values]
        if drop_src:
            d = d.drop_sel(source=drop_src)
        out.append(d)
    return out


def subset_df(df, **kwargs):
    mask = pd.Series([True] * len(df))
    for key, value in kwargs.items():
        if isinstance(value, (list, tuple, set)):
            mask &= df[key].isin(value)
        else:
            mask &= df[key] == value
    return df[mask]


def main(args: Namespace) -> None:
    """Main function to verify results from KENDA-1 data."""

    # remove duplicated but not identical values from analyses (rounding errors)
    dfs = [xr.open_dataset(f) for f in args.verif_files]
    # 1) Ensure each dataset has unique lead_time values
    dfs = [_ensure_unique_lead_time(d) for d in dfs]
    # 2) For sources present in multiple datasets, keep the one with most lead_times
    dfs = _select_best_sources(dfs)
    # 3) Concatenate by source; outer join to keep the union of lead_times
    ds = xr.concat(dfs, dim="source", join="outer")

    # extract only  non-spatial variables to pd.DataFrame
    nonspatial_vars = [d for d in ds.data_vars if "spatial" not in d]
    all_df = (
        ds[nonspatial_vars].to_array("stack").to_dataframe(name="value").reset_index()
    )
    all_df[["param", "metric"]] = all_df["stack"].str.split(".", n=1, expand=True)
    all_df.drop(columns=["stack"], inplace=True)
    all_df["lead_time"] = all_df["lead_time"].dt.total_seconds() / 3600

    metrics = all_df["metric"].unique()
    params = all_df["param"].unique()
    seasons = all_df["season"].unique() if args.stratify else ["all"]
    init_hours = (
        all_df["init_hour"].unique() if args.stratify else [-999]
    )  # numeric code to indicate all init hours

    for metric, param, season, init_hour in itertools.product(
        metrics, params, seasons, init_hours
    ):
        LOG.info(
            f"Processing metric: {metric}, param: {param}, season: {season}, init_hour: {init_hour}"
        )

        def _subset_df(df):
            return subset_df(
                df,
                metric=metric,
                param=param,
                season=season,
                init_hour=init_hour,
            )

        sub_df = _subset_df(all_df).dropna()

        # breakpoint()
        fig, ax = plt.subplots(figsize=(10, 6))

        title = f"{metric} - {param}"
        title += f"- {season} - {init_hour}" if args.stratify else ""
        for source, df in sub_df.groupby("source"):
            df.plot(
                x="lead_time",
                y="value",
                kind="line",
                marker="o",
                title=title,
                xlabel="Lead Time [h]",
                ylabel=metric,
                label=source,
                color="black" if "analysis" in source else None,
                ax=ax,
            )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        fn = f"{metric}_{param}"
        fn += f"_{season}_{init_hour}.png" if args.stratify else ".png"
        plt.savefig(args.output_dir / fn)
        plt.close(fig)


if __name__ == "__main__":
    parser = ArgumentParser(description="Verify results from KENDA-1 data.")
    parser.add_argument(
        "verif_files",
        type=Path,
        nargs="+",
        help="Paths to verification files.",
        # "--verif_files", type=Path, nargs="+", help="Paths to verification files.",
        # default = list(Path("output/data").glob("*/*/verif_aggregated.nc")), required=False
    )
    parser.add_argument(
        "--stratify",
        action="store_true",
        help="Stratify results by hour, season, and init_hour.",
        default=False,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="plots",
        help="Path to save the aggregated results.",
    )
    args = parser.parse_args()
    main(args)
