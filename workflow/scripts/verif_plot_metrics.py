from pathlib import Path
import itertools
from argparse import ArgumentParser, Namespace
import logging

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


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

    # override to remove duplicated but not identical values from analyses (rounding errors)
    # TODO: fix approach
    ds = xr.open_mfdataset(args.verif_files, compat="override")

    # extract only  non-spatial variables to pd.DataFrame
    nonspatial_vars = [d for d in ds.data_vars if "spatial" not in d]
    all_df = ds[nonspatial_vars].to_array("stack").to_dataframe(name = "value").reset_index()
    all_df[["param", "metric"]] = all_df["stack"].str.split(".", n = 1, expand=True)
    all_df.drop(columns = ["stack"], inplace=True)
    all_df[["hour", "season", "init_hour"]] = "all"
    all_df["lead_time"] = all_df["lead_time"].dt.total_seconds() / 3600

    metrics = all_df["metric"].unique()
    params = all_df["param"].unique()
    hours = all_df["hour"].unique() if args.stratify else ["all"]
    seasons = all_df["season"].unique() if args.stratify else ["all"]
    init_hours = all_df["init_hour"].unique() if args.stratify else ["all"]
    

    for metric, param, hour, season, init_hour in itertools.product(
        metrics, params, hours, seasons, init_hours
    ):
        LOG.info(
            f"Processing metric: {metric}, param: {param}, hour: {hour}, "
            f"season: {season}, init_hour: {init_hour}"
        )

        def _subset_df(df):
            return subset_df(
                df,
                metric=metric,
                param=param,
                hour=hour,
                season=season,
                init_hour=init_hour,
            )

        sub_df = _subset_df(all_df)

        # breakpoint()
        fig, ax = plt.subplots(figsize=(10, 6))

        title = f"{metric} - {param}"
        title += f"- {hour} - {season} - {init_hour}" if args.stratify else ""
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
        fn += f"_{hour}_{season}_{init_hour}.png" if args.stratify else ".png"
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
