from pathlib import Path
import itertools
from argparse import ArgumentParser, Namespace
import logging

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


def read_verif_file(Path: str) -> pd.DataFrame:
    """Read a verification file and return it as a DataFrame."""

    df = pd.read_csv(
        Path, date_format="ISO8601", dtype={"lead_time": str}, index_col=None
    )
    df["lead_time"] = pd.to_timedelta(df["lead_time"])
    return df


def _check_same_columns(dfs: list[pd.DataFrame]) -> None:
    """Check if all DataFrames have the same columns."""

    columns = [set(df.columns) for df in dfs]
    if not all(col == columns[0] for col in columns):
        raise ValueError("DataFrames do not have the same columns.")


def _check_same_column_values(dfs: list[pd.DataFrame], column: str) -> None:
    """Check if all DataFrames have the same unique values in a column."""
    values = [set(df[column].unique()) for df in dfs]
    if not all(metric == values[0] for metric in values):
        LOG.warning(
            f"DataFrames do not have the same unique values in column '{column}'."
            f" Found: {values}"
        )
        # return the intersection of all values
        return values[0].intersection(*values[1:])
    return values[0]


def main(args: Namespace) -> None:
    """Main function to verify results from KENDA-1 data."""

    dfs = [read_verif_file(file) for file in args.verif_files]

    _check_same_columns(dfs)
    metrics = _check_same_column_values(dfs, "metric")
    params = _check_same_column_values(dfs, "param")

    hours = _check_same_column_values(dfs, "hour") if args.stratify else ["all"]
    seasons = _check_same_column_values(dfs, "season") if args.stratify else ["all"]
    init_hours = (
        _check_same_column_values(dfs, "init_hour") if args.stratify else ["all"]
    )

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

        subsets_dfs = [_subset_df(df) for df in dfs]
        # breakpoint()
        fig, ax = plt.subplots(figsize=(10, 6))

        title = f"{metric} - {param}"
        title += f"- {hour} - {season} - {init_hour}" if args.stratify else ""
        for i, sdf in enumerate(subsets_dfs):
            # convert lead time to integer hours for plotting
            sdf["lead_time"] = sdf["lead_time"].dt.total_seconds() / 3600
            for label, df in sdf.groupby("label"):
                df.plot(
                    x="lead_time",
                    y="value",
                    kind="line",
                    marker="o",
                    title=title,
                    xlabel="Lead Time [h]",
                    ylabel=metric,
                    label=label,
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
        # default = list(Path("output/data").glob("*/*/verif_aggregated.csv")), required=False
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
