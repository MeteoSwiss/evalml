# /// script
# dependencies = [
#   "pandas",
#   "zarr",
#   "meteodata-lab",
# ]
# ///
# TODO: duplicated code from workflow/scripts/verify_cosmoe_fct.py
from pathlib import Path
import itertools
from argparse import ArgumentParser, Namespace
import logging

import pandas as pd

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


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
    

def read_verif_file(Path: str) -> pd.DataFrame:
    """Read a verification file and return it as a DataFrame."""

    df = pd.read_csv(
        Path,
        date_format="ISO8601",
        parse_dates=["ref_time", "valid_time", "time"],
        dtype={"lead_time": str},
        index_col=0
    )
    df["lead_time"] = pd.to_timedelta(df["lead_time"])
    return df


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean metric values aggregated by all combinations of hour, season, and init_hour."""
    

    # extract features
    df = df.copy()
    df["hour"] = df["time"].dt.hour
    df["init_hour"] = df["ref_time"].dt.hour
    df["season"] = df["time"].apply(get_season)

    # generate all combinations of original and "all" for ["hour", "season", "init_hour"]
    features = ["hour", "season", "init_hour"]
    groupings = []

    for combination in itertools.product([True, False], repeat=3):
        modified_df = df.copy()
        for include_original, col in zip(combination, features):
            if not include_original:
                modified_df[col] = "all"
        groupings.append(modified_df)

    # concatenate all versions
    df_extended = pd.concat(groupings, ignore_index=True)

    # aggregate
    aggregated = df_extended.groupby(
        ["metric", "lead_time", "param", "hour", "season", "init_hour"],
        dropna=False  # optional, ensures NaN values are not dropped
    ).agg(value_mean=("value", "mean"), value_count=("value", "count"), value_sum=("value","sum")).reset_index()
    
    return aggregated

def main(args: Namespace) -> None:
    """Main function to verify results from KENDA-1 data."""
    
    LOG.info("Reading %d verification files", len(args.verif_files))
    df = pd.concat([read_verif_file(f) for f in args.verif_files], ignore_index=True)

    LOG.info("Concatenated DataFrame: \n %s", df.head())

    LOG.info("Aggregating results")
    results = aggregate_results(df)

    LOG.info("Aggregated results: \n %s", results.head())

    # Save results to CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    LOG.info("Results saved to %s", args.output)

if __name__ == "__main__":
    parser = ArgumentParser(description="Verify results from KENDA-1 data.")
    parser.add_argument("verif_files", type=Path, nargs="+",
                        help="Paths to verification files.")
    parser.add_argument("--output", type=Path, default="verif_results.csv",
                        help="Path to save the aggregated results."),
    args = parser.parse_args()
    main(args)

    # example usage:
    # uv run workflow/scripts/verif_results.py /users/fzanetta/projects/mch-anemoi-evaluation/output/7c58e59d24e949c9ade3df635bbd37e2/*/verif.csv