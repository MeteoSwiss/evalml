import itertools
import logging
from argparse import ArgumentParser
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt

from plotting.metric_lead_time_panel import plot_panel
from verification.loading import load_long_df, subset_df

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main(args: Namespace) -> None:
    """Main function to verify results from KENDA-1 data."""

    all_df = load_long_df(args.verif_files)

    metrics = all_df["metric"].unique()
    params = all_df["param"].unique()
    seasons = all_df["season"].unique() if args.stratify else ["all"]
    regions = all_df["region"].unique() if args.stratify else ["all"]
    init_hours = (
        all_df["init_hour"].unique() if args.stratify else [-999]
    )  # numeric code to indicate all init hours

    for region, metric, param, season, init_hour in itertools.product(
        regions, metrics, params, seasons, init_hours
    ):
        LOG.info(
            f"Processing region: {region}, metric: {metric}, param: {param}, season: {season}, init_hour: {init_hour}"
        )

        sub_df = subset_df(
            all_df,
            region=region,
            metric=metric,
            param=param,
            season=season,
            init_hour=init_hour,
        ).dropna()
        if sub_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        title = f"{metric} - {param} - {region}"
        title += f"- {season} - {init_hour}" if args.stratify else ""
        plot_panel(
            ax,
            sub_df,
            metric=metric,
            param=param,
            title=title,
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
