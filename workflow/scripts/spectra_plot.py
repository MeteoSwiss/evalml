"""Overlay truth + all participants into one figure per (variable, lead time)."""

import logging
from argparse import ArgumentParser

from spectra.compute import plot_experiment_spectra

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main():
    ap = ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--participants", nargs="+", required=True)
    ap.add_argument("--variables", required=True)
    ap.add_argument("--lead_times", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    plot_experiment_spectra(
        args.truth,
        args.participants,
        args.output_dir,
        variables=args.variables.split(","),
        lead_times=[int(s) for s in args.lead_times.split(",")],
    )


if __name__ == "__main__":
    main()
