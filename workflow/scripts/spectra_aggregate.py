"""Average per-init spectra over init times for one participant."""

import logging
from argparse import ArgumentParser
from pathlib import Path

from spectra.compute import aggregate_spectra

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main():
    ap = ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="per-init spectra.nc files.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    agg = aggregate_spectra(args.inputs)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    agg.to_netcdf(out)
    LOG.info("Saved aggregated spectra to %s", out)


if __name__ == "__main__":
    main()
