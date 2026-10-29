"""Write where one participant's forecast has a value, on the truth points.

The masks of all experiment participants are intersected by verification_metrics.py
(--valid_masks) so that the listed parameters are scored on the same points
for every participant; see verification.valid_mask.
"""

import logging
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from verification_metrics import load_aligned_data  # noqa: E402

from data_input import parse_steps  # noqa: E402
from verification.valid_mask import forecast_valid_mask  # noqa: E402

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main(args):
    # Loaded exactly as for verification, so the mask matches the scored points.
    fcst, _ = load_aligned_data(args, args.params)
    mask = forecast_valid_mask(fcst, args.params)
    LOG.info("Valid mask for %s: \n%s", args.params, mask)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mask.to_netcdf(args.output)
    LOG.info("Saved valid mask to %s", args.output)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forecast",
        type=Path,
        required=True,
        help="Path to the grib forecast directory or the baseline archive.",
    )
    parser.add_argument("--truth", type=Path, required=True, help="Truth data.")
    parser.add_argument(
        "--reftime",
        type=lambda s: datetime.strptime(s, "%Y%m%d%H%M"),
        required=True,
        help="Reference time as YYYYmmddHHMM.",
    )
    parser.add_argument(
        "--params",
        type=lambda x: x.split(","),
        required=True,
        help="Comma-separated parameters of experiment.score_on_common_points.",
    )
    parser.add_argument(
        "--steps",
        type=parse_steps,
        required=True,
        help="Forecast steps in the format 'start/stop/step'.",
    )
    parser.add_argument(
        "--member",
        type=str,
        default="000",
        help="Ensemble member to load (see verification_metrics.py).",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output NetCDF.")
    main(parser.parse_args())
