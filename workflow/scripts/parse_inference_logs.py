"""CLI wrapper: parse anemoi-inference logs and write system metrics JSON."""

import argparse
import json
import logging
from pathlib import Path

from diagnostics import parse_logs

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main(args: argparse.Namespace) -> None:
    records = parse_logs(
        log_files=args.logs,
        label_map=json.loads(args.label_map),
        gpu_map=json.loads(args.gpu_map),
        log_dir=args.log_dir,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(records, fh, indent=2)
    LOG.info("Saved system metrics to %s", args.output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Parse inference logs for system metrics.")
    p.add_argument("--logs", nargs="+", required=True, help="Inference log file paths.")
    p.add_argument(
        "--label_map", required=True, help="JSON dict mapping run_id → source label."
    )
    p.add_argument(
        "--gpu_map",
        default="{}",
        help="JSON dict mapping run_id → GPU count (default: 1).",
    )
    p.add_argument(
        "--log_dir",
        required=True,
        help="Root of inference_execute logs; used to extract run_id from file path.",
    )
    p.add_argument("--output", required=True, help="Output JSON file path.")
    main(p.parse_args())
