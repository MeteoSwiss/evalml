import logging
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import jinja2

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def _parse_steps(steps: str) -> int:
    # check that steps is in the format "start/stop/step"
    if "/" not in steps:
        raise ValueError(f"Expected steps in format 'start/stop/step', got '{steps}'")
    if len(steps.split("/")) != 3:
        raise ValueError(f"Expected steps in format 'start/stop/step', got '{steps}'")
    start, end, step = map(int, steps.split("/"))
    return list(range(start, end + 1, step))


def program_summary_log(args):
    """Log a welcome message with the script information."""
    LOG.info("=" * 80)
    LOG.info("Generating MEC namelist")
    LOG.info("=" * 80)
    LOG.info("Template:        %s", args.template)
    LOG.info("Valid time:  %s", args.init_time.strftime("%Y%m%d%H%M"))
    LOG.info("Lead times:           %s", args.steps)
    LOG.info("Output namelist: %s", args.namelist)
    LOG.info("=" * 80)


def main(args):
    program_summary_log(args)
    # Include stop_h (inclusive). Produce strings like 0000,0600,1200,...,12000
    lead_hours = args.steps
    leadtimes = ",".join(f"{h:02d}00" for h in lead_hours)

    # Render template with init_time and computed leadtimes
    context = {"init_time": f"{args.init_time:%Y%m%d%H%M}", "leadtimes": leadtimes}
    template_path = Path(args.template)
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(template_path.parent)))
    template = env.get_template(template_path.name)
    namelist = template.render(**context)
    # Ensure file ends with a newline (prevent editors/tools from removing final RETURN)
    if not namelist.endswith("\n"):
        namelist += "\n"

    out_path = Path(str(args.namelist))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(namelist)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Render a Jinja2 MEC namelist template for a given initialisation time and lead times."
    )

    parser.add_argument(
        "--steps",
        type=_parse_steps,
        default="0/120/6",
        help="Forecast lead times formatted as 'start/end/step' (hours), e.g. '0/120/6'.",
    )

    parser.add_argument(
        "--init_time",
        type=lambda s: datetime.strptime(s, "%Y%m%d%H%M"),
        default="202010010000",
        help="Initialisation time in YYYYMMDDHHmm format, e.g. '202010010000'.",
    )

    parser.add_argument(
        "--template",
        type=str,
        help="Path to the Jinja2 namelist template file.",
    )

    parser.add_argument(
        "--namelist",
        type=str,
        help="Full path to the output namelist file to be written.",
    )

    args = parser.parse_args()

    main(args)
