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


def main(args):
    # Include stop_h (inclusive). Produce strings like 0000,0600,1200,...,12000
    lead_hours = args.steps
    leadtimes = ",".join(f"{h:02d}00" for h in lead_hours)

    # Render template with init_time and computed leadtimes
    context = {"init_time": f"{args.init_time:%Y%m%d%H%M}", "leadtimes": leadtimes}
    template_path = Path(args.template)
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(template_path.parent)))
    template = env.get_template(template_path.name)
    namelist = template.render(**context)
    LOG.info(f"MEC namelist created: {namelist}")

    out_path = Path(str(args.namelist))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(namelist)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--steps", type=_parse_steps, default="0/120/6")

    parser.add_argument(
        "--init_time",
        type=lambda s: datetime.strptime(s, "%Y%m%d%H%M"),
        default="202010010000",
        help="Valid time for the data in ISO format.",
    )

    parser.add_argument(
        "--template",
        type=str,
    )

    parser.add_argument(
        "--namelist",
        type=str,
        help="Anything useful",
    )

    args = parser.parse_args()

    main(args)
