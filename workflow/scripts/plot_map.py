from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

State = dict[str, np.ndarray | dict[str, np.ndarray]]

from argparse import ArgumentParser, Namespace
from functools import partial
import logging
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np

from src.plotting import StatePlotter
#from src.calc import process_augment_state
from src.compat import load_state_from_raw
from src.colormaps import CMAP_DEFAULTS

State = dict[str, np.ndarray | dict[str, np.ndarray]]

LOG = logging.getLogger(__name__)
LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)


def plot_state(
    plotter: StatePlotter,
    pred_state: State,
    paramlist: list[str],
    projection: str = "orthographic",
    region: str = "europe",
) -> None:

    for param in paramlist:

        # # plot individual fields
        fig, [gax] = plotter.init_geoaxes(
            projection=projection, region=region, coastlines=True, zorder=2
        )
        fig.set_size_inches(10, 10)

        if param == "uv":
            field = np.sqrt(
                pred_state["fields"]["10u"] ** 2 + pred_state["fields"]["10v"] ** 2
            )
        elif param == "2t":
            field = pred_state["fields"][param] - 273.15
        else:
            field = pred_state["fields"][param]

        plotter.plot_field(
            ax=gax,
            field=field,
            zorder=1,
            validtime=pred_state["valid_time"].strftime("%Y%m%d%H%M"),
            **CMAP_DEFAULTS[param],
        )

        validtime = pred_state["valid_time"].strftime("%Y%m%d%H%M")
        leadtime = int(pred_state["lead_time"].total_seconds() // 3600)
        fn = f"{validtime}_{leadtime:03}_{param}_{projection}_{region}.png"
        plt.savefig(plotter.out_dir / fn, bbox_inches="tight", dpi=400)
        plt.clf()
        plt.cla()
        plt.close()


def process_plot_leadtime(
    file: Path,
    # dataset: Dataset,
    paramlist: list[str],
    plots_dir: Path,
):
    LOG.info(f"Started plotting {file.name}...")

    LOG.info(f"Loading predicted and true states")
    pred_state = load_state_from_raw(file)
    print(pred_state.keys())

    LOG.info(f"Augmenting states")
    #pred_state = process_augment_state(pred_state)

    LOG.info(f"Initializing plotter")
    plotter = StatePlotter(
        pred_state["longitudes"],
        pred_state["latitudes"],
        plots_dir,
    )

    for region in ["europe", "globe", "switzerland"]:
        for proj in ["orthographic"]:
            plot_state(
                plotter,
                pred_state,
                paramlist,
                projection=proj,
                region=region,
            )
    LOG.info(f"Done plotting {file}.")
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)
    return 0


def create_animation(
    plot_dir: Path,
    param: str,
    projection: str,
    region: str,
    name_prefix: str | None = None,
) -> None:

    out_dir = plot_dir / "animations"
    out_dir.mkdir(exist_ok=True, parents=True)
    name_prefix = f"{name_prefix}_" if name_prefix else ""
    cmd = f"convert -delay 80 -loop 0"

    # # animations of prediction plots
    plots_glob = f"{plot_dir}/*{param}_{projection}_{region}.png"
    gif_fn = f"{out_dir}/{name_prefix}{param}_{projection}_{region}.gif"
    print("CMD", plots_glob)
    os.system(f"{cmd} {plots_glob} {gif_fn}")


class ScriptConfig(Namespace):
    checkpoint_run_id: Path
    model_name: None | str
    # date: datetime
    raw_dir: Path
    paramlist: list[str]
    out_dir: Path


def main(cfg: ScriptConfig) -> None:

    LOG.info(f"Plotting inference results for {cfg}")
    # set up output directory

    out_dir = Path(cfg.out_dir)
    print(out_dir)
    plots_dir = Path(out_dir)

    raw_dir = Path(cfg.raw_dir)
    raw_files = sorted(raw_dir.glob(f"*.npz"))
    _map_fn = partial(
        process_plot_leadtime,
        paramlist=cfg.paramlist,
        plots_dir=plots_dir,
    )

    with ProcessPoolExecutor(max_workers=len(raw_files)) as executor:
        results = executor.map(_map_fn, raw_files)

    # wait for all processes to finish
    for _ in results:
        pass

    LOG.info(f"Creating animations")
    for param in cfg.paramlist:
        for region in ["europe", "globe", "switzerland"]:
            for proj in ["orthographic"]:
                create_animation(
                    plots_dir,
                    param,
                    proj,
                    region,
                    name_prefix=cfg.model_name,
                )


if __name__ == "__main__":

    ROOT = Path(__file__).parent
    OUT_DIR = ROOT / "output"

    parser = ArgumentParser()

    parser.add_argument("--input", type=str, default=None, help="Directory to raw data")
    parser.add_argument("--date", type=str, default=None, help="reference datetime")
    parser.add_argument("--output", type=str, help="output directory")

    args = parser.parse_args()

    config = ScriptConfig(
        checkpoint_run_id="",
        model_name="icon",
        raw_dir=args.input,
        paramlist=["10u", "2t"],
        out_dir=args.output,
    )

    main(config)
