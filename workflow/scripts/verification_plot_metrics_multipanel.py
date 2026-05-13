"""Build a multi-panel metric-vs-lead-time figure from aggregated verification files.

The panel layout (rows, cols, per-panel selectors) is supplied as a JSON spec
either inline (``--spec_json '<json>'``) or as a path to a JSON file
(``--spec_path /path/to/spec.json``). The spec schema mirrors
``MultipanelPlotSpec`` in ``src/evalml/config.py``.
"""

import json
import logging
import string
from argparse import ArgumentParser
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt

from plotting.metric_lead_time_panel import plot_panel
from plotting.source_colors import source_color_map
from plotting.units import metric_units
from verification import decode_metric
from verification.loading import load_long_df, subset_df


def _panel_label(idx: int) -> str:
    """Return 'a)', 'b)', ..., 'z)', 'aa)', ... for the given 0-based index."""
    letters = string.ascii_lowercase
    if idx < len(letters):
        return f"{letters[idx]})"
    a, b = divmod(idx, len(letters))
    return f"{letters[a - 1]}{letters[b]})"


LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def _load_spec(args: Namespace) -> dict:
    if args.spec_json:
        return json.loads(args.spec_json)
    return json.loads(args.spec_path.read_text())


def main(args: Namespace) -> None:
    spec = _load_spec(args)
    rows = int(spec["rows"])
    cols = int(spec["cols"])
    panels = spec["panels"]
    if len(panels) != rows * cols:
        raise ValueError(
            f"panels has length {len(panels)}, expected rows*cols = {rows * cols}"
        )

    all_df = load_long_df(args.verif_files)
    color_map = source_color_map(all_df["source"].unique())

    figsize = tuple(spec.get("figsize") or (4.5 * cols, 3.5 * rows))
    fig, axes = plt.subplots(rows, cols, sharex=True, figsize=figsize, squeeze=False)

    legend_entries: dict[str, object] = {}
    for idx, panel in enumerate(panels):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        metric = panel["metric"]
        param = panel["param"]
        sub = subset_df(
            all_df,
            metric=metric,
            param=param,
            region=panel.get("region", "all"),
            season=panel.get("season", "all"),
            init_hour=panel.get("init_hour", -999),
        ).dropna()
        if sub.empty:
            LOG.warning(
                "No data for panel %d (metric=%s, param=%s, region=%s, season=%s, init_hour=%s)",
                idx,
                metric,
                param,
                panel.get("region", "all"),
                panel.get("season", "all"),
                panel.get("init_hour", -999),
            )

        is_bottom = r == rows - 1
        is_left = c == 0
        title = panel.get("title", f"{metric} - {param}")
        units = metric_units(metric, param)
        ylabel = (
            (f"{decode_metric(metric)} [{units}]" if units else decode_metric(metric))
            if is_left
            else None
        )
        plot_panel(
            ax,
            sub,
            metric=metric,
            param=param,
            title=title,
            panel_label=_panel_label(idx),
            xlabel="Lead Time [h]" if is_bottom else None,
            ylabel=ylabel,
            show_legend=False,
            color_map=color_map,
        )
        if panel.get("ylim"):
            ax.set_ylim(panel["ylim"])

        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            legend_entries.setdefault(label, handle)

    if spec.get("title"):
        fig.suptitle(spec["title"])

    if legend_entries:
        fig.legend(
            list(legend_entries.values()),
            list(legend_entries.keys()),
            loc="lower center",
            ncol=min(len(legend_entries), 4),
            bbox_to_anchor=(0.5, 0.0),
        )

    top = 0.92 if spec.get("title") else 0.96
    fig.subplots_adjust(
        left=0.09,
        right=0.97,
        top=top,
        bottom=0.13,
        hspace=0.45,
        wspace=0.28,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "verif_files",
        type=Path,
        nargs="+",
        help="Paths to aggregated verification netCDFs.",
    )
    spec_group = parser.add_mutually_exclusive_group(required=True)
    spec_group.add_argument(
        "--spec_json",
        type=str,
        default=None,
        help="Inline JSON string describing the panel layout.",
    )
    spec_group.add_argument(
        "--spec_path",
        type=Path,
        default=None,
        help="Path to a JSON file describing the panel layout.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PNG path.",
    )
    args = parser.parse_args()
    main(args)
