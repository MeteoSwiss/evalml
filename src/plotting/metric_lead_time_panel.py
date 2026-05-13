"""Per-axes plotting helper for verification metrics vs. lead time."""
import pandas as pd
from matplotlib.axes import Axes

from verification import decode_metric

from .units import metric_units


def _default_ylabel(metric: str, param: str | None) -> str:
    label = decode_metric(metric)
    units = metric_units(metric, param) if param is not None else ""
    return f"{label} [{units}]" if units else label


def plot_panel(
    ax: Axes,
    sub_df: pd.DataFrame,
    *,
    metric: str,
    param: str | None = None,
    title: str | None = None,
    panel_label: str | None = None,
    xlabel: str | None = "Lead Time [h]",
    ylabel: str | None = None,
    show_legend: bool = True,
    color_map: dict[str, str] | None = None,
) -> None:
    """Plot one metric-vs-lead-time panel onto `ax`.

    `sub_df` must already be filtered to a single (metric, param, region, season,
    init_hour) combo and contain at least the columns: source, lead_time, value.
    One line per source is drawn.

    If `ylabel` is None and `param` is provided, the y-axis label is built as
    "<decoded metric> [<units>]" via plotting.units.metric_units.

    `panel_label` (e.g. "a)") is rendered left-aligned at the same height as
    the centred title.

    If `color_map` is given, each source's line is drawn in
    ``color_map[source]``; sources missing from the map fall back to
    matplotlib's default color cycle. Use ``plotting.source_colors.source_color_map``
    to build a map that matches the dashboard.
    """
    if ylabel is None:
        ylabel = _default_ylabel(metric, param)
    for source, df in sub_df.groupby("source"):
        df.plot(
            x="lead_time",
            y="value",
            kind="line",
            marker="o",
            title=title,
            xlabel=xlabel or "",
            ylabel=ylabel or "",
            label=source,
            color=(color_map or {}).get(source),
            ax=ax,
            legend=show_legend,
        )
    if panel_label:
        ax.set_title(panel_label, loc="left", fontweight="bold")
