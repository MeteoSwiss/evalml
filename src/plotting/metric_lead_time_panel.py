"""Per-axes plotting helper for verification metrics vs. lead time."""
import pandas as pd
from matplotlib.axes import Axes


def plot_panel(
    ax: Axes,
    sub_df: pd.DataFrame,
    *,
    metric: str,
    title: str | None = None,
    xlabel: str | None = "Lead Time [h]",
    ylabel: str | None = None,
    show_legend: bool = True,
) -> None:
    """Plot one metric-vs-lead-time panel onto `ax`.

    `sub_df` must already be filtered to a single (metric, param, region, season,
    init_hour) combo and contain at least the columns: source, lead_time, value.
    One line per source is drawn; sources whose name contains "analysis" are
    forced to black.
    """
    if ylabel is None:
        ylabel = metric
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
            color="black" if "analysis" in source else None,
            ax=ax,
            legend=show_legend,
        )
