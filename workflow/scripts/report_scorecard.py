"""Render a verification scorecard PNG comparing a model run against a baseline.

Each cell in the scorecard is a dot whose colour encodes which source is better
and whose area encodes the magnitude of the relative difference:

    diff = (model − baseline) / |baseline| × 100  [%]

Positive diff is better for higher-is-better metrics (R², ETS, POD) and worse
for lower-is-better metrics (RMSE, MAE, STDE, FAR).
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation

from verification import decode_metric

# ── Constants ─────────────────────────────────────────────────────────────────

# Sentinel values that select the "aggregate over all" slice for each
# stratification dimension that is not the active stratification axis.
_STRAT_ALL_VALUES = {"season": "all", "init_hour": -999}

DEFAULT_PLOT_CFG = {
    "rcparams": {
        "font_family": "sans-serif",
        "font_sans": "Liberation Sans",
        "dpi": 250,
    },
    "colors": {
        "model_better": "#4878d0",  # seaborn muted blue
        "baseline_better": "#d65f5f",  # seaborn muted red
        "neutral": "#cccccc",  # |diff|% below neutral_threshold_pct
        "missing": "#999999",  # NaN / no-data marker
        "leads": "#666666",  # lead-time tick labels
        "hline": "#cccccc",  # horizontal separators between variable groups
        "vline": "black",  # vertical separators between stratification slices
    },
    "fonts": {  # font sizes in points
        "title": 24,
        "group": 20,  # variable group labels (U_10M, T_2M, …)
        "metric": 18,  # metric labels         (RMSE, MAE, …)
        "slice": 20,  # slice header labels; increasing this auto-widens columns
        "leads": 16,
        "legend": 18,
    },
    "dots": {
        "neutral_threshold_pct": 5,  # |diff|% below which the dot is drawn neutral
        "size_cap_pct": 30,  # |diff|% at which the dot reaches max area
        "max_area": 250,  # max dot area (matplotlib scatter `s` units)
        "alpha": 0.9,
        "missing_marker_size": 8,  # size of the "×" marker for NaN cells
        "missing_marker_lw": 1.5,  # line width of the "×" marker
    },
    "figure": {  # sizing rules (inches)
        "col_width": 0.26,  # minimum width per lead-time column; grows if slice labels need more space
        "width_min": 5,  # minimum total figure width
        "width_pad": 5,  # extra width reserved for left-side labels (group + metric names)
        "left_margin_in": 2.5,  # physical left margin (inches) before the data area starts
        "row_height": 0.42,  # height per metric row
        "title_margin_in": 1.5,  # fixed space above the axes reserved for the title
    },
    "layout": {  # spacing in data coordinates unless noted
        "slice_gap": 1.5,  # empty columns inserted between adjacent stratification blocks
        "metric_x": -1.5,  # x position of metric labels (data coords, left of x=0)
        "group_metric_gap_pt": 15,  # physical gap (pt) between the metric label and the group label
        "slice_label_pad": 1.05,  # scale factor applied to the longest slice label width when sizing columns
        "slice_y": 1.9,  # y position of slice title labels (text baseline, data coords)
        "slice_y_pad": 0.2,  # extra y added above the slice label height for the top y-limit
        "leads_y": 0.45,  # y position of lead-time tick labels (data coords)
    },
    "hline": {  # horizontal separators between variable groups
        "start_pad_pt": 0,  # offset (pt) from the left edge of the metric label to the line start
        "x_end": 1.0,  # axes fraction where the line ends
        "linewidth": 0.7,
    },
    "vline": {  # vertical separators between stratification slices
        "label_gap": 0.4,  # gap (metric rows) between the slice label bottom and the top of the line
        "linewidth": 0.8,
    },
    "legend": {
        "width_in": 6.5,  # physical width (inches) of the dot row
        "dot_below_pt": 40,  # gap (pt) below the axes bottom to dot centres
        "label_below_pt": 54,  # gap (pt) below the axes bottom to labels
        "sample_pcts": [30, 15, 5],  # |diff|% values shown on each side of neutral
        "label_fontsize_factor": 0.85,
        "side_text_offset": 0.03,  # gap (axes fraction) between the outermost dot and its side label
        "missing_dot_offset_pt": 50,  # additional vertical gap (pt) for the "no data" row
    },
}

# ── Small pure helpers ────────────────────────────────────────────────────────


def _parse_var_metrics(spec: str) -> tuple[str, list[str] | None]:
    """Parse a ``'VAR:M1,M2,...'`` CLI token into ``(variable, [metrics])``.

    Returns ``None`` for the metrics list when no colon is present, which is
    interpreted downstream as "use all available metrics for this variable".
    """
    if ":" in spec:
        var, metrics_str = spec.split(":", 1)
        return var.strip(), [m.strip() for m in metrics_str.split(",") if m.strip()]
    return spec.strip(), None


def _timedelta_to_hours(td) -> int:
    """Convert a numpy or pandas timedelta to an integer number of hours."""
    return int(pd.Timedelta(td).total_seconds() / 3600)


def _resolve_metric(var_name: str, metric: str, data_vars) -> list[str]:
    """Return the data-variable names that match ``var_name.metric`` (or ``var_name.metric_*``).

    The trailing-underscore fallback handles threshold-suffixed variants such as
    ``TOT_PREC.ETS_1mm``.
    """
    exact = f"{var_name}.{metric}"
    if exact in data_vars:
        return [exact]
    return [v for v in data_vars if v.startswith(f"{var_name}.{metric}_")]


def _scaled_dot_area(diff_pct: float, dots: dict) -> float:
    """Map a relative difference percentage to a scatter dot area.

    Area grows linearly with ``|diff_pct|`` and is capped at ``dots['max_area']``.
    """
    return float(
        min(abs(diff_pct) / dots["size_cap_pct"] * dots["max_area"], dots["max_area"])
    )


def _is_model_better(
    diff_pct: float, metric: str, lower_is_better: list, higher_is_better: list
) -> bool | None:
    """Return True if the model is better, False if the baseline is better, None if unknown.

    Accounts for threshold-suffixed metric names (e.g. ``ETS_1mm`` matches ``ETS``).
    """

    def _matches(name, candidates):
        return any(name == c or name.startswith(f"{c}_") for c in candidates)

    if _matches(metric, higher_is_better):
        return diff_pct > 0  # model score is higher → model is better
    if _matches(metric, lower_is_better):
        return diff_pct < 0  # model score is lower  → model is better
    return None


def _format_slice_label(s, strat_dim: str) -> str:
    """Format a stratification coordinate value as a human-readable label."""
    if isinstance(s, str):
        return s.upper() if strat_dim == "season" and s != "all" else s.capitalize()
    # Numeric init_hour: -999 is the "all hours" sentinel, others are formatted as HHZ
    return "All" if int(s) == -999 else f"{int(s):02d}Z"


# ── Data pipeline ─────────────────────────────────────────────────────────────


def _build_config(args) -> dict:
    """Assemble the run configuration dict from parsed CLI arguments."""
    if args.variable:
        variables = dict(_parse_var_metrics(s) for s in args.variable)
    else:
        # Sensible default: RMSE for the core surface variables
        variables = {
            "U_10M": ["RMSE"],
            "V_10M": ["RMSE"],
            "T_2M": ["RMSE"],
            "PMSL": ["RMSE"],
            "TD_2M": ["RMSE"],
            "TOT_PREC": ["RMSE"],
        }

    return {
        "model": {
            "path": args.verif_run,
            "source": args.run_source,
            "label": args.run_label if args.run_label is not None else args.run_source,
        },
        "baseline": {
            "path": args.verif_baseline,
            "source": args.baseline_source,
            "label": args.baseline_label
            if args.baseline_label is not None
            else args.baseline_source,
        },
        "stratification": args.stratification,
        "lead_times": args.lead_times,
        # All recognised metrics — every entry must also appear in metric_directions.
        "all_metrics": ["RMSE", "MAE", "STDE", "R2", "ETS", "POD", "FAR"],
        "metric_directions": {
            "lower_is_better": ["RMSE", "MAE", "STDE", "FAR"],
            "higher_is_better": ["R2", "ETS", "POD"],
        },
        # None value → use all_metrics for that variable; list → restrict to that subset.
        "variables": variables,
        "plot": DEFAULT_PLOT_CFG,
    }


def _load_relative_diff(cfg: dict) -> xr.Dataset:
    """Load model and baseline datasets and return the relative difference in %.

    The result is ``(model − baseline) / |baseline| × 100``, with non-finite
    values (division by zero, inf) masked to NaN.  Both datasets are sliced to
    their common variables and lead times before the computation.
    """
    model_source = cfg["model"]["source"]
    baseline_source = cfg["baseline"]["source"]
    strat_dim = cfg["stratification"]

    # Fix all non-active stratification dimensions to their "aggregate all" value.
    sel_coords: dict = {"source": model_source}
    for dim, all_value in _STRAT_ALL_VALUES.items():
        if dim != strat_dim:
            sel_coords[dim] = all_value

    model_ds = xr.open_dataset(cfg["model"]["path"])
    baseline_ds = xr.open_dataset(cfg["baseline"]["path"])

    if strat_dim != "region":
        sel_coords["region"] = model_ds["region"].values[0]

    for label, ds in [("model", model_ds), ("baseline", baseline_ds)]:
        if "n_samples" not in ds.data_vars:
            raise ValueError(
                f"'n_samples' is missing from the {label} dataset '{cfg[label]['path']}'.\n"
                f"This file was likely produced before n_samples tracking was introduced.\n"
                f"Fix: delete '{cfg[label]['path']}' and rerun the pipeline."
            )
    model_n = int(model_ds["n_samples"].sel(season="all", init_hour=-999).item())
    baseline_n = int(baseline_ds["n_samples"].sel(season="all", init_hour=-999).item())
    if model_n != baseline_n:
        fewer = (
            cfg["model"]["path"] if model_n < baseline_n else cfg["baseline"]["path"]
        )
        raise ValueError(
            f"n_samples mismatch: model has {model_n} and baseline has {baseline_n} "
            f"forecast dates.\n"
            f"Both runs must cover the same set of dates for a valid scorecard.\n"
            f"Fix: delete '{fewer}' and rerun the pipeline."
        )

    model_ds = model_ds.sel(**sel_coords).squeeze(drop=True)
    baseline_ds = baseline_ds.sel(**{**sel_coords, "source": baseline_source}).squeeze(
        drop=True
    )

    common_vars = [v for v in model_ds.data_vars if v in baseline_ds.data_vars]
    common_leads = sorted(set(model_ds.step.values) & set(baseline_ds.step.values))

    if not common_vars:
        raise ValueError(
            "No variables in common between model and baseline.\n"
            f"  model:    {sorted({v.split('.')[0] for v in model_ds.data_vars})}\n"
            f"  baseline: {sorted({v.split('.')[0] for v in baseline_ds.data_vars})}"
        )
    if not common_leads:
        raise ValueError("No lead times in common between model and baseline.")

    model_slice = model_ds[common_vars].sel(step=common_leads)
    baseline_slice = baseline_ds[common_vars].sel(step=common_leads)

    rel_diff = (model_slice - baseline_slice) / abs(baseline_slice) * 100
    rel_diff = rel_diff.where(
        np.isfinite(rel_diff)
    )  # mask ±inf and NaN from zero baseline

    model_ds.close()
    baseline_ds.close()
    return rel_diff


def _filter_diff(diff: xr.Dataset, cfg: dict) -> xr.Dataset:
    """Subset *diff* to the requested lead times and variable/metric combinations.

    Lead times are selected by the ``'start/stop/step'`` grid in ``cfg['lead_times']``.
    Variables and metrics are selected according to ``cfg['variables']``.
    """
    all_metrics = cfg["all_metrics"]
    result = diff

    if cfg.get("lead_times"):
        start, stop, step = (int(x) for x in cfg["lead_times"].split("/"))
        requested = {pd.Timedelta(h, "h") for h in range(start, stop + 1, step)}
        available = {pd.Timedelta(lt) for lt in result.step.values}
        keep = sorted(requested & available)
        if not keep:
            raise ValueError(
                f"No lead times match '{cfg['lead_times']}'. "
                f"Available (h): {sorted(_timedelta_to_hours(lt) for lt in available)}"
            )
        result = result.sel(step=keep)

    keep = [
        var
        for var_name, var_metrics in cfg["variables"].items()
        for metric in (var_metrics or all_metrics)
        for var in _resolve_metric(var_name, metric, result.data_vars)
    ]
    result = result[keep]

    if not result.data_vars:
        raise ValueError("No variables left after filtering.")

    return result


# ── Rendering helpers ─────────────────────────────────────────────────────────


def _measure_label_sizes(plot: dict, rows: list, slices: list, strat_dim: str) -> tuple:
    """Return pixel-accurate label dimensions by rendering text in throwaway figures.

    Returns:
        slice_label_w_in:  width of the longest slice label in inches.
        slice_label_h_rows: height of the longest slice label in row units.
        metric_label_w_pt:  width of the longest metric label in points.
    """
    fonts = plot["fonts"]
    dpi = plot["rcparams"]["dpi"]
    font_sans = plot["rcparams"]["font_sans"]
    row_height = plot["figure"]["row_height"]

    longest_slice = max((_format_slice_label(s, strat_dim) for s in slices), key=len)
    fig, _ = plt.subplots(dpi=dpi)
    t = fig.axes[0].text(
        0,
        0,
        longest_slice,
        fontsize=fonts["slice"],
        fontweight="bold",
        fontfamily=font_sans,
    )
    fig.canvas.draw()
    bbox = t.get_window_extent(fig.canvas.get_renderer())
    slice_label_w_in = bbox.width / fig.dpi
    slice_label_h_rows = bbox.height / fig.dpi / row_height
    plt.close(fig)

    longest_metric = max((decode_metric(metric) for _, metric in rows), key=len)
    fig2, _ = plt.subplots(dpi=dpi)
    t2 = fig2.axes[0].text(
        0, 0, longest_metric, fontsize=fonts["metric"], fontfamily=font_sans
    )
    fig2.canvas.draw()
    metric_label_w_pt = (
        t2.get_window_extent(fig2.canvas.get_renderer()).width * 72 / fig2.dpi
    )
    plt.close(fig2)

    return slice_label_w_in, slice_label_h_rows, metric_label_w_pt


def _draw_data_rows(
    ax, diff, rows, slices, strat_dim, n_leads, neutral_dot_size, group_transform, cfg
) -> list:
    """Draw the variable/metric label column and the dot grid.

    Each data row corresponds to one (variable_group, metric) pair.  Dots are
    coloured blue for model-better, red for baseline-better, grey for neutral.
    Missing data is shown as an "×" marker.

    Returns:
        group_separator_ys: y-positions (data coords) where horizontal group
            separators should be drawn (i.e. between consecutive variable groups).
    """
    plot = cfg["plot"]
    layout = plot["layout"]
    colors = plot["colors"]
    fonts = plot["fonts"]
    dots = plot["dots"]
    lower_is_better = cfg["metric_directions"].get("lower_is_better") or []
    higher_is_better = cfg["metric_directions"].get("higher_is_better") or []

    cur_group = None
    group_separator_ys = []

    for row_idx, (group, metric) in enumerate(rows):
        y = -row_idx  # rows run top-to-bottom from y=0
        var_name = f"{group}.{metric}"

        if group != cur_group:
            # New variable group: draw the bold group label shifted left of the metric labels.
            ax.text(
                layout["metric_x"],
                y,
                group,
                ha="right",
                va="center",
                fontsize=fonts["group"],
                fontweight="bold",
                transform=group_transform,
            )
            if row_idx > 0:
                group_separator_ys.append(
                    y + 0.5
                )  # separator sits halfway between rows
            cur_group = group

        ax.text(
            layout["metric_x"],
            y,
            decode_metric(metric),
            ha="right",
            va="center",
            fontsize=fonts["metric"],
        )

        for sec_idx, slice_val in enumerate(slices):
            x_off = sec_idx * (n_leads + layout["slice_gap"])
            row_data = diff[var_name].sel({strat_dim: slice_val}).values

            for lt_idx, d in enumerate(row_data):
                x = x_off + lt_idx
                if np.isnan(d):
                    ax.plot(
                        x,
                        y,
                        "x",
                        color=colors["missing"],
                        ms=dots["missing_marker_size"],
                        mew=dots["missing_marker_lw"],
                    )
                    continue
                if abs(d) < dots["neutral_threshold_pct"]:
                    color, size = colors["neutral"], neutral_dot_size
                else:
                    better = _is_model_better(
                        d, metric, lower_is_better, higher_is_better
                    )
                    color = (
                        colors["model_better"] if better else colors["baseline_better"]
                    )
                    size = _scaled_dot_area(d, dots)
                ax.scatter(x, y, s=size, c=color, alpha=dots["alpha"], linewidths=0)

    return group_separator_ys


def _draw_slice_headers(ax, slices, n_leads, lead_hours, y_bottom, strat_dim, cfg):
    """Draw slice title labels, vertical separators, and lead-time tick labels."""
    plot = cfg["plot"]
    layout = plot["layout"]
    fonts = plot["fonts"]
    colors = plot["colors"]
    vline = plot["vline"]

    for sec_idx, slice_val in enumerate(slices):
        x_off = sec_idx * (n_leads + layout["slice_gap"])

        ax.text(
            x_off + (n_leads - 1) / 2,
            layout["slice_y"],
            _format_slice_label(slice_val, strat_dim),
            ha="center",
            va="bottom",
            fontsize=fonts["slice"],
            fontweight="bold",
        )

        if sec_idx > 0:
            # Vertical separator line centred in the gap between this and the previous slice block
            x_sep = x_off - (layout["slice_gap"] + 1) / 2
            ax.plot(
                [x_sep, x_sep],
                [y_bottom, layout["slice_y"] - vline["label_gap"]],
                color=colors["vline"],
                lw=vline["linewidth"],
            )

        for lt_idx, h in enumerate(lead_hours):
            ax.text(
                x_off + lt_idx,
                layout["leads_y"],
                f"{h}h",
                ha="center",
                va="bottom",
                fontsize=fonts["leads"],
                rotation=90,
                color=colors["leads"],
            )


def _draw_legend(
    ax,
    fig,
    dot_specs,
    x_dots,
    has_missing,
    small_fs,
    neutral_dot_size,
    model_source,
    baseline_source,
    cfg,
):
    """Draw the dot-size reference legend below the axes.

    The legend shows a row of example dots (from most-baseline-better on the left
    to most-model-better on the right) with percentage labels, plus an optional
    "no data" row if any NaN cells are present.
    """
    plot = cfg["plot"]
    legend = plot["legend"]
    dots = plot["dots"]
    fonts = plot["fonts"]
    colors = plot["colors"]

    # Each transform shifts the artist a fixed number of points below the axes bottom.
    dot_trans = ax.transAxes + ScaledTranslation(
        0, -legend["dot_below_pt"] / 72, fig.dpi_scale_trans
    )
    label_trans = ax.transAxes + ScaledTranslation(
        0, -legend["label_below_pt"] / 72, fig.dpi_scale_trans
    )
    miss_dot_trans = ax.transAxes + ScaledTranslation(
        0,
        -(legend["dot_below_pt"] + legend["missing_dot_offset_pt"]) / 72,
        fig.dpi_scale_trans,
    )
    miss_label_trans = ax.transAxes + ScaledTranslation(
        0,
        -(legend["label_below_pt"] + legend["missing_dot_offset_pt"]) / 72,
        fig.dpi_scale_trans,
    )

    for x, (diff_pct, color, label) in zip(x_dots, dot_specs):
        size = (
            neutral_dot_size
            if color == colors["neutral"]
            else _scaled_dot_area(diff_pct, dots)
        )
        ax.scatter(
            [x],
            [0],
            s=size,
            c=color,
            alpha=dots["alpha"],
            linewidths=0,
            transform=dot_trans,
            clip_on=False,
        )
        ax.text(
            x,
            0,
            label,
            ha="center",
            va="top",
            fontsize=small_fs,
            transform=label_trans,
            clip_on=False,
        )

    ax.text(
        x_dots[0] - legend["side_text_offset"],
        0,
        f"{baseline_source} better ←",
        ha="right",
        va="center",
        fontsize=fonts["legend"],
        transform=dot_trans,
        clip_on=False,
    )
    ax.text(
        x_dots[-1] + legend["side_text_offset"],
        0,
        f"→ {model_source} better",
        ha="left",
        va="center",
        fontsize=fonts["legend"],
        transform=dot_trans,
        clip_on=False,
    )

    if has_missing:
        x_neutral = x_dots[len(legend["sample_pcts"])]  # neutral dot is at the centre
        ax.plot(
            [x_neutral],
            [0],
            "x",
            color=colors["missing"],
            markersize=dots["missing_marker_size"],
            mew=dots["missing_marker_lw"],
            transform=miss_dot_trans,
            clip_on=False,
        )
        ax.text(
            x_neutral,
            0,
            "No data",
            ha="center",
            va="top",
            fontsize=small_fs,
            transform=miss_label_trans,
            clip_on=False,
        )


# ── Top-level render ──────────────────────────────────────────────────────────


def _render_scorecard(diff: xr.Dataset, cfg: dict, outfn: Path):
    """Compose and save the scorecard figure to *outfn*."""
    plot = cfg["plot"]
    figure = plot["figure"]
    layout = plot["layout"]
    hline = plot["hline"]
    legend = plot["legend"]
    dots = plot["dots"]
    fonts = plot["fonts"]
    model_source = cfg["model"]["label"]
    baseline_source = cfg["baseline"]["label"]
    strat_dim = cfg.get("stratification", "region")

    plt.rcParams["font.family"] = plot["rcparams"]["font_family"]
    plt.rcParams["font.sans-serif"] = [plot["rcparams"]["font_sans"]]
    plt.rcParams["figure.dpi"] = plot["rcparams"]["dpi"]

    # Decompose data-var names (e.g. "T_2M.RMSE") into (group, metric) pairs.
    rows = [tuple(v.rsplit(".", 1)) for v in diff.data_vars]
    slices = list(diff[strat_dim].values)
    n_leads = diff.sizes["step"]
    lead_hours = [_timedelta_to_hours(lt) for lt in diff.step.values]
    has_missing = any(np.isnan(diff[v].values).any() for v in diff.data_vars)

    slice_label_w_in, slice_label_h_rows, metric_label_w_pt = _measure_label_sizes(
        plot, rows, slices, strat_dim
    )
    # Group labels are drawn further left than metric labels by this many points.
    group_label_offset_pt = metric_label_w_pt + layout["group_metric_gap_pt"]

    # Column width must be at least wide enough for the slice header label.
    col_width = max(
        figure["col_width"],
        slice_label_w_in * layout["slice_label_pad"] / (n_leads + layout["slice_gap"]),
    )
    # Total data-coordinate width: n_slices blocks separated by slice_gap columns.
    plot_width = len(slices) * (n_leads + layout["slice_gap"]) - layout["slice_gap"]
    fig_width = max(figure["width_min"], plot_width * col_width + figure["width_pad"])

    # y=0 is the first row; rows extend downward, so y_bottom is negative.
    y_bottom = -(len(rows) - 0.5)
    y_top = layout["slice_y"] + slice_label_h_rows + layout["slice_y_pad"]

    small_fs = fonts["legend"] * legend["label_fontsize_factor"]
    # Legend height accounts for: label text (×1.4 for line height) + dot row + optional "no data" row.
    legend_h_in = (
        legend["label_below_pt"]
        + legend.get("missing_dot_offset_pt", 0) * has_missing
        + small_fs * 1.4
    ) / 72
    fig_height = (
        figure["title_margin_in"]
        + legend_h_in
        + figure["row_height"] * (y_top - y_bottom)
    )

    # Left x-limit: push far enough left to fit the label margin.
    xlim_left = layout["metric_x"] - figure["left_margin_in"] / col_width

    # Build the subtitle line showing which stratification dimensions are fixed.
    fixed_dims = {
        dim: val for dim, val in _STRAT_ALL_VALUES.items() if dim != strat_dim
    }
    title_parts = []
    if "region" in fixed_dims:
        title_parts.append(f"region={fixed_dims['region']}")
    if "season" in fixed_dims:
        title_parts.append(f"season={fixed_dims['season']}")
    if "init_hour" in fixed_dims:
        init_h = fixed_dims["init_hour"]
        title_parts.append(f"init={'all' if init_h == -999 else f'{init_h:02d}Z'}")

    neutral_dot_size = _scaled_dot_area(dots["neutral_threshold_pct"], dots)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(xlim_left, plot_width)
    ax.set_ylim(y_bottom, y_top)

    # Transform that shifts group labels a fixed number of points left of the metric labels.
    group_transform = ax.transData + ScaledTranslation(
        -group_label_offset_pt / 72, 0, fig.dpi_scale_trans
    )

    fig.text(
        0.01,
        0.99,
        f"{model_source} vs {baseline_source}"
        + (f"\n{'   '.join(title_parts)}" if title_parts else ""),
        fontsize=fonts["title"],
        fontweight="bold",
        ha="left",
        va="top",
        transform=fig.transFigure,
    )

    group_separator_ys = _draw_data_rows(
        ax,
        diff,
        rows,
        slices,
        strat_dim,
        n_leads,
        neutral_dot_size,
        group_transform,
        cfg,
    )
    _draw_slice_headers(ax, slices, n_leads, lead_hours, y_bottom, strat_dim, cfg)
    ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(
        top=1 - figure["title_margin_in"] / fig_height,
        bottom=legend_h_in / fig_height,
    )

    # Draw horizontal separators between variable groups after tight_layout so
    # ax.get_position() reflects the final axes extent.
    ax_w_in = ax.get_position().width * fig.get_figwidth()
    metric_label_frac = (layout["metric_x"] - xlim_left) / (plot_width - xlim_left)
    hline_x_start = (
        metric_label_frac - ((metric_label_w_pt - hline["start_pad_pt"]) / 72) / ax_w_in
    )
    for sep_y in group_separator_ys:
        ax.axhline(
            y=sep_y,
            xmin=hline_x_start,
            xmax=hline["x_end"],
            color=plot["colors"]["hline"],
            lw=hline["linewidth"],
        )

    # Build the legend dot specs: baseline-better on the left, model-better on the right.
    # The outermost entries use ≤/≥ to indicate they represent the size cap.
    sample_pcts = legend["sample_pcts"]
    neutral_pct = dots["neutral_threshold_pct"]
    colors = plot["colors"]
    dot_specs = (
        [(sample_pcts[0], colors["baseline_better"], f"≤-{sample_pcts[0]}%")]
        + [(p, colors["baseline_better"], f"-{p}%") for p in sample_pcts[1:]]
        + [(neutral_pct, colors["neutral"], f"|Δ|<{neutral_pct}%")]
        + [(p, colors["model_better"], f"+{p}%") for p in reversed(sample_pcts[1:])]
        + [(sample_pcts[0], colors["model_better"], f"≥+{sample_pcts[0]}%")]
    )

    # Centre the legend horizontally under the data area.
    x_span = min(legend["width_in"] / ax_w_in, 0.8)
    legend_center_x = ((plot_width - 1) / 2 - xlim_left) / (plot_width - xlim_left)
    x_dots = np.linspace(
        legend_center_x - x_span / 2, legend_center_x + x_span / 2, len(dot_specs)
    )

    _draw_legend(
        ax,
        fig,
        dot_specs,
        x_dots,
        has_missing,
        small_fs,
        neutral_dot_size,
        model_source,
        baseline_source,
        cfg,
    )

    outfn.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfn, bbox_inches="tight")
    print(f"saved: {outfn}")


# ── Entry point ───────────────────────────────────────────────────────────────


def main(args):
    """Run the full scorecard pipeline: load → diff → filter → render."""
    cfg = _build_config(args)
    diff = _load_relative_diff(cfg)
    diff_filtered = _filter_diff(diff, cfg)
    _render_scorecard(diff_filtered, cfg, Path(args.output))


if __name__ == "__main__":
    parser = ArgumentParser(description="Render a verification scorecard PNG.")
    parser.add_argument(
        "--verif_run",
        type=str,
        required=True,
        help="Path to the model verif_aggregated.nc file.",
    )
    parser.add_argument(
        "--verif_baseline",
        type=str,
        required=True,
        help="Path to the baseline verif_aggregated.nc file.",
    )
    parser.add_argument(
        "--run_source",
        type=str,
        required=True,
        help="Value of the 'source' dim to select inside --verif_run.",
    )
    parser.add_argument(
        "--baseline_source",
        type=str,
        required=True,
        help="Value of the 'source' dim to select inside --verif_baseline.",
    )
    parser.add_argument(
        "--run_label",
        type=str,
        default=None,
        help="Human-readable label for the model run (used in plot titles/legend). Defaults to --run_source.",
    )
    parser.add_argument(
        "--baseline_label",
        type=str,
        default=None,
        help="Human-readable label for the baseline (used in plot titles/legend). Defaults to --baseline_source.",
    )
    parser.add_argument(
        "--lead_times",
        type=str,
        default="6/33/6",
        help="Lead-time grid 'start/stop/step' in hours (default: 6/33/6).",
    )
    parser.add_argument(
        "--stratification",
        type=str,
        default="region",
        help="Dimension name to use as scorecard columns (default: region).",
    )
    parser.add_argument(
        "--variable",
        action="append",
        default=None,
        help=(
            "Variable + optional metric subset, format 'VAR:M1,M2'. "
            "Repeat once per variable. Omit ':...' to use all metrics for that variable. "
            "If omitted entirely, falls back to a minimal RMSE-only set."
        ),
    )
    parser.add_argument("--output", type=str, required=True, help="Output PNG path.")

    args = parser.parse_args()
    main(args)
