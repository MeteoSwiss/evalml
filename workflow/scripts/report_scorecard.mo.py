import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    from argparse import ArgumentParser
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import xarray as xr

    import matplotlib.pyplot as plt
    from matplotlib.transforms import ScaledTranslation

    from verification import decode_metric

    return ArgumentParser, Path, ScaledTranslation, decode_metric, np, pd, plt, xr


@app.cell
def config(ArgumentParser, Path):
    """Build config from CLI args plus stable plotting defaults."""

    def _parse_var_metrics(spec: str):
        """Parse a 'VAR:M1,M2,...' (or 'VAR' alone) item into (var, [metrics])."""
        if ":" in spec:
            var, metrics = spec.split(":", 1)
            return var.strip(), [m.strip() for m in metrics.split(",") if m.strip()]
        return spec.strip(), None  # None → use all_metrics for this variable

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
            "Repeat once per variable. Omit ':...' to use all metrics. "
            "If not given at all, falls back to a minimal RMSE-only set."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output PNG path.",
    )

    args = parser.parse_args()

    if args.variable:
        vars_metrics = dict(_parse_var_metrics(s) for s in args.variable)
    else:
        vars_metrics = {
            "U_10M": ["RMSE"],
            "V_10M": ["RMSE"],
            "T_2M": ["RMSE"],
            "PMSL": ["RMSE"],
            "TD_2M": ["RMSE"],
            "TOT_PREC": ["RMSE"],
        }

    outfn = Path(args.output)

    cfg = {
        "model": {"path": args.verif_run, "source": args.run_source},
        "baseline": {"path": args.verif_baseline, "source": args.baseline_source},
        "stratification": args.stratification,
        "lead_times": args.lead_times,
        "all_metrics": [
            "RMSE",
            "MAE",
            "STDE",
            "R2",
            "ETS",
            "POD",
            "FAR",
        ],  # every entry must appear in metric_directions
        "metric_directions": {
            "lower_is_better": ["RMSE", "MAE", "STDE", "FAR"],
            "higher_is_better": ["R2", "ETS", "POD"],
        },
        "variables": vars_metrics,  # null entry → use all_metrics; list → restrict to that subset
        "plot": {
            "rcparams": {
                "font_family": "sans-serif",
                "font_sans": "Liberation Sans",
                "dpi": 250,
            },
            "colors": {
                "model_better": "#4878d0",  # seaborn muted blue
                "baseline_better": "#d65f5f",  # seaborn muted red
                "neutral": "#cccccc",  # |diff|% below dots.neutral_threshold_pct
                "missing": "#999999",  # NaN / no data marker
                "leads": "#666666",  # lead-time tick labels
                "hline": "#cccccc",  # horizontal separators between variable groups
                "vline": "black",  # vertical separators between slices
            },
            "fonts": {  # font sizes in points
                "title": 24,
                "group": 20,  # variable group labels (U_10M, T_2M, ...)
                "metric": 18,  # metric labels         (RMSE, MAE, ...)
                "slice": 20,  # slice header labels; increasing this auto-widens the columns
                "leads": 16,
                "legend": 18,
            },
            "dots": {
                "neutral_threshold_pct": 5,  # |diff|% below which the dot is drawn neutral grey
                "size_cap_pct": 30,  # |diff|% at which the dot reaches max area
                "max_area": 250,  # max dot area (matplotlib scatter `s` units)
                "alpha": 0.9,  # dot transparency
                "missing_marker_size": 8,  # size of the "x" marker drawn for NaN values
                "missing_marker_lw": 1.5,  # line width of the "x" marker
            },
            "figure": {  # sizing rules (inches)
                "col_width": 0.26,  # minimum width per lead-time column; actual width grows automatically if slice labels need more space
                "width_min": 5,  # minimum total figure width
                "width_pad": 5,  # extra width for left-side labels (group + metric)
                "row_height": 0.42,  # height per metric row
                "height_pad": 2.5,  # extra height for title, legend, margins
                "title_margin_in": 1.5,  # fixed space above axes for the title (independent of figure height)
            },
            "layout": {  # data-coordinate offsets controlling spacing
                "slice_gap": 1.5,  # empty columns between adjacent slice blocks
                "metric_x": -1.5,  # x position of metric labels (data coords)
                "group_metric_gap_pt": 15,  # physical gap (points) between the longest metric label and the group label
                "slice_label_pad": 1.05,  # factor applied to the longest slice label width when widening columns to fit it
                "slice_y": 1.9,  # y position of slice title labels (text baseline)
                "slice_y_pad": 0.2,  # extra y above the measured slice label height for the top ylim
                "leads_y": 0.45,  # y position of the lead-time tick labels
            },
            "hline": {  # horizontal separators between variable groups
                "start_pad_pt": 0,  # offset (points) from the measured left edge of the metric label to the hline start
                "x_end": 1.0,  # axes fraction where the line ends
                "linewidth": 0.7,
            },
            "vline": {  # vertical separators between slices
                "label_gap": 0.4,  # gap (metric rows) between the slice label bottom and the top of the separator line
                "linewidth": 0.8,
            },
            "legend": {
                "width_in": 6.5,  # physical width (inches) of the dot row
                "dot_below_pt": 40,  # fixed gap below axes bottom to dot centres (points); keep in sync with label_below_pt
                "label_below_pt": 54,  # fixed gap below axes bottom to labels (points)
                "sample_pcts": [
                    30,
                    15,
                    5,
                ],  # |diff|% values shown on each side of neutral
                "label_fontsize_factor": 0.85,  # factor applied to fonts.legend for the small labels
                "side_text_offset": 0.03,  # gap (axes fraction) between outer dot and side label
                "missing_dot_offset_pt": 50,  # extra vertical gap (points) below the main legend row for the "no data" example
            },
        },
    }
    return cfg, outfn


@app.cell
def align_and_diff(cfg, np, xr):
    """Open + select model and baseline, then compute (model − baseline) / |baseline| in %."""

    all_metrics = cfg["all_metrics"]
    lower_is_better = cfg["metric_directions"].get("lower_is_better") or []
    higher_is_better = cfg["metric_directions"].get("higher_is_better") or []

    model_source = cfg["model"]["source"]
    baseline_source = cfg["baseline"]["source"]

    # Build selection dict: collapse all non-stratification dims to their "all"/aggregated value.
    _ALL = {"region": "all", "season": "all", "init_hour": -999}
    _strat = cfg["stratification"]
    _sel = {"source": model_source}
    for dim, all_val in _ALL.items():
        if dim != _strat:
            _sel[dim] = all_val

    # squeeze(drop=True) collapses singleton dims left over from .sel().
    model = xr.open_dataset(cfg["model"]["path"]).sel(**_sel).squeeze(drop=True)
    baseline = (
        xr.open_dataset(cfg["baseline"]["path"])
        .sel(**{**_sel, "source": baseline_source})
        .squeeze(drop=True)
    )

    common_vars = [v for v in model.data_vars if v in baseline.data_vars]
    common_leads = sorted(set(model.lead_time.values) & set(baseline.lead_time.values))

    if not common_vars:
        raise ValueError(
            "No variables in common between model and baseline.\n"
            f"  model:    {sorted({v.split('.')[0] for v in model.data_vars})}\n"
            f"  baseline: {sorted({v.split('.')[0] for v in baseline.data_vars})}"
        )
    if not common_leads:
        raise ValueError("No lead times in common between model and baseline.")

    m = model[common_vars].sel(lead_time=common_leads)
    b = baseline[common_vars].sel(lead_time=common_leads)

    # Relative difference in percent; baseline=0 → ±inf masked to NaN
    # so the plot renders those cells as a grey "x".
    diff = (m - b) / abs(b) * 100
    diff = diff.where(np.isfinite(diff))

    model.close()
    baseline.close()
    return (
        all_metrics,
        baseline_source,
        diff,
        higher_is_better,
        lower_is_better,
        model_source,
    )


@app.cell
def filter_diff(all_metrics, cfg, diff, pd):
    """Filter diff by slices, lead-time grid and variable/metric selection."""

    def to_h(td):
        return int(pd.Timedelta(td).total_seconds() / 3600)

    diff_filtered = diff

    if cfg.get("lead_times"):
        start, stop, step = (int(x) for x in cfg["lead_times"].split("/"))
        requested = {pd.Timedelta(h, "h") for h in range(start, stop + 1, step)}
        _available = {pd.Timedelta(lt) for lt in diff_filtered.lead_time.values}
        keep = sorted(requested & _available)
        if not keep:
            raise ValueError(
                f"No lead times match '{cfg['lead_times']}'. "
                f"Available (h): {sorted(to_h(lt) for lt in _available)}"
            )
        diff_filtered = diff_filtered.sel(lead_time=keep)

    def _resolve_metric(var_name, m, data_vars):
        """Resolve one metric name to matching data_var names.

        Handles two cases:
        - exact match:      "RMSE" → ["U_10M.RMSE"]
        - prefix expansion: "ETS"  → all "VAR.ETS_*" in the dataset
        """
        exact = f"{var_name}.{m}"
        if exact in data_vars:
            return [exact]
        return [v for v in data_vars if v.startswith(f"{var_name}.{m}_")]

    # Order follows cfg["variables"] for groups; within each group, follows all_metrics.
    if cfg.get("variables"):
        keep = [
            v
            for _var_name, _var_metrics in cfg["variables"].items()
            for _m in (_var_metrics or all_metrics)
            for v in _resolve_metric(_var_name, _m, diff_filtered.data_vars)
        ]
        diff_filtered = diff_filtered[keep]
    else:
        diff_filtered = diff_filtered[
            [
                v
                for v in diff_filtered.data_vars
                if any(
                    s == m or s.startswith(f"{m}_")
                    for m in all_metrics
                    for s in [v.split(".")[-1]]
                )
            ]
        ]

    if not diff_filtered.data_vars:
        raise ValueError("No variables left after filtering.")
    return diff_filtered, to_h


@app.cell
def plot_scorecard(
    ScaledTranslation,
    baseline_source,
    decode_metric,
    cfg,
    diff_filtered,
    higher_is_better,
    lower_is_better,
    model_source,
    np,
    outfn,
    plt,
    to_h,
):
    """Render the scorecard PNG; colours, fonts, dot sizes and legend all read from cfg['plot']."""
    plot = cfg["plot"]
    colors = plot["colors"]
    fonts = plot["fonts"]
    dots = plot["dots"]
    figure = plot["figure"]
    layout = plot["layout"]
    hline = plot["hline"]
    vline = plot["vline"]
    legend = plot["legend"]

    plt.rcParams["font.family"] = plot["rcparams"]["font_family"]
    plt.rcParams["font.sans-serif"] = [plot["rcparams"]["font_sans"]]
    plt.rcParams["figure.dpi"] = plot["rcparams"]["dpi"]

    def is_model_better(d, metric):
        """True if a positive `d` (signed diff %) means the model beats the baseline."""
        if any(metric == m or metric.startswith(f"{m}_") for m in higher_is_better):
            return d > 0
        if any(metric == m or metric.startswith(f"{m}_") for m in lower_is_better):
            return d < 0
        return None

    def dot_size(d):
        """Linear ramp from 0 to dots.max_area, capped at dots.size_cap_pct."""
        return float(
            min(abs(d) / dots["size_cap_pct"] * dots["max_area"], dots["max_area"])
        )

    # ── Figure size and layout precomputation ────────────────────────────────────
    rows = [tuple(v.rsplit(".", 1)) for v in diff_filtered.data_vars]
    strat_dim = cfg.get("stratification", "region")
    _slices = list(diff_filtered[strat_dim].values)
    n_leads = diff_filtered.sizes["lead_time"]
    lead_hours = [to_h(lt) for lt in diff_filtered.lead_time.values]

    def _fmt_slice(s):
        if isinstance(s, str):
            return s.upper() if strat_dim == "season" else s.capitalize()
        return "All" if int(s) == -999 else f"{int(s):02d}Z"

    # Measure the longest slice label on a throwaway figure so col_width can grow to prevent adjacent headers from overlapping;
    # its height (in metric rows) also fixes the top ylim and the top of the vertical separators.
    _longest_slice = max((_fmt_slice(s) for s in _slices), key=len)
    _fig_tmp, _ax_tmp = plt.subplots(dpi=plot["rcparams"]["dpi"])
    _t = _ax_tmp.text(
        0,
        0,
        _longest_slice,
        fontsize=fonts["slice"],
        fontweight="bold",
        fontfamily=plot["rcparams"]["font_sans"],
    )
    _fig_tmp.canvas.draw()
    _slice_bbox = _t.get_window_extent(_fig_tmp.canvas.get_renderer())
    _text_w_in = _slice_bbox.width / _fig_tmp.dpi
    _slice_h_rows = _slice_bbox.height / _fig_tmp.dpi / figure["row_height"]
    plt.close(_fig_tmp)

    # Measure the longest metric label on a throwaway figure so the group labels can sit just left of it without overlapping.
    _longest_metric = max((decode_metric(m) for _, m in rows), key=len)
    _fig_tmp2, _ax_tmp2 = plt.subplots(dpi=plot["rcparams"]["dpi"])
    _t2 = _ax_tmp2.text(
        0,
        0,
        _longest_metric,
        fontsize=fonts["metric"],
        fontfamily=plot["rcparams"]["font_sans"],
    )
    _fig_tmp2.canvas.draw()
    _metric_w_pt = (
        _t2.get_window_extent(_fig_tmp2.canvas.get_renderer()).width
        * 72
        / _fig_tmp2.dpi
    )
    plt.close(_fig_tmp2)
    group_offset_pt = _metric_w_pt + layout["group_metric_gap_pt"]

    col_width = max(
        figure["col_width"],
        _text_w_in * layout["slice_label_pad"] / (n_leads + layout["slice_gap"]),
    )
    plot_width = len(_slices) * (n_leads + layout["slice_gap"]) - layout["slice_gap"]
    fig_width = max(figure["width_min"], plot_width * col_width + figure["width_pad"])
    # Last row is at y=-(len(rows)-1); add 0.5 units of clearance below it.
    y_bottom = -(len(rows) - 0.5)
    y_top = layout["slice_y"] + _slice_h_rows + layout["slice_y_pad"]
    has_missing = any(
        np.isnan(diff_filtered[v].values).any() for v in diff_filtered.data_vars
    )
    small_fs = fonts["legend"] * legend["label_fontsize_factor"]
    _legend_h_in = (
        legend["label_below_pt"]
        + legend.get("missing_dot_offset_pt", 0) * has_missing
        + small_fs * 1.4
    ) / 72
    _overhead_in = figure["title_margin_in"] + _legend_h_in
    fig_height = _overhead_in + figure["row_height"] * (y_top - y_bottom)

    # Left margin fixed in inches so content does not shift right when col_width is large.
    _left_margin_col = figure.get("left_margin_in", 2.5) / col_width
    xlim_left = layout["metric_x"] - _left_margin_col

    _ALL = {"region": "all", "season": "all", "init_hour": -999}
    _strat = cfg["stratification"]
    _fixed = {dim: val for dim, val in _ALL.items() if dim != _strat}
    _init_raw = _fixed.get("init_hour")
    _title_parts = []
    if "season" in _fixed:
        _title_parts.append(f"season={_fixed['season']}")
    if _init_raw is not None:
        _title_parts.append(
            f"init={'all' if _init_raw == -999 else f'{_init_raw:02d}Z'}"
        )
    neutral_size = dot_size(dots["neutral_threshold_pct"])

    # ── Figure ───────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # xlim/ylim are pinned here because data→axes-fraction conversions below
    # depend on these bounds; changing them later would shift those positions.
    ax.set_xlim(xlim_left, plot_width)
    ax.set_ylim(y_bottom, y_top)

    # Group labels: same x as metric labels, shifted left by the measured metric label width
    # so the group label never overlaps with metric labels regardless of figure width.
    group_transform = ax.transData + ScaledTranslation(
        -group_offset_pt / 72,
        0,
        fig.dpi_scale_trans,
    )

    # Title in figure coords so it is unaffected by axes margin changes.
    fig.text(
        0.01,
        0.99,
        f"{model_source} vs {baseline_source}"
        + (f"\n{'   '.join(_title_parts)}" if _title_parts else ""),
        fontsize=fonts["title"],
        fontweight="bold",
        ha="left",
        va="top",
        transform=fig.transFigure,
    )

    # ── Dots and row labels ──────────────────────────────────────────────────────
    # y is negated so row 0 sits at the top.
    cur_group = None
    group_separator_ys = []
    for row_idx, (group, metric) in enumerate(rows):
        y = -row_idx
        name = f"{group}.{metric}"

        if group != cur_group:
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
                group_separator_ys.append(y + 0.5)
            cur_group = group

        ax.text(
            layout["metric_x"],
            y,
            decode_metric(metric),
            ha="right",
            va="center",
            fontsize=fonts["metric"],
        )

        for sec_idx, _slice in enumerate(_slices):
            x_off = sec_idx * (n_leads + layout["slice_gap"])
            for lt_idx, d in enumerate(
                diff_filtered[name].sel({strat_dim: _slice}).values
            ):
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
                    color, size = colors["neutral"], neutral_size
                else:
                    color = (
                        colors["model_better"]
                        if is_model_better(d, metric)
                        else colors["baseline_better"]
                    )
                    size = dot_size(d)
                ax.scatter(x, y, s=size, c=color, alpha=dots["alpha"], linewidths=0)

    # ── Slice headers, lead-time ticks, vertical separators ─────────────────────
    for sec_idx, _slice in enumerate(_slices):
        x_off = sec_idx * (n_leads + layout["slice_gap"])

        ax.text(
            x_off + (n_leads - 1) / 2,
            layout["slice_y"],
            _fmt_slice(_slice),
            ha="center",
            va="bottom",
            fontsize=fonts["slice"],
            fontweight="bold",
        )

        if sec_idx > 0:
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

    ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(
        top=1 - figure["title_margin_in"] / fig_height,
        bottom=_legend_h_in / fig_height,
    )

    # Horizontal group separators start at the left edge of the longest metric label and run to the right edge of the axes.
    # axhline's xmin is an axes fraction, so compute it here, after layout, when the axes width is final.
    metric_frac = (layout["metric_x"] - xlim_left) / (plot_width - xlim_left)
    ax_w_in = ax.get_position().width * fig.get_figwidth()
    hline_x_start = (
        metric_frac - ((_metric_w_pt - hline["start_pad_pt"]) / 72) / ax_w_in
    )
    for _sep_y in group_separator_ys:
        ax.axhline(
            y=_sep_y,
            xmin=hline_x_start,
            xmax=hline["x_end"],
            color=colors["hline"],
            lw=hline["linewidth"],
        )

    # ── Legend ───────────────────────────────────────────────────────────────────
    sample_pcts = legend["sample_pcts"]
    neutral_pct = dots["neutral_threshold_pct"]

    dot_specs = (
        [(p, colors["baseline_better"], f"-{p}%") for p in sample_pcts]
        + [(neutral_pct, colors["neutral"], f"|Δ|<{neutral_pct}%")]
        + [(p, colors["model_better"], f"+{p}%") for p in reversed(sample_pcts)]
    )
    dot_specs[0] = (sample_pcts[0], colors["baseline_better"], f"≤-{sample_pcts[0]}%")
    dot_specs[-1] = (sample_pcts[0], colors["model_better"], f"≥+{sample_pcts[0]}%")

    # Legend dot row uses axes-fraction (stable across plot widths), is centred on the dot grid,
    # and is capped at 0.8 width to leave room for the side labels.
    x_span = min(legend["width_in"] / ax_w_in, 0.8)
    _data_ctr = ((plot_width - 1) / 2 - xlim_left) / (plot_width - xlim_left)
    x_dots = np.linspace(_data_ctr - x_span / 2, _data_ctr + x_span / 2, len(dot_specs))

    # Anchor at a fixed physical offset below the axes so spacing is independent
    # of figure height.
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

    # clip_on=False because the legend sits below the axes box.
    for x, (_val, col, lbl) in zip(x_dots, dot_specs):
        s = neutral_size if col == colors["neutral"] else dot_size(_val)
        ax.scatter(
            [x],
            [0],
            s=s,
            c=col,
            alpha=dots["alpha"],
            linewidths=0,
            transform=dot_trans,
            clip_on=False,
        )
        ax.text(
            x,
            0,
            lbl,
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
        x_neutral = x_dots[len(sample_pcts)]
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

    outfn.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfn, bbox_inches="tight")
    print(f"saved: {outfn}")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
