import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    from argparse import ArgumentParser
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import xarray as xr

    import matplotlib.pyplot as plt
    from matplotlib.transforms import ScaledTranslation

    return ArgumentParser, Path, ScaledTranslation, np, os, pd, plt, xr


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
        "--regions",
        type=str,
        nargs="+",
        default=[
            "all",
            "mittelland",
            "berge",
            "alpennordseite",
            "alpensuedseite",
            "jura",
        ],
        help="Space-separated list of regions to include.",
    )
    parser.add_argument(
        "--variable",
        action="append",
        default=None,
        help=(
            "Variable + optional metric subset, format 'VAR:M1,M2'. "
            "Repeat once per variable. Omit ':...' to use all metrics. "
            "If not given at all, falls back to a default 6-variable set."
        ),
    )
    parser.add_argument(
        "--season",
        type=str,
        default="all",
        help="Value of the 'season' dim to select (default: all).",
    )
    parser.add_argument(
        "--init_hour",
        type=int,
        default=-999,
        help="Value of the 'init_hour' dim (default: -999 = aggregated).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output PNG path.",
    )

    args = parser.parse_args()

    # Fall back to a default 6-variable set when --variable is not given.
    if args.variable:
        vars_metrics = dict(_parse_var_metrics(s) for s in args.variable)
    else:
        vars_metrics = {
            "U_10M": ["RMSE", "MAE", "STDE", "CORR", "R2"],
            "V_10M": ["RMSE", "MAE", "STDE", "CORR", "R2"],
            "T_2M": ["RMSE", "MAE", "STDE", "CORR", "R2"],
            "PMSL": ["RMSE", "MAE", "STDE", "CORR", "R2"],
            "TD_2M": ["RMSE", "MAE", "STDE", "CORR", "R2"],
            "TOT_PREC": ["RMSE", "MAE", "STDE", "CORR", "R2"],
        }

    outfn = Path(args.output)

    cfg = {
        "model": {"path": args.verif_run, "source": args.run_source},
        "baseline": {"path": args.verif_baseline, "source": args.baseline_source},
        "season": args.season,
        "init_hour": args.init_hour,
        "regions": args.regions,
        "lead_times": args.lead_times,
        "all_metrics": [
            "RMSE",
            "MAE",
            "STDE",
            "CORR",
            "R2",
        ],  # every entry must appear in metric_directions
        "metric_directions": {
            "lower_is_better": ["RMSE", "MAE", "STDE"],
            "higher_is_better": ["CORR", "R2"],
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
                "vline": "black",  # vertical separators between regions
            },
            "fonts": {  # font sizes in points
                "title": 24,
                "group": 20,  # variable group labels (U_10M, T_2M, ...)
                "metric": 18,  # metric labels         (RMSE, MAE, ...)
                "region": 20,  # region header labels; increasing this auto-widens the columns
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
                "col_width": 0.26,  # minimum width per lead-time column; actual width grows automatically if region labels need more space
                "width_min": 5,  # minimum total figure width
                "width_pad": 5,  # extra width for left-side labels (group + metric)
                "row_height": 0.42,  # height per metric row
                "height_pad": 2.5,  # extra height for title, legend, margins
                "title_margin_in": 1.5,  # fixed space above axes for the title (independent of figure height)
            },
            "layout": {  # data-coordinate offsets controlling spacing
                "region_gap": 1.5,  # empty columns between adjacent region blocks
                "metric_x": -1.5,  # x position of metric labels (data coords)
                "group_offset_pt": 75,  # physical gap (points) from metric label to group label
                "region_y": 1.9,  # y position of region title labels
                "region_y_pad": 0.8,  # extra y above region_y for the top ylim
                "leads_y": 0.35,  # y position of the lead-time tick labels
            },
            "hline": {  # horizontal separators between variable groups
                "gap_pt": -42,  # physical gap (points) from metric label to line start
                "x_end": 1.0,  # axes fraction where the line ends
                "linewidth": 0.7,
            },
            "vline": {  # vertical separators between regions
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
                "side_text_offset": 0.02,  # gap (axes fraction) between outer dot and side label
                "missing_dot_offset_pt": 16,  # extra vertical gap for the "no data" example (points)
            },
        },
    }

    cfg
    return cfg, outfn


# todo: shorten or remove this part & other validations
@app.cell
def validate_config(cfg, os, xr):
    """Validate config keys, metric directions, lead-time grid and dataset selections; fail fast."""
    errors = []

    # 1) Required top-level keys.
    REQUIRED_TOP_KEYS = (
        "model",
        "baseline",
        "season",
        "init_hour",
        "all_metrics",
        "metric_directions",
        "plot",
    )
    for _key in REQUIRED_TOP_KEYS:
        if _key not in cfg:
            errors.append(f"Missing required key: '{_key}'")

    # 2) model/baseline must each have `path` and `source`.
    for role in ("model", "baseline"):
        if role in cfg:
            for sub in ("path", "source"):
                if sub not in (cfg[role] or {}):
                    errors.append(f"cfg['{role}'] is missing '{sub}'")

    # 3) metric_directions must partition all_metrics (no overlap, no gap).
    if "metric_directions" in cfg and "all_metrics" in cfg:
        md_cfg = cfg["metric_directions"] or {}
        lb = md_cfg.get("lower_is_better")
        hb = md_cfg.get("higher_is_better")
        if lb is None:
            errors.append(
                "metric_directions.lower_is_better is missing or null (use [] for empty)"
            )
        elif not isinstance(lb, list):
            errors.append(
                f"metric_directions.lower_is_better must be a list, got: {type(lb).__name__}"
            )
        if hb is None:
            errors.append(
                "metric_directions.higher_is_better is missing or null (use [] for empty)"
            )
        elif not isinstance(hb, list):
            errors.append(
                f"metric_directions.higher_is_better must be a list, got: {type(hb).__name__}"
            )
        if isinstance(lb, list) and isinstance(hb, list):
            lb_set, hb_set = set(lb), set(hb)
            all_set = set(cfg["all_metrics"] or [])
            overlap = lb_set & hb_set
            _missing = all_set - (lb_set | hb_set)
            extra = (lb_set | hb_set) - all_set
            if overlap:
                errors.append(
                    f"Metrics in both lower_is_better and higher_is_better: {sorted(overlap)}"
                )
            if _missing:
                errors.append(
                    f"Metrics in all_metrics with no direction assigned: {sorted(_missing)}"
                )
            if extra:
                errors.append(
                    f"Metrics in metric_directions but not in all_metrics: {sorted(extra)}"
                )

    # 4) Per-variable metric overrides must be a subset of all_metrics.
    if "all_metrics" in cfg and cfg.get("variables"):
        all_m = set(cfg["all_metrics"] or [])
        for _var_name, _var_metrics in cfg["variables"].items():
            for _m in _var_metrics or []:
                if _m not in all_m:
                    errors.append(
                        f"variables.{_var_name}: metric '{_m}' is not in all_metrics"
                    )

    # 5) lead_times format: "start/stop/step" with integers and step > 0.
    if cfg.get("lead_times"):
        try:
            parts = str(cfg["lead_times"]).split("/")
            assert len(parts) == 3, "must have exactly 3 parts"
            lt_start, lt_stop, lt_step = int(parts[0]), int(parts[1]), int(parts[2])
            if lt_step <= 0:
                errors.append(f"lead_times step must be > 0, got {lt_step}")
            if lt_stop < lt_start:
                errors.append(
                    f"lead_times stop ({lt_stop}) must be >= start ({lt_start})"
                )
        except (ValueError, AssertionError) as e:
            errors.append(
                f"lead_times must be 'start/stop/step' integers, got: '{cfg['lead_times']}' ({e})"
            )

    # 6) Files exist; source/season/init_hour/regions are valid in each dataset.
    if not errors:
        for role in ("model", "baseline"):
            path = cfg[role]["path"]
            source = cfg[role]["source"]
            if not os.path.exists(path):
                errors.append(f"{role} file not found:\n    {path}")
                continue
            ds_meta = xr.open_dataset(path)

            # Check that each requested scalar selection exists in the dataset.
            for dim, _val in [
                ("source", source),
                ("season", cfg["season"]),
                ("init_hour", cfg["init_hour"]),
            ]:
                if dim in ds_meta.dims and _val not in ds_meta[dim].values:
                    errors.append(
                        f"{role}: '{_val}' not found in dim '{dim}'.\n"
                        f"    Available: {list(ds_meta[dim].values)}"
                    )

            # Regions are a list, so check each element individually.
            if cfg.get("regions"):
                if "region" not in ds_meta.dims:
                    errors.append(
                        f"{role}: 'regions' specified in config but dataset has no 'region' dimension"
                    )
                else:
                    _available = list(ds_meta["region"].values)
                    _unknown = [r for r in cfg["regions"] if r not in _available]
                    if _unknown:
                        errors.append(
                            f"{role}: region(s) {_unknown} not found.\n    Available: {_available}"
                        )

            ds_meta.close()

    # 7) Variables in config must exist in the model dataset.
    if not errors and cfg.get("variables"):
        ds_check = xr.open_dataset(cfg["model"]["path"])
        prefixes = {v.split(".")[0] for v in ds_check.data_vars}
        ds_check.close()
        for _var_name in cfg["variables"]:
            if _var_name not in prefixes:
                errors.append(f"Variable '{_var_name}' not found in model dataset")

    # 8) Plot section: required subsections must exist.
    if "plot" in cfg:
        required_plot_subkeys = (
            "rcparams",
            "colors",
            "fonts",
            "dots",
            "figure",
            "layout",
            "hline",
            "vline",
            "legend",
        )
        for k in required_plot_subkeys:
            if k not in (cfg["plot"] or {}):
                errors.append(f"plot.{k} is missing")

    # 9) Report.
    if errors:
        for e in errors:
            print(f"❌  {e}")
        raise ValueError(
            f"Config validation failed ({len(errors)} error(s)). Fix the config and re-run."
        )
    print("✅  Config OK")
    return


@app.cell
def align_and_diff(cfg, np, xr):
    """Open + select model and baseline, then compute (model − baseline) / |baseline| in %."""

    all_metrics = cfg["all_metrics"]
    lower_is_better = cfg["metric_directions"].get("lower_is_better") or []
    higher_is_better = cfg["metric_directions"].get("higher_is_better") or []

    model_source = cfg["model"]["source"]
    baseline_source = cfg["baseline"]["source"]

    # squeeze(drop=True) collapses singleton dims left over from .sel().
    model = (
        xr.open_dataset(cfg["model"]["path"])
        .sel(
            source=model_source,
            season=cfg["season"],
            init_hour=cfg["init_hour"],
        )
        .squeeze(drop=True)
    )

    baseline = (
        xr.open_dataset(cfg["baseline"]["path"])
        .sel(
            source=baseline_source,
            season=cfg["season"],
            init_hour=cfg["init_hour"],
        )
        .squeeze(drop=True)
    )

    # Restrict to the variables and lead times present in both datasets.
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
    diff
    return (
        all_metrics,
        b,
        baseline_source,
        diff,
        higher_is_better,
        lower_is_better,
        m,
        model_source,
    )


@app.cell
def filter_diff(all_metrics, b, cfg, diff, m, pd):
    """Filter diff by regions, lead-time grid and variable/metric selection."""

    def to_h(td):
        """Convert timedelta64[ns] into integer hours. Raises if not a whole number of hours."""
        h = pd.Timedelta(td).total_seconds() / 3600
        if h != int(h):
            raise ValueError(f"Lead time {td} is not a whole number of hours ({h}h)")
        return int(h)

    model_leads_h = sorted(to_h(lt) for lt in m.lead_time.values)
    baseline_leads_h = sorted(to_h(lt) for lt in b.lead_time.values)

    # ── Regions ──────────────────────────────────────────────────────────────────
    diff_filtered = diff
    if cfg.get("regions"):
        _available = list(diff_filtered.region.values)
        _unknown = [r for r in cfg["regions"] if r not in _available]
        if _unknown:
            raise ValueError(
                f"Unknown region(s): {_unknown}\n  Available: {_available}"
            )
        diff_filtered = diff_filtered.sel(region=cfg["regions"])

    # ── Lead times ───────────────────────────────────────────────────────────────
    if cfg.get("lead_times"):
        start, stop, step = (int(x) for x in cfg["lead_times"].split("/"))
        requested = {pd.Timedelta(h, "h") for h in range(start, stop + 1, step)}
        _available = {pd.Timedelta(lt) for lt in diff_filtered.lead_time.values}
        keep = sorted(requested & _available)
        _missing = sorted(requested - _available)

        if _missing:
            print(
                f"⚠️  Lead times not in dataset (skipped): {[to_h(lt) for lt in _missing]}h"
            )
            print(f"    model    (h): {model_leads_h}")
            print(f"    baseline (h): {baseline_leads_h}")
        if not keep:
            raise ValueError(
                f"No lead times match '{cfg['lead_times']}'.\n"
                f"  model    (h): {model_leads_h}\n"
                f"  baseline (h): {baseline_leads_h}"
            )
        diff_filtered = diff_filtered.sel(lead_time=keep)

    # ── Variables and metrics ────────────────────────────────────────────────────
    # Order follows cfg["variables"] for groups; within each group, follows all_metrics.
    if cfg.get("variables"):
        keep = []
        for _var_name, _var_metrics in cfg["variables"].items():
            allowed = _var_metrics or all_metrics
            for _m in allowed:
                _key = f"{_var_name}.{_m}"
                if _key in diff_filtered.data_vars:
                    keep.append(_key)
                else:
                    print(f"⚠️  {_key} not found in dataset, skipped")
        diff_filtered = diff_filtered[keep]
    else:
        diff_filtered = diff_filtered[
            [v for v in diff_filtered.data_vars if v.split(".")[-1] in all_metrics]
        ]

    if not diff_filtered.data_vars:
        raise ValueError(
            "No variables left after filtering. "
            "Check that variables/metrics in the config exist in both datasets."
        )

    diff_filtered
    return diff_filtered, to_h


@app.cell
def plot_scorecard(
    ScaledTranslation,
    baseline_source,
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
        if metric in higher_is_better:
            return d > 0
        if metric in lower_is_better:
            return d < 0
        return None

    def dot_size(d):
        """Linear ramp from 0 to dots.max_area, capped at dots.size_cap_pct."""
        return float(
            min(abs(d) / dots["size_cap_pct"] * dots["max_area"], dots["max_area"])
        )

    # ── Figure size and layout precomputation ────────────────────────────────────
    # One row per (variable, metric), one column per lead time.
    rows = [tuple(v.rsplit(".", 1)) for v in diff_filtered.data_vars]
    regions_ = list(diff_filtered.region.values)
    n_leads = diff_filtered.sizes["lead_time"]
    lead_hours = [to_h(lt) for lt in diff_filtered.lead_time.values]

    # Measure the longest region label on a throwaway figure so col_width can grow
    # to prevent adjacent region headers from overlapping.
    _longest_region = max(regions_, key=len).capitalize()
    _fig_tmp, _ax_tmp = plt.subplots(dpi=plot["rcparams"]["dpi"])
    _t = _ax_tmp.text(
        0,
        0,
        _longest_region,
        fontsize=fonts["region"],
        fontweight="bold",
        fontfamily=plot["rcparams"]["font_sans"],
    )
    _fig_tmp.canvas.draw()
    _text_w_in = (
        _t.get_window_extent(_fig_tmp.canvas.get_renderer()).width / _fig_tmp.dpi
    )
    plt.close(_fig_tmp)

    col_width = max(figure["col_width"], _text_w_in / (n_leads + layout["region_gap"]))
    plot_width = len(regions_) * (n_leads + layout["region_gap"]) - layout["region_gap"]
    fig_width = max(figure["width_min"], plot_width * col_width + figure["width_pad"])
    # Last row is at y=-(len(rows)-1); add 0.5 units of clearance below it.
    y_bottom = -(len(rows) - 0.5)
    y_top = layout["region_y"] + layout["region_y_pad"]
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

    init_label = "all" if cfg["init_hour"] == -999 else f"{cfg['init_hour']:02d}Z"
    neutral_size = dot_size(dots["neutral_threshold_pct"])

    # ── Figure ───────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # xlim/ylim are pinned here because data→axes-fraction conversions below
    # depend on these bounds; changing them later would shift those positions.
    ax.set_xlim(xlim_left, plot_width)
    ax.set_ylim(y_bottom, y_top)

    # Group labels: same x as metric labels, shifted left by a fixed point offset
    # so the gap stays constant regardless of figure width.
    group_transform = ax.transData + ScaledTranslation(
        -layout["group_offset_pt"] / 72,
        0,
        fig.dpi_scale_trans,
    )

    # Horizontal separators start just right of the metric labels. axhline expects
    # axes-fraction, so convert metric_x from data coords and add the physical gap.
    xlim_range = plot_width - xlim_left
    metric_frac = (layout["metric_x"] - xlim_left) / xlim_range
    ax_w_in = fig_width * ax.get_position().width
    gap_frac = (hline["gap_pt"] / 72) / ax_w_in
    hline_x_start = metric_frac + gap_frac

    # Title in figure coords so it is unaffected by axes margin changes.
    fig.text(
        0.01,
        0.99,
        f"{model_source} vs {baseline_source}\n"
        f"season={cfg['season']}   init={init_label}",
        fontsize=fonts["title"],
        fontweight="bold",
        ha="left",
        va="top",
        transform=fig.transFigure,
    )

    # ── Dots and row labels ──────────────────────────────────────────────────────
    # y is negated so row 0 sits at the top.
    cur_group = None
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
                ax.axhline(
                    y=y + 0.5,
                    xmin=hline_x_start,
                    xmax=hline["x_end"],
                    color=colors["hline"],
                    lw=hline["linewidth"],
                )
            cur_group = group

        ax.text(
            layout["metric_x"],
            y,
            metric,
            ha="right",
            va="center",
            fontsize=fonts["metric"],
        )

        # Marker per (region, lead_time) cell:
        #   NaN                   -> grey "x"
        #   |diff|% < threshold   -> neutral grey dot (tie)
        #   else                  -> coloured dot sized by |diff|%, blue=model, red=baseline
        for sec_idx, region in enumerate(regions_):
            x_off = sec_idx * (n_leads + layout["region_gap"])
            for lt_idx, d in enumerate(diff_filtered[name].sel(region=region).values):
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

    # ── Region headers, lead-time ticks, vertical separators ─────────────────────
    for sec_idx, region in enumerate(regions_):
        x_off = sec_idx * (n_leads + layout["region_gap"])

        ax.text(
            x_off + (n_leads - 1) / 2,
            layout["region_y"],
            region.capitalize(),
            ha="center",
            va="bottom",
            fontsize=fonts["region"],
            fontweight="bold",
        )

        if sec_idx > 0:
            x_sep = x_off - (layout["region_gap"] + 1) / 2
            ax.plot(
                [x_sep, x_sep],
                [y_bottom, y_top],
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

    # ── Legend ───────────────────────────────────────────────────────────────────
    sample_pcts = legend["sample_pcts"]
    neutral_pct = dots["neutral_threshold_pct"]

    # Legend dots, left to right: baseline-better (red) large→small,
    # neutral (grey), model-better (blue) small→large.
    dot_specs = (
        [(p, colors["baseline_better"], f"-{p}%") for p in sample_pcts]
        + [(neutral_pct, colors["neutral"], f"|Δ|<{neutral_pct}%")]
        + [(p, colors["model_better"], f"+{p}%") for p in reversed(sample_pcts)]
    )
    dot_specs[0] = (sample_pcts[0], colors["baseline_better"], f"≤-{sample_pcts[0]}%")
    dot_specs[-1] = (sample_pcts[0], colors["model_better"], f"≥+{sample_pcts[0]}%")

    # Legend uses axes-fraction so it stays stable across plot widths.
    # Cap at 0.8 to leave room for the side labels.
    ax_w_in = ax.get_position().width * fig.get_figwidth()
    x_span = min(legend["width_in"] / ax_w_in, 0.8)
    _data_ctr = (plot_width / 2 - xlim_left) / (plot_width - xlim_left)
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
        ax.plot(
            [0.5],
            [0],
            "x",
            color=colors["missing"],
            markersize=dots["missing_marker_size"],
            mew=dots["missing_marker_lw"],
            transform=miss_dot_trans,
            clip_on=False,
        )
        ax.text(
            0.5,
            0,
            "No data",
            ha="center",
            va="top",
            fontsize=small_fs,
            color=colors["missing"],
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
