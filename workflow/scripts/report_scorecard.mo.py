import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Verification scorecard
    """)
    return


@app.cell
def _():
    import os

    import numpy as np
    import pandas as pd
    import xarray as xr
    # import yaml

    import matplotlib.pyplot as plt
    from matplotlib.transforms import ScaledTranslation

    return ScaledTranslation, np, os, pd, plt, xr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Configuration

    All settings are hardcoded in the cell below. The "params" block at the top
    holds the values that the Snakemake rule will pass as placeholders for this run.
    """)
    return


@app.cell
def _():
    # ── Params & participants (placeholder — eventually overridden by Snakemake/CLI) ─────────────
    lead_times = "6/33/6"
    # regions = ["all"]
    # vars_metrics = {"T_2M": ["RMSE", "MAE"]}
    regions = ["all", "mittelland", "berge", "alpennordseite", "alpensuedseite", "jura"]
    vars_metrics = {"U_10M": ["RMSE", "MAE", "STDE", "CORR", "R2"], "V_10M": ["RMSE", "MAE", "STDE", "CORR", "R2"], "T_2M": ["RMSE", "MAE", "STDE", "CORR", "R2"], "PMSL": ["RMSE", "MAE", "STDE", "CORR", "R2"], "TD_2M": ["RMSE", "MAE", "STDE", "CORR", "R2"], "TOT_PREC": ["RMSE", "MAE", "STDE", "CORR", "R2"]}

    _model = {
            "path":   "/scratch/mch/ned/evalml/output/data/runs/interpolator-tmp-d5aa-on-forecaster-c304-1e7e/86c2/verif_aggregated.nc",
            "source": "Varda-Single",
        }

    _baseline = {
            "path":   "/scratch/mch/ned/evalml/output/data/baselines/ICON-CH1-EPS/verif_aggregated.nc",
            "source": "ICON-CH1-ctrl",
        }


    # ── Stable settings ─────────────────────────
    cfg = {
        "model": _model,                    # from params
        "baseline": _baseline,              # from params

        "season":     "all",
        "init_hour":  -999,                 # convention used by the dataset: -999 = aggregated over all init hours

        "regions":    regions,              # from params
        "lead_times": lead_times,           # from params; "start/stop/step" in hours; null = use all common lead times

        "all_metrics": ["RMSE", "MAE", "STDE", "CORR", "R2"],   # every entry must appear in metric_directions
        "metric_directions": {
            "lower_is_better":  ["RMSE", "MAE", "STDE"],
            "higher_is_better": ["CORR", "R2"],
        },

        "variables":  vars_metrics,         # from params; null entry → use all_metrics; list → restrict to that subset

        "plot": {

            "rcparams": {
                "font_family": "sans-serif",
                "font_sans":   "Liberation Sans",
                "dpi":         250,
            },

            "colors": {
                "model_better":    "#4878d0",   # seaborn muted blue   — model outperforms baseline
                "baseline_better": "#d65f5f",   # seaborn muted red    — baseline outperforms model
                "neutral":         "#cccccc",   # |diff|% below dots.neutral_threshold_pct
                "missing":         "#999999",   # NaN / no data marker
                "leads":           "#666666",   # lead-time tick labels
                "hline":           "#cccccc",   # horizontal separators between variable groups
                "vline":           "black",     # vertical separators between regions
            },

            "fonts": {              # font sizes in points
                "title":  24,
                "group":  20,        # variable group labels (U_10M, T_2M, ...)
                "metric": 18,        # metric labels         (RMSE, MAE, ...)
                "region": 20,        # region header labels; increasing this auto-widens the columns
                "leads":  16,
                "legend": 18,
            },

            "dots": {
                "neutral_threshold_pct": 5,     # |diff|% below which the dot is drawn neutral grey
                "size_cap_pct":          30,    # |diff|% at which the dot reaches max area
                "max_area":              250,   # max dot area (matplotlib scatter `s` units)
                "alpha":                 0.9,   # dot transparency (old: 0.85)
                "missing_marker_size":   8,     # size of the "x" marker drawn for NaN values
                "missing_marker_lw":     1.5,   # line width of the "x" marker
            },

            "figure": {                  # sizing rules (inches)
                "col_width":  0.26,         # minimum width per lead-time column; actual width grows automatically if region labels need more space
                "width_min":  5,           # minimum total figure width
                "width_pad":  5,            # extra width for left-side labels (group + metric)
                "row_height": 0.42,         # height per metric row
                "height_pad": 2.5,          # extra height for title, legend, margins
                "title_margin_in": 1.5,    # fixed space above axes for the title (independent of figure height)
            },

            "layout": {               # data-coordinate offsets controlling spacing
                "region_gap":      1.5,   # empty columns between adjacent region blocks
                "metric_x":        -1.5,  # x position of metric labels (data coords)
                "group_offset_pt": 75,    # physical gap (points) from metric label to group label
                "region_y":        1.9,   # y position of region title labels
                "region_y_pad":    0.8,   # extra y above region_y for the top ylim
                "leads_y":         0.35,  # y position of the lead-time tick labels
            },

            "hline": {              # horizontal separators between variable groups
                "gap_pt":    -42,      # physical gap (points) from metric label to line start
                "x_end":     1.0,      # axes fraction where the line ends
                "linewidth": 0.7,
            },

            "vline": {                # vertical separators between regions
                "linewidth": 0.8,
            },

            "legend": {
            "width_in":              6.5,        # physical width (inches) of the dot row
            "dot_below_pt":          40,         # fixed gap below axes bottom to dot centres (points); if you add n to this, add it to label_below_pt too
            "label_below_pt":        54,         # fixed gap below axes bottom to labels (points)
            "sample_pcts":           [30, 15, 5],# |diff|% values shown on each side of neutral
            "label_fontsize_factor": 0.85,       # factor applied to fonts.legend for the small labels
            "side_text_offset":      0.02,       # gap (axes fraction) between outer dot and side label
            "missing_dot_offset_pt": 16,         # extra vertical gap for the "no data" example (points)
            },
        },
    }

    cfg
    return (cfg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Validate the configuration

    Fail fast with explicit messages if the config is incomplete or inconsistent
    with the actual datasets. Every check below corresponds to an assumption made
    later in the pipeline.
    """)
    return


@app.cell
def _(cfg, os, xr):
    errors = []

    # ── 1. Required top-level keys ────────────────────────────────────────────────
    REQUIRED_TOP_KEYS = ("model", "baseline", "season", "init_hour",
                         "all_metrics", "metric_directions", "plot")
    for _key in REQUIRED_TOP_KEYS:
        if _key not in cfg:
            errors.append(f"Missing required key: '{_key}'")

    # ── 2. model/baseline must each have `path` and `source` ──────────────────────
    for role in ("model", "baseline"):
        if role in cfg:
            for sub in ("path", "source"):
                if sub not in (cfg[role] or {}):
                    errors.append(f"cfg['{role}'] is missing '{sub}'")

    # ── 3. metric_directions must partition all_metrics (no overlap, no gap) ──────
    if "metric_directions" in cfg and "all_metrics" in cfg:
        md_cfg = cfg["metric_directions"] or {}
        lb = md_cfg.get("lower_is_better")
        hb = md_cfg.get("higher_is_better")
        if lb is None:
            errors.append("metric_directions.lower_is_better is missing or null (use [] for empty)")
        elif not isinstance(lb, list):
            errors.append(f"metric_directions.lower_is_better must be a list, got: {type(lb).__name__}")
        if hb is None:
            errors.append("metric_directions.higher_is_better is missing or null (use [] for empty)")
        elif not isinstance(hb, list):
            errors.append(f"metric_directions.higher_is_better must be a list, got: {type(hb).__name__}")
        if isinstance(lb, list) and isinstance(hb, list):
            lb_set, hb_set = set(lb), set(hb)
            all_set = set(cfg["all_metrics"] or [])
            overlap = lb_set & hb_set
            _missing = all_set - (lb_set | hb_set)
            extra   = (lb_set | hb_set) - all_set
            if overlap:
                errors.append(f"Metrics in both lower_is_better and higher_is_better: {sorted(overlap)}")
            if _missing:
                errors.append(f"Metrics in all_metrics with no direction assigned: {sorted(_missing)}")
            if extra:
                errors.append(f"Metrics in metric_directions but not in all_metrics: {sorted(extra)}")

    # ── 4. Per-variable metric overrides must be a subset of all_metrics ──────────
    if "all_metrics" in cfg and cfg.get("variables"):
        all_m = set(cfg["all_metrics"] or [])
        for _var_name, _var_metrics in cfg["variables"].items():
            for _m in (_var_metrics or []):
                if _m not in all_m:
                    errors.append(f"variables.{_var_name}: metric '{_m}' is not in all_metrics")

    # ── 5. lead_times format: "start/stop/step" with integers and step > 0 ────────
    if cfg.get("lead_times"):
        try:
            parts = str(cfg["lead_times"]).split("/")
            assert len(parts) == 3, "must have exactly 3 parts"
            lt_start, lt_stop, lt_step = int(parts[0]), int(parts[1]), int(parts[2])
            if lt_step <= 0:
                errors.append(f"lead_times step must be > 0, got {lt_step}")
            if lt_stop < lt_start:
                errors.append(f"lead_times stop ({lt_stop}) must be >= start ({lt_start})")
        except (ValueError, AssertionError) as e:
            errors.append(f"lead_times must be 'start/stop/step' integers, got: '{cfg['lead_times']}' ({e})")

    # ── 6. Files exist; source/season/init_hour/regions are valid in each dataset ─
    if not errors:
        for role in ("model", "baseline"):
            path   = cfg[role]["path"]
            source = cfg[role]["source"]
            if not os.path.exists(path):
                errors.append(f"{role} file not found:\n    {path}")
                continue
            ds_meta = xr.open_dataset(path)

            # Check that each requested scalar selection exists in the dataset.
            for dim, _val in [("source", source),
                              ("season", cfg["season"]),
                              ("init_hour", cfg["init_hour"])]:
                if dim in ds_meta.dims and _val not in ds_meta[dim].values:
                    errors.append(
                        f"{role}: '{_val}' not found in dim '{dim}'.\n"
                        f"    Available: {list(ds_meta[dim].values)}"
                    )

            # Regions are a list, so check each element individually.
            if cfg.get("regions"):
                if "region" not in ds_meta.dims:
                    errors.append(f"{role}: 'regions' specified in config but dataset has no 'region' dimension")
                else:
                    _available = list(ds_meta["region"].values)
                    _unknown   = [r for r in cfg["regions"] if r not in _available]
                    if _unknown:
                        errors.append(f"{role}: region(s) {_unknown} not found.\n    Available: {_available}")

            ds_meta.close()

    # ── 7. Variables in config must exist in the model dataset ────────────────────
    if not errors and cfg.get("variables"):
        ds_check = xr.open_dataset(cfg["model"]["path"])
        prefixes = {v.split(".")[0] for v in ds_check.data_vars}
        ds_check.close()
        for _var_name in cfg["variables"]:
            if _var_name not in prefixes:
                errors.append(f"Variable '{_var_name}' not found in model dataset")

    # ── 8. Plot section: required subsections must exist ──────────────────────────
    if "plot" in cfg:
        required_plot_subkeys = ("rcparams", "colors", "fonts", "dots",
                                 "figure", "layout", "hline", "vline", "legend")
        for k in required_plot_subkeys:
            if k not in (cfg["plot"] or {}):
                errors.append(f"plot.{k} is missing")

    # ── Report ────────────────────────────────────────────────────────────────────
    if errors:
        for e in errors:
            print(f"❌  {e}")
        raise ValueError(f"Config validation failed ({len(errors)} error(s)). Fix the config and re-run.")
    print("✅  Config OK")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. "Align" datasets and compute relative difference

    Open both datasets, select the requested season / init_hour / source so
    that each Dataset is reduced to `(region, lead_time)` plus the data variables.
    Then compute the relative difference `(model − baseline) / |baseline|` in
    percent. This is what the scorecard visualises.

    Sign convention: a *positive* `diff` means the model value is higher than the
    baseline value. Whether that is good or bad depends on the metric (handled
    later via `metric_directions`).
    """)
    return


@app.cell
def _(cfg, np, xr):
    # Convenience aliases used throughout the rest of the notebook.
    all_metrics      = cfg["all_metrics"]
    lower_is_better  = cfg["metric_directions"].get("lower_is_better")  or []
    higher_is_better = cfg["metric_directions"].get("higher_is_better") or []

    model_source    = cfg["model"]["source"]
    baseline_source = cfg["baseline"]["source"]

    # Open + select. `squeeze(drop=True)` collapses any singleton dimensions
    # (e.g. `eps` when only one ensemble member exists).
    model = xr.open_dataset(cfg["model"]["path"]).sel(
        source=model_source, season=cfg["season"], init_hour=cfg["init_hour"],
    ).squeeze(drop=True)

    baseline = xr.open_dataset(cfg["baseline"]["path"]).sel(
        source=baseline_source, season=cfg["season"], init_hour=cfg["init_hour"],
    ).squeeze(drop=True)

    # Restrict to the variables and lead times present in both datasets.
    common_vars  = [v for v in model.data_vars if v in baseline.data_vars]
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

    # Relative difference in percent.
    # baseline = 0 → ±inf; we mask those to NaN and the plot shows them as a grey "x".
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Apply config filters

    Three filters are applied to `diff`, in order:
    1. **regions**: keep only the regions listed in the config (and in the order they appear).
    2. **lead_times**: keep only the lead times that match the `start/stop/step` grid *and* exist in the data.
    3. **variables / metrics**: keep only the requested combinations (in the order they appear in all_metrics).

    Any filter is a no-op if its config value is `null`.
    """)
    return


@app.cell
def _(all_metrics, b, cfg, diff, m, pd):
    def to_h(td):
        """Convert timedelta64[ns] into integer hours. Raises if not a whole number of hours."""
        h = pd.Timedelta(td).total_seconds() / 3600
        if h != int(h):
            raise ValueError(f"Lead time {td} is not a whole number of hours ({h}h)")
        return int(h)

    model_leads_h    = sorted(to_h(lt) for lt in m.lead_time.values)
    baseline_leads_h = sorted(to_h(lt) for lt in b.lead_time.values)

    # ── Regions ──────────────────────────────────────────────────────────────────
    diff_filtered = diff
    if cfg.get("regions"):
        _available = list(diff_filtered.region.values)
        _unknown   = [r for r in cfg["regions"] if r not in _available]
        if _unknown:
            raise ValueError(f"Unknown region(s): {_unknown}\n  Available: {_available}")
        diff_filtered = diff_filtered.sel(region=cfg["regions"])

    # ── Lead times ───────────────────────────────────────────────────────────────
    if cfg.get("lead_times"):
        start, stop, step = (int(x) for x in cfg["lead_times"].split("/"))
        requested = {pd.Timedelta(h, "h") for h in range(start, stop + 1, step)}
        _available = {pd.Timedelta(lt) for lt in diff_filtered.lead_time.values}
        keep      = sorted(requested & _available)
        _missing   = sorted(requested - _available)

        if _missing:
            print(f"⚠️  Lead times not in dataset (skipped): {[to_h(lt) for lt in _missing]}h")
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
        diff_filtered = diff_filtered[[v for v in diff_filtered.data_vars if v.split(".")[-1] in all_metrics]]

    if not diff_filtered.data_vars:
        raise ValueError(
            "No variables left after filtering. "
            "Check that variables/metrics in the config exist in both datasets."
        )

    diff_filtered
    return diff_filtered, to_h


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Plot the scorecard

    Each row is a `(variable, metric)` pair, each column is a `(region, lead_time)`
    pair. The dot's colour indicates which dataset performs better for that metric;
    the dot's size grows linearly with `|diff|%` and saturates at `dots.size_cap_pct`.
    Values smaller than `dots.neutral_threshold_pct` are drawn neutral grey, and NaN
    values appear as a grey "x".

    Every visual parameter (colours, fonts, dot sizes, layout offsets, legend)
    is read from `cfg["plot"]`.
    """)
    return


@app.cell
def _(
    ScaledTranslation,
    baseline_source,
    cfg,
    diff_filtered,
    higher_is_better,
    lower_is_better,
    model_source,
    np,
    plt,
    to_h,
):
    # ── Read the plot config ──────────────────────────────────────────────────────
    plot   = cfg["plot"]
    colors = plot["colors"]
    fonts  = plot["fonts"]
    dots   = plot["dots"]
    figure = plot["figure"]
    layout = plot["layout"]
    hline  = plot["hline"]
    vline  = plot["vline"]
    legend = plot["legend"]

    # rcParams = matplotlib's global defaults; setting them once applies to every plot below.
    plt.rcParams["font.family"]     = plot["rcparams"]["font_family"]
    plt.rcParams["font.sans-serif"] = [plot["rcparams"]["font_sans"]]
    plt.rcParams["figure.dpi"]      = plot["rcparams"]["dpi"]


    # ── Helpers ──────────────────────────────────────────────────────────────────
    def is_model_better(d, metric):
        """True if a positive `d` (signed diff %) means the model beats the baseline."""
        if metric in higher_is_better: return d > 0
        if metric in lower_is_better:  return d < 0
        return None

    def dot_size(d):
        """Linear ramp from 0 to dots.max_area, capped at dots.size_cap_pct."""
        return float(min(abs(d) / dots["size_cap_pct"] * dots["max_area"], dots["max_area"]))


    # ── Figure size and layout precomputation ────────────────────────────────────
    # Layout building blocks: one row per (variable, metric), one column per lead time.
    rows       = [tuple(v.rsplit(".", 1)) for v in diff_filtered.data_vars]
    regions_   = list(diff_filtered.region.values)
    n_leads    = diff_filtered.sizes["lead_time"]                                        # .sizes -> {dim: length};
    lead_hours = [to_h(lt) for lt in diff_filtered.lead_time.values]

    # Measure the longest region label so col_width is wide enough that adjacent
    # region headers never overlap. We render the text on a throwaway figure and
    # read its bounding box from the renderer.
    _longest_region = max(regions_, key=len).capitalize()
    _fig_tmp, _ax_tmp = plt.subplots(dpi=plot["rcparams"]["dpi"])
    _t = _ax_tmp.text(0, 0, _longest_region, fontsize=fonts["region"], fontweight="bold",
                      fontfamily=plot["rcparams"]["font_sans"])
    _fig_tmp.canvas.draw()
    _text_w_in = _t.get_window_extent(_fig_tmp.canvas.get_renderer()).width / _fig_tmp.dpi  # pixels -> inches
    plt.close(_fig_tmp)

    # Final figure dimensions in inches and data-coordinate bounds.
    # x-axis uses "column units" (one per lead time, plus region_gap empty units between regions);
    # y-axis uses "row units" (one per metric row).
    col_width  = max(figure["col_width"], _text_w_in / (n_leads + layout["region_gap"]))
    plot_width = len(regions_) * (n_leads + layout["region_gap"]) - layout["region_gap"]
    fig_width  = max(figure["width_min"], plot_width * col_width + figure["width_pad"])
    fig_height = len(rows) * figure["row_height"] + figure["height_pad"]
    y_bottom   = -(len(rows) - 0.5)                                             # = -(last_row_y) - 0.5: last row is at y=-(len(rows)-1), add 0.5 units of clearance below it
    y_top      = layout["region_y"] + layout["region_y_pad"]

    # Left margin fixed in inches so content does not shift right when col_width is large.
    _left_margin_col = figure.get("left_margin_in", 2.5) / col_width
    xlim_left        = layout["metric_x"] - _left_margin_col

    # Small precomputed values reused later.
    init_label   = "all" if cfg["init_hour"] == -999 else f"{cfg['init_hour']:02d}Z"
    neutral_size = dot_size(dots["neutral_threshold_pct"])
    has_missing  = any(np.isnan(diff_filtered[v].values).any() for v in diff_filtered.data_vars)


    # ── Figure ───────────────────────────────────────────────────────────────────
    # fig = the whole canvas (physical size in inches); ax = the drawing area inside it.
    # matplotlib has three coordinate systems we use below:
    #   • transData   -> the data values you pass to ax.scatter / ax.text (column units, row units here)
    #   • transAxes   -> 0..1 fraction of the axes box (handy for legends and overlays)
    #   • transFigure -> 0..1 fraction of the whole figure (used for the title)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # xlim/ylim pin the x/y coordinate ranges of the axes box: here x goes from
    # xlim_left to plot_width (column units), y from y_bottom to y_top (row units).
    # Must be set here because further below we manually convert those ranges to
    # axes-fraction (0..1) — if xlim/ylim changed later, those positions would shift.
    ax.set_xlim(xlim_left, plot_width)
    ax.set_ylim(y_bottom, y_top)

    # Group labels: same x as metric labels, shifted left by a fixed number of points.
    # ScaledTranslation(dx, dy, dpi_scale_trans) builds an offset of (dx, dy) inches;
    # adding it to ax.transData gives "data position + fixed inch offset" so the gap
    # stays constant regardless of figure width.
    # Unit conversion: typography uses points, 1 inch = 72 pt -> divide pt by 72 to get inches.
    group_transform = ax.transData + ScaledTranslation(
        -layout["group_offset_pt"] / 72, 0, fig.dpi_scale_trans,
    )

    # The horizontal separator lines must start just to the right of the metric
    # labels. The problem is that metric labels are placed in data coords, but
    # the line-drawing function expects positions in axes-fraction (0..1).
    # So we convert: metric label x → 0..1, then add the small physical gap.
    xlim_range    = plot_width - xlim_left
    metric_frac   = (layout["metric_x"] - xlim_left) / xlim_range   # data coords → 0..1
    ax_w_in       = fig_width * ax.get_position().width              # axes width in inches
    gap_frac      = (hline["gap_pt"] / 72) / ax_w_in                 # gap: pt → inches → 0..1
    hline_x_start = metric_frac + gap_frac


    # Title placed in figure coords (top-left), so it stays put regardless of axes margins.
    fig.text(
        0.01, 0.99,
        f"{model_source} vs {baseline_source}\n"
        f"season={cfg['season']}   init={init_label}",
        fontsize=fonts["title"], fontweight="bold", ha="left", va="top",        # ha/va = horizontal/vertical alignment of the text anchor
        transform=fig.transFigure,
    )


    # ── Dots and row labels ──────────────────────────────────────────────────────
    # y is negated because matplotlib y grows upward, but we want row 0 at the top.
    cur_group = None
    for row_idx, (group, metric) in enumerate(rows):
        y    = -row_idx
        name = f"{group}.{metric}"

        # When entering a new variable group: write its bold name (e.g. "U_10M")
        # and draw a thin horizontal separator above the row (skipped on the very first row).
        if group != cur_group:
            ax.text(layout["metric_x"], y, group,
                    ha="right", va="center", fontsize=fonts["group"], fontweight="bold",
                    transform=group_transform)                                  # data coords + the fixed inch offset built above
            if row_idx > 0:
                ax.axhline(y=y + 0.5, xmin=hline_x_start, xmax=hline["x_end"],  # xmin/xmax in axes-fraction (0..1)
                           color=colors["hline"], lw=hline["linewidth"])
            cur_group = group

        # Metric label on the left of the row (e.g. "RMSE", "MAE").
        ax.text(layout["metric_x"], y, metric,
                ha="right", va="center", fontsize=fonts["metric"])

        # For every (region, lead_time) cell of this row, draw one marker.
        # Three cases:
        #   • NaN data            -> small grey "x"
        #   • |diff|% < threshold -> small grey neutral dot (a tie)
        #   • otherwise           -> coloured dot, sized by |diff|%, blue if model wins, red otherwise
        for sec_idx, region in enumerate(regions_):
            x_off = sec_idx * (n_leads + layout["region_gap"])                  # left edge of region block, in data coords
            for lt_idx, d in enumerate(diff_filtered[name].sel(region=region).values):
                x = x_off + lt_idx
                if np.isnan(d):
                    ax.plot(x, y, "x", color=colors["missing"],
                            ms=dots["missing_marker_size"], mew=dots["missing_marker_lw"])  # ms = markersize (pt), mew = marker edge width (pt)
                    continue
                if abs(d) < dots["neutral_threshold_pct"]:
                    color, size = colors["neutral"], neutral_size
                else:
                    color = colors["model_better"] if is_model_better(d, metric) else colors["baseline_better"]
                    size  = dot_size(d)
                ax.scatter(x, y, s=size, c=color, alpha=dots["alpha"], linewidths=0)        # s = marker area in pt²


    # ── Region headers, lead-time ticks, vertical separators ─────────────────────
    # Second pass over regions to draw the elements that frame each region block:
    for sec_idx, region in enumerate(regions_):
        x_off = sec_idx * (n_leads + layout["region_gap"])

        # Region name, bold, centred above the block.
        ax.text(x_off + (n_leads - 1) / 2, layout["region_y"], region.capitalize(),
                ha="center", va="bottom", fontsize=fonts["region"], fontweight="bold")

        # Vertical line between this block and the previous one (skipped before the first block).
        if sec_idx > 0:
            x_sep = x_off - (layout["region_gap"] + 1) / 2
            ax.plot([x_sep, x_sep], [y_bottom, y_top],
                    color=colors["vline"], lw=vline["linewidth"])

        # Rotated lead-time tick labels (e.g. "6h", "12h", ...) below the bottom row.
        for lt_idx, h in enumerate(lead_hours):
            ax.text(x_off + lt_idx, layout["leads_y"], f"{h}h",
                    ha="center", va="bottom", fontsize=fonts["leads"],
                    rotation=90, color=colors["leads"])

    ax.axis("off")                  # hide default x/y axis frame, ticks and labels (we draw our own)

    title_margin_in   = figure["title_margin_in"]
    small_fs          = fonts["legend"] * legend["label_fontsize_factor"]
    legend_bottom_in  = (legend["label_below_pt"] + legend.get("missing_dot_offset_pt", 0) * has_missing + small_fs * 1.4) / 72
    plt.tight_layout()
    plt.subplots_adjust(
    top=1 - title_margin_in / fig_height,
    bottom=legend_bottom_in / fig_height,
    )



    # ── Legend ───────────────────────────────────────────────────────────────────
    sample_pcts = legend["sample_pcts"]                          # e.g. [30, 15, 5]
    neutral_pct = dots["neutral_threshold_pct"]

    # List of (diff_value, color, label) tuples — one per legend dot, left to right:
    # baseline-better (red) large→small | neutral (grey) | model-better (blue) small→large
    dot_specs = (
        [(p, colors["baseline_better"], f"-{p}%") for p in sample_pcts]
        + [(neutral_pct, colors["neutral"], f"|Δ|<{neutral_pct}%")]
        + [(p, colors["model_better"], f"+{p}%") for p in reversed(sample_pcts)]
    )
    dot_specs[0]  = (sample_pcts[0], colors["baseline_better"], f"≤-{sample_pcts[0]}%")
    dot_specs[-1] = (sample_pcts[0], colors["model_better"],    f"≥+{sample_pcts[0]}%")

    # The legend sits below the plot area, so all positions are in axes-fraction
    # (0 = left edge of axes, 1 = right edge) rather than data coords — this way
    # the legend never moves no matter how many columns the plot has.

    # Convert the desired physical width (inches) to axes-fraction so we know
    # how much of the 0..1 range the dot row should occupy.
    # Cap at 0.8 to leave room for the "baseline better ←" / "→ model better" labels on the sides.
    ax_w_in  = ax.get_position().width * fig.get_figwidth()   # axes box width in inches
    x_span   = min(legend["width_in"] / ax_w_in, 0.8)         # desired legend width in axes-fraction
    x_dots   = np.linspace(0.5 - x_span / 2, 0.5 + x_span / 2, len(dot_specs))  # x positions of each dot, evenly spaced and centred
    small_fs = fonts["legend"] * legend["label_fontsize_factor"]                  # smaller font for the % labels under each dot

    # Builds transforms anchored at a fixed physical distance below the axes bottom,
    # so spacing is constant regardless of figure height (unlike axes-fraction coords).
    # y=0 in ax.transAxes = axes bottom; ScaledTranslation then shifts down by the given points.
    dot_trans   = ax.transAxes + ScaledTranslation(0, -legend["dot_below_pt"]   / 72, fig.dpi_scale_trans)
    label_trans = ax.transAxes + ScaledTranslation(0, -legend["label_below_pt"] / 72, fig.dpi_scale_trans)
    miss_dot_trans   = ax.transAxes + ScaledTranslation(
        0, -(legend["dot_below_pt"]   + legend["missing_dot_offset_pt"]) / 72, fig.dpi_scale_trans)
    miss_label_trans = ax.transAxes + ScaledTranslation(
        0, -(legend["label_below_pt"] + legend["missing_dot_offset_pt"]) / 72, fig.dpi_scale_trans)

    # Draw each dot and its label underneath.
    # clip_on=False is needed because the legend sits below the axes box.
    for x, (_val, col, lbl) in zip(x_dots, dot_specs):
        s = neutral_size if col == colors["neutral"] else dot_size(_val)
        ax.scatter([x], [0], s=s, c=col, alpha=dots["alpha"], linewidths=0,
                   transform=dot_trans, clip_on=False)
        ax.text(x, 0, lbl, ha="center", va="top",
                fontsize=small_fs, transform=label_trans, clip_on=False)

    # "baseline better ←" to the left of the dot row, "→ model better" to the right.
    ax.text(x_dots[0] - legend["side_text_offset"], 0, f"{baseline_source} better ←",
            ha="right", va="center", fontsize=fonts["legend"],
            transform=dot_trans, clip_on=False)
    ax.text(x_dots[-1] + legend["side_text_offset"], 0, f"→ {model_source} better",
            ha="left", va="center", fontsize=fonts["legend"],
            transform=dot_trans, clip_on=False)

    # If any NaN values exist in the data, add a small example showing what the "x" marker means.
    if has_missing:
        ax.plot([0.5], [0], "x", color=colors["missing"],
                markersize=dots["missing_marker_size"], mew=dots["missing_marker_lw"],
                transform=miss_dot_trans, clip_on=False)
        ax.text(0.5, 0, "No data",
                ha="center", va="top", fontsize=small_fs, color=colors["missing"],
                transform=miss_label_trans, clip_on=False)


    plt.show()
    return (init_label,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Appendix: distribution of |diff|%

    Just a tuning aid for `plot.dots.size_cap_pct`: the histogram below shows where
    the bulk of the data lies, which informs at what value the dot size should
    saturate.
    """)
    return


@app.cell
def _(baseline_source, cfg, diff_filtered, init_label, model_source, np, plt):
    all_vals = np.concatenate([diff_filtered[v].values.flatten() for v in diff_filtered.data_vars])
    abs_vals = np.abs(all_vals[np.isfinite(all_vals)])

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.set_title(
        f"|diff|% distribution — {model_source} vs {baseline_source}\n"
        f"season={cfg['season']}   init={init_label}",
        fontsize=11, loc="left",
    )
    _ax.hist(abs_vals, bins=60, color="steelblue", alpha=0.75)
    _ax.set_xlabel("|diff|  (%)")
    _ax.set_ylabel("count")
    label_y = {"p50": 0.95, "p90": 0.95, "p95": 0.75, "p99": 0.95}
    for label, pct in [("p50", 50), ("p90", 90), ("p95", 95), ("p99", 99)]:
        _val = np.percentile(abs_vals, pct)
        _ax.axvline(_val, color="red", ls="--", alpha=0.6, lw=1)
        _ax.text(_val, _ax.get_ylim()[1] * label_y[label], f" {label}={_val:.0f}",
                color="red", fontsize=9, va="top")
    plt.tight_layout()
    plt.show()

    print(f"min={abs_vals.min():.2f}  max={abs_vals.max():.2f}  "
          f"median={np.median(abs_vals):.2f}  mean={abs_vals.mean():.2f}\n")
    n = len(abs_vals)
    for label, mask in [("< 50%  ", abs_vals < 50),
                        ("50–100%", (abs_vals >= 50) & (abs_vals <= 100)),
                        ("> 100% ", abs_vals > 100)]:
        count = mask.sum()
        print(f"  {label}  {count:>6}  ({100*count/n:.1f}%)")

    print()
    edges = list(range(0, 110, 10)) + [float("inf")]
    for lo, hi in zip(edges, edges[1:]):
        mask  = (abs_vals >= lo) & (abs_vals < hi)
        count = mask.sum()
        label = f"{lo:>3}–{int(hi):>3}%" if hi != float("inf") else f"{lo:>3}%+    "
        print(f"  {label}  {count:>6}  ({100*count/n:.1f}%)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # To-do

    - fix title top-left
    """)
    return


if __name__ == "__main__":
    app.run()
