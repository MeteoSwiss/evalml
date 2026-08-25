import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt

    # Repo root: cwd when run from repo root, else walk up (marimo runs with
    # cwd = the notebook's own directory). __file__ is undefined in a kernel.
    PROJECT_ROOT = Path.cwd().resolve()
    if not (PROJECT_ROOT / "workflow").is_dir():
        for _p in [PROJECT_ROOT] + list(PROJECT_ROOT.parents):
            if (_p / "workflow").is_dir() and (_p / "src").is_dir():
                PROJECT_ROOT = _p
                break
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    sys.path.insert(0, str(PROJECT_ROOT / "workflow" / "scripts"))

    from evalml.publication import style as _style

    plt.style.use(_style.mplstyle_path())

    from report_scorecard import (
        DEFAULT_PLOT_CFG,
        _draw_data_rows as draw_data_rows,
        _draw_legend as draw_legend,
        _draw_slice_headers as draw_slice_headers,
        _filter_diff as filter_diff,
        _load_relative_diff as load_relative_diff,
        _measure_label_sizes as measure_label_sizes,
        _parse_var_metrics as parse_var_metrics,
        _scaled_dot_area as scaled_dot_area,
        _timedelta_to_hours as timedelta_to_hours,
    )

    from evalml.publication.style import (
        COLOR_SKILL_BASELINE_BETTER,
        COLOR_SKILL_MODEL_BETTER,
        PARAM_UNITS,
        param_label,
        region_label,
    )

    # region coordinate values are already human-readable after assign_coords below,
    # so skip the .capitalize() that _format_slice_label would otherwise apply.
    import report_scorecard as _rsc

    _orig_fmt = _rsc._format_slice_label

    def _fmt(s, strat_dim):
        if strat_dim == "region" and isinstance(s, str):
            return s
        return _orig_fmt(s, strat_dim)

    _rsc._format_slice_label = _fmt
    return (
        COLOR_SKILL_BASELINE_BETTER,
        COLOR_SKILL_MODEL_BETTER,
        DEFAULT_PLOT_CFG,
        PARAM_UNITS,
        PROJECT_ROOT,
        Path,
        draw_data_rows,
        draw_legend,
        draw_slice_headers,
        filter_diff,
        load_relative_diff,
        measure_label_sizes,
        mo,
        param_label,
        parse_var_metrics,
        plt,
        region_label,
        scaled_dot_area,
        sys,
        timedelta_to_hours,
    )


@app.cell
def _():
    import logging

    LOG = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return (LOG,)


@app.cell
def _(PROJECT_ROOT, mo, sys):
    import os as _os

    # Interactive defaults: try manifest auto-discovery so opening the notebook
    # without arguments still points at a sensible path.
    _manifest_default = _os.environ.get("EVALML_MANIFEST")
    _output_default = None
    if not _manifest_default:
        try:
            from evalml.publication.manifest import (
                default_manifest_path as _dmp,
                truth_slug as _ts,
            )
            import json as _json

            _found = _dmp()
            _manifest_default = str(_found)
            _truth = _json.loads(_found.read_text()).get("truth", {})
            _output_default = str(
                PROJECT_ROOT
                / f"output/figures/{_truth.get('slug') or _ts(_truth.get('label', ''))}/scorecards"
            )
        except Exception as _exc:
            _manifest_default = str(
                PROJECT_ROOT / "output/manifests/manifest_varda-single_paper_stations.json"
            )
            print(
                f"[publication_scorecard] manifest auto-discovery failed ({_exc}); "
                f"defaulting to {_manifest_default}"
            )

    _cli = mo.cli_args()
    if not _cli and len(sys.argv) > 1:
        import argparse

        _p = argparse.ArgumentParser()
        _p.add_argument("--manifest", default=_manifest_default)
        _p.add_argument("--output", default=_output_default)
        _p.add_argument("--candidate", default=None)
        _a, _ = _p.parse_known_args()
        manifest_path = _a.manifest
        output_dir = _a.output
        candidate = _a.candidate
    else:
        manifest_path = _cli.get("manifest", default=_manifest_default)
        output_dir = _cli.get("output", default=_output_default)
        candidate = _cli.get("candidate", default="Varda-single")
    return candidate, manifest_path, output_dir


@app.cell
def _(LOG, PROJECT_ROOT, candidate, manifest_path, output_dir):
    from evalml.publication.manifest import load_manifest

    mani = load_manifest(manifest_path)
    cand = mani.get_candidate(candidate)
    slug = mani.truth.get("slug", "truth")

    cand_info = {
        "id": cand.id,
        "label": cand.label,
        "verif_aggregated": PROJECT_ROOT / cand.paths["verif_aggregated"],
    }

    section_cfgs = [
        {
            "name": "short_range",
            "baseline": "ICON-CH1-CTRL",
            "lead_times": "6/33/6",
            "variables": ["T_2M", "SP_10M", "TOT_PREC1", "PMSL"],
        },
        {
            "name": "medium_range",
            "baseline": "ICON-CH2-CTRL",
            "lead_times": "24/120/24",
            "variables": ["T_2M", "SP_10M", "TOT_PREC6", "PMSL"],
        },
    ]
    for d in section_cfgs:
        _base = mani.resolve_baseline(d["baseline"])
        d.update(
            {
                "stratification": "region",
                "scores": ["RMSE", "ETS"],
                "base_id": _base.id,
                "base_label": _base.label,
                "base_verif": PROJECT_ROOT / _base.paths["verif_aggregated"],
            }
        )
        LOG.info("scorecard: section %r — baseline %r", d["name"], d["base_label"])

    resolved_output = output_dir or str(
        PROJECT_ROOT / f"output/figures/{slug}/scorecards"
    )
    return cand_info, resolved_output, section_cfgs


@app.cell
def _(
    COLOR_SKILL_BASELINE_BETTER,
    COLOR_SKILL_MODEL_BETTER,
    DEFAULT_PLOT_CFG,
    LOG,
    PARAM_UNITS,
    cand_info,
    filter_diff,
    load_relative_diff,
    measure_label_sizes,
    mo,
    param_label,
    parse_var_metrics,
    plt,
    region_label,
    section_cfgs,
    timedelta_to_hours,
):
    import time as _time

    def _build_panel_cfg(sec, plot_cfg):
        variables = dict(parse_var_metrics(s) for s in sec["variables"])
        return {
            "model": {
                "path": cand_info["verif_aggregated"],
                "source": cand_info["id"],
                "label": cand_info["label"],
            },
            "baseline": {
                "path": sec["base_verif"],
                "source": sec["base_id"],
                "label": sec["base_label"],
            },
            "stratification": sec["stratification"],
            "lead_times": sec["lead_times"],
            "all_metrics": sec["scores"],
            "metric_directions": {
                "lower_is_better": ["RMSE", "MAE", "STDE", "FAR"],
                "higher_is_better": ["R2", "ETS", "POD"],
            },
            "variables": variables,
            "plot": plot_cfg,
        }

    def _panel_layout(diff, cfg):
        plot = cfg["plot"]
        figure = plot["figure"]
        layout = plot["layout"]
        strat_dim = cfg.get("stratification", "region")
        rows = [tuple(v.rsplit(".", 1)) for v in diff.data_vars]
        slices = list(diff[strat_dim].values)
        n_leads = diff.sizes["step"]
        lead_hours = [timedelta_to_hours(lt) for lt in diff.step.values]
        slice_label_w_in, slice_label_h_rows, metric_label_w_pt = measure_label_sizes(
            plot, rows, slices, strat_dim
        )
        col_width = max(
            figure["col_width"],
            slice_label_w_in
            * layout["slice_label_pad"]
            / (n_leads + layout["slice_gap"]),
        )
        plot_width = len(slices) * (n_leads + layout["slice_gap"]) - layout["slice_gap"]
        y_bottom = -(len(rows) - 0.5)
        y_top = layout["slice_y"] + slice_label_h_rows + layout["slice_y_pad"]
        xlim_left = layout["metric_x"] - figure["left_margin_in"] / col_width
        data_w_in = max(
            figure["width_min"], plot_width * col_width + figure["width_pad"]
        )
        return dict(
            rows=rows,
            slices=slices,
            n_leads=n_leads,
            lead_hours=lead_hours,
            col_width=col_width,
            plot_width=plot_width,
            y_bottom=y_bottom,
            y_top=y_top,
            xlim_left=xlim_left,
            slice_label_h_rows=slice_label_h_rows,
            metric_label_w_pt=metric_label_w_pt,
            data_w_in=data_w_in,
        )

    import copy
    import textwrap as _tw
    import xarray as xr

    _PARAM_WRAP = 20  # wrap parameter names after ~20 chars
    _REGION_WRAP = 12  # wrap region names after ~12 chars

    def _top_align(label):
        """Prepend (n-1) empty lines so va='center' aligns the first visible line with y."""
        n = label.count("\n") + 1
        return "\n" * (n - 1) + label

    _THRESHOLD_OPS = {"_gt_": ">", "_ge_": ">=", "_lt_": "<", "_le_": "<="}
    _K_TO_C_PARAMS = {"T_2M", "TD_2M"}

    def _metric_with_unit(param, metric):
        """Return a display-ready metric string with physical unit (and K→°C for temperature).

        For temperature params the value is converted and the result is fully decoded so that
        decode_metric() (called later inside report_scorecard) leaves it unchanged.
        For other params the raw encoded metric gets a unit suffix appended; decode_metric
        will then expand '_gt_' → '>' and 'p' → '.' as usual.
        """
        for op_enc, op_sym in _THRESHOLD_OPS.items():
            if op_enc in metric:
                score, val_raw = metric.split(op_enc, 1)
                val = float(val_raw.replace("p", "."))
                if param in _K_TO_C_PARAMS:
                    return f"{score} {op_sym} {val - 273.15:g} °C"
                unit = PARAM_UNITS.get(param, "")
                return f"{metric} {unit}" if unit else metric
        return metric

    excluded_data_vars = [
        "TOT_PREC1.ETS_gt_0p1",
        "TOT_PREC1.ETS_gt_10p0",
        "TOT_PREC6.ETS_gt_0p1",
        "TOT_PREC6.ETS_gt_1p0",
        "TOT_PREC6.ETS_gt_10p0",
        "TOT_PREC6.ETS_gt_50p0",
    ]

    plot_cfg = copy.deepcopy(DEFAULT_PLOT_CFG)
    plot_cfg["colors"]["model_better"] = COLOR_SKILL_MODEL_BETTER
    plot_cfg["colors"]["baseline_better"] = COLOR_SKILL_BASELINE_BETTER
    plot_cfg["figure"]["title_margin_in"] = 0.9  # space between title and axes
    plot_cfg["figure"]["inter_panel_gap_in"] = 1.2  # extra gap above non-first panels
    plot_cfg["figure"]["col_width"] = 0.25      # slightly narrower than default (0.26)
    plot_cfg["figure"]["row_height"] = 0.34     # ~20% shorter than default (0.42)
    plot_cfg["figure"]["left_margin_in"] = 3.0  # physical space for param/metric labels
    plot_cfg["figure"]["width_pad"] = 2.5       # ≈ left_margin_in + small right buffer
    plot_cfg["dots"]["max_area"] = 180          # keep dots within col_width (0.25 in ≈ 18 pt)

    # Inherit font sizes from the mplstyle rather than using the hardcoded defaults.
    _fs = plt.rcParams["font.size"]          # 13 pt — values / annotations
    _fs_title = plt.rcParams["axes.titlesize"]  # 16 pt — titles / identifiers
    plot_cfg["fonts"]["title"] = _fs_title
    plot_cfg["fonts"]["group"] = _fs_title   # param-group labels (bold, identify variable)
    plot_cfg["fonts"]["slice"] = _fs_title   # region column headers
    plot_cfg["fonts"]["metric"] = _fs        # metric-row labels
    plot_cfg["fonts"]["leads"] = _fs         # lead-time tick labels
    plot_cfg["fonts"]["legend"] = _fs        # legend side-text and dot labels
    plot_cfg["legend"]["label_fontsize_factor"] = 1.0  # keep dot labels same size as side-text

    LOG.info("scorecard: loading data for %d section(s)", len(section_cfgs))
    panels = []
    for _sec in section_cfgs:
        _t0 = _time.perf_counter()
        LOG.info(
            "scorecard: loading section %r (candidate vs %s)",
            _sec["name"],
            _sec["base_label"],
        )
        _cfg = _build_panel_cfg(_sec, plot_cfg)
        _diff = load_relative_diff(_cfg)
        _diff = filter_diff(_diff, _cfg)
        _diff = _diff.drop_vars([v for v in excluded_data_vars if v in _diff.data_vars])
        _diff = _diff.sel(region = xr.DataArray(["icon", "jura", "mittelland", "alpen"], dims = "region"))
        _diff = _diff.rename(
            {
                v: f"{_top_align(_tw.fill(param_label(v.rsplit('.', 1)[0]), width=_PARAM_WRAP))}.{_metric_with_unit(v.rsplit('.', 1)[0], v.rsplit('.', 1)[1])}"
                for v in _diff.data_vars
            }
        )
        _strat_dim = _cfg.get("stratification", "region")
        if _strat_dim in _diff.coords:
            _diff = _diff.assign_coords(
                {
                    _strat_dim: [
                        _tw.fill(region_label(str(r)), width=_REGION_WRAP)
                        for r in _diff[_strat_dim].values
                    ]
                }
            )
        _lay = _panel_layout(_diff, _cfg)
        panels.append((_diff, _cfg, _sec["name"], _lay))
        LOG.info(
            "scorecard: section %r done in %.1fs — %d rows × %d slices",
            _sec["name"],
            _time.perf_counter() - _t0,
            len(_lay["rows"]),
            len(_lay["slices"]),
        )

    mo.md(
        f"**Loaded {len(panels)} panel(s):** "
        + ", ".join(f"{name} ({len(lay['rows'])} rows)" for _, _, name, lay in panels)
    )
    return panels, plot_cfg


@app.cell
def _(
    LOG,
    Path,
    draw_data_rows,
    draw_legend,
    draw_slice_headers,
    panels,
    plot_cfg,
    plt,
    resolved_output,
    scaled_dot_area,
):
    import numpy as np
    from matplotlib.transforms import ScaledTranslation

    figure_cfg = plot_cfg["figure"]
    fonts = plot_cfg["fonts"]
    legend_cfg = plot_cfg["legend"]
    dots = plot_cfg["dots"]
    hline_cfg = plot_cfg["hline"]
    layout_cfg = plot_cfg["layout"]

    has_missing = any(
        any(np.isnan(diff[v].values).any() for v in diff.data_vars)
        for diff, _, _, _ in panels
    )

    small_fs = fonts["legend"] * legend_cfg["label_fontsize_factor"]
    legend_h_in = (
        legend_cfg["label_below_pt"]
        + legend_cfg.get("missing_dot_offset_pt", 0) * has_missing
        + small_fs * 1.4
    ) / 72

    # Horizontal layout: panels side by side.
    panel_widths = [lay["data_w_in"] for _, _, _, lay in panels]
    panel_height = (
        figure_cfg["title_margin_in"]
        + legend_h_in
        + figure_cfg["row_height"]
        * max(lay["y_top"] - lay["y_bottom"] for _, _, _, lay in panels)
    )
    fig_width = sum(panel_widths)
    fig_height = panel_height

    LOG.info("scorecard: rendering figure (%.1f × %.1f in)", fig_width, fig_height)
    _fig = plt.figure(figsize=(fig_width, fig_height))
    _subfigs = _fig.subfigures(1, len(panels), width_ratios=panel_widths)
    if len(panels) == 1:
        _subfigs = [_subfigs]

    neutral_dot_size = scaled_dot_area(dots["neutral_threshold_pct"], dots)

    _axes_info = []
    for _i, (_subfig, (_diff, _cfg, _section_name, _lay)) in enumerate(
        zip(_subfigs, panels)
    ):
        _strat_dim = _cfg.get("stratification", "region")
        _model_source = _cfg["model"]["label"]
        _baseline_source = _cfg["baseline"]["label"]
        _subfig_w_in = panel_widths[_i]

        _ax = _subfig.add_subplot(1, 1, 1)
        _ax.set_xlim(_lay["xlim_left"], _lay["plot_width"])
        _ax.set_ylim(_lay["y_bottom"], _lay["y_top"])
        _ax.axis("off")

        _subfig.subplots_adjust(
            top=1 - figure_cfg["title_margin_in"] / panel_height,
            bottom=legend_h_in / panel_height,
        )

        # Align the group-label left edge with the title's left anchor (subfig x=0.01).
        # _ax.get_position() gives the axes bbox in subfig fraction after subplots_adjust.
        _title_x = 0.01
        _ax_pos = _ax.get_position()
        _metric_x_frac = (layout_cfg["metric_x"] - _lay["xlim_left"]) / (
            _lay["plot_width"] - _lay["xlim_left"]
        )
        _group_dx_in = (
            _title_x - _ax_pos.x0 - _ax_pos.width * _metric_x_frac
        ) * _subfig_w_in
        _group_transform = _ax.transData + ScaledTranslation(
            _group_dx_in,
            0,
            _fig.dpi_scale_trans,
        )
        _letter = chr(ord("a") + _i)
        _title_y = 0.99
        _subfig.text(
            _title_x,
            _title_y,
            f"{_letter}) {_section_name} with {_baseline_source} as baseline",
            fontsize=fonts["title"],
            ha="left",
            va="top",
        )

        # _draw_data_rows uses ha="right" for group labels; override it on the axes
        # instance so it uses ha="left" (matching the title anchor above).
        _orig_ax_text = _ax.text

        def _left_text(*args, **kwargs):
            if kwargs.get("transform") is _group_transform:
                kwargs = {**kwargs, "ha": "left"}
            return _orig_ax_text(*args, **kwargs)

        _ax.text = _left_text
        _sep_ys = draw_data_rows(
            _ax,
            _diff,
            _lay["rows"],
            _lay["slices"],
            _strat_dim,
            _lay["n_leads"],
            neutral_dot_size,
            _group_transform,
            _cfg,
        )
        _ax.text = _orig_ax_text
        draw_slice_headers(
            _ax,
            _lay["slices"],
            _lay["n_leads"],
            _lay["lead_hours"],
            _lay["y_bottom"],
            _strat_dim,
            _cfg,
        )
        _axes_info.append((_ax, _lay, _sep_ys, _cfg, _model_source, _baseline_source))

    def _combined_label(labels):
        if len(set(labels)) == 1:
            return labels[0]
        prefix = ""
        for chars in zip(*labels):
            if len(set(chars)) == 1:
                prefix += chars[0]
            else:
                break
        suffix = ""
        for chars in zip(*[l[::-1] for l in labels]):
            if len(set(chars)) == 1:
                suffix = chars[0] + suffix
            else:
                break
        n = len(suffix)
        middles = [l[len(prefix): len(l) - n if n else len(l)] for l in labels]
        return prefix + "/".join(middles) + suffix

    _unified_baseline = _combined_label([bs for _, _, _, _, _, bs in _axes_info])
    _unified_model = _combined_label([ms for _, _, _, _, ms, _ in _axes_info])

    for _ax, _lay, _sep_ys, _cfg, _model_source, _baseline_source in _axes_info:
        _colors = _cfg["plot"]["colors"]
        _ax_w_in = _ax.get_window_extent().width / _fig.dpi

        _metric_frac = (layout_cfg["metric_x"] - _lay["xlim_left"]) / (
            _lay["plot_width"] - _lay["xlim_left"]
        )
        _hline_x0 = (
            _metric_frac
            - ((_lay["metric_label_w_pt"] - hline_cfg["start_pad_pt"]) / 72) / _ax_w_in
        )
        for _sy in _sep_ys:
            _ax.axhline(
                y=_sy,
                xmin=_hline_x0,
                xmax=hline_cfg["x_end"],
                color=_colors["hline"],
                lw=hline_cfg["linewidth"],
            )

    # Single spanning legend across all panels.
    _leg_colors = _axes_info[0][3]["plot"]["colors"]
    _sample_pcts = legend_cfg["sample_pcts"]
    _neutral_pct = dots["neutral_threshold_pct"]
    _dot_specs = (
        [(_sample_pcts[0], _leg_colors["baseline_better"], f"≤-{_sample_pcts[0]}%")]
        + [(p, _leg_colors["baseline_better"], f"-{p}%") for p in _sample_pcts[1:]]
        + [(_neutral_pct, _leg_colors["neutral"], f"|Δ|<{_neutral_pct}%")]
        + [
            (p, _leg_colors["model_better"], f"+{p}%")
            for p in reversed(_sample_pcts[1:])
        ]
        + [(_sample_pcts[0], _leg_colors["model_better"], f"≥+{_sample_pcts[0]}%")]
    )
    _x_span = min(legend_cfg["width_in"] / fig_width, 0.8)
    _x_dots = np.linspace(0.5 - _x_span / 2, 0.5 + _x_span / 2, len(_dot_specs))
    _legend_ax = _fig.add_axes([0, legend_h_in / panel_height, 1, 1e-6])
    _legend_ax.set_xlim(0, 1)
    _legend_ax.axis("off")
    draw_legend(
        _legend_ax,
        _fig,
        _dot_specs,
        _x_dots,
        has_missing,
        small_fs,
        neutral_dot_size,
        _unified_model,
        _unified_baseline,
        _axes_info[0][3],
    )

    _out = Path(resolved_output)
    _out.mkdir(parents=True, exist_ok=True)
    _pdf = _out / "publication_scorecard.pdf"
    _png = _out / "publication_scorecard.png"
    LOG.info("scorecard: saving figures to %s", _out)
    _fig.savefig(_pdf, bbox_inches="tight")
    _fig.savefig(_png, dpi=200, bbox_inches="tight")
    plt.close(_fig)
    LOG.info("scorecard: figures saved")
    (_out / "publication_scorecards.html").write_text(
        "<!doctype html><html><body>"
        '<img src="publication_scorecard.png" style="max-width:100%">'
        "</body></html>"
    )
    return


if __name__ == "__main__":
    app.run()
