import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    import pathlib

    import marimo as mo
    import matplotlib.pyplot as plt

    _script_dir = pathlib.Path(__file__).resolve().parent  # workflow/scripts/
    sys.path.append(str(_script_dir))
    project_root = _script_dir.parent.parent

    plt.style.use(_script_dir / "publication.mplstyle")

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

    from publication_style import (
        COLOR_SKILL_BASELINE_BETTER,
        COLOR_SKILL_MODEL_BETTER,
    )

    return (
        COLOR_SKILL_BASELINE_BETTER,
        COLOR_SKILL_MODEL_BETTER,
        DEFAULT_PLOT_CFG,
        draw_data_rows,
        draw_legend,
        draw_slice_headers,
        filter_diff,
        load_relative_diff,
        measure_label_sizes,
        mo,
        parse_var_metrics,
        pathlib,
        project_root,
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
def _(mo, project_root, sys):
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
                project_root
                / f"output/figures/{_truth.get('slug') or _ts(_truth.get('label', ''))}/scorecards"
            )
        except Exception as _exc:
            _manifest_default = str(
                project_root / "output/publication/SwissMetNet/manifest.json"
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
        candidate = _cli.get("candidate", default=None)
    return candidate, manifest_path, output_dir


@app.cell
def _(manifest_path):
    manifest_path
    return


@app.cell
def _(LOG, candidate, manifest_path, output_dir, project_root):
    from evalml.publication.manifest import load_manifest

    mani = load_manifest(manifest_path)
    cand = mani.get_candidate(candidate)
    slug = mani.truth.get("slug", "truth")

    cand_info = {
        "id": cand.id,
        "label": cand.label,
        "verif_aggregated": project_root / cand.paths["verif_aggregated"],
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
                "base_verif": project_root / _base.paths["verif_aggregated"],
            }
        )
        LOG.info("scorecard: section %r — baseline %r", d["name"], d["base_label"])

    resolved_output = output_dir or str(project_root / f"output/figures/{slug}/scorecards")
    return cand_info, resolved_output, section_cfgs


@app.cell
def _(
    COLOR_SKILL_BASELINE_BETTER,
    COLOR_SKILL_MODEL_BETTER,
    DEFAULT_PLOT_CFG,
    LOG,
    cand_info,
    filter_diff,
    load_relative_diff,
    measure_label_sizes,
    mo,
    parse_var_metrics,
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
            slice_label_w_in * layout["slice_label_pad"] / (n_leads + layout["slice_gap"]),
        )
        plot_width = len(slices) * (n_leads + layout["slice_gap"]) - layout["slice_gap"]
        y_bottom = -(len(rows) - 0.5)
        y_top = layout["slice_y"] + slice_label_h_rows + layout["slice_y_pad"]
        xlim_left = layout["metric_x"] - figure["left_margin_in"] / col_width
        data_w_in = max(figure["width_min"], plot_width * col_width + figure["width_pad"])
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

    plot_cfg = copy.deepcopy(DEFAULT_PLOT_CFG)
    plot_cfg["colors"]["model_better"] = COLOR_SKILL_MODEL_BETTER
    plot_cfg["colors"]["baseline_better"] = COLOR_SKILL_BASELINE_BETTER

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
        + ", ".join(
            f"{name} ({len(lay['rows'])} rows)" for _, _, name, lay in panels
        )
    )
    return panels, plot_cfg


@app.cell
def _(
    LOG,
    draw_data_rows,
    draw_legend,
    draw_slice_headers,
    panels,
    pathlib,
    plot_cfg,
    resolved_output,
    scaled_dot_area,
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.transforms import ScaledTranslation

    figure_cfg = plot_cfg["figure"]
    fonts = plot_cfg["fonts"]
    legend_cfg = plot_cfg["legend"]
    dots = plot_cfg["dots"]
    hline_cfg = plot_cfg["hline"]
    layout_cfg = plot_cfg["layout"]

    plt.rcParams["font.family"] = plot_cfg["rcparams"]["font_family"]
    plt.rcParams["font.sans-serif"] = [plot_cfg["rcparams"]["font_sans"]]
    plt.rcParams["figure.dpi"] = plot_cfg["rcparams"]["dpi"]

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

    # Vertical layout: one panel per section, stacked top-to-bottom.
    panel_widths = [lay["data_w_in"] for _, _, _, lay in panels]
    panel_heights = [
        figure_cfg["title_margin_in"]
        + legend_h_in
        + figure_cfg["row_height"] * (lay["y_top"] - lay["y_bottom"])
        for _, _, _, lay in panels
    ]
    fig_width = max(panel_widths)
    fig_height = sum(panel_heights)

    LOG.info("scorecard: rendering figure (%.1f × %.1f in)", fig_width, fig_height)
    _fig = plt.figure(figsize=(fig_width, fig_height))
    _subfigs = _fig.subfigures(len(panels), 1, height_ratios=panel_heights)
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
        _panel_h = panel_heights[_i]

        _ax = _subfig.add_subplot(1, 1, 1)
        _ax.set_xlim(_lay["xlim_left"], _lay["plot_width"])
        _ax.set_ylim(_lay["y_bottom"], _lay["y_top"])
        _ax.axis("off")

        _subfig.subplots_adjust(
            top=1 - figure_cfg["title_margin_in"] / _panel_h,
            bottom=legend_h_in / _panel_h,
        )

        _group_transform = _ax.transData + ScaledTranslation(
            -(_lay["metric_label_w_pt"] + layout_cfg["group_metric_gap_pt"]) / 72,
            0,
            _fig.dpi_scale_trans,
        )
        _letter = chr(ord("a") + _i)
        _subfig.text(
            0.01,
            0.99,
            f"{_letter}) {_section_name} with {_baseline_source} as baseline",
            fontsize=fonts["title"],
            fontweight="bold",
            ha="left",
            va="top",
        )

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

    for _ax, _lay, _sep_ys, _cfg, _model_source, _baseline_source in _axes_info:
        _colors = _cfg["plot"]["colors"]
        _ax_w_in = _ax.get_window_extent().width / _fig.dpi

        _metric_frac = (layout_cfg["metric_x"] - _lay["xlim_left"]) / (
            _lay["plot_width"] - _lay["xlim_left"]
        )
        _hline_x0 = _metric_frac - (
            (_lay["metric_label_w_pt"] - hline_cfg["start_pad_pt"]) / 72
        ) / _ax_w_in
        for _sy in _sep_ys:
            _ax.axhline(
                y=_sy,
                xmin=_hline_x0,
                xmax=hline_cfg["x_end"],
                color=_colors["hline"],
                lw=hline_cfg["linewidth"],
            )

        _sample_pcts = legend_cfg["sample_pcts"]
        _neutral_pct = dots["neutral_threshold_pct"]
        _dot_specs = (
            [(_sample_pcts[0], _colors["baseline_better"], f"≤-{_sample_pcts[0]}%")]
            + [(p, _colors["baseline_better"], f"-{p}%") for p in _sample_pcts[1:]]
            + [(_neutral_pct, _colors["neutral"], f"|Δ|<{_neutral_pct}%")]
            + [(p, _colors["model_better"], f"+{p}%") for p in reversed(_sample_pcts[1:])]
            + [(_sample_pcts[0], _colors["model_better"], f"≥+{_sample_pcts[0]}%")]
        )
        _x_span = min(legend_cfg["width_in"] / _ax_w_in, 0.8)
        _cx = ((_lay["plot_width"] - 1) / 2 - _lay["xlim_left"]) / (
            _lay["plot_width"] - _lay["xlim_left"]
        )
        _x_dots = np.linspace(_cx - _x_span / 2, _cx + _x_span / 2, len(_dot_specs))
        draw_legend(
            _ax,
            _fig,
            _dot_specs,
            _x_dots,
            has_missing,
            small_fs,
            neutral_dot_size,
            _model_source,
            _baseline_source,
            _cfg,
        )

    _out = pathlib.Path(resolved_output)
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
