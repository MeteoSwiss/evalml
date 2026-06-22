import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path
    import marimo as mo

    _script_dir = Path(__file__).resolve().parent  # workflow/scripts/
    sys.path.append(str(_script_dir))
    project_root = _script_dir.parent.parent  # repo root
    return Path, mo, project_root, sys


@app.cell
def _(mo, sys):
    _VERIF_DEFAULT = (
        "output/data/baselines/baseline-7e02/verif_aggregated_2b83.nc"
        " output/data/baselines/baseline-ce47/verif_aggregated_2b83.nc"
        " output/data/baselines/baseline-7342/verif_aggregated_2b83.nc"
        " output/data/baselines/baseline-e0f0/verif_aggregated_2b83.nc"
        " output/data/runs/temporal_downscaler-f927-1ee3-on-forecaster-c304-23e7/495c/verif_aggregated_2b83.nc"
    )
    _SOURCE_DEFAULT = (
        "ICON-CH1-CTRL,ICON-CH2-CTRL,ICON-CH1-EPS mean,ICON-CH2-EPS mean,Varda-Single"
    )

    _cli = mo.cli_args()
    if not _cli and len(sys.argv) > 1:
        import argparse

        _p = argparse.ArgumentParser()
        _p.add_argument("--verif_files", default=_VERIF_DEFAULT)
        _p.add_argument("--sources", default=_SOURCE_DEFAULT)
        _p.add_argument("--output", default="figures/debug")
        _parsed, _ = _p.parse_known_args()
        _default_verif = _parsed.verif_files
        _default_sources = _parsed.sources
        _default_output = _parsed.output
        _is_script = True
    else:
        _default_verif = _cli.get("verif_files", default=_VERIF_DEFAULT)
        _default_sources = _cli.get("sources", default=_SOURCE_DEFAULT)
        _default_output = _cli.get("output", default="figures/debug")
        _is_script = bool(_cli)

    verif_input = mo.ui.text(
        value=_default_verif, label="Verification files (space-separated)"
    )
    sources_input = mo.ui.text(
        value=_default_sources, label="Sources (comma-separated)"
    )
    output_input = mo.ui.text(value=_default_output, label="Output directory")

    mo.vstack([verif_input, sources_input, output_input]) if not _is_script else None
    return output_input, sources_input, verif_input


@app.cell
def _(output_input, sources_input, verif_input):
    verif_files = verif_input.value.split() if verif_input.value else []
    sources = [s.strip() for s in sources_input.value.split(",") if s.strip()]
    output_dir = output_input.value
    return output_dir, sources, verif_files


@app.cell
def _(Path, project_root, verif_files):
    import xarray as xr
    from verification_plot_metrics import (
        _ensure_unique_lead_time as ensure_unique_lead_time,
        _select_best_sources as select_best_sources,
        decode_metric,
    )

    def _abs(f):
        p = Path(f)
        return p if p.is_absolute() else project_root / p

    dfs = [xr.open_dataset(_abs(f)) for f in verif_files]
    dfs = [ensure_unique_lead_time(d) for d in dfs]
    dfs = select_best_sources(dfs)
    ds = xr.concat(dfs, dim="source", join="outer")

    # extract only  non-spatial variables to pd.DataFrame
    nonspatial_vars = [d for d in ds.data_vars if "spatial" not in d and "." in d]
    df = ds[nonspatial_vars].to_array("stack").to_dataframe(name="value").reset_index()
    df[["param", "metric"]] = df["stack"].str.split(".", n=1, expand=True)
    df["metric"] = df.metric.apply(decode_metric)
    df.drop(columns=["stack"], inplace=True)
    df["step"] = df["step"].dt.total_seconds() / 3600
    # convert numeric column init_hour to string in format HH:00 UTC and replace -999 with "all"
    df["init_hour"] = df["init_hour"].astype(str).str.zfill(2) + ":00 UTC"
    df["init_hour"] = df["init_hour"].where(df["init_hour"] != "-999:00 UTC", "all")
    return (df,)


@app.cell
def _(df):
    # Filter to aggregated-over-all-strata slice for overview plots
    df_all = df[
        (df["region"] == "all") & (df["season"] == "all") & (df["init_hour"] == "all")
    ].copy()
    return (df_all,)


@app.cell
def _(Path, df_all, mo, output_dir, sources):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    from publication_style import line_style, param_label

    plt.style.use(Path(__file__).resolve().parent / "publication.mplstyle")

    _PARAMS = ["T_2M", "TOT_PREC", "U_10M"]
    _METRICS = ["BIAS", "RMSE"]
    # ETS panels: TOT_PREC > 0, TOT_PREC > 5.0, U_10M at largest threshold
    _prec_ets = sorted(
        m
        for m in df_all[df_all["param"] == "TOT_PREC"]["metric"].unique()
        if "ETS" in m
    )
    _wind_ets = sorted(
        m for m in df_all[df_all["param"] == "U_10M"]["metric"].unique() if "ETS" in m
    )

    def _find_ets(metrics, threshold):
        return next((m for m in metrics if m.endswith(f"> {threshold}")), None)

    _ets_panels = [
        ("TOT_PREC", _find_ets(_prec_ets, "0.0"), "mm"),
        ("TOT_PREC", _find_ets(_prec_ets, "5.0"), "mm"),
        ("U_10M", _wind_ets[-1] if _wind_ets else None, "m/s"),
    ]

    _out = Path(output_dir)
    _out.mkdir(parents=True, exist_ok=True)

    _fig, _axes = plt.subplots(
        len(_METRICS) + 1,
        len(_PARAMS),
        figsize=(4 * len(_PARAMS), 3 * (len(_METRICS) + 1)),
        sharex=True,
    )

    _xscale_kw = dict(
        functions=(
            lambda x: np.sign(x) * np.abs(x) ** 0.7,
            lambda x: np.sign(x) * np.abs(x) ** (1 / 0.7),
        )
    )
    _xticks = mticker.FixedLocator([0, 3, 6, 12, 24, 36, 48, 72, 96, 120])

    # Rows 0…N-1: one panel per (metric, param)
    for _row, _metric in enumerate(_METRICS):
        for _col, _param in enumerate(_PARAMS):
            _ax = _axes[_row, _col]
            _data = df_all[(df_all["param"] == _param) & (df_all["metric"] == _metric)]
            for _src in sources:
                _grp = _data[_data["source"] == _src].sort_values("step")
                if _grp.empty:
                    continue
                _ax.plot(_grp["step"], _grp["value"], label=_src, **line_style(_src))
            _ax.set_xscale("function", **_xscale_kw)
            _ax.xaxis.set_major_locator(_xticks)
            if _row == 0:
                _ax.set_title(param_label(_param))
            if _col == 0:
                _ax.set_ylabel(_metric)

    # Last row: three specific ETS panels
    _ets_row = len(_METRICS)
    for _col, (_ets_param, _ets_metric, _ets_unit) in enumerate(_ets_panels):
        _ax = _axes[_ets_row, _col]
        if _ets_metric is None:
            _ax.set_visible(False)
            continue
        _data = df_all[
            (df_all["param"] == _ets_param) & (df_all["metric"] == _ets_metric)
        ]
        for _src in sources:
            _grp = _data[_data["source"] == _src].sort_values("step")
            if _grp.empty:
                continue
            _ax.plot(_grp["step"], _grp["value"], label=_src, **line_style(_src))
        _ax.set_xscale("function", **_xscale_kw)
        _ax.xaxis.set_major_locator(_xticks)
        _panel_label = (
            _ets_metric.replace("ETS", param_label(_ets_param)) + f" {_ets_unit}"
        )
        _ax.text(
            0.97, 0.97, _panel_label, transform=_ax.transAxes, ha="right", va="top"
        )
        _ax.set_xlabel("Lead time (h)")
        if _col == 0:
            _ax.set_ylabel("ETS")

    _axes[0, 0].set_xlim(-1, 126)

    _handles, _labels = _axes[0, 0].get_legend_handles_labels()
    _fig.legend(
        _handles,
        _labels,
        loc="lower center",
        ncol=len(sources),
        bbox_to_anchor=(0.5, 0.06),
    )
    _fig.tight_layout()
    _fig.subplots_adjust(bottom=0.22)

    # Extra gap between last metric row and ETS row
    _extra_gap = 0.04
    for _c in range(len(_PARAMS)):
        _pos = _axes[_ets_row, _c].get_position()
        _axes[_ets_row, _c].set_position(
            [_pos.x0, _pos.y0 - _extra_gap, _pos.width, _pos.height]
        )

    # Light-grey background behind the TOT_PREC panels (incl. their titles/labels).
    # Derived from the axes' *tight* bounding boxes so it stays aligned at any
    # font size. Two overlapping rectangles form the stepped shape: a narrow one
    # over the TOT_PREC metric column (col 1) and a wider one under the two
    # TOT_PREC ETS panels (cols 0-1). Axes keep white faces, so grey shows only
    # in the margins around them.
    from matplotlib.patches import Rectangle as _Rect

    _PREC_BG = "#f0f0f0"
    _pad = 0.008

    _fig.canvas.draw()  # ensure renderer + final positions for tight bboxes
    _rend = _fig.canvas.get_renderer()
    _inv = _fig.transFigure.inverted()

    def _union_bbox(_axs):
        _pts = [_inv.transform(_a.get_tightbbox(_rend).get_points()) for _a in _axs]
        return (
            min(_p[0][0] for _p in _pts),
            min(_p[0][1] for _p in _pts),
            max(_p[1][0] for _p in _pts),
            max(_p[1][1] for _p in _pts),
        )

    _ux0, _uy0, _ux1, _uy1 = _union_bbox([_axes[_r, 1] for _r in range(len(_METRICS))])
    _lx0, _ly0, _lx1, _ly1 = _union_bbox([_axes[_ets_row, 0], _axes[_ets_row, 1]])
    _mid = (_uy0 + _ly1) / 2  # split point in the gap between metric and ETS blocks
    for _x0, _y0, _x1, _y1 in [
        (_ux0, _mid, _ux1, _uy1),  # upper block (col 1)
        (_lx0, _ly0, _lx1, _mid),  # lower block (cols 0-1), overlaps at _mid
    ]:
        _fig.add_artist(
            _Rect(
                (_x0 - _pad, _y0 - _pad),
                (_x1 - _x0) + 2 * _pad,
                (_y1 - _y0) + 2 * _pad,
                facecolor=_PREC_BG,
                edgecolor="none",
                zorder=-1,
                transform=_fig.transFigure,
            )
        )

    _fname = _out / "publication_figures_leadtime.pdf"
    _fig.savefig(_fname, bbox_inches="tight")
    _fig.savefig(_fname.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(_fig)

    (_out / "publication_figures.html").write_text(
        "<!doctype html><html><body>"
        '<img src="publication_figures_leadtime.png" style="max-width:100%">'
        "</body></html>"
    )

    mo.image(str(_fname.with_suffix(".png")))
    return


@app.cell
def _(df_all):
    df_all
    return


if __name__ == "__main__":
    app.run()
