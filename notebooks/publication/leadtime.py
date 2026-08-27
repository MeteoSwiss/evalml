import marimo

__generated_with = "0.23.3"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path

    # Repo root = two levels up from notebooks/publication/. Notebooks run with cwd
    # at the repo root when invoked via nbconvert; fall back by walking up from cwd.
    PROJECT_ROOT = Path.cwd().resolve()
    if not (PROJECT_ROOT / "workflow").is_dir():
        # Walk up until we find the repo root (contains both workflow/ and src/).
        for _p in [PROJECT_ROOT] + list(PROJECT_ROOT.parents):
            if (_p / "workflow").is_dir() and (_p / "src").is_dir():
                PROJECT_ROOT = _p
                break
    # verification_plot_metrics lives in workflow/scripts (shared with the main
    # workflow, deliberately not moved); plotting/data_input/verification come from
    # the editable src/ install. Style is a proper package import (no path hack).
    sys.path.insert(0, str(PROJECT_ROOT / "workflow" / "scripts"))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    import matplotlib.pyplot as plt
    from evalml.publication import style

    plt.style.use(style.mplstyle_path())
    return PROJECT_ROOT, Path, plt


@app.cell
def _(PROJECT_ROOT, Path):
    import pandas as _pd
    import xarray as _xr
    from evalml.publication.manifest import load_manifest, figures_dir
    from verification_plot_metrics import (
        _ensure_unique_lead_time as ensure_unique_lead_time,
        _select_best_sources as select_best_sources,
        decode_metric as _decode_metric,
    )

    m = load_manifest(PROJECT_ROOT / "output/manifests/manifest_varda-single_paper_analysis.json")
    m.validate_request("figures")

    pairs = m.verif_paths()
    output_dir = str(PROJECT_ROOT / figures_dir(m.output_root, m.truth["label"]) / "leadtime")

    def _abs(f):
        p = Path(f)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def _load_participant_df(path, label):
        """Load one verification .nc and return a tidy DataFrame labelled with label."""
        try:
            _ds = _xr.open_dataset(_abs(path))
        except Exception as e:
            print(f"Skipping {label} ({path}): {e}")
            return None
        [_ds] = select_best_sources([ensure_unique_lead_time(_ds)])
        _src_vals = [s for s in _ds.source.values if not str(s).startswith("truth-")]
        if not _src_vals:
            return None
        _ds = _ds.sel(source=_src_vals)
        _vars = [v for v in _ds.data_vars if "spatial" not in v and "." in v]
        if not _vars:
            return None
        _df = (
            _ds[_vars]
            .to_array("stack")
            .to_dataframe(name="value")
            .reset_index()
        )
        _df[["param", "metric_raw"]] = _df["stack"].str.split(".", n=1, expand=True)
        _df["metric"] = _df["metric_raw"].apply(_decode_metric)
        _df["step"] = _df["step"].dt.total_seconds() / 3600
        _df["init_hour"] = _df["init_hour"].astype(str).str.zfill(2) + ":00 UTC"
        _df["init_hour"] = _df["init_hour"].where(_df["init_hour"] != "-999:00 UTC", "all")
        _df["source"] = label
        return _df.drop(columns=["stack"])

    _loaded = [(lbl, _load_participant_df(p, lbl)) for p, lbl in pairs]

    df = _pd.concat([part for _, part in _loaded if part is not None], ignore_index=True)
    sources = [lbl for lbl, part in _loaded if part is not None]
    return df, output_dir, sources


@app.cell
def _(df):
    df_all = df[
        (df["region"] == "icon") & (df["season"] == "all") & (df["init_hour"] == "all")
    ].copy()
    return (df_all,)


@app.cell
def _(df, sources):
    import pandas as _pd

    def compute_skill_df(dataframe, all_sources):
        """Compute 1 - Score(other) / Score(Varda-Single) per source.

        General formula: skill = (S_fcst - S_baseline) / (S_perfect - S_baseline)
        where S_fcst = comparison source, S_baseline = Varda (reference),
        S_perfect = 1 for ETS metrics, 0 for all others (RMSE, BIAS², MAE, …).
        BIAS is squared first so it is a proper non-negative score before the ratio.
        """
        _varda_src = next((s for s in all_sources if "Varda" in s and "single" in s), None)
        _other_srcs = [s for s in all_sources if s != _varda_src]
        _join_cols = ["param", "metric", "metric_raw", "step", "region", "season", "init_hour"]
        _varda = (
            dataframe[dataframe["source"] == _varda_src][_join_cols + ["value"]]
            .rename(columns={"value": "_v"})
        )
        _result_parts = []
        for _src in _other_srcs:
            _src_df = dataframe[dataframe["source"] == _src].copy()
            if _src_df.empty:
                continue
            _merged = _src_df.merge(_varda, on=_join_cols, how="left")
            _is_ets = _merged["metric_raw"].str.startswith("ETS")
            _is_bias = _merged["metric_raw"] == "BIAS"
            _s_perfect = _is_ets.astype(float)  # 1 for ETS, 0 for RMSE/BIAS²/…
            # Square BIAS to make it a non-negative score before applying the formula
            _s_fcst = _merged["value"].where(~_is_bias, _merged["value"] ** 2)
            _s_baseline = _merged["_v"].where(~_is_bias, _merged["_v"] ** 2)
            _denom = _s_perfect - _s_baseline
            _merged["value"] = ((_s_fcst - _s_baseline) / _denom).clip(lower=-10, upper=1)
            _merged.loc[_denom.abs() < 1e-12, "value"] = float("nan")
            _result_parts.append(_merged.drop(columns=["_v"]))
        if not _result_parts:
            return dataframe.iloc[:0].copy()
        return _pd.concat(_result_parts, ignore_index=True)

    skill_sources = [s for s in sources if not ("Varda" in s and "single" in s)]
    _df_skill = compute_skill_df(df, sources)
    df_skill_all = _df_skill[
        (_df_skill["region"] == "icon")
        & (_df_skill["season"] == "all")
        & (_df_skill["init_hour"] == "all")
        & (_df_skill["metric"] != "BIAS")
    ].copy()
    return df_skill_all, skill_sources


@app.cell
def _():
    import matplotlib.pyplot as _plt
    import matplotlib.ticker as _mticker
    import numpy as _np

    from evalml.publication.style import (
        line_style as _line_style,
        mplstyle_path as _mplstyle_path,
        figure_width as _figure_width,
    )

    _XSCALE_KW = dict(
        functions=(
            lambda x: _np.sign(x) * _np.abs(x) ** 0.7,
            lambda x: _np.sign(x) * _np.abs(x) ** (1 / 0.7),
        )
    )
    _XTICKS = _mticker.FixedLocator([0, 3, 6, 12, 24, 36, 48, 72, 96, 120])

    def plot_panels(panels, df, sources, legend_ncol=None):
        """Draw one figure from a panel-spec DataFrame.

        panels columns: row_id, col_id, param_name, metric, param_text,
                        title_x, title_y, horizontal_line
        df may contain raw scores or pre-computed skill scores; sources
        must match the source values present in df.
        legend_ncol: columns in the figure legend (default: all sources in one row).
        """
        with _plt.style.context(_mplstyle_path()):
            nrows = panels["row_id"].max() + 1
            ncols = panels["col_id"].max() + 1
            # Fixed page width (2-column); height proportional to the grid.
            _w = _figure_width(2)
            fig, axes = _plt.subplots(
                nrows, ncols, figsize=(_w, _w * 0.75 * nrows / ncols), sharex=True
            )
            for _, p in panels.iterrows():
                ax = axes[p.row_id, p.col_id]
                data = df[(df["param"] == p.param_name) & (df["metric"] == p.metric)]
                for src in sources:
                    grp = data[data["source"] == src].sort_values("step")
                    if grp.empty:
                        continue
                    ax.plot(grp["step"], grp["value"], label=src, **_line_style(src))
                    if ("Varda" in src and "single" in src) or "AIFS" in src:
                        m6 = grp[grp["step"] % 6 == 0]
                        ax.plot(
                            m6["step"],
                            m6["value"],
                            linestyle="none",
                            marker="o",
                            markersize=5,
                            color=_line_style(src)["color"],
                        )
                ax.set_xscale("function", **_XSCALE_KW)
                ax.xaxis.set_major_locator(_XTICKS)
                ax.axhline(
                    p.horizontal_line, color="black", linestyle="dashed", linewidth=0.7, zorder=1.5
                )
                if p.param_text:
                    ax.text(
                        0.97,
                        0.97,
                        p.param_text,
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                    )
                if p.title_x:
                    ax.set_title(p.title_x, loc="center", y=1.05)
                if p.title_y:
                    ax.set_title(
                        p.title_y, x=-0.25, y=0.5, rotation=90, va="center", loc="left"
                    )
                _margin = 0.05
                _ymin, _ymax = ax.get_ylim()
                # Clamp first, then compute span so the margin doesn't inflate
                # _hi when extreme negative values widen the raw autoscale range.
                _lo_clamp = max(-2, _ymin)
                _clamped_span = _ymax - _lo_clamp
                _lo = _lo_clamp - _margin * _clamped_span
                _hi = _ymax + _margin * _clamped_span
                ax.set_ylim(_lo, _hi)
                ax.yaxis.set_major_locator(_mticker.MaxNLocator(nbins=4))
                if p.row_id == nrows - 1:
                    ax.set_xlabel("Lead time (h)")
            axes[0, 0].set_xlim(-1, 126)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            _order = sorted(
                range(len(labels)), key=lambda i: (0 if "Varda" in labels[i] else 1)
            )
            handles = [handles[i] for i in _order]
            labels = [labels[i] for i in _order]
            _ncol = legend_ncol if legend_ncol is not None else len(sources)
            _legend_rows = (len(labels) + _ncol - 1) // _ncol
            _subplot_bottom = 0.12 + 0.08 * _legend_rows
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=_ncol,
                bbox_to_anchor=(0.5, _subplot_bottom - 0.05),
                fontsize=_plt.rcParams["axes.labelsize"],
            )
            fig.tight_layout()
            fig.subplots_adjust(bottom=_subplot_bottom)
            return fig

    return (plot_panels,)


@app.cell
def _(
    Path,
    df_all,
    df_skill_all,
    output_dir,
    plot_panels,
    plt,
    skill_sources,
    sources,
):
    import matplotlib.pyplot as _plt
    import pandas as _pd
    from evalml.publication.style import param_label as _param_label

    _PARAMS = ["T_2M", "TOT_PREC1", "SP_10M"]

    def _find_ets(metrics, op, threshold):
        return next((m for m in metrics if m.endswith(f"{op} {threshold}")), None)

    def _ets_specs(df):
        _t2m = sorted(m for m in df[df["param"] == "T_2M"]["metric"].unique() if "ETS" in m)
        _prec = sorted(m for m in df[df["param"] == "TOT_PREC1"]["metric"].unique() if "ETS" in m)
        _wind = sorted(m for m in df[df["param"] == "SP_10M"]["metric"].unique() if "ETS" in m)
        return [
            (0, 0, "T_2M", _find_ets(_t2m, "<", "273.15"), "< 0 °C"),
            (0, 1, "TOT_PREC1", _find_ets(_prec, ">", "0.0"), "> 0 mm"),
            (0, 2, "SP_10M", _find_ets(_wind, ">", "5.0"), "> 5 m/s"),
            (1, 0, "T_2M", _find_ets(_t2m, ">", "298.15"), "> 25 °C"),
            (1, 1, "TOT_PREC1", _find_ets(_prec, ">", "5.0"), "> 5 mm"),
            (1, 2, "SP_10M", _find_ets(_wind, ">", "10.0"), "> 10 m/s"),
        ]

    def _build_combined_panels(df, metric_labels):
        """Build panel spec: score rows from metric_labels, then ETS rows."""
        _score_rows = [
            {
                "row_id": row_id,
                "col_id": col_id,
                "param_name": param,
                "metric": metric,
                "param_text": "",
                "title_x": _param_label(param) if row_id == 0 else "",
                "title_y": label if col_id == 0 else "",
                "horizontal_line": 0,
            }
            for row_id, (metric, label) in enumerate(metric_labels.items())
            for col_id, param in enumerate(_PARAMS)
        ]
        _ets_offset = len(metric_labels)
        _ets_rows = [
            {
                "row_id": ets_row + _ets_offset,
                "col_id": col_id,
                "param_name": param,
                "metric": metric,
                "param_text": param_text,
                "title_x": "",
                "title_y": "ETS" if col_id == 0 else "",
                "horizontal_line": None,
            }
            for ets_row, col_id, param, metric, param_text in _ets_specs(df)
        ]
        return _pd.DataFrame(_score_rows + _ets_rows)

    _out = Path(output_dir)
    _out.mkdir(parents=True, exist_ok=True)

    _panels = _build_combined_panels(df_all, {"RMSE": "RMSE", "BIAS": "BIAS"})
    _fig = plot_panels(_panels, df_all, sources, legend_ncol=(len(sources) + 1) // 2)
    _fname = _out / "publication_leadtime.pdf"
    _fig.savefig(_fname, bbox_inches="tight")
    _fig.savefig(_fname.with_suffix(".png"), dpi=250, bbox_inches="tight")
    _plt.close(_fig)

    _panels_skill = _build_combined_panels(
        df_skill_all, {"RMSE": "RMSE skill"}
    )
    _fig_skill = plot_panels(
        _panels_skill, df_skill_all, skill_sources, legend_ncol=(len(skill_sources) + 1) // 2
    )
    _fname_skill = _out / "publication_leadtime_skill.pdf"
    _fig_skill.savefig(_fname_skill, bbox_inches="tight")
    _fig_skill.savefig(_fname_skill.with_suffix(".png"), dpi=250, bbox_inches="tight")
    _plt.close(_fig_skill)

    (_out / "publication_leadtime.html").write_text(
        "<!doctype html><html><body>"
        '<img src="publication_leadtime.png" style="max-width:100%"><br>'
        '<img src="publication_leadtime_skill.png" style="max-width:100%">'
        "</body></html>"
    )

    plt.show()
    return


if __name__ == "__main__":
    app.run()
