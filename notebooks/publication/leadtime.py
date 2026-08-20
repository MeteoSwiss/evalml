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
    import warnings
    import xarray as xr
    from evalml.publication.manifest import load_manifest, figures_dir
    from verification_plot_metrics import (
        _ensure_unique_lead_time as ensure_unique_lead_time,
        _select_best_sources as select_best_sources,
        decode_metric,
    )

    # Auto-discovers output/publication/<truth>/manifest.json (or $EVALML_MANIFEST,
    # or set truth=... below when several truths exist).
    m = load_manifest(PROJECT_ROOT / "output/manifests/manifest_varda-single_paper_stations.json")
    m.validate_request("figures")

    pairs = m.verif_paths()  # [(path, label), ...]
    sources = [label for _, label in pairs]
    output_dir = str(PROJECT_ROOT / figures_dir(m.output_root, m.truth["label"]) / "leadtime")

    def _abs(f):
        p = Path(f)
        return p if p.is_absolute() else PROJECT_ROOT / p

    # Build a mapping from internal source IDs (stored in the .nc source coordinate)
    # to the human-readable labels from the manifest.  Each participant file contains
    # the participant's own internal ID plus a truth-* source; we skip the truth entry.
    _id_to_label = {}
    for _path, _label in pairs:
        _ds_tmp = xr.open_dataset(_abs(_path))
        for _sid in _ds_tmp.source.values:
            if not str(_sid).startswith("truth-"):
                _id_to_label[str(_sid)] = _label
        _ds_tmp.close()

    _dfs = [xr.open_dataset(_abs(p)) for p, _ in pairs]
    _dfs = [ensure_unique_lead_time(d) for d in _dfs]
    _dfs = select_best_sources(_dfs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = xr.concat(_dfs, dim="source", join="outer")

    # Rename source coordinate values from internal IDs to manifest labels.
    _new_source_vals = [_id_to_label.get(str(s), str(s)) for s in ds.source.values]
    ds = ds.assign_coords(source=_new_source_vals)
    return decode_metric, ds, output_dir, sources


@app.cell
def _(decode_metric, ds):
    def ds_to_df(dataset):
        """Convert a verification xarray Dataset to a tidy DataFrame."""
        _nonspatial_vars = [
            d for d in dataset.data_vars if "spatial" not in d and "." in d
        ]
        _df = (
            dataset[_nonspatial_vars]
            .to_array("stack")
            .to_dataframe(name="value")
            .reset_index()
        )
        _df[["param", "metric"]] = _df["stack"].str.split(".", n=1, expand=True)
        _df["metric"] = _df["metric"].apply(decode_metric)
        _df.drop(columns=["stack"], inplace=True)
        _df["step"] = _df["step"].dt.total_seconds() / 3600
        _df["init_hour"] = _df["init_hour"].astype(str).str.zfill(2) + ":00 UTC"
        _df["init_hour"] = _df["init_hour"].where(
            _df["init_hour"] != "-999:00 UTC", "all"
        )
        return _df

    df = ds_to_df(ds)
    return df, ds_to_df


@app.cell
def _(df):
    df_all = df[
        (df["region"] == "icon") & (df["season"] == "all") & (df["init_hour"] == "all")
    ].copy()
    return (df_all,)


@app.cell
def _(ds, ds_to_df, sources):
    import xarray as _xr

    def compute_skill_ds(dataset, srcs):
        """Compute 1 - Score(baseline) / Score(Varda-Single) for all baselines.

        Only processes sources explicitly listed in srcs (minus Varda-Single),
        so dataset sources not requested by the user (e.g. truth sources) are
        excluded. For BIAS variables, scores are squared before taking the ratio
        so that the denominator is always non-negative.
        """
        _varda_src = next((s for s in srcs if "Varda" in s and "Single" in s), None)
        _baseline_srcs = [s for s in srcs if s != _varda_src]
        _available = set(dataset.source.values.tolist())
        _baseline_srcs = [s for s in _baseline_srcs if s in _available]
        _varda = dataset.sel(source=_varda_src)
        _baselines = dataset.sel(source=_baseline_srcs)
        _skill_vars = {}
        for _var in _baselines.data_vars:
            _metric_raw = _var.split(".", 1)[1] if "." in _var else ""
            _b = _baselines[_var].astype(float)
            _v = _varda[_var].astype(float)
            if _metric_raw == "BIAS":
                _b = _b**2
                _v = _v**2
            _skill_vars[_var] = _xr.where(_v != 0, 1 - _b / _v, float("nan"))
        return _xr.Dataset(_skill_vars)

    skill_sources = [s for s in sources if not ("Varda" in s and "Single" in s)]
    ds_skill = compute_skill_ds(ds, sources)
    _df_skill = ds_to_df(ds_skill)
    df_skill_all = _df_skill[
        (_df_skill["region"] == "all")
        & (_df_skill["season"] == "all")
        & (_df_skill["init_hour"] == "all")
    ].copy()
    return df_skill_all, skill_sources


@app.cell
def _():
    import matplotlib.pyplot as _plt
    import matplotlib.ticker as _mticker
    import numpy as _np

    from evalml.publication.style import line_style as _line_style

    _XSCALE_KW = dict(
        functions=(
            lambda x: _np.sign(x) * _np.abs(x) ** 0.7,
            lambda x: _np.sign(x) * _np.abs(x) ** (1 / 0.7),
        )
    )
    _XTICKS = _mticker.FixedLocator([0, 3, 6, 12, 24, 36, 48, 72, 96, 120])

    def plot_panels(panels, df, sources):
        """Draw one figure from a panel-spec DataFrame.

        panels columns: row_id, col_id, param_name, metric, param_text,
                        title_x, title_y, zero_line
        df may contain raw scores or pre-computed skill scores; sources
        must match the source values present in df.
        """
        nrows = panels["row_id"].max() + 1
        ncols = panels["col_id"].max() + 1
        fig, axes = _plt.subplots(
            nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True
        )
        for _, p in panels.iterrows():
            ax = axes[p.row_id, p.col_id]
            data = df[(df["param"] == p.param_name) & (df["metric"] == p.metric)]
            for src in sources:
                grp = data[data["source"] == src].sort_values("step")
                if grp.empty:
                    continue
                ax.plot(grp["step"], grp["value"], label=src, **_line_style(src))
                if "Varda" in src and "Single" in src:
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
            if p.zero_line:
                ax.axhline(
                    0, color="black", linestyle="dashed", linewidth=0.7, zorder=0
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
            if p.row_id == nrows - 1:
                ax.set_xlabel("Lead time (h)")
        axes[0, 0].set_xlim(-1, 126)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        _order = sorted(
            range(len(labels)), key=lambda i: (0 if "Varda" in labels[i] else 1)
        )
        handles = [handles[i] for i in _order]
        labels = [labels[i] for i in _order]
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(sources),
            bbox_to_anchor=(0.5, 0.02),
        )
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.2)
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
    _METRICS = ["RMSE", "BIAS"]

    _panels = _pd.DataFrame(
        [
            {
                "row_id": row_id,
                "col_id": col_id,
                "param_name": param,
                "metric": metric,
                "param_text": "",
                "title_x": _param_label(param) if row_id == 0 else "",
                "title_y": metric if col_id == 0 else "",
                "zero_line": metric == "BIAS",
            }
            for row_id, metric in enumerate(_METRICS)
            for col_id, param in enumerate(_PARAMS)
        ]
    )

    _out = Path(output_dir)
    _out.mkdir(parents=True, exist_ok=True)
    _fig = plot_panels(_panels, df_all, sources)
    _fname = _out / "publication_figures_rmse_bias.pdf"
    _fig.savefig(_fname, bbox_inches="tight")
    _fig.savefig(_fname.with_suffix(".png"), dpi=200, bbox_inches="tight")
    _plt.close(_fig)

    import matplotlib.pyplot as _plt
    import pandas as _pd
    from evalml.publication.style import param_label as _param_label

    def _find_ets(metrics, op, threshold):
        return next((m for m in metrics if m.endswith(f"{op} {threshold}")), None)

    _t2m_ets = sorted(
        m for m in df_all[df_all["param"] == "T_2M"]["metric"].unique() if "ETS" in m
    )
    _prec_ets = sorted(
        m
        for m in df_all[df_all["param"] == "TOT_PREC1"]["metric"].unique()
        if "ETS" in m
    )
    _wind_ets = sorted(
        m for m in df_all[df_all["param"] == "SP_10M"]["metric"].unique() if "ETS" in m
    )

    _specs = [
        (
            0,
            0,
            "T_2M",
            _find_ets(_t2m_ets, "<", "273.15"),
            f"{_param_label('T_2M')} < 0 °C",
        ),
        (
            0,
            1,
            "TOT_PREC1",
            _find_ets(_prec_ets, ">", "0.0"),
            f"{_param_label('TOT_PREC1')} > 0 mm",
        ),
        (
            0,
            2,
            "SP_10M",
            _find_ets(_wind_ets, ">", "5.0"),
            f"{_param_label('SP_10M')} > 5 m/s",
        ),
        (
            1,
            0,
            "T_2M",
            _find_ets(_t2m_ets, ">", "298.15"),
            f"{_param_label('T_2M')} > 25 °C",
        ),
        (
            1,
            1,
            "TOT_PREC1",
            _find_ets(_prec_ets, ">", "5.0"),
            f"{_param_label('TOT_PREC1')} > 5 mm",
        ),
        (
            1,
            2,
            "SP_10M",
            _find_ets(_wind_ets, ">", "10.0"),
            f"{_param_label('SP_10M')} > 10 m/s",
        ),
    ]
    _panels = _pd.DataFrame(
        [
            {
                "row_id": row_id,
                "col_id": col_id,
                "param_name": param,
                "metric": metric,
                "param_text": param_text,
                "title_x": _param_label(param) if row_id == 0 else "",
                "title_y": "ETS" if col_id == 0 else "",
                "zero_line": False,
            }
            for row_id, col_id, param, metric, param_text in _specs
        ]
    )

    _out = Path(output_dir)
    _out.mkdir(parents=True, exist_ok=True)
    _fig = plot_panels(_panels, df_all, sources)
    _fname = _out / "publication_figures_ets.pdf"
    _fig.savefig(_fname, bbox_inches="tight")
    _fig.savefig(_fname.with_suffix(".png"), dpi=200, bbox_inches="tight")
    _plt.close(_fig)

    import matplotlib.pyplot as _plt
    import pandas as _pd
    from evalml.publication.style import param_label as _param_label

    _PARAMS = ["T_2M", "TOT_PREC1", "SP_10M"]
    _METRICS = ["RMSE", "BIAS"]
    _SKILL_LABELS = {"RMSE": "RMSE skill", "BIAS": "Bias² skill"}

    _panels = _pd.DataFrame(
        [
            {
                "row_id": row_id,
                "col_id": col_id,
                "param_name": param,
                "metric": metric,
                "param_text": "",
                "title_x": _param_label(param) if row_id == 0 else "",
                "title_y": _SKILL_LABELS[metric] if col_id == 0 else "",
                "zero_line": True,
            }
            for row_id, metric in enumerate(_METRICS)
            for col_id, param in enumerate(_PARAMS)
        ]
    )

    _out = Path(output_dir)
    _out.mkdir(parents=True, exist_ok=True)
    _fig = plot_panels(_panels, df_skill_all, skill_sources)
    _fname = _out / "publication_figures_rmse_bias_skill.pdf"
    _fig.savefig(_fname, bbox_inches="tight")
    _fig.savefig(_fname.with_suffix(".png"), dpi=200, bbox_inches="tight")
    _plt.close(_fig)

    import matplotlib.pyplot as _plt
    import pandas as _pd
    from evalml.publication.style import param_label as _param_label

    def _find_ets(metrics, op, threshold):
        return next((m for m in metrics if m.endswith(f"{op} {threshold}")), None)

    _t2m_ets = sorted(
        m
        for m in df_skill_all[df_skill_all["param"] == "T_2M"]["metric"].unique()
        if "ETS" in m
    )
    _prec_ets = sorted(
        m
        for m in df_skill_all[df_skill_all["param"] == "TOT_PREC1"]["metric"].unique()
        if "ETS" in m
    )
    _wind_ets = sorted(
        m
        for m in df_skill_all[df_skill_all["param"] == "SP_10M"]["metric"].unique()
        if "ETS" in m
    )

    _specs = [
        (
            0,
            0,
            "T_2M",
            _find_ets(_t2m_ets, "<", "273.15"),
            f"{_param_label('T_2M')} < 0 °C",
        ),
        (
            0,
            1,
            "TOT_PREC1",
            _find_ets(_prec_ets, ">", "0.0"),
            f"{_param_label('TOT_PREC1')} > 0 mm",
        ),
        (
            0,
            2,
            "SP_10M",
            _find_ets(_wind_ets, ">", "5.0"),
            f"{_param_label('SP_10M')} > 5 m/s",
        ),
        (
            1,
            0,
            "T_2M",
            _find_ets(_t2m_ets, ">", "298.15"),
            f"{_param_label('T_2M')} > 25 °C",
        ),
        (
            1,
            1,
            "TOT_PREC1",
            _find_ets(_prec_ets, ">", "5.0"),
            f"{_param_label('TOT_PREC1')} > 5 mm",
        ),
        (
            1,
            2,
            "SP_10M",
            _find_ets(_wind_ets, ">", "10.0"),
            f"{_param_label('SP_10M')} > 10 m/s",
        ),
    ]
    _panels = _pd.DataFrame(
        [
            {
                "row_id": row_id,
                "col_id": col_id,
                "param_name": param,
                "metric": metric,
                "param_text": param_text,
                "title_x": _param_label(param) if row_id == 0 else "",
                "title_y": "ETS skill" if col_id == 0 else "",
                "zero_line": True,
            }
            for row_id, col_id, param, metric, param_text in _specs
        ]
    )

    _out = Path(output_dir)
    _out.mkdir(parents=True, exist_ok=True)
    _fig = plot_panels(_panels, df_skill_all, skill_sources)
    _fname = _out / "publication_figures_ets_skill.pdf"
    _fig.savefig(_fname, bbox_inches="tight")
    _fig.savefig(_fname.with_suffix(".png"), dpi=200, bbox_inches="tight")
    _plt.close(_fig)

    (_out / "publication_figures.html").write_text(
        "<!doctype html><html><body>"
        '<img src="publication_figures_rmse_bias.png" style="max-width:100%"><br>'
        '<img src="publication_figures_ets.png" style="max-width:100%"><br>'
        '<img src="publication_figures_rmse_bias_skill.png" style="max-width:100%"><br>'
        '<img src="publication_figures_ets_skill.png" style="max-width:100%">'
        "</body></html>"
    )

    plt.show()
    return


if __name__ == "__main__":
    app.run()
