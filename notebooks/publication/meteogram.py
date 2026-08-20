import marimo

__generated_with = "0.23.7"
app = marimo.App()


@app.cell
def _():
    import sys
    import time
    import logging
    from pathlib import Path

    # Repo root: cwd when run from repo root, else walk up (nbconvert runs with
    # cwd = the notebook's own directory; no module-level path variable is available
    # in a Jupyter kernel, so we never rely on one).
    PROJECT_ROOT = Path.cwd().resolve()
    if not (PROJECT_ROOT / "workflow").is_dir():
        for _p in [PROJECT_ROOT] + list(PROJECT_ROOT.parents):
            if (_p / "workflow").is_dir() and (_p / "src").is_dir():
                PROJECT_ROOT = _p
                break
    sys.path.insert(0, str(PROJECT_ROOT / "workflow" / "scripts"))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    LOG = logging.getLogger("meteogram")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    import matplotlib.pyplot as plt
    from evalml.publication import style

    plt.style.use(style.mplstyle_path())
    return LOG, PROJECT_ROOT, Path, plt, time


@app.cell
def _():
    from evalml.publication.manifest import load_manifest, figures_dir

    m = load_manifest()
    _mg = m.publication.get("meteogram") or {}

    # Configured case; override any of these variables in-cell to retarget.
    date = _mg.get("init_time", "202504010000")
    station = _mg.get("station", "SIO")
    display_params = _mg.get("params", ["T_2M", "TOT_PREC1", "SP_10M", "DD_10M"])

    m.validate_request("meteogram", init_time=date)

    _cand = m.get_candidate()
    forecast = m.grib_dir(_cand, date)
    forecast_steps = _cand.steps
    forecast_label = _cand.label
    output_dir = str(figures_dir(m.output_root, m.truth["label"]) / "meteogram")

    # Parse the structured baseline spec "root|steps|member|label;..." from the manifest.
    def _parse_baselines(raw):
        out = []
        for spec in [s for s in raw.split(";") if s.strip()]:
            root, steps, member, label = spec.split("|")
            out.append({"root": root, "steps": steps, "member": member, "label": label})
        return out

    baselines = _parse_baselines(m.meteogram_baseline_specs())
    return (
        baselines,
        date,
        display_params,
        forecast,
        forecast_label,
        forecast_steps,
        output_dir,
        station,
    )


@app.cell
def _(
    LOG,
    PROJECT_ROOT,
    Path,
    baselines,
    date,
    display_params,
    forecast,
    forecast_label,
    forecast_steps,
    station,
    time,
):
    from datetime import datetime

    from data_input import (
        load_forecast_data,
        load_obs_data_from_jretrieve,
        parse_steps,
    )
    from meteogram_derivations import (
        add_derived,
        expand_to_base_params,
        station_timeseries_to_long,
    )
    from verification.spatial import map_forecast_to_truth

    import pandas as pd

    from evalml.publication.style import OBS_LABEL

    def _abs(p):
        p = Path(p)
        return p if p.is_absolute() else PROJECT_ROOT / p

    init_time = datetime.strptime(str(date), "%Y%m%d%H%M")
    base_params = expand_to_base_params(display_params)

    # Observations for the plotted station from the MeteoSwiss DWH.
    obs_source = f"jretrievedwh:locations={station}"
    LOG.info("meteogram: loading observations from %s", obs_source)
    _t0 = time.perf_counter()
    obs_steps = parse_steps(forecast_steps)
    obs = load_obs_data_from_jretrieve(obs_source, init_time, obs_steps, base_params)
    obs_station = add_derived(obs.sel(values=[station]), display_params)
    _sel = obs.sel(values=[station])
    station_target = _sel.drop_vars(list(_sel.data_vars))
    LOG.info("meteogram: observations loaded in %.1fs", time.perf_counter() - _t0)

    frames = [station_timeseries_to_long(obs_station, OBS_LABEL, display_params)]

    LOG.info("meteogram: loading candidate forecast from %s", forecast)
    _t0 = time.perf_counter()
    cand = load_forecast_data(
        _abs(forecast), init_time, parse_steps(forecast_steps), base_params
    )
    LOG.info("meteogram: candidate GRIB loaded in %.1fs", time.perf_counter() - _t0)
    _t0 = time.perf_counter()
    cand_st = add_derived(map_forecast_to_truth(cand, station_target), display_params)
    LOG.info(
        "meteogram: candidate remapped to station in %.1fs", time.perf_counter() - _t0
    )
    frames.append(station_timeseries_to_long(cand_st, forecast_label, display_params))

    for b in baselines:
        LOG.info("meteogram: loading baseline %s from %s", b["label"], b["root"])
        _t0 = time.perf_counter()
        bds = load_forecast_data(
            Path(b["root"]),
            init_time,
            parse_steps(b["steps"]),
            base_params,
            member=b["member"],
        )
        LOG.info(
            "meteogram: baseline %s loaded in %.1fs",
            b["label"],
            time.perf_counter() - _t0,
        )
        _t0 = time.perf_counter()
        bst = add_derived(map_forecast_to_truth(bds, station_target), display_params)
        LOG.info(
            "meteogram: baseline %s remapped in %.1fs",
            b["label"],
            time.perf_counter() - _t0,
        )
        frames.append(station_timeseries_to_long(bst, b["label"], display_params))

    df = pd.concat(frames, ignore_index=True)
    source_order = [OBS_LABEL] + [b["label"] for b in baselines] + [forecast_label]
    return OBS_LABEL, df, init_time, source_order


@app.cell
def _(
    LOG,
    OBS_LABEL,
    Path,
    df,
    display_params,
    init_time,
    output_dir,
    plt,
    source_order,
    station,
    time,
):
    import matplotlib.ticker as mticker
    from evalml.publication.style import line_style, param_label

    LOG.info("meteogram: rendering plot")
    _t0 = time.perf_counter()
    _UNITS = {
        "T_2M": "K",
        "TOT_PREC": "mm",
        "TOT_PREC1": "mm",
        "SP_10M": "m/s",
        "DD_10M": "deg",
    }
    _fig, _axes = plt.subplots(
        len(display_params), 1, figsize=(8, 2.6 * len(display_params)), sharex=True
    )
    if len(display_params) == 1:
        _axes = [_axes]
    for _ax, _p in zip(_axes, display_params):
        _sub = df[df["param"] == _p]
        for _src in source_order:
            _g = _sub[_sub["source"] == _src].sort_values("valid_time")
            if _g.empty:
                continue
            _style = line_style(_src)
            if _p == "DD_10M":
                _style = {**_style, "linestyle": "none", "marker": ".", "markersize": 5}
            elif _src == OBS_LABEL:
                _style = {
                    **_style,
                    "linestyle": "-",
                    "marker": "none",
                    "linewidth": 1.5,
                }
            _lead = (_g["valid_time"] - init_time).dt.total_seconds() / 3600.0
            _ax.plot(_lead, _g["value"], label=_src, **_style)
        _ax.set_ylabel(_UNITS.get(_p, _p))
        _ax.text(
            0.01, 0.97, param_label(_p), transform=_ax.transAxes, ha="left", va="top"
        )
        if _p == "DD_10M":
            _ax.set_ylim(0, 360)
            _ax.set_yticks([0, 90, 180, 270, 360])
        _ax.xaxis.set_major_locator(mticker.MultipleLocator(24))
        _ax.xaxis.set_minor_locator(
            mticker.MultipleLocator(6)
        )  # Wind direction is circular: draw everything (incl. observations)
        _ax.grid(
            True, axis="x", which="major", color="0.6", linewidth=0.8, linestyle="--"
        )  # as markers so the 0<->360 wraparound doesn't create spurious
        _ax.grid(
            True, axis="x", which="minor", color="0.8", linewidth=0.6, linestyle=":"
        )  # vertical segments.
    _axes[-1].set_xlabel("Lead time (h)")
    _axes[0].set_xlim(left=0)
    _handles, _labels = _axes[
        0
    ].get_legend_handles_labels()  # Observations as a continuous line (no markers) in all other panels.
    _fig.legend(
        _handles,
        _labels,
        loc="lower center",
        ncol=len(source_order),
        bbox_to_anchor=(0.5, 0.0),
    )
    _fig.suptitle(f"{station} — Init time {init_time:%Y-%m-%d %H:%M}")
    _fig.tight_layout(rect=[0, 0.05, 1, 0.99])
    LOG.info("meteogram: plot rendered in %.1fs", time.perf_counter() - _t0)
    _out = Path(output_dir)
    _out.mkdir(parents=True, exist_ok=True)
    _fname = _out / "publication_meteogram.pdf"
    LOG.info("meteogram: saving figures to %s", _out)
    _t0 = time.perf_counter()
    _fig.savefig(_fname, bbox_inches="tight")
    _fig.savefig(_fname.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(_fig)
    LOG.info("meteogram: figures saved in %.1fs", time.perf_counter() - _t0)
    (_out / "publication_meteogram.html").write_text(
        '<!doctype html><html><body><img src="publication_meteogram.png" style="max-width:100%"></body></html>'
    )
    plt.show()  # Lead-time x-axis (hours since init): major every 24 h, minor every 6 h
    return


if __name__ == "__main__":
    app.run()
