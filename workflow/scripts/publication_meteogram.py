import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    import time
    from pathlib import Path

    import marimo as mo

    _script_dir = Path(__file__).resolve().parent  # workflow/scripts/
    sys.path.append(str(_script_dir))
    project_root = _script_dir.parent.parent
    return Path, mo, project_root, sys, time


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
def _(mo, sys):
    # Interactive defaults point at the copied dataset so the notebook runs as-is.
    _RUN = "output/data/runs/temporal_downscaler-f927-1ee3-on-forecaster-c304-23e7/495c"
    _FORECAST_DEFAULT = f"{_RUN}/202504010000/grib"
    _BASELINES_DEFAULT = (
        "/store_new/mch/msopr/osm/ICON-CH1-EPS|0/33/1|mean|ICON-CH1-EPS mean;"
        "/store_new/mch/msopr/osm/ICON-CH2-EPS|0/120/1|mean|ICON-CH2-EPS mean"
    )

    _cli = mo.cli_args()
    if not _cli and len(sys.argv) > 1:
        import argparse

        _p = argparse.ArgumentParser()
        _p.add_argument("--forecast", default=_FORECAST_DEFAULT)
        _p.add_argument("--forecast_steps", default="0/120/1")
        _p.add_argument("--forecast_label", default="Varda-Single")
        _p.add_argument("--baseline", action="append", default=None)
        _p.add_argument("--date", default="202504010000")
        _p.add_argument("--station", default="KLO")
        _p.add_argument("--params", default="T_2M,TOT_PREC,SP_10M,DD_10M")
        _p.add_argument("--output", default="figures/meteogram")
        _a, _ = _p.parse_known_args()
        forecast = _a.forecast
        forecast_steps = _a.forecast_steps
        forecast_label = _a.forecast_label
        baselines_raw = ";".join(_a.baseline) if _a.baseline else _BASELINES_DEFAULT
        date = _a.date
        station = _a.station
        params_raw = _a.params
        output_dir = _a.output
        _is_script = True
    else:
        forecast = _cli.get("forecast", default=_FORECAST_DEFAULT)
        forecast_steps = _cli.get("forecast_steps", default="0/120/1")
        forecast_label = _cli.get("forecast_label", default="Varda-Single")
        baselines_raw = _cli.get("baseline", default=_BASELINES_DEFAULT)
        date = _cli.get("date", default="202504010000")
        station = _cli.get("station", default="KLO")
        params_raw = _cli.get("params", default="T_2M,TOT_PREC,SP_10M,DD_10M")
        output_dir = _cli.get("output", default="figures/meteogram")
        _is_script = bool(_cli)
    return (
        baselines_raw,
        date,
        forecast,
        forecast_label,
        forecast_steps,
        output_dir,
        params_raw,
        station,
    )


@app.cell
def _(baselines_raw, params_raw):
    # Parse the structured baseline string: "root|steps|member|label;root|..."
    def _parse_baselines(raw):
        out = []
        for spec in [s for s in raw.split(";") if s.strip()]:
            root, steps, member, label = spec.split("|")
            out.append({"root": root, "steps": steps, "member": member, "label": label})
        return out

    baselines = _parse_baselines(baselines_raw)
    display_params = [p.strip() for p in params_raw.split(",") if p.strip()]
    return baselines, display_params


@app.cell
def _(
    LOG,
    Path,
    baselines,
    date,
    display_params,
    forecast,
    forecast_label,
    forecast_steps,
    project_root,
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

    from publication_style import OBS_LABEL

    def _abs(p):
        p = Path(p)
        return p if p.is_absolute() else project_root / p

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
    mo,
    output_dir,
    source_order,
    station,
    time,
):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    from publication_style import line_style, param_label

    LOG.info("meteogram: rendering plot")
    _t0 = time.perf_counter()
    plt.style.use(Path(__file__).resolve().parent / "publication.mplstyle")

    _UNITS = {"T_2M": "K", "TOT_PREC": "mm", "SP_10M": "m/s", "DD_10M": "deg"}

    _fig, _axes = plt.subplots(
        len(display_params),
        1,
        figsize=(8, 2.6 * len(display_params)),
        sharex=True,
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
            # Wind direction is circular: draw as markers (no lines) so the
            # 0<->360 wraparound doesn't create spurious vertical segments.
            if _p == "DD_10M" and _src != OBS_LABEL:
                _style = {**_style, "linestyle": "none", "marker": ".", "markersize": 5}
            _lead = (_g["valid_time"] - init_time).dt.total_seconds() / 3600.0
            _ax.plot(_lead, _g["value"], label=_src, **_style)
        _ax.set_ylabel(_UNITS.get(_p, _p))
        _ax.text(
            0.01, 0.97, param_label(_p), transform=_ax.transAxes, ha="left", va="top"
        )
        if _p == "DD_10M":
            _ax.set_ylim(0, 360)
            _ax.set_yticks([0, 90, 180, 270, 360])
        # Lead-time x-axis (hours since init): major every 24 h, minor every 6 h
        _ax.xaxis.set_major_locator(mticker.MultipleLocator(24))
        _ax.xaxis.set_minor_locator(mticker.MultipleLocator(6))
        _ax.grid(
            True, axis="x", which="major", color="0.6", linewidth=0.8, linestyle="--"
        )
        _ax.grid(
            True, axis="x", which="minor", color="0.8", linewidth=0.6, linestyle=":"
        )

    _axes[-1].set_xlabel("Lead time (h)")
    _axes[0].set_xlim(left=0)
    _handles, _labels = _axes[0].get_legend_handles_labels()
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
        "<!doctype html><html><body>"
        '<img src="publication_meteogram.png" style="max-width:100%">'
        "</body></html>"
    )

    mo.image(str(_fname.with_suffix(".png")))
    return


if __name__ == "__main__":
    app.run()
