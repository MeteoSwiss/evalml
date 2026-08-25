import marimo

__generated_with = "0.23.7"
app = marimo.App()


@app.cell
def _():
    import sys
    import logging
    from pathlib import Path
    from datetime import datetime

    # Repo root: cwd when run from repo root, else walk up (a kernel has no
    # __file__, so we never rely on one).
    PROJECT_ROOT = Path.cwd().resolve()
    if not (PROJECT_ROOT / "workflow").is_dir():
        for _p in [PROJECT_ROOT] + list(PROJECT_ROOT.parents):
            if (_p / "workflow").is_dir() and (_p / "src").is_dir():
                PROJECT_ROOT = _p
                break
    sys.path.insert(0, str(PROJECT_ROOT / "workflow" / "scripts"))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    LOG = logging.getLogger("meteogram")

    import matplotlib.pyplot as plt
    from evalml.publication import style

    plt.style.use(style.mplstyle_path())
    # A multi-panel, page-width figure is scaled down in the paper, so bump the
    # font sizes above the shared defaults for legibility.
    plt.rcParams.update(
        {
            "axes.titlesize": 18,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 15,
        }
    )
    return LOG, PROJECT_ROOT, Path, datetime, plt


@app.cell
def _(PROJECT_ROOT):
    # ============================ WHAT TO PLOT ============================
    # A figure is a grid of PANELS. Each panel is one dict:
    #   {"row": r, "col": c, "param": "T_2M", "station": "ALT"}          # a param at a station
    #   {..., "param": "PMSL", "station": "ALT", "minus_station": "LUG"} # station A - B difference
    # Optional per-panel keys:
    #   init          init time (YYYYMMDDHHMM); defaults to INIT below
    #   ylabel        y-axis label (defaults to the shared human-readable name)
    #   unit          unit shown under the ylabel (defaults to a lookup)
    #   scale         multiply values before plotting (e.g. 0.01 for Pa -> hPa)
    #   zero_line     draw a dashed y=0 line (useful for differences)
    #   title         panel title (comparison_panels puts station/init on the top row)
    # DD_10M is auto-drawn as circular markers (0-360). Panel letters (a),(b),...
    # follow the PANELS list order.

    from evalml.publication.manifest import load_manifest

    # -- which manifest / init / baselines / figure size ------------------
    MANIFEST = PROJECT_ROOT / "output/manifests/manifest_varda-single_meteogram_20250321.json"
    INIT = "202503210000"
    BASELINE_LABELS = {"ICON-CH1-CTRL", "ICON-CH2-CTRL"}  # CTRL (fast) or the EPS means
    FIGSIZE = (13.0, 6.4)          # None -> auto (5.4*ncols, 2.7*nrows)
    SUPTITLE = "ALT — Init time 2025-03-21 00:00"
    PANEL_LABELS = True            # draw (a), (b), ...

    def comparison_panels(cases, params):
        """Build a params x cases grid: cols = cases (station, init), rows = params.
        Top-row panels get a '<station> - <init>' title.
        """
        panels = []
        for col, (station, init) in enumerate(cases):
            for row, p in enumerate(params):
                panel = {"row": row, "col": col, "param": p, "station": station, "init": init}
                if row == 0:
                    panel["title"] = f"{station} — {init[:4]}-{init[4:6]}-{init[6:8]} {init[8:10]}:{init[10:12]}"
                panels.append(panel)
        return panels

    # -- the figure -------------------------------------------------------
    # Default: the ALT pressure-difference 2x2 (left: T_2M, PMSL diff; right: DD, SP).
    PANELS = [
        {"row": 0, "col": 0, "param": "T_2M", "station": "ALT"},
        {"row": 1, "col": 0, "param": "PMSL", "station": "ALT", "minus_station": "LUG",
         "scale": 0.01, "unit": "hPa", "ylabel": "Pressure diff. ALT\N{MINUS SIGN}LUG",
         "zero_line": True},
        {"row": 0, "col": 1, "param": "DD_10M", "station": "ALT"},
        {"row": 1, "col": 1, "param": "SP_10M", "station": "ALT"},
    ]

    # Example — the two-station comparison figure (SIO + KLO, EPS means):
    #   MANIFEST = PROJECT_ROOT / "output/manifests/manifest_varda-single_meteogram_20250627.json"
    #   INIT = "202506271800"
    #   BASELINE_LABELS = {"ICON-CH1-EPS mean", "ICON-CH2-EPS mean"}
    #   FIGSIZE, SUPTITLE = None, None
    #   PANELS = comparison_panels(
    #       cases=[("SIO", INIT), ("KLO", INIT)],
    #       params=["T_2M", "SP_10M", "DD_10M"],
    #   )
    # =====================================================================

    m = load_manifest(MANIFEST)
    return (
        BASELINE_LABELS,
        FIGSIZE,
        INIT,
        PANELS,
        PANEL_LABELS,
        SUPTITLE,
        m,
    )


@app.cell
def _(BASELINE_LABELS, INIT, LOG, PANELS, Path, datetime, m):
    import pandas as pd

    from data_input import load_forecast_data, load_obs_data_from_jretrieve, parse_steps
    from meteogram_derivations import add_derived, expand_to_base_params, station_timeseries_to_long
    from verification.spatial import map_forecast_to_truth
    from evalml.publication.style import OBS_LABEL

    cand = m.get_candidate()
    steps, cand_label = cand.steps, cand.label

    def _parse_baselines(raw):
        out = []
        for spec in [s for s in raw.split(";") if s.strip()]:
            root, st, member, lbl = spec.split("|")
            out.append({"root": root, "steps": st, "member": member, "label": lbl})
        return out

    baselines = [b for b in _parse_baselines(m.meteogram_baseline_specs()) if b["label"] in BASELINE_LABELS]
    source_order = [OBS_LABEL] + [b["label"] for b in baselines] + [cand_label]

    # Which (init -> stations) and which params the panels reference.
    PARAMS = sorted({p["param"] for p in PANELS})
    needed: dict[str, set] = {}
    for _p in PANELS:
        _it = _p.get("init", INIT)
        needed.setdefault(_it, set()).add(_p["station"])
        if _p.get("minus_station"):
            needed[_it].add(_p["minus_station"])

    base_params = expand_to_base_params(PARAMS)

    def _per_station_long(ds_all, label, stations, init):
        parts = []
        for st in stations:
            dss = add_derived(ds_all.sel(values=[st]), PARAMS)
            ldf = station_timeseries_to_long(dss, label, PARAMS)
            ldf["station"] = st
            ldf["init"] = init
            parts.append(ldf)
        return parts

    frames = []
    for init, stset in needed.items():
        stations = sorted(stset)
        LOG.info("loading init %s for stations %s", init, stations)
        m.validate_request("meteogram", init_time=init)
        init_time = datetime.strptime(init, "%Y%m%d%H%M")

        obs = load_obs_data_from_jretrieve(
            "jretrievedwh:locations=" + ",".join(stations), init_time, parse_steps(steps), base_params
        )
        station_target = obs.sel(values=stations).drop_vars(list(obs.data_vars))

        frames += _per_station_long(obs, OBS_LABEL, stations, init)

        LOG.info("  candidate")
        cds = load_forecast_data(Path(m.grib_dir(cand, init)), init_time, parse_steps(steps), base_params)
        frames += _per_station_long(map_forecast_to_truth(cds, station_target), cand_label, stations, init)

        for b in baselines:
            LOG.info("  baseline %s", b["label"])
            bds = load_forecast_data(Path(b["root"]), init_time, parse_steps(b["steps"]), base_params, member=b["member"])
            frames += _per_station_long(map_forecast_to_truth(bds, station_target), b["label"], stations, init)

    df = pd.concat(frames, ignore_index=True)
    return OBS_LABEL, df, source_order


@app.cell
def _(
    FIGSIZE,
    INIT,
    LOG,
    OBS_LABEL,
    PANELS,
    PANEL_LABELS,
    Path,
    SUPTITLE,
    datetime,
    df,
    m,
    plt,
    source_order,
):
    import matplotlib.ticker as mticker
    from evalml.publication.manifest import figures_dir
    from evalml.publication.style import line_style, param_label

    _UNITS = {"T_2M": "K", "PMSL": "hPa", "SP_10M": "m/s", "DD_10M": "deg", "U_10M": "m/s", "V_10M": "m/s"}

    def _series(src, panel):
        """(lead_hours, values) for one source in one panel; handles differences."""
        init = panel.get("init", INIT)
        init_time = datetime.strptime(init, "%Y%m%d%H%M")

        def _pick(station):
            g = df[
                (df["source"] == src) & (df["station"] == station)
                & (df["param"] == panel["param"]) & (df["init"] == init)
            ].sort_values("valid_time")
            return g

        a = _pick(panel["station"])
        if a.empty:
            return None, None
        if panel.get("minus_station"):
            b = _pick(panel["minus_station"])
            merged = a.merge(b, on="valid_time", suffixes=("_a", "_b"))
            if merged.empty:
                return None, None
            lead = (merged["valid_time"] - init_time).dt.total_seconds() / 3600.0
            return lead, (merged["value_a"] - merged["value_b"])
        lead = (a["valid_time"] - init_time).dt.total_seconds() / 3600.0
        return lead, a["value"]

    nrows = max(p["row"] for p in PANELS) + 1
    ncols = max(p["col"] for p in PANELS) + 1
    figsize = FIGSIZE or (5.4 * ncols, 2.7 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, squeeze=False)

    used = {(p["row"], p["col"]) for p in PANELS}
    for r in range(nrows):
        for c in range(ncols):
            if (r, c) not in used:
                axes[r][c].axis("off")

    legend = {}
    for i, p in enumerate(PANELS):
        ax = axes[p["row"]][p["col"]]
        scale = p.get("scale", 1.0)
        is_dd = p["param"] == "DD_10M"
        for src in source_order:
            lead, vals = _series(src, p)
            if lead is None:
                continue
            stl = line_style(src)
            if is_dd:
                stl = {**stl, "linestyle": "none", "marker": ".", "markersize": 5}
            elif src == OBS_LABEL:
                stl = {**stl, "linestyle": "-", "marker": "none", "linewidth": 1.5}
            (ln,) = ax.plot(lead, vals * scale, label=src, **stl)
            legend.setdefault(src, ln)
        if is_dd:
            ax.set_ylim(0, 360)
            ax.set_yticks([0, 90, 180, 270, 360])
        if p.get("zero_line"):
            ax.axhline(0, color="black", linewidth=0.7, linestyle="dashed", zorder=0)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(24))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(6))
        ax.grid(True, axis="x", which="major", color="0.6", linewidth=0.8, linestyle="--")
        ax.grid(True, axis="x", which="minor", color="0.85", linewidth=0.6, linestyle=":")
        ax.set_xlim(left=0)
        unit = p.get("unit", _UNITS.get(p["param"], ""))
        ylabel = p.get("ylabel", param_label(p["param"]))
        ax.set_ylabel(f"{ylabel}\n({unit})" if unit else ylabel)
        if p.get("title"):
            ax.set_title(p["title"])
        if p["row"] == nrows - 1:
            ax.set_xlabel("Lead time (h)")
        if PANEL_LABELS:
            ax.text(0.015, 0.94, f"({chr(97 + i)})", transform=ax.transAxes,
                    fontweight="bold", va="top", ha="left")

    order = [OBS_LABEL] + [s for s in source_order if s != OBS_LABEL]
    handles = [legend[s] for s in order if s in legend]
    labels = [s for s in order if s in legend]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, 0.0))
    if SUPTITLE:
        fig.suptitle(SUPTITLE, y=0.99)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97 if SUPTITLE else 0.99])

    out = Path(figures_dir(m.output_root, m.truth["label"])) / "meteogram"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "publication_meteogram.pdf", bbox_inches="tight")
    fig.savefig(out / "publication_meteogram.png", dpi=200, bbox_inches="tight")
    LOG.info("meteogram: saved to %s", out)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
