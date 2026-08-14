"""Publication case-snapshot precipitation maps.

For each hand-picked case, one row of maps: the candidate forecast vs the gridded
(analysis) truth, showing the accumulated precipitation valid at a fixed lead
time, with the case's SAL (Structure/Amplitude/Location) scores annotated on the
forecast panel. Cases are chosen objectively from the SAL ranking (one
well-forecast, one poorly-forecast) to illustrate the seasonal skill contrast.

Fields load through ``data_input.load_forecast_data`` / ``load_truth_data`` using
the accumulated precipitation parameter (e.g. ``TOT_PREC6``), so the mapped field
is exactly what the SAL score was computed on. Plotting uses evalml's native ICON
``StatePlotter`` (earthkit.plots) with the standard precipitation colormap.

Outputs (into ``--output``):
  publication_case_snapshots.{png,pdf}   rows = cases, cols = forecast | truth
  publication_case_snapshots.html        report index
"""

import sys
from argparse import ArgumentParser
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

_script_dir = Path(__file__).resolve().parent
sys.path.append(str(_script_dir))
sys.path.append(str(_script_dir.parent.parent / "src"))
plt.style.use(_script_dir / "publication.mplstyle")

from data_input import (  # noqa: E402
    load_forecast_data,
    load_truth_data,
    open_truth_zarr,
)
from plot_forecast_frame import get_style  # noqa: E402
from plotting import DOMAINS, StatePlotter  # noqa: E402

DATETIME_FMT = "%Y%m%d%H%M"


def _unpack(da):
    """Flatten a field to (values>=0, lon, lat) 1-D arrays."""
    field = np.maximum(np.asarray(da.values, dtype=float).ravel(), 0.0)
    lon = np.asarray(da["longitude"].values).ravel()
    lat = np.asarray(da["latitude"].values).ravel()
    return field, lon, lat


def load_case(grib_dir, truth_path, lazy_truth, reftime, accum_param, leadtime):
    """Return (forecast, truth) as (field, lon, lat) for the +leadtime window."""
    fcst = load_forecast_data(Path(grib_dir), reftime, [leadtime], [accum_param])
    if "step" in fcst.dims:
        fcst = fcst.sel(step=np.timedelta64(leadtime, "h"))
    fda = fcst[accum_param].squeeze()

    truth = load_truth_data(
        Path(truth_path), reftime, [leadtime], [accum_param], lazy_ds=lazy_truth
    ).isel(time=0)
    tda = truth[accum_param].squeeze()
    return _unpack(fda), _unpack(tda)


def sal_annotation(sal_csv, reftime_str):
    """S/A/L annotation string for one init, or None if unavailable."""
    try:
        df = pd.read_csv(sal_csv, comment="#")
    except FileNotFoundError:
        return None
    df["reftime"] = df["reftime"].astype(str)
    row = df[df["reftime"] == reftime_str]
    if row.empty or not np.isfinite(row["S"].iloc[0]):
        return None
    r = row.iloc[0]
    return f"S={r['S']:+.2f}  A={r['A']:+.2f}  L={r['L']:.2f}"


def build_figure(panels, plotters, domain_name, param, accumulation, out_paths):
    """panels: list of dicts (field, plotter_key, title, sal_text). 2 cols/case."""
    ncols = 2
    nrows = len(panels) // ncols
    domain = DOMAINS[domain_name]
    lead = plotters["fcst"]
    fig = lead.init_geoaxes(
        nrows=nrows,
        ncols=ncols,
        projection=domain["projection"],
        bbox=domain["extent"],
        name=domain_name,
        size=(9.0, 4.4 * nrows),
    )
    style_kwargs = get_style(param, "mm", accu=accumulation)
    subplots = []
    for k, panel in enumerate(panels):
        row, col = divmod(k, ncols)
        subplot = fig.add_map(row=row, column=col)
        subplots.append(subplot)
        plotters[panel["plotter_key"]].plot_field(
            subplot,
            panel["field"],
            colorbar=False,
            title=panel["title"],
            **style_kwargs,
        )
        if panel["sal_text"]:
            subplot.ax.text(
                0.02,
                0.02,
                panel["sal_text"],
                transform=subplot.ax.transAxes,
                fontsize=7.5,
                va="bottom",
                ha="left",
                zorder=10,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "black",
                    "linewidth": 0.5,
                    "pad": 2.0,
                },
            )
    try:
        fig.legend()
    except Exception as exc:  # fall back to a colorbar on the last panel
        print(f"figure-level legend failed ({exc}); using last-panel colorbar")
        subplots[-1].legend()

    for path in out_paths:
        fig.save(path, bbox_inches="tight", dpi=200)
        print(f"Wrote {path}")


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-label", dest="candidate_label", required=True)
    parser.add_argument("--truth", required=True, help="Truth zarr root (or spec).")
    parser.add_argument("--truth-label", dest="truth_label", default="analysis")
    parser.add_argument("--param", default="TOT_PREC", help="Base precip param.")
    parser.add_argument("--leadtime", type=int, default=12)
    parser.add_argument("--accumulation", type=int, default=6)
    parser.add_argument("--domain", default="centraleurope")
    parser.add_argument(
        "--case",
        nargs=3,
        action="append",
        metavar=("INIT", "GRIB_DIR", "SAL_CSV"),
        required=True,
        help="Case init time (YYYYMMDDHHMM), its GRIB directory, and the SAL CSV "
        "to read the annotated S/A/L from. Repeat per case.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    args = parser.parse_args()

    accum_param = f"{args.param}{args.accumulation}"
    args.output.mkdir(parents=True, exist_ok=True)

    truth_path = Path(args.truth)
    lazy_truth = (
        open_truth_zarr(truth_path, [accum_param])
        if truth_path.suffix == ".zarr"
        else None
    )

    panels = []
    plotters = {}
    for init, grib_dir, sal_csv in args.case:
        reftime = datetime.strptime(init, DATETIME_FMT)
        (ffield, flon, flat), (tfield, tlon, tlat) = load_case(
            grib_dir, truth_path, lazy_truth, reftime, accum_param, args.leadtime
        )
        if "fcst" not in plotters:  # cases share the ICON-CH1 grids; build once
            plotters["fcst"] = StatePlotter(flon, flat, args.output)
            plotters["truth"] = StatePlotter(tlon, tlat, args.output)
        valid = reftime + timedelta(hours=args.leadtime)
        panels.append(
            {
                "field": ffield,
                "plotter_key": "fcst",
                "title": (
                    f"{args.candidate_label} +{args.leadtime} h\n"
                    f"init {reftime:%Y-%m-%d %H} UTC"
                ),
                "sal_text": sal_annotation(sal_csv, init),
            }
        )
        panels.append(
            {
                "field": tfield,
                "plotter_key": "truth",
                "title": (
                    f"{args.truth_label} analysis\nvalid {valid:%Y-%m-%d %H} UTC"
                ),
                "sal_text": None,
            }
        )

    build_figure(
        panels,
        plotters,
        args.domain,
        args.param,
        args.accumulation,
        [
            args.output / "publication_case_snapshots.pdf",
            args.output / "publication_case_snapshots.png",
        ],
    )
    (args.output / "publication_case_snapshots.html").write_text(
        '<!doctype html><html><body><img src="publication_case_snapshots.png" '
        'style="max-width:100%"></body></html>'
    )


if __name__ == "__main__":
    main()
