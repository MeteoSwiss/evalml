"""Plot the raw time series of one (init, station), all sources in one panel.

The spectra and the RMSE tables both reduce the series to a single number. This
puts the series themselves on screen, so a number like "Multistep keeps 40% of
the 3-6 h power" can be checked against what the curve actually looks like.

Series are already at station locations: each of the 145 points is the model
grid point nearest an SMN station, so no further interpolation happens here.

By default the (init, station) cell is the one where Multistep-stage-C scored
its lowest RMSE for that season and parameter, read from the CSV written by
rmse_by_station_init.py. That is a best case, not a typical one: it shows what
the model looks like when it is doing well. --init and --station override it.

--reference picks which of those CSVs selects the cell and what the lower panel
takes errors against, and --sources limits which curves are drawn, e.g.

    uv run workflow/scripts/plot_series_single_init.py --season winter2024 \\
        --reference REA-L --sources REA-L,Varda-single-stage-C,Multistep-stage-C
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from spectra_forecast import (FORECAST_SOURCES, INTERP_LABEL, OBS_LABEL,
                              TRUTH_LABEL, load_season)

UNITS = {"T_2M": "K", "TD_2M": "K", "SP_10M": "m/s"}  # points are stored in SI
STYLE = {OBS_LABEL: ("0.35", "-", 2.2),
         TRUTH_LABEL: ("k", "-", 1.6),
         FORECAST_SOURCES[0]: ("C0", "-", 1.4),
         FORECAST_SOURCES[1]: ("C3", "-", 1.4),
         INTERP_LABEL: ("C2", "--", 1.2)}


def pick_cell(rmse_csv: Path) -> tuple[str, str, float]:
    """(init, station, rmse) of the lowest Multistep RMSE in this table."""
    d = pd.read_csv(rmse_csv, dtype={"init": str})
    d = d[d["source"] == FORECAST_SOURCES[1]].dropna(subset=["rmse"])
    r = d.loc[d["rmse"].idxmin()]
    return r["init"], r["station"], r["rmse"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default="winter2024")
    ap.add_argument("--param", default="T_2M", choices=["T_2M", "TD_2M", "SP_10M"])
    ap.add_argument("--init", help="YYYYMMDDHHMM; default is the best Multistep init")
    ap.add_argument("--station", help="e.g. ABO; default is the best Multistep station")
    ap.add_argument("--points", type=Path, default=Path("output/points"))
    ap.add_argument("--truth", type=Path, default=Path("output/points_truth"))
    ap.add_argument("--reference", default=OBS_LABEL,
                    choices=[OBS_LABEL, TRUTH_LABEL],
                    help="which RMSE table selects the cell, and what the error "
                         "panel is taken against")
    ap.add_argument("--sources", default=",".join(STYLE),
                    help="comma-separated subset of the sources to draw")
    ap.add_argument("--rmse-dir", type=Path, default=Path("output/rmse"))
    ap.add_argument("--out-dir", type=Path, default=Path("output/series"))
    args = ap.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    if bad := [s for s in sources if s not in STYLE]:
        raise SystemExit(f"unknown source(s) {bad}; choose from {list(STYLE)}")

    tag = f"{args.season}_{args.param}_vs-{args.reference}"
    rmse = None
    if args.init and args.station:
        init, station = args.init, args.station
    else:
        csv = args.rmse_dir / f"rmse_by_station_init_{tag}.csv"
        if not csv.exists():
            raise SystemExit(f"{csv} missing: run rmse_by_station_init.py "
                             f"--reference {args.reference} first")
        init, station, rmse = pick_cell(csv)
        print(f"{tag}: best Multistep cell is {init} / {station} "
              f"(RMSE vs {args.reference} {rmse:.2f} {UNITS[args.param]})")

    data, inits = load_season(args.points, args.truth, args.season, args.param)
    if OBS_LABEL not in data:
        raise SystemExit(f"no observations for {args.param}")
    with xr.open_dataset(args.points / args.season / FORECAST_SOURCES[0] /
                         f"{inits[0]}.nc") as d:
        stations = list(d["station"].values.astype(str))
    if init not in inits:
        raise SystemExit(f"init {init} not among the {len(inits)} {args.season} inits")
    if station not in stations:
        raise SystemExit(f"station {station} not in the 145-station sample")
    i, j = inits.index(init), stations.index(station)

    # Valid times rather than lead hours, so the diurnal cycle is readable.
    with xr.open_dataset(args.points / args.season / FORECAST_SOURCES[0] /
                         f"{init}.nc") as d:
        t = pd.to_datetime(d["valid_time"].values)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, axe) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True,
                                  height_ratios=[2.4, 1])
    ref = data[args.reference][i, :, j]
    for s in sources:
        c, ls, lw = STYLE[s]
        if s not in data:
            continue
        ax.plot(t, data[s][i, :, j], color=c, ls=ls, lw=lw, label=s)
        if s != args.reference:  # the reference minus itself is a flat zero
            axe.plot(t, data[s][i, :, j] - ref, color=c, ls=ls, lw=lw)
    axe.axhline(0, color=STYLE[args.reference][0], lw=2.0)

    ax.set_ylabel(f"{args.param} ({UNITS[args.param]})")
    axe.set_ylabel(f"error vs {args.reference}")
    axe.set_xlabel(f"valid time (init {init}, leads 0-120 h)")
    for a in (ax, axe):
        a.grid(alpha=0.3)
        # One tick per day: the run is 5 days, hourly ticks are unreadable.
        a.xaxis.set_major_locator(matplotlib.dates.DayLocator())
        a.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
    ax.legend(fontsize=8, ncol=5, loc="best")
    sub = (f"   (Multistep RMSE vs {args.reference} {rmse:.2f}, its best cell)"
           if rmse is not None else "")
    fig.suptitle(f"{args.season} {args.param} at {station}, init {init}{sub}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"series_{tag}_{init}_{station}.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
