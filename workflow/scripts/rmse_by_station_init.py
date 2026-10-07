"""RMSE per (station, init) against SwissMetNet, pooled over all lead times.

Companion to spectra_by_station.py: the spectra say how much sub-6 h variance
each source has, this says how close the series actually are. A source can lose
most of its 3-6 h power and still score well, so the two are read together.

--reference picks what the sources are scored against. SwissMetNet (the default,
the same choice as in spectra_by_station.py) measures the gap to reality, which
includes representativeness since a point instrument is not a 1 km cell; REA-L
measures the gap to what the models were actually trained to reproduce. Every
source other than the reference is scored, so an REA-L run also reports how far
the observations sit from REA-L. The reference is part of the output filename.

One RMSE per (source, init, station) over steps 0-120. Pairs where either side
is missing are dropped, and a (source, init, station) cell with fewer than
--min-steps valid pairs is set to NaN rather than scored on a short sample.
Only T_2M, TD_2M and SP_10M exist in the observations; PS can be scored
against REA-L only.

`--extra-sources` scores further hourly sources (e.g. the ICON baselines), and
`--max-step` restricts the lead times, e.g. to 0-33 h where ICON-CH1 ends. A
restricted run gets a `_0-<max>h` suffix so it cannot overwrite the full one.

    uv run workflow/scripts/rmse_by_station_init.py --season winter2024
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default="winter2024")
    ap.add_argument("--param", default="T_2M", choices=["T_2M", "TD_2M", "SP_10M", "PS"])
    ap.add_argument("--points", type=Path, default=Path("output/points"))
    ap.add_argument("--truth", type=Path, default=Path("output/points_truth"))
    ap.add_argument("--min-steps", type=int, default=96,
                    help="valid pairs required out of 121 to score a cell")
    ap.add_argument("--reference", default=OBS_LABEL,
                    choices=[OBS_LABEL, TRUTH_LABEL],
                    help="score against the station observations (default) or "
                         "against REA-L, the models' training target")
    ap.add_argument("--out-dir", type=Path, default=Path("output/rmse"))
    ap.add_argument("--extra-sources", nargs="+", default=[])
    ap.add_argument("--max-step", type=int, default=120)
    args = ap.parse_args()

    data, inits = load_season(args.points, args.truth, args.season, args.param,
                              tuple(args.extra_sources))
    if args.reference not in data:
        raise SystemExit(f"no {args.reference} for {args.param}")
    data = {k: v[:, :args.max_step + 1] for k, v in data.items()}
    # Everything except the reference is scored, so an REA-L run still reports
    # how far the observations sit from it.
    srcs = [s for s in (OBS_LABEL, TRUTH_LABEL, *FORECAST_SOURCES, INTERP_LABEL,
                        *args.extra_sources)
            if s != args.reference and s in data]

    # Station order is taken from the forecast files; check the observations
    # were stacked in the same order rather than trusting it.
    with xr.open_dataset(args.points / args.season / FORECAST_SOURCES[0] /
                         f"{inits[0]}.nc") as d:
        stations = d["station"].values.astype(str)
    obs_stations = xr.open_dataset("output/points_obs/obs_hourly.nc")["station"].values.astype(str)
    if not np.array_equal(stations, obs_stations):
        raise SystemExit("station order differs between forecasts and observations")

    ref = data[args.reference]
    rows = []
    for s in srcs:
        err = data[s] - ref                      # (init, step, station)
        ok = np.isfinite(err)
        n = ok.sum(axis=1)
        sq = np.where(ok, err, 0.0) ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            rmse = np.sqrt(sq.sum(axis=1) / n)
            bias = np.where(ok, err, 0.0).sum(axis=1) / n
            # Standard deviation of the error: the part of the RMSE left once
            # the mean offset over the run is removed, i.e. the timing and
            # amplitude part rather than a constant warm or cold shift.
            # Clipped at 0 because rounding can make the difference -1e-16.
            stde = np.sqrt(np.clip(rmse**2 - bias**2, 0.0, None))
        short = n < args.min_steps
        rmse[short] = np.nan
        bias[short] = np.nan
        stde[short] = np.nan
        rows.append(pd.DataFrame({
            "season": args.season, "param": args.param,
            "reference": args.reference, "source": s,
            "init": np.repeat(inits, len(stations)),
            "station": np.tile(stations, len(inits)),
            "rmse": rmse.ravel(), "bias": bias.ravel(), "stde": stde.ravel(),
            "n": n.ravel()}))
    out = pd.concat(rows, ignore_index=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # The reference is part of the name: the two sets answer different questions
    # and must not overwrite each other.
    tag = f"{args.season}_{args.param}_vs-{args.reference}"
    if args.max_step != 120:
        tag += f"_0-{args.max_step}h"
    out.to_csv(args.out_dir / f"rmse_by_station_init_{tag}.csv", index=False)

    print(f"\n=== {args.season} {args.param}: RMSE vs {args.reference}, "
          f"{len(inits)} inits x {len(stations)} stations ===")
    print(f"{'source':22s} {'cells':>7s} {'rmse':>7s} {'median':>7s} "
          f"{'p90':>7s} {'bias':>7s} {'stde':>7s}")
    for s in srcs:
        d = out[out["source"] == s]
        print(f"{s:22s} {d['rmse'].notna().sum():7d} {d['rmse'].mean():7.2f} "
              f"{d['rmse'].median():7.2f} {d['rmse'].quantile(.9):7.2f} "
              f"{d['bias'].mean():7.2f} {d['stde'].mean():7.2f}")

    by_st = out.pivot_table(index="station", columns="source", values="rmse")
    by_in = out.pivot_table(index="init", columns="source", values="rmse")
    print(f"\n--- worst 8 stations (by {FORECAST_SOURCES[1]}) ---")
    print(by_st.nlargest(8, FORECAST_SOURCES[1])[srcs].to_string(
        float_format=lambda v: f"{v:.2f}"))
    print(f"\n--- worst 8 inits (by {FORECAST_SOURCES[1]}) ---")
    print(by_in.nlargest(8, FORECAST_SOURCES[1])[srcs].to_string(
        float_format=lambda v: f"{v:.2f}"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {s: c for s, c in ((OBS_LABEL, "0.45"), (TRUTH_LABEL, "k"),
                               (FORECAST_SOURCES[0], "C0"),
                               (FORECAST_SOURCES[1], "C3"), (INTERP_LABEL, "C2"),
                               *zip(args.extra_sources, ("C4", "C5", "C6")))
             if s in srcs}
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5))
    order = by_st[FORECAST_SOURCES[1]].sort_values().index
    for s, c in style.items():
        axes[0].plot(range(len(order)), by_st.loc[order, s], ".", color=c,
                     ms=5, label=s)
    axes[0].set_xticks(range(len(order)))
    axes[0].set_xticklabels(order, rotation=90, fontsize=4)
    axes[0].set_xlabel("station (sorted by Multistep RMSE)")

    # Markers, not lines: a season is two calendar blocks (Jan/Feb and Dec) and
    # a connected line would draw a trend across the nine months in between.
    x = pd.to_datetime(by_in.index, format="%Y%m%d%H%M")
    for s, c in style.items():
        axes[1].plot(x, by_in[s], ".", color=c, ms=6, label=s)
    axes[1].set_xlabel("init")

    for ax in axes:
        ax.set_ylabel(f"RMSE of {args.param}")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=4)
    fig.suptitle(f"{args.season} {args.param}: RMSE against {args.reference}, "
                 f"pooled over leads 0-{args.max_step} h, averaged over inits (top) and "
                 f"stations (bottom)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out_dir / f"rmse_by_station_init_{tag}.png", dpi=150)
    print(f"\nwrote {args.out_dir}/rmse_by_station_init_{tag}.{{csv,png}}")


if __name__ == "__main__":
    main()
