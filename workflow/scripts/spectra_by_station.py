"""Stratify the 3-6 h spectral deficit by station.

Tests the terrain part of the exp_plan_01 hypothesis: if the models miss
km-scale, terrain-driven variability, the deficit should be worst in complex
terrain and mildest on the flat Plateau. A flat result across station types
would instead point at a generic smoothing property of MSE training.

Reference is the station observations, not REA-L: REA-L is itself missing about
half the observed sub-6 h variance, and that loss is station-dependent, so
scoring against it would mix REA-L's own smoothing into the answer.

Sampling: periodograms are averaged over inits and over two NON-overlapping
lead-time windows, [0,48) and [48,96), before any ratio is taken. Ratios of
single-case periodograms would be both wildly scattered and biased upward, since
each bin has only ~2 degrees of freedom (see spectra_single_cases.py).

Terrain proxies, from resources/smn_grid_points.csv:
  elevation        station height, separating valley floors from summits
  |elevation_diff| station minus model height. Large where the 1 km orography
                   cannot resolve the local topography, so it stands in for
                   unresolved terrain complexity.

    uv run workflow/scripts/spectra_by_station.py --season winter2024
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from spectra_forecast import (FORECAST_SOURCES, INTERP_LABEL, OBS_LABEL,
                              TRUTH_LABEL, WINDOW_H, load_season, psd)

WINDOWS = (0, 48)  # non-overlapping


def station_classes(tarball: Path) -> dict[str, str]:
    """nat_abbr -> T2m_based_class, from the spaziurat.lf R package.

    Read with `rdata` rather than `pyreadr`: the latter assumes UTF-8 and the
    table carries accented station names in another encoding. Stations absent
    from the package table get NA.
    """
    import tarfile, tempfile
    import rdata

    member = "spaziurat.lf/data/stat.class.info.rda"
    with tarfile.open(tarball) as t, tempfile.TemporaryDirectory() as tmp:
        t.extract(t.getmember(member), tmp)
        d = rdata.conversion.convert(
            rdata.parser.parse_file(Path(tmp) / member))["stat.class.info"]
    return dict(zip(d["nat_abbr"], d["T2m_based_class"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default="winter2024")
    ap.add_argument("--param", default="T_2M")
    ap.add_argument("--points", type=Path, default=Path("output/points"))
    ap.add_argument("--truth", type=Path, default=Path("output/points_truth"))
    ap.add_argument("--stations", type=Path, default=Path("resources/smn_grid_points.csv"))
    ap.add_argument("--classes", type=Path,
                    default=Path("resources/spaziurat.lf_2.6.1.tar.gz"),
                    help="R package holding stat.class.info")
    ap.add_argument("--out-dir", type=Path, default=Path("output/spectra"))
    args = ap.parse_args()

    data, inits = load_season(args.points, args.truth, args.season, args.param)
    if OBS_LABEL not in data:
        raise SystemExit(f"no observations for {args.param}")
    srcs = [OBS_LABEL, TRUTH_LABEL, *FORECAST_SOURCES, INTERP_LABEL]

    nstat = data[OBS_LABEL].shape[2]
    acc = {s: np.zeros((WINDOW_H // 2 + 1, nstat)) for s in srcs}
    n = 0
    for w in WINDOWS:
        sl = slice(w, w + WINDOW_H)
        ok = np.ones(len(inits), bool)
        for s in srcs:
            ok &= ~np.isnan(data[s][:, sl]).any(axis=(1, 2))
        for i in np.flatnonzero(ok):
            for s in srcs:
                f, p = psd(data[s][i, sl])
                acc[s] += p
            n += 1
        print(f"  window {w}-{w + WINDOW_H} h: {ok.sum()}/{len(inits)} inits")
    mean = {s: acc[s] / n for s in srcs}
    print(f"  {n} (init, window) cases per station")

    f = psd(data[OBS_LABEL][0, 0:WINDOW_H])[0]
    period = np.divide(1.0, f, out=np.full_like(f, np.inf), where=f > 0)
    band = (period >= 3) & (period <= 6)

    st = pd.read_csv(args.stations)
    # Station names live only in the original sample file, not in the grid mapping.
    names = pd.read_csv("output/station_selection/station_sample.csv")
    name_of = dict(zip(names["nat_abbr"], names["name"]))
    out = pd.DataFrame({"nat_abbr": st["nat_abbr"],
                        "name": st["nat_abbr"].map(name_of),
                        "T2m_based_class": st["nat_abbr"].map(station_classes(args.classes)),
                        "elevation": st["elevation"],
                        "elev_diff": st["elevation_diff"]})
    for s in srcs:
        out[s] = mean[s][band].sum(axis=0)
    for s in srcs[1:]:
        out[f"frac_{s}"] = out[s] / out[OBS_LABEL]
    # Same again against REA-L, the models' own training target. frac_ measures
    # the gap to reality (which includes representativeness, since a point
    # instrument is not a 1 km cell); FRAC_ measures the gap to what the models
    # were actually trained to reproduce.
    for s in srcs[2:]:
        out[f"FRAC_{s}"] = out[s] / out[TRUTH_LABEL]

    tag = f"{args.season}_{args.param}"
    out.to_csv(args.out_dir / f"by_station_{tag}.csv", index=False)

    print(f"\n=== {tag}: 3-6 h power as a fraction of observed ===")
    for s in srcs[1:]:
        c = out[f"frac_{s}"]
        print(f"  {s:22s} median {c.median():6.1%}   p10 {c.quantile(.1):6.1%}   p90 {c.quantile(.9):6.1%}")

    print("\n--- by elevation band (median fraction of observed) ---")
    bins = [0, 600, 1000, 1500, 4000]
    lab = ["<600 m", "600-1000", "1000-1500", ">1500 m"]
    out["band"] = pd.cut(out["elevation"], bins=bins, labels=lab)
    cols = [f"frac_{s}" for s in (TRUTH_LABEL, *FORECAST_SOURCES)]
    tab = out.groupby("band", observed=True)[cols].median()
    tab.insert(0, "n", out.groupby("band", observed=True).size())
    print(tab.to_string(float_format=lambda v: f"{v:.1%}"))

    print("\n--- by |station - model| elevation mismatch ---")
    out["mm"] = pd.cut(out["elev_diff"].abs(), bins=[-1, 25, 75, 150, 10000],
                       labels=["<25 m", "25-75", "75-150", ">150 m"])
    tab2 = out.groupby("mm", observed=True)[cols].median()
    tab2.insert(0, "n", out.groupby("mm", observed=True).size())
    print(tab2.to_string(float_format=lambda v: f"{v:.1%}"))

    best = out.nlargest(5, f"frac_{FORECAST_SOURCES[0]}")
    worst = out.nsmallest(5, f"frac_{FORECAST_SOURCES[0]}")
    print(f"\n--- {FORECAST_SOURCES[0]}: best / worst stations ---")
    for lbl, d in (("best ", best), ("worst", worst)):
        s = ", ".join(f"{r.nat_abbr}({getattr(r, 'frac_' + FORECAST_SOURCES[0].replace('-', '_'), 0):.0%}"
                      f" {r.elevation:.0f}m)" if False else
                      f"{r.nat_abbr}({d.loc[r.Index, 'frac_' + FORECAST_SOURCES[0]]:.0%}, {r.elevation:.0f}m)"
                      for r in d.itertuples())
        print(f"  {lbl}: {s}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, xcol, xlab in ((axes[0], "elevation", "station elevation (m)"),
                           (axes[1], out["elev_diff"].abs().name or "elev_diff",
                            "|station - model elevation| (m)")):
        x = out["elevation"] if xcol == "elevation" else out["elev_diff"].abs()
        for s, c in ((TRUTH_LABEL, "k"), (FORECAST_SOURCES[0], "C0"),
                     (FORECAST_SOURCES[1], "C3")):
            ax.scatter(x, 100 * out[f"frac_{s}"], s=14, alpha=0.65, color=c, label=s)
        ax.set_xlabel(xlab)
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("3-6 h power, % of observed")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{tag}: per-station 3-6 h variance relative to SwissMetNet "
                 f"({n} init/window cases per station)", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out_dir / f"by_station_{tag}.png", dpi=150)
    print(f"\nwrote {args.out_dir}/by_station_{tag}.{{csv,png}}")


if __name__ == "__main__":
    main()
