"""3-6 h filtered time series for a look at what the band actually contains.

For 10 characteristic stations, 4 variables and 10 random inits per season, one
PNG per (season, variable, station) with one panel per init: the 3-6 h component
(scales.component, i.e. exactly the analysis filter) over lead 24-96 h for
SwissMetNet (where observed), REA-L, Varda-single, Multistep and ICON-CH2.
The y axis is shared within a file. Inits are drawn once per season (seed 0)
among those with every source complete for every variable. `--single` writes
one 16:9 figure per init instead, with its own y axis, to
output/filtered_series_3-6h_single/<season>/<param>/<station>/<init>.png.

    uv run workflow/scripts/plot_filtered_series.py [--single]
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from spectra_forecast import OBS_LABEL, TRUTH_LABEL, load_season
from gap_robustness import station_classes
import scales

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--single", action="store_true")
args = ap.parse_args()
OUT = Path("output/filtered_series_3-6h" + ("_single" if args.single else ""))
STATIONS = ["PAY", "KLO", "GVE", "LUG", "SIO", "DAV", "SAE", "JUN", "ABO", "CHU"]
PARAMS = ["T_2M", "TD_2M", "SP_10M", "PS"]
UNIT = {"T_2M": "K", "TD_2M": "K", "SP_10M": "m/s", "PS": "Pa"}
SRC = [(OBS_LABEL, "SwissMetNet", "0.6", 2.2), (TRUTH_LABEL, "REA-L", "k", 1.8),
       ("Varda-single-stage-C", "Varda-single", "C0", 1.2),
       ("Multistep-stage-C", "Multistep", "C1", 1.2), ("ICON-CH2-CTRL", "ICON-CH2", "C2", 1.2)]
SC, NSTEP = "3-6h", 121
lead = np.arange(NSTEP)[scales.interior(SC)]

st = pd.read_csv("resources/smn_grid_points.csv")
cls = dict(zip(st.nat_abbr, st.nat_abbr.map(station_classes(
    Path("resources/spaziurat.lf_2.6.1.tar.gz")))))
elev = dict(zip(st.nat_abbr, st.elevation))
for season in ("winter2024", "summer2024"):
    data = {p: load_season(Path("output/points"), Path("output/points_truth"), season, p,
                           ("ICON-CH2-CTRL",)) for p in PARAMS}
    inits = data[PARAMS[0]][1]
    with xr.open_dataset(f"output/points/{season}/Varda-single-stage-C/{inits[0]}.nc") as d:
        col = {s: k for k, s in enumerate(d["station"].values.astype(str))}
    ok = np.ones(len(inits), bool)
    for p in PARAMS:
        for s, *_ in SRC[1:]:
            ok &= ~np.isnan(data[p][0][s][:, :NSTEP]).any(axis=(1, 2))
    pick = np.sort(np.random.default_rng(0).choice(np.flatnonzero(ok), 10, replace=False))
    for p in PARAMS:
        dat = data[p][0]
        d = OUT / season / p
        d.mkdir(parents=True, exist_ok=True)
        for stn in STATIONS:
            j = col[stn]
            if args.single:
                (d / stn).mkdir(exist_ok=True)
                for i in pick:
                    fig, ax = plt.subplots(figsize=(12, 6.75))
                    for s, lab, c, lw in SRC:
                        if s in dat and not np.isnan(dat[s][i, :NSTEP, j]).any():
                            ax.plot(lead, scales.component(dat[s][i, :NSTEP, j][:, None], SC)[:, 0],
                                    color=c, lw=lw + 0.4, label=lab)
                    ax.axhline(0, color="0.7", lw=0.6)
                    ax.set_xlim(lead[0], lead[-1]); ax.set_xticks(range(24, 97, 6))
                    ax.set_xlabel("lead time [h]")
                    ax.set_ylabel(f"3-6 h component of {p} [{UNIT[p]}]")
                    ax.grid(alpha=0.3); ax.legend(ncol=5, fontsize=9, loc="upper right")
                    ax.set_title(f"{stn} ({cls.get(stn)}, {elev[stn]:.0f} m), {season}, {p}, "
                                 f"init {inits[i]}: 3-6 h filtered series", fontsize=11)
                    fig.tight_layout()
                    fig.savefig(d / stn / f"{inits[i]}.png", dpi=110)
                    plt.close(fig)
                continue
            fig, axes = plt.subplots(len(pick), 1, figsize=(11, 15), sharex=True, sharey=True)
            for ax, i in zip(axes, pick):
                for s, lab, c, lw in SRC:
                    if s not in dat or np.isnan(dat[s][i, :NSTEP, j]).any():
                        continue
                    y = scales.component(dat[s][i, :NSTEP, j][:, None], SC)[:, 0]
                    ax.plot(lead, y, color=c, lw=lw, label=lab)
                ax.axhline(0, color="0.7", lw=0.6)
                ax.text(0.005, 0.92, f"init {inits[i]}", transform=ax.transAxes, fontsize=8,
                        va="top", color="0.3")
                ax.grid(alpha=0.25)
            axes[0].legend(ncol=5, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.02))
            axes[-1].set_xlabel("lead time [h]")
            axes[-1].set_xticks(range(24, 97, 12))
            axes[len(pick) // 2].set_ylabel(f"3-6 h component of {p} [{UNIT[p]}]")
            fig.suptitle(f"{stn} ({cls.get(stn)}, {elev[stn]:.0f} m), {season}, {p}: 3-6 h "
                         "filtered series, 10 random inits", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.975))
            fig.savefig(d / f"{stn}.png", dpi=110)
            plt.close(fig)
    print(season, "inits:", [inits[i] for i in pick])
print(f"wrote {OUT}/<season>/<param>/<station>.png")
