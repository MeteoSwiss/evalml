"""How a time-scale component is extracted, on one real series.

Left: REA-L at one station for one init, (a) as is with its linear trend,
(b) what enters the extraction (detrended; with --anom also minus the station's
mean daily cycle), (c) the component on the scale's lead window, with the
discarded leads greyed and its variance (the quantity every FRAC is built from)
stated. Right: the method itself, (d) its kernel, i.e. the response to a single
1-hour spike in the middle of the record, and (e) the fraction of variance kept
at each period.

Default --scale 3-6h is the 4th-order bandpass of bandpass_timedomain.py; the
other scales are those of scales.py. The init is picked at the 75th percentile
of the station's component variance, so the example is typical-to-active.

    uv run workflow/scripts/plot_bandpass_explainer.py --station PAY --season summer2024
    uv run workflow/scripts/plot_bandpass_explainer.py --scale diurnal --anom
"""
import argparse, sys
from pathlib import Path
import numpy as np
import xarray as xr
from scipy.signal import sosfilt
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from spectra_forecast import TRUTH_LABEL, load_season
import scales

NSTEP = 121
HOW = {"3-6h": "4th-order Butterworth bandpass, 1/6-1/3 cycles per hour, applied forward and "
               "backward (sosfiltfilt)",
       "6-12h": "2nd-order Butterworth bandpass, 1/12-1/6 cycles per hour, applied forward "
                "and backward (sosfiltfilt)",
       "diurnal": "least-squares fit of 24 h and 12 h harmonics over the lead window",
       "synoptic": "24 h running mean applied twice (47 h triangular kernel)"}
BAND = {"3-6h": (3, 6), "6-12h": (6, 12), "diurnal": None, "synoptic": (48, 150)}

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--station", default="PAY")
ap.add_argument("--season", default="summer2024")
ap.add_argument("--param", default="T_2M")
ap.add_argument("--scale", default="3-6h", choices=scales.SCALES)
ap.add_argument("--anom", action="store_true", help="remove the mean daily cycle first")
ap.add_argument("--out", type=Path, help="default output/explainer/bandpass_explained[_<scale>][_anom].png")
args = ap.parse_args()
sc = args.scale
out = args.out or Path("output/explainer/bandpass_explained"
                       + ("" if sc == "3-6h" else f"_{sc}") + ("_anom" if args.anom else "") + ".png")

data, inits = load_season(Path("output/points"), Path("output/points_truth"),
                          args.season, args.param)
with xr.open_dataset(f"output/points/{args.season}/Varda-single-stage-C/{inits[0]}.nc") as d:
    j = list(d["station"].values.astype(str)).index(args.station)
raw = data[TRUTH_LABEL][:, :NSTEP, j]
series = scales.remove_daily_clim({"t": data[TRUTH_LABEL]}, inits)["t"][:, :NSTEP, j] \
    if args.anom else raw
inner = scales.interior(sc)
t = np.arange(NSTEP)
ok = np.flatnonzero(np.isfinite(series).all(axis=1))
cv = np.array([scales.component(series[i][:, None], sc).var() for i in ok])
i = ok[np.argsort(cv)[int(0.75 * (len(cv) - 1))]]
x0, x = raw[i], series[i]
trend0 = x0 - (x0 - np.polyval(np.polyfit(t, x0, 1), t))
xd = x - np.polyval(np.polyfit(t, x, 1), t)
y = scales.component(x[:, None], sc)[:, 0]
ti = t[inner]

fig = plt.figure(figsize=(13, 8.5))
gs = fig.add_gridspec(3, 2, width_ratios=[1.6, 1], hspace=0.45, wspace=0.25)
ax_a, ax_b, ax_c = (fig.add_subplot(gs[k, 0]) for k in range(3))
ax_d = fig.add_subplot(gs[0:2, 1]); ax_e = fig.add_subplot(gs[2, 1])
unit = "K" if args.param in ("T_2M", "TD_2M") else ""

ax_a.plot(t, x0, color="0.2", lw=1.6, label=f"REA-L {args.param}")
ax_a.plot(t, trend0, color="C1", lw=1.4, ls="--", label="linear trend (removed)")
ax_a.set_title("(a) original series, lead 0-120 h", loc="left", fontsize=10)
ax_b.plot(t, xd, color="0.2", lw=1.6)
ax_b.set_title("(b) detrended" + (", minus this station's mean daily cycle" if args.anom else "")
               + ": this goes into the extraction", loc="left", fontsize=10)
ax_c.plot(ti, y, color="C0", lw=1.8)
ax_c.set_title(f"(c) the {scales.label(sc)} component. Its variance = {y.var():.3f} {unit}² "
               f"(vs {xd[inner].var():.2f} {unit}² in (b) over the same leads)",
               loc="left", fontsize=10)
for ax in (ax_a, ax_b, ax_c):
    for lo, hi in ((0, inner.start), (inner.stop - 1, NSTEP - 1)):
        ax.axvspan(lo, hi, color="0.88", zorder=0)
    ax.set_xlim(0, NSTEP - 1); ax.set_xticks(range(0, 121, 12))
    ax.set_ylabel(unit); ax.grid(alpha=0.3)
ax_c.axhline(0, color="0.5", lw=0.8)
ax_c.set_xlabel("lead time [h]")
ax_c.text(inner.start / 2, ax_c.get_ylim()[1] * 0.8, "discarded", ha="center",
          fontsize=8, color="0.35")
ax_a.legend(fontsize=8, loc="best")

# (d) response to a spike at lead 60 h, the middle of every lead window
imp = np.zeros(NSTEP); imp[60] = 1.0
lag = ti - 60
if sc in ("3-6h", "6-12h"):
    ax_d.plot(np.arange(NSTEP) - 60, sosfilt(scales._SOS[sc], imp), color="0.55", lw=1.4,
              label="one pass (causal: responds after the spike)")
ax_d.plot(lag, scales.component(imp[:, None], sc)[:, 0], color="C0", lw=2.0,
          label={"3-6h": "forward + backward (what is applied):\nsymmetric, so no time shift",
                 "6-12h": "forward + backward (what is applied):\nsymmetric, so no time shift",
                 "diurnal": "harmonic fit: the spike leaks into the\nfitted 24 h + 12 h waves over the whole window",
                 "synoptic": "double running mean: a 47 h triangle,\nsymmetric, so no time shift"}[sc])
ax_d.axhline(0, color="0.5", lw=0.8); ax_d.axvline(0, color="0.5", lw=0.8, ls=":")
ax_d.set_xlim(lag[0], lag[-1])
ax_d.set_xlabel("time relative to the spike [h]")
ax_d.set_title("(d) kernel: response to a single 1-hour spike at lead 60 h", loc="left", fontsize=10)
ax_d.legend(fontsize=8, loc="lower left"); ax_d.grid(alpha=0.3)

per = np.geomspace(2.2, 150, 300)
if BAND[sc]:
    ax_e.axvspan(*BAND[sc], color="C0", alpha=0.12, lw=0)
for h in ((24, 12) if sc == "diurnal" else ()):
    ax_e.axvline(h, color="C0", alpha=0.3, lw=6)
ax_e.plot(per, scales.response(sc, per), color="C0", lw=2.0)
ax_e.set_xscale("log"); ax_e.set_xlim(2.2, 150)
tk = [3, 4, 6, 8, 12, 24, 48, 96]
ax_e.set_xticks(tk); ax_e.set_xticklabels(tk)
ax_e.set_ylim(0, 1.05); ax_e.set_xlabel("period of a pure sine [h]")
ax_e.set_ylabel("fraction of variance kept")
ax_e.set_title("(e) what is kept, by period", loc="left", fontsize=10)
ax_e.grid(alpha=0.3)

fig.suptitle(f"The {scales.label(sc)} component, step by step: {args.station}, {args.season}, "
             f"init {inits[i]} (75th percentile of this station's component variance)\n"
             f"{HOW[sc]}; leads {inner.start}-{inner.stop - 1} h kept"
             + ("; mean daily cycle removed first" if args.anom else ""), fontsize=11)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}  (init {inits[i]}, component var {y.var():.4f})")
