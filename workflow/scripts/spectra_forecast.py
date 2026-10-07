"""Stage 2: temporal spectra of the forecasts against REA-L, from the cached points.

Reads only output/points/ and output/points_truth/ (~96 MB), so it runs anywhere
and can be rerun freely. See exp_plan_01.md Step 1.

Four sources per season, all at the same 145 grid points and the same valid times:

  Varda-single-stage-C   the 6-hourly forecaster plus the learned downscaler
  Multistep-stage-C      hourly states carried forward
  Varda-interp-6h        the 6-hourly forecaster, linearly interpolated to hourly.
                         Not a competitor: it is the floor the learned downscaler
                         must beat to be doing more than interpolating.
  REA-L                  truth, on matched valid times

Windows: three 48 h lead-time windows at steps [0,48), [36,84), [72,120), so the
spectrum can be read against lead time. Hann taper and linear detrend are both
essential; without them leakage from the diurnal and synoptic peaks swamps the
3-6 h band we are measuring.

Excluded windows: 10 Varda-single runs contain all-NaN blocks from the known
GRIB 9999-geopotential collision (see the Data_for_Nadja DATA_CORRECTION_NOTE).
Any (init, window) that is incomplete in *any* source is dropped from *every*
source, so the four curves always describe the same sample.

Precipitation: step 0 has no accumulation, so its first window is [1,49) and it
is reported separately. Its spectrum is dominated by intermittency rather than
smooth variability, so do not pool it with the others.

    uv run workflow/scripts/spectra_forecast.py --season winter2024
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import periodogram

LOG = logging.getLogger("spectra_forecast")

WINDOW_H = 48
WINDOW_STARTS = (0, 36, 72)
FORECAST_SOURCES = ("Varda-single-stage-C", "Multistep-stage-C")
INTERP_SOURCE = "Varda-forecaster-6h"
INTERP_LABEL = "Varda-interp-6h"
TRUTH_LABEL = "REA-L"
OBS_LABEL = "SwissMetNet"
OBS_FILE = Path("output/points_obs/obs_hourly.nc")
# forecast name -> REA-L name
PARAMS = {"T_2M": "T_2M", "TD_2M": "TD_2M", "U_10M": "U_10M",
          "V_10M": "V_10M", "PS": "PS", "TOT_PREC1": "TOT_PREC_1H"}


def psd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(time, points) -> (freq, power). Linear detrend + Hann taper."""
    return periodogram(x, fs=1.0, window="hann", detrend="linear",
                       scaling="density", axis=0)


def interp_to_hourly(x6: np.ndarray, steps6: np.ndarray, steps1: np.ndarray) -> np.ndarray:
    """Linear interpolation of the 6-hourly series onto hourly steps, per point.

    Done on the extracted 145-point series, never on full fields.
    """
    out = np.empty((len(steps1), x6.shape[1]), dtype=np.float64)
    for j in range(x6.shape[1]):
        out[:, j] = np.interp(steps1, steps6, x6[:, j])
    return out


def series(ds, param: str) -> np.ndarray | None:
    """Pull `param` from a dataset, deriving 10 m wind speed where needed.

    Speed is a nonlinear function of U and V, so it cannot be recovered from
    their spectra and must be formed on the series before any transform. SMN
    measures speed natively, so the models are converted to speed rather than
    the observations to components; that also avoids interpolating a circular
    wind direction.
    """
    if param == "SP_10M":
        if "SP_10M" in ds:
            return ds["SP_10M"].values.astype(float)
        if "U_10M" in ds and "V_10M" in ds:
            return np.hypot(ds["U_10M"].values.astype(float),
                            ds["V_10M"].values.astype(float))
        return None
    return ds[param].values.astype(float) if param in ds else None


def load_season(root: Path, truth_root: Path, season: str, param: str,
                extra: tuple[str, ...] = ()):
    """Return dict source -> (n_init, 121, n_station) and the init list.

    `extra` names further hourly sources under root/season/ (e.g. the ICON
    baselines). A source with fewer than 121 steps (ICON-CH1 stops at 33 h) is
    padded with NaN, so the matched-sample masks drop it wherever it is absent.
    """
    hourly_steps = np.arange(121)
    truth = xr.open_mfdataset(sorted(truth_root.glob("*.nc")), combine="by_coords")
    tvar = PARAMS.get(param)
    tser = truth[tvar].load() if tvar and tvar in truth else None
    # REA-L stores precipitation in m, the forecasts in kg m-2 (mm): the
    # inference config rescales by 1000 on the forecast side only. Without this
    # the truth spectrum sits a factor 1e6 below the models' in power.
    if param == "TOT_PREC1":
        tser = tser * 1000.0
    if param == "SP_10M":
        tser = np.hypot(truth["U_10M"].load(), truth["V_10M"].load())

    inits = sorted(p.stem for p in (root / season / FORECAST_SOURCES[0]).glob("*.nc"))
    out: dict[str, np.ndarray] = {}

    for src in (*FORECAST_SOURCES, *extra):
        arr = []
        for init in inits:
            with xr.open_dataset(root / season / src / f"{init}.nc") as d:
                a = series(d, param)
                full = np.full((121, d.sizes["station"]), np.nan)
                if a is not None:
                    full[d["step"].values] = a
                arr.append(full)
        out[src] = np.stack(arr)

    obs = xr.open_dataset(OBS_FILE) if OBS_FILE.exists() else None
    oser = obs[param].load() if obs is not None and param in obs else None

    arr, tarr, oarr = [], [], []
    for init in inits:
        with xr.open_dataset(root / season / INTERP_SOURCE / f"{init}.nc") as d:
            a = series(d, param)
            if a is not None:
                arr.append(interp_to_hourly(a, d["step"].values.astype(float),
                                            hourly_steps.astype(float)))
            else:  # TOT_PREC1 absent from the 6-hourly run by design
                arr.append(np.full((121, d.sizes["station"]), np.nan))
        with xr.open_dataset(root / season / FORECAST_SOURCES[0] / f"{init}.nc") as d:
            vt = d["valid_time"].values
            tarr.append(np.asarray(tser.sel(time=vt).values)
                        if tser is not None else np.full((121, d.sizes["station"]), np.nan))
            if oser is not None:
                oarr.append(np.asarray(oser.sel(time=vt).values))
    out[INTERP_LABEL] = np.stack(arr)
    out[TRUTH_LABEL] = np.stack(tarr)
    if oarr:
        out[OBS_LABEL] = np.stack(oarr)
    return out, inits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", required=True)
    ap.add_argument("--param", default="T_2M")
    ap.add_argument("--points", type=Path, default=Path("output/points"))
    ap.add_argument("--truth", type=Path, default=Path("output/points_truth"))
    ap.add_argument("--out-dir", type=Path, default=Path("output/spectra"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    data, inits = load_season(args.points, args.truth, args.season, args.param)
    LOG.info("%s %s: %d inits, sources %s", args.season, args.param,
             len(inits), list(data))

    starts = list(WINDOW_STARTS)
    if args.param == "TOT_PREC1":
        starts[0] = 1  # no accumulation over the hour before init

    results, counts, freqs = {}, {}, None
    for w, s in enumerate(starts):
        sl = slice(s, s + WINDOW_H)
        # A window is usable only where every source is finite, so all four
        # curves describe the same sample.
        ok = np.ones(len(inits), bool)
        for src, a in data.items():
            if np.isnan(a[:, sl]).all():
                continue  # source lacks this param entirely
            ok &= ~np.isnan(a[:, sl]).any(axis=(1, 2))
        counts[s] = int(ok.sum())
        LOG.info("window %d-%d h: %d/%d inits usable", s, s + WINDOW_H, ok.sum(), len(inits))
        for src, a in data.items():
            sub = a[ok][:, sl]
            if not np.isfinite(sub).all():
                continue
            spec = [psd(sub[i])[1] for i in range(sub.shape[0])]
            f = psd(sub[0])[0]
            results[(src, s)] = np.mean(spec, axis=(0, 2))
            freqs = f

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.season}_{args.param}"
    np.savez(args.out_dir / f"spectra_{tag}.npz",
             freq=freqs,
             **{f"{src}__{s}": v for (src, s), v in results.items()},
             window_starts=np.array(starts),
             usable=np.array([counts[s] for s in starts]),
             n_inits=len(inits))

    f = freqs
    period = np.divide(1.0, f, out=np.full_like(f, np.inf), where=f > 0)
    band = (period >= 3) & (period <= 6)
    six = np.argmin(np.abs(period - 6.0))
    print(f"\n=== {tag} ===")
    for s in starts:
        print(f"\n lead {s}-{s+WINDOW_H} h  ({counts[s]}/{len(inits)} inits)")
        ref = results.get((TRUTH_LABEL, s))
        for src in (TRUTH_LABEL, OBS_LABEL, *FORECAST_SOURCES, INTERP_LABEL):
            v = results.get((src, s))
            if v is None:
                continue
            frac = v[band].sum() / ref[band].sum()
            print(f"   {src:22s} 3-6h power {v[band].sum():10.4g}  "
                  f"({frac:6.1%} of REA-L)   at 6h {v[six]:9.4g}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {TRUTH_LABEL: ("k", "-", 2.0),
             OBS_LABEL: ("0.45", "-", 1.6),
             "Varda-single-stage-C": ("C0", "-", 1.4),
             "Multistep-stage-C": ("C3", "-", 1.4),
             INTERP_LABEL: ("C2", "--", 1.2)}
    m = f > 0
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    for ax, s in zip(axes, starts):
        for src, (c, ls, lw) in style.items():
            v = results.get((src, s))
            if v is None:
                continue
            ax.loglog(period[m], v[m], color=c, ls=ls, lw=lw, label=src)
        ax.axvspan(3, 6, color="0.85", zorder=0)
        ax.axvspan(2, 3, color="0.93", zorder=0)
        ax.axvline(6, color="0.4", lw=0.8, ls=":")
        ax.invert_xaxis()
        # Label real periods; the default log decades are unreadable here.
        ticks = [48, 24, 12, 8, 6, 4, 3, 2]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks], fontsize=8)
        ax.set_xticks([], minor=True)
        ax.set_xlabel("period (h)")
        ax.set_title(f"lead {s}-{s + WINDOW_H} h   ({counts[s]}/{len(inits)} inits)",
                     fontsize=10)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel(f"PSD of {args.param}")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"{args.season} {args.param}: temporal spectra at 145 SMN grid points. "
                 f"Shaded 3-6 h is the claim band; 2-3 h is too close to the 2 h Nyquist "
                 f"limit to trust.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out_dir / f"spectra_{tag}.png", dpi=150)

    print(f"\nwrote {args.out_dir}/spectra_{tag}.{{npz,png}}")


if __name__ == "__main__":
    main()
