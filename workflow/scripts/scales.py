"""Time-scale components of a 121 h hourly series, shared by the variance scripts.

`component(x, scale)` takes one init's series, shape (121, n_station), and returns
the part at that time scale on the scale's interior lead times. Band variance is
its variance, the timing correlation its covariance with REA-L's, as for 3-6 h.
Every scale starts from the same linearly detrended 121 h record. Except for
3-6h, which is left bit-identical to the original method (its interior mean is
already ~0), the component is centred on its interior, because the timing
correlation takes the covariance as mean(forecast x truth).

  3-6h      4th-order Butterworth bandpass 1/6-1/3 cyc/h, zero phase (sosfiltfilt),
            24 h discarded at each end. Exactly the original method.
  6-12h     2nd-order Butterworth bandpass 1/12-1/6 cyc/h, zero phase, 26 h
            discarded. 4th order would need ~45 h guards and leave 31 h; at 2nd
            order the kernel is below 1 % of its peak beyond 26 h.
  diurnal   least-squares fit of 24 h and 12 h harmonics (plus a constant) over
            lead 24-96 h; the fitted harmonics are the component. A bandpass
            cannot isolate 12-24 h in a 121 h record: its kernel is as long as
            the record.
  synoptic  24 h running mean applied twice (a 47 h triangular kernel), on lead
            24-96 h. It zeroes 24 h and all its harmonics exactly and passes
            periods of about two days and longer: power passed is ~0.16 at 48 h,
            ~0.66 at 96 h. Detrending first removes the straight-line part of the
            synoptic evolution, so this is synoptic variability beyond a trend.

`remove_daily_clim` subtracts each source's mean daily cycle (per station and
season, by hour of day over all inits and leads) before any scale is taken. The
daily cycle is not a pure 24 h sine: its harmonics at 12, 8, 6, 4.8, 4 h ... fall
into every sub-daily band and are sun-driven, hence predictable for free. The
anomaly variant keeps only the weather-dependent part (a stronger-than-usual
cycle on a clear day, a front at an unusual hour).

    uv run workflow/scripts/scales.py [--plot out.png]   # response of each scale
"""
import numpy as np
from scipy.signal import butter, detrend, sosfiltfilt

NSTEP = 121
SCALES = {  # name -> interior lead-time slice, label
    "3-6h": (slice(24, 97), "3-6 h"),
    "6-12h": (slice(26, 95), "6-12 h"),
    "diurnal": (slice(24, 97), "daily cycle (24 + 12 h)"),
    "synoptic": (slice(24, 97), "synoptic (> ~2 days)"),
}
_SOS = {"3-6h": butter(4, [1 / 6, 1 / 3], btype="band", fs=1.0, output="sos"),
        "6-12h": butter(2, [1 / 12, 1 / 6], btype="band", fs=1.0, output="sos")}
_TRI = np.convolve(np.ones(24) / 24, np.ones(24) / 24)   # 47 h, centred on index 23


def interior(scale: str) -> slice:
    return SCALES[scale][0]


def label(scale: str) -> str:
    return SCALES[scale][1]


def component(x: np.ndarray, scale: str) -> np.ndarray:
    """(121, n) raw series -> (n_interior, n) component on the scale's interior."""
    xd = detrend(x, axis=0, type="linear")
    sl = interior(scale)
    if scale == "3-6h":
        return sosfiltfilt(_SOS[scale], xd, axis=0)[sl]
    if scale in _SOS:
        c = sosfiltfilt(_SOS[scale], xd, axis=0)[sl]
    elif scale == "diurnal":
        t = np.arange(NSTEP)[sl].astype(float)
        A = np.column_stack([np.ones_like(t)] + [f(2 * np.pi * t / P) for P in (24, 12)
                                                  for f in (np.cos, np.sin)])
        coef = np.linalg.lstsq(A, xd[sl], rcond=None)[0]
        c = A[:, 1:] @ coef[1:]
    elif scale == "synoptic":
        lo = np.apply_along_axis(np.convolve, 0, xd, _TRI, mode="valid")  # centres 23..97
        c = lo[sl.start - 23:sl.stop - 23]
    else:
        raise ValueError(scale)
    return c - c.mean(axis=0)


def remove_daily_clim(data: dict, inits) -> dict:
    """source -> (n_init, 121, n) with each source's own mean daily cycle removed."""
    hod = (np.array([int(i[8:10]) for i in inits])[:, None] + np.arange(NSTEP)) % 24
    out = {}
    for s, a in data.items():
        a = a[:, :NSTEP]
        clim = np.stack([np.nanmean(a[hod == h], axis=0) for h in range(24)])  # (24, n)
        out[s] = a - clim[hod]
    return out


def response(scale: str, periods, nphase: int = 24) -> np.ndarray:
    """Fraction of a pure sine's variance that survives, averaged over phase."""
    t = np.arange(NSTEP)[:, None]
    out = []
    for P in periods:
        ph = np.linspace(0, 2 * np.pi, nphase, endpoint=False)[None, :]
        out.append(component(np.sin(2 * np.pi * t / P + ph), scale).var(axis=0).mean() / 0.5)
    return np.array(out)


if __name__ == "__main__":
    import sys
    P = [2.5, 3, 4, 4.5, 6, 8, 9, 12, 16, 24, 36, 48, 72, 96]
    print("period h  " + " ".join(f"{p:5g}" for p in P))
    for s in SCALES:
        print(f"{s:9s} " + " ".join(f"{v:5.2f}" for v in response(s, P)))
    if "--plot" in sys.argv:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out = sys.argv[sys.argv.index("--plot") + 1]
        per = np.geomspace(2.2, 150, 400)
        fig, ax = plt.subplots(figsize=(9, 4))
        for k, s in enumerate(SCALES):
            ax.plot(per, response(s, per), color=f"C{k}", lw=2, label=label(s))
        for h in (24, 12, 8, 6, 4.8, 4):
            ax.axvline(h, color="0.6", lw=0.8, ls=":")
        ax.text(24, 1.07, "daily cycle and its harmonics (dotted)", ha="right", fontsize=8,
                color="0.4")
        ax.set_xscale("log"); ax.set_xlim(2.2, 150); ax.set_ylim(0, 1.12)
        ticks = [3, 4, 6, 8, 12, 24, 48, 96]
        ax.set_xticks(ticks); ax.set_xticklabels(ticks)
        ax.set_xlabel("period of a pure sine [h]"); ax.set_ylabel("fraction of variance kept")
        ax.set_title("What each time scale picks out of a 121 h series (lead 24-96 h)",
                     fontsize=10)
        ax.legend(fontsize=9, loc="center right"); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out, dpi=150)
        print(f"wrote {out}")
