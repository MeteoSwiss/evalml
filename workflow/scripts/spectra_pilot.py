"""Pilot: does REA-L-CH1 carry usable 2-6 h temporal power at station points?

Falsification step for Step 1 of exp_plan_01.md. Step 1 claims Varda shows a
deficit in temporal spectral power below ~6 h that the multistep model partly
recovers. That test is only meaningful if the truth itself has power in that
band. REA-L-CH1 is a reanalysis and may well be temporally smooth there, in
which case neither model can be shown to be missing anything and Step 1 is
unfalsifiable. This script settles that before any forecast GRIB is touched.

Run:
    uv run python workflow/scripts/spectra_pilot.py --season winter
    uv run python workflow/scripts/spectra_pilot.py --season summer
    uv run python workflow/scripts/spectra_pilot.py --self-test

Outputs a .npz of the mean spectra and a .png, per season.

Reference curve: the same REA-L series subsampled to 6 h and linearly
interpolated back to hourly. That is the ceiling on what any 6-hourly forecaster
plus a purely interpolating reconstruction could retain, so the gap between it
and REA-L is the power that is in principle available for the multistep design
to win. It is not a model of Varda's downscaler, which is learned and can beat
interpolation; it is the floor that the downscaler must beat to be doing
anything at all.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import zarr
from scipy.signal import periodogram
from scipy.spatial import cKDTree

LOG = logging.getLogger("spectra_pilot")

REAL_ZARR = "/store_new/mch/msopr/ml/datasets/mch-realch1-fdb-1km-2005-2025-1h-pl13-v1.0.zarr"
SEASONS = {"winter": (1, 2, 12), "summer": (6, 7, 8)}
WINDOW_H = 48  # hours per spectral window; see exp_plan_01.md Step 1


def psd(series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Periodogram of hourly series, shape (time, points) -> (freq, power).

    Linear detrend and a Hann taper are both essential: diurnal and synoptic
    power sits orders of magnitude above the 2-6 h band, and without them
    spectral leakage from those low frequencies swamps exactly the band of
    interest. Frequencies are in cycles per hour.
    """
    f, p = periodogram(
        series, fs=1.0, window="hann", detrend="linear", scaling="density", axis=0
    )
    return f, p


def degrade_6h(series: np.ndarray) -> np.ndarray:
    """Subsample to 6 h and linearly interpolate back to hourly."""
    n = series.shape[0]
    t = np.arange(n)
    t6 = t[::6]
    out = np.empty_like(series)
    for j in range(series.shape[1]):
        out[:, j] = np.interp(t, t6, series[t6, j])
    return out


def self_test() -> None:
    """Check the estimator recovers a known weak high-frequency signal that sits
    under a much stronger diurnal cycle and a trend."""
    rng = np.random.default_rng(0)
    n = WINDOW_H
    hours = np.arange(n)
    # 3 h signal at amplitude 0.2 K, under a 5 K diurnal cycle and a 4 K trend.
    base = (
        5.0 * np.sin(2 * np.pi * hours / 24)
        + 0.2 * np.sin(2 * np.pi * hours / 3)
        + 4.0 * hours / n
    )
    x = base[:, None] + 0.01 * rng.standard_normal((n, 50))
    f, p = psd(x)
    period = np.divide(1.0, f, out=np.full_like(f, np.inf), where=f > 0)
    peak = period[1:][np.argmax(p[1:, 0])]
    near3 = p[np.argmin(np.abs(period - 3.0)), 0]
    floor = np.median(p[(period > 2) & (period < 6), 0])
    assert 2.7 < peak < 3.4 or near3 > 10 * floor, (
        f"3 h peak not recovered: strongest period {peak:.2f} h, "
        f"power at 3 h {near3:.3e} vs band median {floor:.3e}"
    )
    # Same signal, but 6-hourly sampling cannot represent a 3 h period at all
    # (Nyquist is 12 h): the degraded series must lose it.
    _, pd_ = psd(degrade_6h(x))
    assert pd_[np.argmin(np.abs(period - 3.0)), 0] < 0.05 * near3, (
        "degrade_6h did not remove the 3 h signal"
    )
    print("self-test passed: 3 h signal recovered under a 25x stronger diurnal "
          "cycle, and removed by 6-hourly sampling as expected")


def station_coords(cache: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SwissMetNet station abbreviations, latitudes, longitudes.

    Fetched once from the DWH and then cached, so that every later run and every
    downstream diagnostic uses exactly the same station set and coordinates. A
    drifting station list would silently change the sample between the truth
    spectra and the forecast spectra they are compared against.

    The fetch needs DWH credentials (JRETRIEVE_CLIENT_ID and
    JRETRIEVE_CLIENT_SECRET, in the environment or in .env). Once the cache
    exists, no credentials are needed.
    """
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        LOG.info("using cached station list %s (%d stations)", cache, len(d["abbr"]))
        return d["abbr"], d["lat"], d["lon"]

    from data_input import jretrieve as jr

    jr.check_prerequisites("prod")
    meta = jr.fetch_meta(stations={"group": "SwissMetNet"}, params=["tre200s0"])
    cat = jr.StationCatalog.from_meta(meta)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, abbr=cat.nat_abbr, lat=cat.latitude, lon=cat.longitude,
             elevation=cat.elevation, name=cat.name, station_id=cat.station_id)
    LOG.info("fetched and cached %d stations to %s", cat.n, cache)
    return cat.nat_abbr, cat.latitude, cat.longitude


def grid_points(z, n: int, seed: int = 0) -> np.ndarray:
    """Random subsample of REA-L grid-point indices.

    Fallback for when DWH credentials are unavailable. It answers the pilot's
    question (is there 2-6 h power in REA-L at all) but samples the domain
    uniformly rather than at station locations, so it includes many smooth
    high-altitude and flat points that no station sits on. Expect it to be a
    slightly conservative estimate of the power available at station sites.
    """
    rng = np.random.default_rng(seed)
    return rng.choice(z["latitudes"].shape[0], size=n, replace=False)


def window_starts(dates: np.ndarray, months: tuple[int, ...], n_windows: int) -> list[int]:
    """Indices into `dates` of `n_windows` non-overlapping WINDOW_H blocks,
    spread evenly over the requested months of 2024."""
    yr = dates.astype("datetime64[Y]").astype(int) + 1970
    mo = dates.astype("datetime64[M]").astype(int) % 12 + 1
    eligible = np.flatnonzero((yr == 2024) & np.isin(mo, months))
    # keep only starts whose whole window stays inside the season
    ok = [i for i in eligible if i + WINDOW_H <= len(dates) and mo[i + WINDOW_H - 1] in months]
    if len(ok) < n_windows:
        raise SystemExit(f"only {len(ok)} candidate windows for months {months}")
    return [ok[k] for k in np.linspace(0, len(ok) - 1, n_windows).astype(int)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", choices=sorted(SEASONS))
    ap.add_argument("--param", default="T_2M")
    ap.add_argument("--n-windows", type=int, default=15,
                    help="48 h windows per season. Each costs a full-field zarr "
                         "decompression per hour, so this is the main cost knob.")
    ap.add_argument("--points", choices=("station", "grid"), default="station",
                    help="'station' needs DWH credentials; 'grid' is the fallback")
    ap.add_argument("--n-points", type=int, default=200,
                    help="number of grid points when --points grid")
    ap.add_argument("--stations-cache", type=Path,
                    default=Path("resources/smn_stations.npz"),
                    help="cached SwissMetNet station list; fetched from the DWH "
                         "on first use, then reused without credentials")
    ap.add_argument("--out-dir", type=Path, default=Path("output/spectra_pilot"))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.self_test:
        self_test()
        return
    if not args.season:
        raise SystemExit("--season is required unless --self-test")

    self_test()  # never produce a figure from an unvalidated estimator

    z = zarr.open(REAL_ZARR, mode="r")
    dates = z["dates"][:]
    var_idx = list(z.attrs["variables"]).index(args.param)

    if args.points == "station":
        abbr, slat, slon = station_coords(args.stations_cache)
        LOG.info("matching %d SwissMetNet stations to the REA-L grid", len(abbr))
        glat, glon = z["latitudes"][:], z["longitudes"][:]

        def xyz(lat, lon):
            la, lo = np.radians(lat), np.radians(lon)
            return np.stack(
                [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1
            )

        _, pt = cKDTree(xyz(glat, glon)).query(xyz(slat, slon))
    else:
        pt = grid_points(z, args.n_points)
        abbr = np.array([f"grid{i}" for i in pt])
        LOG.info("using %d random REA-L grid points", len(pt))

    starts = window_starts(dates, SEASONS[args.season], args.n_windows)
    LOG.info("reading %d windows x %d h (each hour = one full-field chunk)",
             len(starts), WINDOW_H)

    specs, specs6, specs_rep = [], [], []
    for k, s in enumerate(starts, 1):
        block = np.empty((WINDOW_H, len(pt)), dtype=np.float32)
        for h in range(WINDOW_H):
            block[h] = z["data"][s + h, var_idx, 0, :][pt]
        LOG.info("window %d/%d starting %s", k, len(starts), dates[s])
        x = block.astype(np.float64)

        # Repair the 00 UTC artefact: join_detect.py shows an isolated one-hour
        # spike in the hourly increments of winter T_2M at 00 UTC, absent from
        # PMSL and from summer, i.e. a surface-side daily update rather than a
        # full segment join. Replacing the 00 UTC value by the mean of its
        # neighbours removes the jump. Comparing the two spectra measures how
        # much the artefact actually contaminates the band of interest, instead
        # of us reasoning about it.
        hod = (dates[s : s + WINDOW_H].astype("datetime64[h]").astype(int)) % 24
        rep = x.copy()
        for h in np.flatnonzero((hod == 0) & (np.arange(WINDOW_H) > 0)
                                & (np.arange(WINDOW_H) < WINDOW_H - 1)):
            rep[h] = 0.5 * (x[h - 1] + x[h + 1])

        f, p = psd(x)
        _, p6 = psd(degrade_6h(x))
        _, pr = psd(rep)
        specs.append(p)
        specs6.append(p6)
        specs_rep.append(pr)

    mean, mean6 = np.mean(specs, axis=(0, 2)), np.mean(specs6, axis=(0, 2))
    mean_rep = np.mean(specs_rep, axis=(0, 2))
    period = np.divide(1.0, f, out=np.full_like(f, np.inf), where=f > 0)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.season}_{args.param}_{args.points}"
    np.savez(args.out_dir / f"spectra_pilot_{tag}.npz", freq=f, period=period,
             psd_real=mean, psd_real_6h_interp=mean6, psd_real_repaired=mean_rep,
             n_windows=len(starts),
             n_stations=len(pt), stations=abbr,
             window_start_dates=np.array([str(dates[s]) for s in starts]))

    # 3-6 h is the honest band: 2-3 h sits next to the 2 h Nyquist limit and is
    # the most alias-contaminated. See exp_plan_01.md Step 1.
    band = (period >= 3) & (period <= 6)
    wide = (period >= 2) & (period <= 6)
    frac = mean[band].sum() / mean[1:].sum()
    ratio = mean[band].sum() / mean6[band].sum()
    art = mean[band].sum() / mean_rep[band].sum()
    print(f"\n=== {tag}: {len(starts)} windows x {len(pt)} points ===")
    print(f"share of total variance in the 3-6 h band : {frac:.4%}  (2-6 h: {mean[wide].sum()/mean[1:].sum():.4%})")
    print(f"REA-L / (6h+linear interp) power in 3-6 h : {ratio:.1f}x")
    print(f"00 UTC artefact inflation of the 3-6 h band: {art:.3f}x "
          f"({(art-1)*100:+.1f}% vs the repaired series)")
    for tgt in (24, 12, 8, 6, 4, 3, 2):
        i = np.argmin(np.abs(period - tgt))
        print(f"  period {tgt:2d} h: REA-L {mean[i]:10.4g}   6h-interp {mean6[i]:10.4g}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    m = f > 0
    ax.loglog(period[m], mean[m], label="REA-L-CH1 (hourly)")
    ax.loglog(period[m], mean_rep[m], lw=0.9, ls="--",
              label="REA-L, 00 UTC artefact repaired")
    ax.loglog(period[m], mean6[m], label="REA-L subsampled 6 h, linearly interpolated")
    ax.axvspan(3, 6, color="0.85", zorder=0, label="3-6 h claim band")
    ax.axvspan(2, 3, color="0.93", zorder=0, label="2-3 h (near Nyquist, untrusted)")
    ax.axvline(6, color="0.4", lw=0.8, ls="--")
    ax.set_xlabel("period (h)")
    ax.set_ylabel(f"PSD of {args.param}")
    ax.set_title(f"{args.season} 2024, {len(starts)} x {WINDOW_H} h windows, "
                 f"{len(pt)} {'SMN station' if args.points == 'station' else 'grid'} points")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_dir / f"spectra_pilot_{tag}.png", dpi=150)
    print(f"\nwrote {args.out_dir}/spectra_pilot_{tag}.{{npz,png}}")


if __name__ == "__main__":
    main()
