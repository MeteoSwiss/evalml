"""Compute per-init SAL (Wernli et al. 2008) precipitation scores.

For a fixed lead time and precip param, loads the forecast and matching truth
slice for each init time in --reftimes, remaps both onto the common near-square
raster (sal_raster) and computes the SAL triple. Writes one CSV row per init —
dry windows included (S/A/L = NaN) — with a commented metadata header. Any init
missing from forecast or truth is a hard error, never a silent skip.

SAL compares a forecast precipitation field to a reference over a fixed domain,
returning three signed, dimensionless components: S (structure), A (amplitude)
and L (location). Object detection and the components are delegated to
pysteps.verification.salscores.sal; this script adds the scoring raster, the
nearest-neighbour remap onto it, and a dry-window gate.
"""

import logging
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pysteps.verification.salscores import sal as _pysteps_sal

from data_input import (
    load_forecast_data,
    load_truth_data,
    open_truth_zarr,
    parse_aggregated_param,
)
from verification.spatial import spherical_nearest_neighbor_indices

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

DATETIME_FMT = "%Y%m%d%H%M"

# pysteps defaults (Wernli et al. 2008, eq. 1): detection threshold is
# thr_factor * the thr_quantile-percentile of the wet precipitation.
DEFAULT_THR_FACTOR = 0.067
DEFAULT_THR_QUANTILE = 0.95

# Defaults for the scoring raster, overridable per experiment via the config
# (SalConfig.grid_extent / grid_step_lat / grid_step_lon, which carry the same
# values). The extent is the bounding box of the ICON-CH1 analysis domain, the
# SAL truth: KENDA-CH1 spans lon -0.8171..17.7106, lat 42.0279..50.5005, so this
# rounds outwards by <=0.29 deg. Scoring much beyond the truth's support is not
# free -- the remap has no distance cutoff, so cells outside it repeat the
# nearest border value, and a larger raster also deflates pysteps' L term via
# its longer diagonal.
DEFAULT_GRID_EXTENT = (-1.0, 18.0, 42.0, 50.5)  # lon_min, lon_max, lat_min, lat_max
DEFAULT_GRID_STEP_LAT = 0.01
DEFAULT_GRID_STEP_LON = 0.0145


def sal_raster(
    extent: tuple[float, float, float, float] = DEFAULT_GRID_EXTENT,
    step_lat: float = DEFAULT_GRID_STEP_LAT,
    step_lon: float = DEFAULT_GRID_STEP_LON,
) -> tuple[np.ndarray, np.ndarray]:
    """The SAL scoring raster over *extent* as ``(lat2d, lon2d)`` meshgrids,
    latitude varying along axis 0. Cells stay inside *extent*: an upper bound is
    included when the span is a whole multiple of the spacing, else the last cell
    falls short of it by up to one step. Pick *step_lat* and *step_lon* so cells
    are metrically near-square at the extent's central latitude (pysteps' L term
    assumes square pixels); the defaults give ~1.1 km cells, exactly square at
    46.4°N (0.0145/0.01 = 1/cos(46.4°))."""
    lon_min, lon_max, lat_min, lat_max = extent
    # The epsilon admits an exact upper bound despite float drift (~1e-11 over
    # ~1e3 steps) while staying ~6 orders below one step, so it cannot overshoot.
    lons = np.arange(lon_min, lon_max + step_lon * 1e-6, step_lon)
    lats = np.arange(lat_min, lat_max + step_lat * 1e-6, step_lat)
    lon2d, lat2d = np.meshgrid(lons, lats)
    return lat2d, lon2d


def remap_indices(
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    tgt_lat2d: np.ndarray,
    tgt_lon2d: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbour flat indices (length ``tgt_lat2d.size``) into the
    flattened source points; reusable across time steps sharing the source grid."""
    return spherical_nearest_neighbor_indices(
        np.asarray(src_lat).ravel(),
        np.asarray(src_lon).ravel(),
        np.asarray(tgt_lat2d).ravel(),
        np.asarray(tgt_lon2d).ravel(),
    )


def remap_field(
    field: np.ndarray,
    indices: np.ndarray,
    shape: tuple[int, int],
    fill: float = 0.0,
) -> np.ndarray:
    """Remap a native field onto the target raster via precomputed *indices*;
    NaNs (e.g. off-domain cells) become *fill* (0 = no precipitation)."""
    flat = np.asarray(field, dtype=float).ravel()
    out = flat[indices].reshape(shape)
    return np.nan_to_num(out, nan=fill)


def compute_sal(
    prediction: np.ndarray,
    observation: np.ndarray,
    thr_factor: float = DEFAULT_THR_FACTOR,
    thr_quantile: float = DEFAULT_THR_QUANTILE,
) -> tuple[float, float, float]:
    """Compute the SAL triple ``(S, A, L)`` for two co-located 2-D fields; NaNs
    count as dry. A window where either field is everywhere dry (max ≤ 0) has no
    detectable objects, so ``(nan, nan, nan)`` is returned rather than raising."""
    pred = np.nan_to_num(np.asarray(prediction, dtype=float))
    obs = np.nan_to_num(np.asarray(observation, dtype=float))
    if not (pred.size and obs.size and pred.max() > 0 and obs.max() > 0):
        return (np.nan, np.nan, np.nan)
    s, a, ell = _pysteps_sal(
        pred, obs, thr_factor=thr_factor, thr_quantile=thr_quantile
    )
    return (float(s), float(a), float(ell))


def _native_1d(da: xr.DataArray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten a field and its 2-D lat/lon coords to matching 1-D arrays,
    dropping size-1 dims (ensemble, vertical, reftime, step/time) first."""
    da = da.squeeze(drop=True)
    field, lat, lon = (
        np.asarray(a, dtype=float).ravel()
        for a in (da.values, da["latitude"].values, da["longitude"].values)
    )
    if not (field.size == lat.size == lon.size):
        raise ValueError(
            f"Field/lat/lon size mismatch after squeeze: {field.size}/{lat.size}/"
            f"{lon.size} (dims={da.dims})."
        )
    return field, lat, lon


def main(args: Namespace) -> None:
    # The accumulation period is encoded in the param name (e.g. TOT_PREC6 -> 6h).
    _, accum_h = parse_aggregated_param(args.param)
    if accum_h is not None and args.step < accum_h:
        raise ValueError(
            f"Lead time {args.step}h < {accum_h}h accumulation of '{args.param}'."
        )
    # SAL needs a resolved (gridded) truth field: reject station observations
    # (jretrievedwh:...) and any other unsupported root before reading data.
    if args.truth.suffix != ".zarr":
        raise ValueError(
            f"SAL needs a gridded analysis zarr as truth, got '{args.truth}'; "
            "station observations (jretrievedwh:...) are not supported."
        )

    lat2d, lon2d = sal_raster(
        tuple(args.grid_extent), args.grid_step_lat, args.grid_step_lon
    )
    shape = lat2d.shape
    LOG.info(
        "SAL raster: %dx%d cells, extent=%s, step=(%g lat, %g lon) deg",
        *shape,
        args.grid_extent,
        args.grid_step_lat,
        args.grid_step_lon,
    )

    reftimes = sorted(datetime.strptime(s, DATETIME_FMT) for s in args.reftimes)
    truth_lazy = open_truth_zarr(args.truth, [args.param])

    # Remap indices depend only on the static source grids: build once, reuse.
    fcst_idx = truth_idx = None
    rows = []

    for reftime in reftimes:
        if args.baseline_root:
            # Baselines read from the operational archive.
            src_root = args.baseline_root
        else:
            # A run stores GRIB under run_root/<reftime>/[grib/] (a missing init
            # then fails in load_forecast_data).
            src_root = args.run_root / reftime.strftime(DATETIME_FMT)
            src_root = src_root / "grib" if (src_root / "grib").is_dir() else src_root

        fcst = load_forecast_data(
            src_root, reftime, [args.step], [args.param], member=args.member
        )
        fcst = fcst.sel(step=np.timedelta64(args.step, "h"))
        truth_ds = load_truth_data(
            args.truth, reftime, [args.step], [args.param], lazy_ds=truth_lazy
        ).isel(time=0)

        fcst_field, fcst_lat, fcst_lon = _native_1d(fcst[args.param])
        truth_field, truth_lat, truth_lon = _native_1d(truth_ds[args.param])

        if fcst_idx is None:
            fcst_idx = remap_indices(fcst_lat, fcst_lon, lat2d, lon2d)
            shared = (
                fcst_lat.shape == truth_lat.shape
                and np.allclose(fcst_lat, truth_lat, atol=1e-6)
                and np.allclose(fcst_lon, truth_lon, atol=1e-6)
            )
            truth_idx = (
                fcst_idx
                if shared
                else remap_indices(truth_lat, truth_lon, lat2d, lon2d)
            )

        fcst_2d = remap_field(fcst_field, fcst_idx, shape)
        truth_2d = remap_field(truth_field, truth_idx, shape)
        s, a, ell = compute_sal(fcst_2d, truth_2d)
        fcst_mean, truth_mean = float(fcst_2d.mean()), float(truth_2d.mean())
        LOG.info(
            "%s  S=%+.3f A=%+.3f L=%.3f  fcst_mean=%.3f truth_mean=%.3f",
            reftime.strftime(DATETIME_FMT),
            s,
            a,
            ell,
            fcst_mean,
            truth_mean,
        )
        rows.append((reftime.strftime(DATETIME_FMT), s, a, ell, fcst_mean, truth_mean))

    # One row per init; fixed metadata in a commented header.
    # S/A/L are NaN for dry windows.
    df = pd.DataFrame(
        rows, columns=["reftime", "S", "A", "L", "fcst_mean", "truth_mean"]
    )
    header = [
        "SAL (Wernli et al. 2008) per-init precipitation scores",
        f"param: {args.param}  accum_h: {accum_h}  step_h: {args.step}  member: {args.member}",
        f"grid_extent: {list(args.grid_extent)}  grid_step: "
        f"({args.grid_step_lat}, {args.grid_step_lon})  grid_cells: {shape[0]}x{shape[1]}",
        f"source: {args.baseline_root or args.run_root}  n_init: {len(df)}",
        "reftime UTC YYYYMMDDHHMM; S/A/L are NaN for dry windows.",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        fh.writelines(f"# {line}\n" for line in header)
        df.to_csv(fh, index=False)
    LOG.info("Saved %d rows to %s", len(df), args.output)


if __name__ == "__main__":
    parser = ArgumentParser(
        description=(
            "Compute per-init SAL precipitation scores for a model run (GRIB) or "
            "a baseline (ICON GRIB / INCA NetCDF archive). Exactly one of "
            "--run_root or --baseline_root must be provided."
        )
    )
    parser.add_argument("--run_root", type=Path, help="output/data/runs/<run_id>.")
    parser.add_argument("--baseline_root", type=Path, help="Baseline archive root.")
    parser.add_argument(
        "--member", default="000", help="ICON baseline: '000', 'median', 'mean', NNN."
    )
    parser.add_argument("--truth", type=Path, required=True, help="Reference zarr.")
    parser.add_argument("--step", type=int, required=True, help="Lead time in hours.")
    parser.add_argument("--param", required=True, help="Precip param, e.g. TOT_PREC6.")
    parser.add_argument(
        "--reftimes", nargs="+", required=True, help="Init times (YYYYMMDDHHMM)."
    )
    parser.add_argument(
        "--grid-extent",
        dest="grid_extent",
        type=float,
        nargs=4,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        default=list(DEFAULT_GRID_EXTENT),
        help="SAL raster extent in degrees (PlateCarree).",
    )
    parser.add_argument(
        "--grid-step-lat",
        dest="grid_step_lat",
        type=float,
        default=DEFAULT_GRID_STEP_LAT,
        help=f"SAL raster latitude spacing in degrees (default {DEFAULT_GRID_STEP_LAT}).",
    )
    parser.add_argument(
        "--grid-step-lon",
        dest="grid_step_lon",
        type=float,
        default=DEFAULT_GRID_STEP_LON,
        help=f"SAL raster longitude spacing in degrees (default {DEFAULT_GRID_STEP_LON}).",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if bool(args.run_root) == bool(args.baseline_root):
        parser.error("Exactly one of --run_root or --baseline_root must be provided.")

    main(args)
