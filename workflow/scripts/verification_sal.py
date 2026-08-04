"""Compute per-init SAL (Structure–Amplitude–Location) precipitation scores.

For a fixed lead time and precip param, iterates over all init times (discovered
under a run directory, or from --reftimes for baselines), loads the forecast and
matching truth slice, remaps both onto a common near-square lat–lon raster (see
verification.sal.build_regular_grid), and computes the Wernli et al. (2008) SAL
triple. One row per init — dry windows included (S/A/L = NaN) — is written to a
CSV with a commented metadata header. Forecasts and truth load via
data_input.load_forecast_data / load_truth_data, which route by source and
de-accumulate transparently (period encoded in the param name, TOT_PREC6 = 6h).
Every configured init must be available across forecast and truth, else a hard
error (never a silent skip).
"""

import logging
from argparse import ArgumentParser, Namespace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from data_input import (
    load_forecast_data,
    load_truth_data,
    open_truth_zarr,
    parse_aggregated_param,
)
from verification.sal import (
    DEFAULT_THR_FACTOR,
    DEFAULT_THR_QUANTILE,
    MIN_TRUTH_POINTS,
    build_regular_grid,
    compute_sal,
    remap_field,
    remap_indices,
)

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

DATETIME_FMT = "%Y%m%d%H%M"

# Default near-isotropic raster: ~1.1 km cells, metrically near-square at
# ~46.5°N, covering the greater-Alpine domain. Overridable via CLI / config.
DEFAULT_GRID_EXTENT = (-1.0, 18.0, 42.0, 50.5)  # lon_min, lon_max, lat_min, lat_max
DEFAULT_GRID_STEP_LAT = 0.01
DEFAULT_GRID_STEP_LON = 0.0145


def iter_init_dirs(run_root: Path) -> list[tuple[datetime, Path]]:
    """Return ``(reftime, grib_dir)`` for every init-time subdir (YYYYMMDDHHMM)
    under *run_root*; GRIB may sit directly in it or in a ``grib/`` subdir."""
    result = []
    for d in sorted(run_root.iterdir()):
        if not d.is_dir():
            continue
        try:
            reftime = datetime.strptime(d.name, DATETIME_FMT)
        except ValueError:
            continue
        grib_dir = d / "grib" if (d / "grib").is_dir() else d
        if not any(grib_dir.glob("*.grib")):
            LOG.debug("No GRIB files in %s, skipping", grib_dir)
            continue
        result.append((reftime, grib_dir))
    return result


def _native_1d(da: xr.DataArray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten a field to 1-D with matching lat/lon arrays, squeezing size-1
    non-spatial dims (ensemble, vertical, reftime, step/time) first."""
    for dim in (
        "z",
        "number",
        "ensemble",
        "eps",
        "forecast_reference_time",
        "step",
        "time",
    ):
        if dim in da.dims and da.sizes[dim] == 1:
            da = da.squeeze(dim, drop=True)
    field = np.asarray(da.values, dtype=float).ravel()
    lat = np.asarray(da["latitude"].values).ravel()
    lon = np.asarray(da["longitude"].values).ravel()
    if not (field.size == lat.size == lon.size):
        raise ValueError(
            f"Field/lat/lon size mismatch after squeeze: field={field.size}, "
            f"lat={lat.size}, lon={lon.size} (dims={da.dims})."
        )
    return field, lat, lon


def main(args: Namespace) -> None:
    LOG.info(
        "SAL verification  param=%s step=%dh  truth=%s  output=%s",
        args.param,
        args.step,
        args.truth,
        args.output,
    )

    # SAL is defined for precipitation only.
    if not args.param.startswith("TOT_PREC"):
        raise ValueError(
            f"SAL is defined for precipitation only; got param '{args.param}'. "
            "Use a TOT_PREC* parameter (e.g. TOT_PREC6, or TOT_PREC for "
            "cumulative-from-start)."
        )
    # The accumulation period is encoded in the param name (e.g. TOT_PREC6 → 6h).
    _, accum_h = parse_aggregated_param(args.param)
    if accum_h is not None:
        if args.step < accum_h:
            raise ValueError(
                f"Lead time {args.step}h < {accum_h}h accumulation of '{args.param}'."
            )
        if args.baseline_root and "INCA" in args.baseline_root.parts and accum_h != 1:
            raise ValueError(
                f"INCA is 1h-accumulation only; '{args.param}' needs {accum_h}h. Use TOT_PREC1."
            )
        LOG.info("Accumulation period: %dh", accum_h)

    grid_extent = tuple(args.grid_extent)
    lats, lons, lat2d, lon2d = build_regular_grid(
        grid_extent, args.grid_step_lat, args.grid_step_lon
    )
    shape = lat2d.shape
    LOG.info(
        "SAL raster: %d x %d cells, extent=%s, step=(%.4f lat, %.4f lon) deg",
        shape[0],
        shape[1],
        grid_extent,
        args.grid_step_lat,
        args.grid_step_lon,
    )

    if args.baseline_root:
        init_items = [
            (rt, None)
            for rt in sorted(datetime.strptime(s, DATETIME_FMT) for s in args.reftimes)
        ]
        LOG.info("Using %d baseline init times from --reftimes", len(init_items))
    else:
        init_items = iter_init_dirs(args.run_root)
        LOG.info("Found %d init time directories", len(init_items))
        if args.reftimes:
            wanted = {datetime.strptime(s, DATETIME_FMT) for s in args.reftimes}
            discovered = {rt for rt, _ in init_items}
            missing = sorted(wanted - discovered)
            if missing:
                raise ValueError(
                    f"{len(missing)} configured init(s) have no GRIB under "
                    f"{args.run_root}: {[m.strftime(DATETIME_FMT) for m in missing]}. "
                    "Blacklist genuinely-absent dates in the experiment config."
                )
            init_items = [(rt, d) for rt, d in init_items if rt in wanted]
            LOG.info("Matched all %d configured init times", len(init_items))

    step_td = timedelta(hours=args.step)

    truth_lazy = (
        open_truth_zarr(args.truth, [args.param])
        if args.truth.suffix == ".zarr"
        else None
    )

    # Fail fast if any required valid time is absent from the truth dataset.
    if truth_lazy is not None:
        truth_times = set(truth_lazy.time.values.astype("datetime64[ns]"))
        required_valid_times = {
            np.datetime64(rt + step_td).astype("datetime64[ns]") for rt, _ in init_items
        }
        missing_truth = sorted(required_valid_times - truth_times)
        if missing_truth:
            raise ValueError(
                f"Truth is missing {len(missing_truth)} required valid time(s) for "
                f"param={args.param}, step={args.step}h (e.g. "
                f"{[str(t) for t in missing_truth[:5]]}). Blacklist absent dates."
            )

    # Remap indices depend only on the static source grids: build once, reuse.
    fcst_idx: np.ndarray | None = None
    truth_idx: np.ndarray | None = None

    rows: list[dict] = []

    for reftime, grib_dir in init_items:
        valid_time = np.datetime64(reftime + step_td).astype("datetime64[ns]")
        LOG.info(
            "Processing reftime=%s valid=%s", reftime.strftime(DATETIME_FMT), valid_time
        )

        # --- load forecast ---
        src_root = args.baseline_root if args.baseline_root else grib_dir
        try:
            fcst = load_forecast_data(
                src_root, reftime, [args.step], [args.param], member=args.member
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not load forecast for initialisation "
                f"{reftime.strftime(DATETIME_FMT)} (lead time {args.step}h) from "
                f"{src_root}: {exc}. All configured initialisations must be "
                "available so that run and baseline SAL scores are computed over "
                "an identical sample; blacklist genuinely-absent dates in the "
                "experiment config."
            ) from exc

        if "step" in fcst.dims:
            fcst = fcst.sel(step=np.timedelta64(args.step, "h"))

        # --- load truth slice ---
        truth_ds = load_truth_data(
            args.truth, reftime, [args.step], [args.param], lazy_ds=truth_lazy
        )
        truth_ds = truth_ds.isel(time=0)

        fcst_field, fcst_lat, fcst_lon = _native_1d(fcst[args.param])
        truth_field, truth_lat, truth_lon = _native_1d(truth_ds[args.param])

        # --- build remap indices once (static source grids) ---
        if fcst_idx is None:
            # SAL needs a resolved (gridded) truth field; warn if it looks sparse.
            if truth_lat.size < MIN_TRUTH_POINTS:
                LOG.warning(
                    "Truth has only %d points (< %d); SAL expects a gridded analysis "
                    "field, not sparse stations — scores may be meaningless.",
                    truth_lat.size,
                    MIN_TRUTH_POINTS,
                )
            LOG.info("Building remap indices for %d forecast points", fcst_lat.size)
            fcst_idx = remap_indices(fcst_lat, fcst_lon, lat2d, lon2d)
            if (
                truth_lat.shape == fcst_lat.shape
                and np.max(np.abs(truth_lat - fcst_lat)) < 1e-6
                and np.max(np.abs(truth_lon - fcst_lon)) < 1e-6
            ):
                LOG.info("Truth shares the forecast grid; reusing indices")
                truth_idx = fcst_idx
            else:
                LOG.info("Building remap indices for %d truth points", truth_lat.size)
                truth_idx = remap_indices(truth_lat, truth_lon, lat2d, lon2d)

        fcst_2d = remap_field(fcst_field, fcst_idx, shape)
        truth_2d = remap_field(truth_field, truth_idx, shape)

        s, a, ell = compute_sal(
            fcst_2d,
            truth_2d,
            thr_factor=args.thr_factor,
            thr_quantile=args.thr_quantile,
        )
        fcst_mean = float(fcst_2d.mean())
        truth_mean = float(truth_2d.mean())
        LOG.info(
            "reftime=%s: S=%s A=%s L=%s (fcst_mean=%.3f truth_mean=%.3f)",
            reftime.strftime(DATETIME_FMT),
            f"{s:+.3f}" if np.isfinite(s) else "nan (dry)",
            f"{a:+.3f}" if np.isfinite(a) else "nan",
            f"{ell:.3f}" if np.isfinite(ell) else "nan",
            fcst_mean,
            truth_mean,
        )

        rows.append(
            {
                "reftime": reftime.strftime(DATETIME_FMT),
                "S": s,
                "A": a,
                "L": ell,
                "fcst_mean": fcst_mean,
                "truth_mean": truth_mean,
            }
        )

    LOG.info("Finished: %d init times processed", len(rows))
    if not rows:
        raise ValueError("No initialisations processed — nothing to write.")

    df = pd.DataFrame(
        rows,
        columns=[
            "reftime",
            "S",
            "A",
            "L",
            "fcst_mean",
            "truth_mean",
        ],
    )

    # One row per init; fixed metadata in a commented header (pandas skips it via
    # read_csv(comment="#")). S/A/L are NaN for dry windows.
    source = str(args.baseline_root if args.baseline_root else args.run_root)
    header = [
        "SAL (Wernli et al. 2008) per-init precipitation scores",
        f"param: {args.param}  accum_h: {accum_h if accum_h is not None else 'n/a'}  step_h: {args.step}",
        f"thr_factor: {args.thr_factor}  thr_quantile: {args.thr_quantile}",
        f"grid_extent: {list(grid_extent)}  grid_step: ({args.grid_step_lat}, {args.grid_step_lon})",
        f"member: {args.member}  source: {source}  n_processed: {len(df)}",
        "reftime UTC YYYYMMDDHHMM; S/A/L are NaN for dry windows.",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        for line in header:
            fh.write(f"# {line}\n")
        df.to_csv(fh, index=False)
    LOG.info("Saved %d rows to %s", len(df), args.output)


if __name__ == "__main__":
    parser = ArgumentParser(
        description=(
            "Compute per-init SAL precipitation scores. Supports model runs "
            "(GRIB) and baselines (ICON GRIB archive or INCA NetCDF archive). "
            "Exactly one of --run_root or --baseline_root must be provided."
        )
    )
    parser.add_argument(
        "--run_root",
        type=Path,
        default=None,
        help="Root directory of a model run (e.g. output/data/runs/<run_id>).",
    )
    parser.add_argument(
        "--baseline_root",
        type=Path,
        default=None,
        help=(
            "Root directory of a baseline archive (ICON-CH1/CH2-EPS operational "
            "GRIB archive, or an INCA NetCDF archive). Requires --reftimes."
        ),
    )
    parser.add_argument(
        "--member",
        type=str,
        default="000",
        help=(
            "Ensemble member for ICON baselines: '000' for control, 'median', "
            "'mean', or a 3-digit ID. Ignored for runs and INCA."
        ),
    )
    parser.add_argument(
        "--truth",
        type=Path,
        required=True,
        help="Path to the reference zarr dataset (or a jretrieve spec).",
    )
    parser.add_argument(
        "--step",
        type=int,
        required=True,
        help="Forecast lead time in hours (e.g. 12).",
    )
    parser.add_argument(
        "--param",
        type=str,
        required=True,
        help="Accumulated precip param, period encoded in the name (e.g. TOT_PREC6).",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Unused; kept for parity with existing Snakemake verification rules.",
    )
    parser.add_argument(
        "--reftimes",
        nargs="+",
        default=None,
        help=(
            "Init times (YYYYMMDDHHMM). For runs: optional restriction of the "
            "discovered init-time directories. For baselines: required."
        ),
    )
    parser.add_argument(
        "--thr-factor",
        dest="thr_factor",
        type=float,
        default=DEFAULT_THR_FACTOR,
        help=f"SAL object-detection threshold factor (default {DEFAULT_THR_FACTOR}).",
    )
    parser.add_argument(
        "--thr-quantile",
        dest="thr_quantile",
        type=float,
        default=DEFAULT_THR_QUANTILE,
        help=f"SAL detection wet quantile (default {DEFAULT_THR_QUANTILE}).",
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
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV file (one row per init, with a commented metadata header).",
    )
    args = parser.parse_args()

    if bool(args.run_root) == bool(args.baseline_root):
        parser.error("Exactly one of --run_root or --baseline_root must be provided.")
    if args.baseline_root and not args.reftimes:
        parser.error(
            "--reftimes is required with --baseline_root: init times cannot be "
            "discovered from the operational archive."
        )

    main(args)
