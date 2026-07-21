"""Compute per-init SAL (Structure–Amplitude–Location) precipitation scores.

For a fixed lead time and (accumulated) precipitation parameter, iterates over
all initialisation times (discovered under a run directory, or taken from
--reftimes for baselines), loads the forecast field and the matching truth
slice, remaps *both* onto a common near-isotropic regular lat–lon raster, and
computes the Wernli et al. (2008) SAL triple for that window. One row per
initialisation — including dry windows (S/A/L = NaN, means retained so a
downstream wet-case filter can drop them) — is written to a CSV file (SAL is a
per-case scalar score with no spatial dimension, so a table is the natural
container; fixed metadata is carried in a commented header).

Unlike the score-map path (verification_scoremaps.py), SAL is a domain-integrated
per-case score, not a per-cell field, so results are kept per initialisation
rather than reduced to running per-cell means. And because the Location term
needs square pixels, both fields are put on a purpose-built regular raster (see
verification.sal.build_regular_grid) instead of the native truth grid.

Forecasts and truth load through data_input.load_forecast_data /
load_truth_data, which route by source and handle de-accumulation transparently:
the accumulation period is encoded in the param name (TOT_PREC6 = 6h). Every
configured initialisation must be available across forecast and truth — a missing
one is a hard error, never a silent skip — so that run and baseline scores are
computed over an identical sample.

Usage
-----
    uv run workflow/scripts/verification_sal.py \\
        output/data/runs/<run_id> \\
        --truth /path/to/truth.zarr \\
        --step 12 \\
        --param TOT_PREC6
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

SEASONS = ["DJF", "MAM", "JJA", "SON"]

# Default near-isotropic raster: ~1.1 km cells, metrically near-square at
# ~46.5°N, covering the greater-Alpine domain. Overridable via CLI / config.
DEFAULT_GRID_EXTENT = (-1.0, 18.0, 42.0, 50.5)  # lon_min, lon_max, lat_min, lat_max
DEFAULT_GRID_STEP_LAT = 0.01
DEFAULT_GRID_STEP_LON = 0.0145


def _season_of(dt: datetime) -> str:
    """Return the meteorological season string for a given datetime."""
    month = dt.month
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def iter_init_dirs(run_root: Path) -> list[tuple[datetime, Path]]:
    """Return ``(reftime, grib_dir)`` pairs for every complete init time.

    Expects subdirectories named ``YYYYMMDDHHMI`` directly under *run_root*.
    GRIB files may live either directly in the init-time directory or inside a
    ``grib/`` subdirectory.
    """
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
    """Flatten a field to 1-D, with matching latitude/longitude arrays.

    Squeezes size-1 non-spatial dims (ensemble, vertical, reference time, the
    already-selected step/time) so only the spatial dimension(s) remain, then
    ravels the field and its ``latitude``/``longitude`` coordinates in a
    consistent (C) order so they index the same points.
    """
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
    LOG.info("=" * 60)
    LOG.info("SAL verification  param=%s  step=%dh", args.param, args.step)
    LOG.info("Run root : %s", args.run_root)
    LOG.info("Truth    : %s", args.truth)
    LOG.info("Output   : %s", args.output)
    LOG.info("=" * 60)

    # The accumulation period is encoded in the param name (e.g. TOT_PREC6 → 6h).
    _, accum_h = parse_aggregated_param(args.param)
    if accum_h is not None:
        if args.step < accum_h:
            raise ValueError(
                f"Lead time {args.step}h is smaller than the {accum_h}h "
                f"accumulation period encoded in '{args.param}'."
            )
        if args.baseline_root and "INCA" in args.baseline_root.parts and accum_h != 1:
            raise ValueError(
                f"INCA provides native 1h accumulations only; '{args.param}' "
                f"requires a {accum_h}h accumulation period. Use TOT_PREC1 for "
                "INCA SAL scores."
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
                    f"{len(missing)} configured initialisation(s) have no GRIB "
                    f"output under {args.run_root}: "
                    f"{[m.strftime(DATETIME_FMT) for m in missing]}. All configured "
                    "initialisations must be available so that run and baseline "
                    "SAL scores are computed over an identical sample; blacklist "
                    "genuinely-absent dates in the experiment config."
                )
            init_items = [(rt, d) for rt, d in init_items if rt in wanted]
            LOG.info("Matched all %d configured init times", len(init_items))

    step_td = timedelta(hours=args.step)

    truth_lazy = (
        open_truth_zarr(args.truth, [args.param])
        if args.truth.suffix == ".zarr"
        else None
    )

    # Fail fast if any required valid time is absent from the truth dataset, so
    # the full set of missing times is reported at once rather than mid-loop.
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
                f"{[str(t) for t in missing_truth[:5]]}). All configured "
                "initialisations must be available so that run and baseline SAL "
                "scores are computed over an identical sample; blacklist "
                "genuinely-absent dates in the experiment config."
            )

    # Remap indices depend only on the (static) source grids, so build them once
    # on the first init and reuse. fcst and truth may share a native grid (e.g.
    # Varda and KENDA-CH1 are both on the ICON-CH1 mesh), in which case the truth
    # indices are reused directly.
    fcst_idx: np.ndarray | None = None
    truth_idx: np.ndarray | None = None

    rows: list[dict] = []

    for reftime, grib_dir in init_items:
        valid_time = np.datetime64(reftime + step_td).astype("datetime64[ns]")
        LOG.info(
            "Processing reftime=%s  valid=%s",
            reftime.strftime(DATETIME_FMT),
            valid_time,
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
                "season": _season_of(reftime),
                "init_hour": reftime.hour,
                "S": s,
                "A": a,
                "L": ell,
                "fcst_mean": fcst_mean,
                "truth_mean": truth_mean,
            }
        )

    LOG.info("Finished: %d init times processed", len(rows))
    if not rows:
        raise ValueError(
            "No initialisations were processed — nothing to write. Check that "
            "--reftimes is non-empty."
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "reftime",
            "season",
            "init_hour",
            "S",
            "A",
            "L",
            "fcst_mean",
            "truth_mean",
        ],
    )

    # SAL is a per-case scalar score (no spatial dimension), so the natural
    # container is a table: one row per initialisation. Fixed metadata (param,
    # lead time, thresholds, raster) goes in a commented header that pandas skips
    # with read_csv(comment="#"). S/A/L are NaN for dry windows; fcst_mean /
    # truth_mean are domain-mean precip over the SAL raster in the loader unit
    # (mm for TOT_PREC*) and let a downstream wet-case filter drop dry windows.
    source = str(args.baseline_root if args.baseline_root else args.run_root)
    header = [
        "SAL (Structure-Amplitude-Location, Wernli et al. 2008) per-init "
        "precipitation scores",
        f"param: {args.param}",
        f"accum_h: {accum_h if accum_h is not None else 'n/a'}",
        f"step_h: {args.step}",
        f"thr_factor: {args.thr_factor}",
        f"thr_quantile: {args.thr_quantile}",
        f"grid_extent: {list(grid_extent)}",
        f"grid_step_lat: {args.grid_step_lat}",
        f"grid_step_lon: {args.grid_step_lon}",
        f"member: {args.member}",
        f"source: {source}",
        f"n_processed: {len(df)}",
        "reftime is the initialisation time (UTC, YYYYMMDDHHMM); "
        "S/A/L are NaN for dry windows.",
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
