"""Compute per-init SAL (Wernli et al. 2008) precipitation scores.

For a fixed lead time and precip param, loads the forecast and matching truth
slice for each init time in --reftimes, remaps both onto the common near-square
raster (verification.sal.sal_raster) and computes the SAL triple. Writes one CSV
row per init — dry windows included (S/A/L = NaN) — with a commented metadata
header. Any init missing from forecast or truth is a hard error, never a silent
skip.
"""

import logging
from argparse import ArgumentParser, Namespace
from datetime import datetime
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
    GRID_EXTENT,
    MIN_TRUTH_POINTS,
    compute_sal,
    remap_field,
    remap_indices,
    sal_raster,
)

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

DATETIME_FMT = "%Y%m%d%H%M"


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

    lat2d, lon2d = sal_raster()
    shape = lat2d.shape

    reftimes = sorted(datetime.strptime(s, DATETIME_FMT) for s in args.reftimes)
    truth_lazy = (
        open_truth_zarr(args.truth, [args.param])
        if args.truth.suffix == ".zarr"
        else None
    )

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
        if "step" in fcst.dims:
            fcst = fcst.sel(step=np.timedelta64(args.step, "h"))
        truth_ds = load_truth_data(
            args.truth, reftime, [args.step], [args.param], lazy_ds=truth_lazy
        ).isel(time=0)

        fcst_field, fcst_lat, fcst_lon = _native_1d(fcst[args.param])
        truth_field, truth_lat, truth_lon = _native_1d(truth_ds[args.param])

        if fcst_idx is None:
            # SAL needs a resolved (gridded) truth field; warn if it looks sparse.
            if truth_lat.size < MIN_TRUTH_POINTS:
                LOG.warning(
                    "Truth has only %d points (< %d): SAL expects a gridded analysis "
                    "field, not sparse stations — scores may be meaningless.",
                    truth_lat.size,
                    MIN_TRUTH_POINTS,
                )
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

    # One row per init; fixed metadata in a commented header (pandas skips it via
    # read_csv(comment="#")). S/A/L are NaN for dry windows.
    df = pd.DataFrame(
        rows, columns=["reftime", "S", "A", "L", "fcst_mean", "truth_mean"]
    )
    header = [
        "SAL (Wernli et al. 2008) per-init precipitation scores",
        f"param: {args.param}  accum_h: {accum_h}  step_h: {args.step}  member: {args.member}",
        f"grid_extent: {list(GRID_EXTENT)}  grid_cells: {shape[0]}x{shape[1]}",
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if bool(args.run_root) == bool(args.baseline_root):
        parser.error("Exactly one of --run_root or --baseline_root must be provided.")

    main(args)
