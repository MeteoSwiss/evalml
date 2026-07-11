"""Compute spatial maps of temporally-aggregated forecast errors.

For a fixed lead time and variable, iterates over all initialisation times
(discovered under a run directory, or taken from --reftimes for baselines),
loads the corresponding forecast field and the matching truth slice from a
reference zarr, maps the forecast onto the truth grid, and accumulates running
error statistics without ever holding the full time series in memory.  The
final BIAS / RMSE / MAE / STDE maps are written to a NetCDF file.

Forecasts load through data_input.load_forecast_data, which routes by source:
ML run directories (GRIB files), INCA (NetCDF archive), or otherwise the ICON
operational GRIB archive. Baselines (--baseline_root) use the latter two paths;
init times are not discovered from the archive but taken from --reftimes. Every
configured initialisation must be available across forecast and truth — a missing
one is a hard error, never a silent skip — so that run and baseline maps are
always computed over an identical sample.

Design note: one Snakemake job per (run, param, lead time), each loading only the
step(s) it needs. Derived and aggregated params (e.g. SP_10M, TOT_PREC6) are
handled transparently by load_forecast_data and load_truth_data; the accumulation
period is encoded in the param name (TOT_PREC6 = 6h) rather than inferred from
step spacing.

Usage
-----
    uv run workflow/scripts/verification_scoremaps.py \\
        output/data/runs/<run_id> \\
        --truth /path/to/truth.zarr \\
        --step 24 \\
        --param T_2M
"""

import logging
from argparse import ArgumentParser, Namespace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from data_input import (
    load_forecast_data,
    load_truth_data,
    open_truth_zarr,
    parse_aggregated_param,
)
from verification.spatial import map_forecast_to_truth

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

DATETIME_FMT = "%Y%m%d%H%M"

SEASONS = ["DJF", "MAM", "JJA", "SON", "all"]
# Init hour buckets. -999 is the "all" sentinel (matches verification_aggregation.py).
INIT_HOURS = [0, 6, 12, 18, -999]


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


# ---------------------------------------------------------------------------
# Init-time discovery
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: Namespace) -> None:
    LOG.info("=" * 60)
    LOG.info("Spatial verification  param=%s  step=%dh", args.param, args.step)
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
                "INCA score maps."
            )
        LOG.info("Accumulation period: %dh", accum_h)

    if args.baseline_root:
        # The operational archive is too large to enumerate up front, so the
        # experiment's configured init times define the work list. Every one of
        # them must be available: a baseline init missing from the archive is a
        # hard error at load time (in the loop below), not a silent skip, so the
        # baseline map covers the same sample as the run maps.
        init_items = [
            (rt, None)
            for rt in sorted(datetime.strptime(s, DATETIME_FMT) for s in args.reftimes)
        ]
        LOG.info("Using %d baseline init times from --reftimes", len(init_items))
    else:
        init_items = iter_init_dirs(args.run_root)
        LOG.info("Found %d init time directories", len(init_items))

        # Restrict to the experiment's configured init times, and require that
        # every configured init was actually discovered: a missing run output
        # directory must fail rather than silently shrink the sample.
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
                    "score maps are computed over an identical sample; blacklist "
                    "genuinely-absent dates in the experiment config."
                )
            init_items = [(rt, d) for rt, d in init_items if rt in wanted]
            LOG.info("Matched all %d configured init times", len(init_items))

    step_td = timedelta(hours=args.step)

    # Running accumulators keyed by (season, init_hour) – initialised on the
    # first successfully processed sample so that we can infer the spatial
    # shape from the data. Each entry is a numpy array over the spatial
    # dimension(s).
    bucket_keys = [(s, h) for s in SEASONS for h in INIT_HOURS]
    accum_n: dict[tuple[str, int], np.ndarray | None] = {k: None for k in bucket_keys}
    accum_sum_e: dict[tuple[str, int], np.ndarray | None] = {
        k: None for k in bucket_keys
    }
    accum_sum_se: dict[tuple[str, int], np.ndarray | None] = {
        k: None for k in bucket_keys
    }
    accum_sum_ae: dict[tuple[str, int], np.ndarray | None] = {
        k: None for k in bucket_keys
    }
    ref_truth_slice: xr.DataArray | None = None  # kept for output coordinates

    truth_lazy = (
        open_truth_zarr(args.truth, [args.param])
        if args.truth.suffix == ".zarr"
        else None
    )

    # Fail fast if any required valid time is absent from the truth dataset.
    # Checking up front means the full set of missing times is reported at once
    # rather than one error per reftime mid-loop.
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
                "initialisations must be available so that run and baseline score "
                "maps are computed over an identical sample; blacklist genuinely-"
                "absent dates in the experiment config."
            )

    n_ok = 0

    for reftime, grib_dir in init_items:
        valid_time = np.datetime64(reftime + step_td).astype("datetime64[ns]")

        LOG.info(
            "Processing reftime=%s  valid=%s",
            reftime.strftime(DATETIME_FMT),
            valid_time,
        )

        first_iter = n_ok == 0

        # --- load forecast ---
        # Derivation (SP_10M) and accumulation (TOT_PREC6) are handled inside
        # load_forecast_data; pass the requested param directly.
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
                "available so that run and baseline score maps are computed over "
                "an identical sample; blacklist genuinely-absent dates in the "
                "experiment config."
            ) from exc

        if "step" in fcst.dims:
            fcst = fcst.sel(step=np.timedelta64(args.step, "h"))

        if first_iter:
            LOG.info("fcst (after step selection): %s", fcst)
            fcst_raw = fcst[args.param].values if args.param in fcst else None
            if fcst_raw is not None:
                n_nan_fcst = int(np.isnan(fcst_raw).sum())
                LOG.info(
                    "fcst[%s]: shape=%s, min=%.4g, max=%.4g, n_nan=%d",
                    args.param,
                    fcst_raw.shape,
                    float(np.nanmin(fcst_raw))
                    if n_nan_fcst < fcst_raw.size
                    else float("nan"),
                    float(np.nanmax(fcst_raw))
                    if n_nan_fcst < fcst_raw.size
                    else float("nan"),
                    n_nan_fcst,
                )

        # --- load truth slice ---
        truth_ds = load_truth_data(
            args.truth, reftime, [args.step], [args.param], lazy_ds=truth_lazy
        )
        truth_ds = truth_ds.isel(time=0)
        truth_slice = truth_ds[args.param]

        if first_iter:
            truth_raw = truth_slice.values
            n_nan_truth = int(np.isnan(truth_raw).sum())
            LOG.info(
                "truth_slice[%s]: shape=%s, min=%.4g, max=%.4g, n_nan=%d",
                args.param,
                truth_raw.shape,
                float(np.nanmin(truth_raw))
                if n_nan_truth < truth_raw.size
                else float("nan"),
                float(np.nanmax(truth_raw))
                if n_nan_truth < truth_raw.size
                else float("nan"),
                n_nan_truth,
            )

        # --- map forecast onto truth grid ---
        try:
            fcst_mapped = map_forecast_to_truth(fcst, truth_ds)
        except Exception as exc:
            raise RuntimeError(
                f"Spatial mapping failed for initialisation "
                f"{reftime.strftime(DATETIME_FMT)} (lead time {args.step}h): {exc}."
            ) from exc

        fcst_param = fcst_mapped[args.param]
        # Squeeze size-1 non-spatial dims so the error array is purely spatial.
        # The earthkit loader keeps `number` (ensemble), `z` (vertical) and
        # `forecast_reference_time` as size-1 dims for a deterministic surface run.
        for dim in ["eps", "ensemble", "number", "z", "forecast_reference_time"]:
            if dim in fcst_param.dims and fcst_param.sizes[dim] == 1:
                fcst_param = fcst_param.squeeze(dim, drop=True)
        fcst_vals = fcst_param.values
        truth_vals = truth_slice.values
        error = fcst_vals - truth_vals  # shape: spatial dims of truth

        if first_iter:
            n_nan_mapped = int(np.isnan(fcst_vals).sum())
            LOG.info(
                "fcst_mapped[%s]: shape=%s, min=%.4g, max=%.4g, n_nan=%d",
                args.param,
                fcst_vals.shape,
                float(np.nanmin(fcst_vals))
                if n_nan_mapped < fcst_vals.size
                else float("nan"),
                float(np.nanmax(fcst_vals))
                if n_nan_mapped < fcst_vals.size
                else float("nan"),
                n_nan_mapped,
            )
            n_nan_err = int(np.isnan(error).sum())
            LOG.info(
                "error: shape=%s, min=%.4g, max=%.4g, n_nan=%d / %d",
                error.shape,
                float(np.nanmin(error)) if n_nan_err < error.size else float("nan"),
                float(np.nanmax(error)) if n_nan_err < error.size else float("nan"),
                n_nan_err,
                error.size,
            )

        n_nan_error = int(np.isnan(error).sum())
        if n_nan_error == error.size:
            LOG.warning(
                "reftime=%s: error is all-NaN (%d points) — nothing accumulated.",
                reftime.strftime(DATETIME_FMT),
                error.size,
            )

        # --- initialise accumulators on first valid sample ---
        if accum_n[("all", -999)] is None:
            for k in bucket_keys:
                accum_n[k] = np.zeros(error.shape, dtype=np.int64)
                accum_sum_e[k] = np.zeros(error.shape, dtype=np.float64)
                accum_sum_se[k] = np.zeros(error.shape, dtype=np.float64)
                accum_sum_ae[k] = np.zeros(error.shape, dtype=np.float64)
            ref_truth_slice = truth_slice

        # --- accumulate into matching (season, init_hour) buckets, plus the
        # "all" rows/cols on each axis (NaN-safe) ---
        season = _season_of(reftime)
        ih = reftime.hour
        valid = ~np.isnan(error)
        for s in (season, "all"):
            for h in (ih, -999):
                accum_n[(s, h)][valid] += 1
                accum_sum_e[(s, h)][valid] += error[valid]
                accum_sum_se[(s, h)][valid] += error[valid] ** 2
                accum_sum_ae[(s, h)][valid] += np.abs(error[valid])
        n_ok += 1

    LOG.info("Finished: %d init times processed", n_ok)

    if n_ok == 0:
        raise ValueError(
            "No initialisations were processed — nothing to write. Check that "
            "--reftimes is non-empty."
        )

    # --- compute aggregate maps per (season, init_hour), then stack ---
    spatial_coords = {
        c: ref_truth_slice[c]
        for c in ref_truth_slice.coords
        if set(ref_truth_slice[c].dims).issubset(set(ref_truth_slice.dims))
        and c != "time"
    }
    spatial_dims = list(ref_truth_slice.dims)
    out_dims = ["season", "init_hour"] + spatial_dims
    out_coords = {"season": SEASONS, "init_hour": INIT_HOURS, **spatial_coords}

    def _strat_da(compute_fn) -> xr.DataArray:
        """Stack per-(season, init_hour) arrays into a (season, init_hour, *spatial) DataArray."""
        out_shape = (len(SEASONS), len(INIT_HOURS)) + ref_truth_slice.shape
        arr = np.empty(out_shape, dtype=np.float32)
        for i, s in enumerate(SEASONS):
            for j, h in enumerate(INIT_HOURS):
                n = accum_n[(s, h)]
                with np.errstate(invalid="ignore", divide="ignore"):
                    arr[i, j] = compute_fn(n, s, h).astype(np.float32)
        return xr.DataArray(arr, dims=out_dims, coords=out_coords)

    out = xr.Dataset(
        {
            f"{args.param}.BIAS": _strat_da(
                lambda n, s, h: np.where(n > 0, accum_sum_e[(s, h)] / n, np.nan)
            ),
            f"{args.param}.RMSE": _strat_da(
                lambda n, s, h: np.where(
                    n > 0, np.sqrt(accum_sum_se[(s, h)] / n), np.nan
                )
            ),
            f"{args.param}.MAE": _strat_da(
                lambda n, s, h: np.where(n > 0, accum_sum_ae[(s, h)] / n, np.nan)
            ),
            f"{args.param}.STDE": _strat_da(
                lambda n, s, h: np.where(
                    n > 0,
                    np.sqrt(
                        np.maximum(
                            accum_sum_se[(s, h)] / n - (accum_sum_e[(s, h)] / n) ** 2,
                            0.0,
                        )
                    ),
                    np.nan,
                )
            ),
            f"{args.param}.N": _strat_da(lambda n, s, h: np.where(n > 0, n, np.nan)),
        },
        attrs={
            "param": args.param,
            "step_h": args.step,
            # Accumulation period of the verified quantity (accumulated params
            # only) — lets consumers tell a 1h INCA map from a 6h ICON map.
            "accum_h": accum_h if accum_h is not None else "n/a",
            "member": args.member,
            "source": str(args.baseline_root if args.baseline_root else args.run_root),
            "n_processed": n_ok,
        },
    )

    LOG.info("Output dataset:\n%s", out)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(args.output)
    LOG.info("Saved to %s", args.output)


if __name__ == "__main__":
    parser = ArgumentParser(
        description=(
            "Compute spatial maps of temporally-aggregated forecast errors. "
            "Supports model runs (GRIB) and baselines (ICON GRIB archive or "
            "INCA NetCDF archive). "
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
            "Root directory of a baseline archive (e.g. the ICON-CH1/CH2-EPS "
            "operational GRIB archive, or an INCA NetCDF archive). Requires "
            "--reftimes."
        ),
    )
    parser.add_argument(
        "--member",
        type=str,
        default="000",
        help=(
            "Ensemble member to load for ICON baselines: '000' for control, "
            "'median' for the pre-computed median, 'mean' to average all "
            "members, or any 3-digit member ID. Ignored for runs and INCA."
        ),
    )
    parser.add_argument(
        "--truth",
        type=Path,
        required=True,
        help="Path to the reference zarr dataset.",
    )
    parser.add_argument(
        "--step",
        type=int,
        required=True,
        help="Forecast lead time in hours (e.g. 24).",
    )
    parser.add_argument(
        "--param",
        type=str,
        required=True,
        help="Variable to verify (e.g. T_2M, TD_2M, U_10M).",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Unused; kept for backwards compatibility with existing Snakemake rules.",
    )
    parser.add_argument(
        "--reftimes",
        nargs="+",
        default=None,
        help=(
            "List of init times (YYYYMMDDHHMM). For runs: optional restriction of "
            "the discovered init-time directories. For baselines: required; "
            "defines the init times to load from the archive."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output NetCDF file.",
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
