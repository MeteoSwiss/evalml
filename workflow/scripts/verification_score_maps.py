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
init times are not discovered from the archive but taken from --reftimes, with
unavailable dates skipped at load time.

Design note: one Snakemake job per (run, param, lead time), each loading only the
step(s) it needs. We deliberately do not load all lead times at once: per-job
memory and output disk scale with N_leadtimes x grid size, which is infeasible at
interpolator (1 h) and nowcasting (10 min) resolutions; that cost is independent
of GRIB read speed, so it does not improve as loading gets faster. For TOT_PREC
the loader (data_input._tot_prec_handling) de-accumulates over the requested
[step - period, step] window, so we just select the target step.

Usage
-----
    uv run workflow/scripts/verification_score_maps.py \\
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

from data_input import load_forecast_data, parse_steps
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


# Maps from standard parameter names to zarr variable names.
# COSMO-2e zarrs use short CF names; COSMO-1e zarrs keep the COSMO names.
_PARAMS_MAP_CO2 = {
    "T_2M": "2t",
    "TD_2M": "2d",
    "U_10M": "10u",
    "V_10M": "10v",
    "PS": "sp",
    "PMSL": "msl",
    "TOT_PREC": "tp",
}
# Derived variables and the components they require.
_DERIVED = {
    "SP_10M": ("U_10M", "V_10M"),
}

# Params whose GRIB/zarr values are cumulative-from-start accumulations and must
# be de-accumulated over a [step - period, step] window before verification.
_ACCUMULATED_PARAMS = {"TOT_PREC"}


def _params_map(truth_root: Path, accum_h: int | None = None) -> dict[str, str]:
    """Map canonical parameter names to truth-zarr variable names.

    COSMO-2e zarrs use short CF names. COSMO-1e / ICON zarrs store precip as
    period accumulations named ``TOT_PREC_<N>H``, where N is the accumulation
    length in hours (matching the verification step spacing); pass it via
    ``accum_h``.
    """
    if "co2" in truth_root.name:
        return _PARAMS_MAP_CO2
    suffix = f"TOT_PREC_{accum_h}H" if accum_h else "TOT_PREC_6H"
    return {k: k.replace("TOT_PREC", suffix) for k in _PARAMS_MAP_CO2}


def _compute_derived(ds: xr.Dataset, param: str) -> xr.DataArray:
    """Compute a derived variable from its components already present in *ds*."""
    if param == "SP_10M":
        return (ds["U_10M"] ** 2 + ds["V_10M"] ** 2) ** 0.5
    raise ValueError(f"No recipe for derived variable '{param}'")


# ---------------------------------------------------------------------------
# Truth loading
# ---------------------------------------------------------------------------
# TODO: consolidate with src/data_input/__init__.py as part of the
# refactor/data-io branch. _open_zarr_component below duplicates
# ~80% of load_analysis_data_from_zarr but returns a lazy DataArray
# rather than a time-sliced Dataset, which is what our streaming
# aggregation needs. The right end-state is a shared lazy-open primitive
# in data_input that both consumers use; not introduced here to avoid
# conflicting with the data-io refactor. Until then this opener must
# mirror the loader's conventions (notably the m -> mm precip conversion
# from MRB-820).


def _open_zarr_component(
    root: Path, param: str, accum_h: int | None = None
) -> xr.DataArray:
    """Open a single native zarr variable lazily as a DataArray."""
    zarr_param = _params_map(root, accum_h)[param]

    ds = xr.open_zarr(root, consolidated=False)
    ds = ds.set_index(time="dates")

    # Extract lat/lon before selecting on variable (they live on cell only).
    spatial_dim = "cell"
    lat = ds["latitudes"] if "latitudes" in ds else None
    lon = ds["longitudes"] if "longitudes" in ds else None

    ds = ds.assign_coords(variable=ds.attrs["variables"])
    ds = ds.sel(variable=zarr_param).squeeze("ensemble", drop=True)

    # Recover 2-D spatial shape when stored as a flat cell dimension.
    if len(ds.attrs["field_shape"]) == 2:
        ny, nx = ds.attrs["field_shape"]
        y_idx, x_idx = np.unravel_index(np.arange(ny * nx), (ny, nx))
        ds = ds.assign_coords(y=(spatial_dim, y_idx), x=(spatial_dim, x_idx))
        ds = ds.set_index(**{spatial_dim: ("y", "x")}).unstack(spatial_dim)
        spatial_dim = None  # now (y, x)

    da = ds["data"].rename(param).drop_vars("variable", errors="ignore")

    # Truth zarrs store precip in m (anemoi convention); all forecast loaders
    # deliver canonical mm (kg m-2) since MRB-820, which put this conversion in
    # load_analysis_data_from_zarr. Mirror it here until this opener is
    # consolidated into data_input (refactor/data-io). Stays lazy (dask).
    if param in _ACCUMULATED_PARAMS:
        da = da * 1000

    # Attach latitude/longitude as coordinates on the spatial dimension(s).
    # Use the full names to match the forecast loader (load_forecast_data) and
    # map_forecast_to_truth, which key on `latitude`/`longitude`.
    if lat is not None and lon is not None:
        if spatial_dim is not None:
            # flat 1-D case: cell/values dim
            da = da.assign_coords(
                latitude=(spatial_dim, lat.values),
                longitude=(spatial_dim, lon.values),
            )
        else:
            # 2-D case: lat/lon still on original flat index — attach via unstack
            da = da.assign_coords(
                latitude=(["y", "x"], lat.values.reshape(ny, nx)),
                longitude=(["y", "x"], lon.values.reshape(ny, nx)),
            )

    return da


def open_truth_zarr(root: Path, param: str, accum_h: int | None = None) -> xr.DataArray:
    """Open the truth zarr lazily and return a DataArray for *param*.

    For derived variables (e.g. SP_10M) the required components are loaded and
    the derivation is applied on the fly.  The returned DataArray has dimensions
    ``(time, y, x)`` or ``(time, values)`` and always exposes ``latitude``/``longitude``.
    ``accum_h`` selects the precip accumulation length (TOT_PREC_<N>H).
    """
    if param in _DERIVED:
        components = {
            c: _open_zarr_component(root, c, accum_h).drop_vars(
                "variable", errors="ignore"
            )
            for c in _DERIVED[param]
        }
        ds = xr.Dataset(components)
        return _compute_derived(ds, param)
    return _open_zarr_component(root, param, accum_h)


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

    # Accumulated params (TOT_PREC) are stored cumulative-from-start, while the
    # truth is a period accumulation whose length equals the verification step
    # spacing (e.g. 6h for steps "0/120/6"). Derive that period so we can (a)
    # request the matching [step - period, step] window from the forecast loader
    # and (b) read the matching TOT_PREC_<period>H truth variable. We do not
    # assume a fixed period; it follows the configured --steps.
    accum_h: int | None = None
    if args.param in _ACCUMULATED_PARAMS:
        if not args.steps:
            raise ValueError(
                f"--steps is required for accumulated param '{args.param}' "
                "(used to derive the accumulation period)."
            )
        spacing = np.diff(parse_steps(args.steps))
        if spacing.size == 0:
            raise ValueError(
                f"Cannot derive an accumulation period from --steps '{args.steps}'."
            )
        accum_h = int(spacing.min())
        if args.step < accum_h:
            raise ValueError(
                f"Lead time {args.step}h is smaller than the {accum_h}h "
                f"accumulation period; cannot form a [step - period, step] "
                f"window for '{args.param}'."
            )
        req_steps = [args.step - accum_h, args.step]
        LOG.info("Accumulation period: %dh  (forecast window %s)", accum_h, req_steps)

        # INCA delivers native 1h precip sums and (unlike the GRIB paths, where
        # the cumulative-from-start diff adapts to the requested window) cannot
        # re-aggregate to a coarser period: the value at the target step would
        # stay a 1h sum while the truth read is TOT_PREC_<accum_h>H — a silent
        # mismatch. Re-aggregation in the loader is a planned follow-up.
        if args.baseline_root and "INCA" in args.baseline_root.parts and accum_h != 1:
            raise ValueError(
                f"INCA provides native 1h accumulations only, but the step "
                f"spacing of --steps '{args.steps}' implies a {accum_h}h "
                f"accumulation period for '{args.param}'. Use 1h-spaced steps "
                f"for INCA score maps."
            )
    else:
        req_steps = [args.step]

    # Open the truth zarr once; individual time slices are loaded on demand.
    truth_da = open_truth_zarr(args.truth, args.param, accum_h)
    # Normalise to datetime64[ns] so membership checks work regardless of zarr precision.
    truth_da = truth_da.assign_coords(
        time=truth_da.time.values.astype("datetime64[ns]")
    )
    # Rename flat spatial dim to 'values' if the zarr uses 'cell'.
    if "cell" in truth_da.dims:
        truth_da = truth_da.rename({"cell": "values"})
    truth_times = set(
        truth_da.time.values
    )  # keep as datetime64, tolist() yields ints for ns precision
    LOG.info("Truth opened lazily: %s", truth_da)

    if args.baseline_root:
        # The operational archive is too large to enumerate up front; the
        # experiment's configured init times define the work list, and dates
        # missing from the archive are skipped at load time below.
        init_items = [
            (rt, None)
            for rt in sorted(datetime.strptime(s, DATETIME_FMT) for s in args.reftimes)
        ]
        LOG.info("Using %d baseline init times from --reftimes", len(init_items))
    else:
        init_items = iter_init_dirs(args.run_root)
        LOG.info("Found %d init time directories", len(init_items))

        # Restrict to the experiment's configured init times if provided.
        if args.reftimes:
            wanted = {datetime.strptime(s, DATETIME_FMT) for s in args.reftimes}
            init_items = [(rt, d) for rt, d in init_items if rt in wanted]
            LOG.info("Filtered to %d init times matching --reftimes", len(init_items))

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

    n_ok = 0
    n_skip = 0

    for reftime, grib_dir in init_items:
        valid_time = np.datetime64(reftime + step_td).astype("datetime64[ns]")

        if valid_time not in truth_times:
            LOG.debug("Valid time %s not in truth, skipping %s", valid_time, reftime)
            n_skip += 1
            continue

        LOG.info(
            "Processing reftime=%s  valid=%s",
            reftime.strftime(DATETIME_FMT),
            valid_time,
        )

        first_iter = n_ok == 0

        # --- load forecast ---
        fct_params = (
            list(_DERIVED[args.param]) if args.param in _DERIVED else [args.param]
        )

        try:
            # For accumulated params (TOT_PREC) req_steps is the [step - period,
            # step] window; for GRIB sources (runs and the ICON archive) the
            # loader de-accumulates the cumulative-from-start field over the
            # requested steps (diff over `step`), so the target step holds the
            # period accumulation; INCA returns native 1h sums, matching the
            # period because accum_h == 1 is enforced above. Instantaneous
            # params request a single step. The target step is selected just
            # below.
            #
            # data_input._tot_prec_handling receives the requested steps and
            # synthesises a zero initial condition when step 0 is requested but
            # absent from the GRIB (anemoi-inference omits TOT_PREC at step 0),
            # which makes the first-lead-time window [0, period] work for ML
            # runs. Windows not containing step 0 are never zero-filled.
            src_root = args.baseline_root if args.baseline_root else grib_dir
            fcst = load_forecast_data(
                src_root, reftime, req_steps, fct_params, member=args.member
            )
        except Exception as exc:
            LOG.warning("Could not load forecast for %s: %s", reftime, exc)
            n_skip += 1
            continue

        # Select the target step. The earthkit loader returns forecasts over the
        # requested steps with a `step` (timedelta64) dimension; for TOT_PREC the
        # loader has already de-accumulated over the window, so the target step
        # holds the period accumulation, and for instantaneous params only the
        # single requested step is present.
        if "step" in fcst.dims:
            fcst = fcst.sel(step=np.timedelta64(args.step, "h"))

        # Compute derived variable if needed.
        if args.param in _DERIVED:
            fcst = fcst.assign({args.param: _compute_derived(fcst, args.param)})

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
        truth_slice = truth_da.sel(time=valid_time).compute()
        # For derived variables truth_da is already the derived DataArray,
        # so wrap it in a Dataset for map_forecast_to_truth compatibility.
        truth_ds = (
            truth_slice.to_dataset(name=args.param)
            if isinstance(truth_slice, xr.DataArray)
            else truth_slice
        )

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
            LOG.warning("Spatial mapping failed for %s: %s", reftime, exc)
            n_skip += 1
            continue

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

    LOG.info("Finished: %d init times processed, %d skipped", n_ok, n_skip)

    if n_ok == 0:
        LOG.error("No data could be processed – no output written.")
        return

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
            "n_skipped": n_skip,
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
        help=(
            "Forecast step spec 'start/stop/step' (e.g. '0/120/6'). Required for "
            "accumulated params (TOT_PREC): the accumulation period is the step "
            "spacing, the forecast is accumulated over [step - period, step], and "
            "the matching TOT_PREC_<period>H truth variable is read. Ignored for "
            "instantaneous params."
        ),
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
