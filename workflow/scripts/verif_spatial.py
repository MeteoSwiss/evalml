"""Compute spatial maps of temporally-aggregated forecast errors.

For a fixed lead time and variable, iterates over all initialisation times found
under a run directory, loads the corresponding GRIB forecast field and the
matching truth slice from a reference zarr, maps the forecast onto the truth
grid, and accumulates running error statistics without ever holding the full
time series in memory.  The final BIAS / RMSE / MAE maps are written to a
NetCDF file.

Usage
-----
    uv run workflow/scripts/verif_spatial.py \\
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

from data_input import load_fct_data_from_grib, load_baseline_from_zarr
from verification.spatial import map_forecast_to_truth

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

DATETIME_FMT = "%Y%m%d%H%M"

SEASONS = ["DJF", "MAM", "JJA", "SON", "all"]


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
_PARAMS_MAP_CO1 = {k: k.replace("TOT_PREC", "TOT_PREC_6H") for k in _PARAMS_MAP_CO2}

# Derived variables and the components they require.
_DERIVED = {
    "SP_10M": ("U_10M", "V_10M"),
}


def _params_map(truth_root: Path) -> dict[str, str]:
    return _PARAMS_MAP_CO2 if "co2" in truth_root.name else _PARAMS_MAP_CO1


def _compute_derived(ds: xr.Dataset, param: str) -> xr.DataArray:
    """Compute a derived variable from its components already present in *ds*."""
    if param == "SP_10M":
        return (ds["U_10M"] ** 2 + ds["V_10M"] ** 2) ** 0.5
    raise ValueError(f"No recipe for derived variable '{param}'")


# ---------------------------------------------------------------------------
# Truth loading
# ---------------------------------------------------------------------------

def _open_zarr_component(root: Path, param: str) -> xr.DataArray:
    """Open a single native zarr variable lazily as a DataArray."""
    zarr_param = _params_map(root)[param]

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

    # Attach lat/lon as coordinates on the spatial dimension(s).
    if lat is not None and lon is not None:
        if spatial_dim is not None:
            # flat 1-D case: cell/values dim
            da = da.assign_coords(lat=(spatial_dim, lat.values), lon=(spatial_dim, lon.values))
        else:
            # 2-D case: lat/lon still on original flat index — attach via unstack
            da = da.assign_coords(lat=(["y", "x"], lat.values.reshape(ny, nx)),
                                  lon=(["y", "x"], lon.values.reshape(ny, nx)))

    return da


def open_truth_zarr(root: Path, param: str) -> xr.DataArray:
    """Open the truth zarr lazily and return a DataArray for *param*.

    For derived variables (e.g. SP_10M) the required components are loaded and
    the derivation is applied on the fly.  The returned DataArray has dimensions
    ``(time, y, x)`` or ``(time, values)`` and always exposes ``lat``/``lon``.
    """
    if param in _DERIVED:
        components = {
            c: _open_zarr_component(root, c).drop_vars("variable", errors="ignore")
            for c in _DERIVED[param]
        }
        ds = xr.Dataset(components)
        return _compute_derived(ds, param)
    return _open_zarr_component(root, param)


# ---------------------------------------------------------------------------
# GRIB step helpers
# ---------------------------------------------------------------------------

# Parameters whose GRIB values are cumulative totals and must be disaggregated
# via diff before verification. For these, the preceding step must also be
# loaded so that load_fct_data_from_grib's diff produces a valid result.
_CUMULATIVE_PARAMS = {"TOT_PREC"}


def _preceding_step(grib_dir: Path, step: int) -> int | None:
    """Return the largest available step smaller than *step* in *grib_dir*."""
    available = sorted(
        int(f.stem.split("_")[-1])
        for f in grib_dir.glob("*.grib")
        if f.stem.split("_")[-1].isdigit()
    )
    smaller = [s for s in available if s < step]
    return smaller[-1] if smaller else None


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


def iter_baseline_init_times(zarr_paths: list[Path], step: int) -> list[datetime]:
    """Return all init times from baseline zarr(s) that have the requested step available."""
    step_td = np.timedelta64(step, "h")
    reftimes = []
    for zarr_path in zarr_paths:
        if not zarr_path.exists():
            LOG.warning("Baseline zarr not found: %s", zarr_path)
            continue
        ds = xr.open_zarr(zarr_path, consolidated=True, decode_timedelta=True)
        if step_td not in ds["step"].values:
            LOG.warning("Step %dh not in %s, skipping", step, zarr_path)
            continue
        for rt in ds["forecast_reference_time"].values:
            ts = (rt - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
            reftimes.append(datetime.utcfromtimestamp(float(ts)))
    return sorted(reftimes)


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

    # Open the truth zarr once; individual time slices are loaded on demand.
    truth_da = open_truth_zarr(args.truth, args.param)
    # Normalise to datetime64[ns] so membership checks work regardless of zarr precision.
    truth_da = truth_da.assign_coords(
        time=truth_da.time.values.astype("datetime64[ns]")
    )
    # Rename flat spatial dim to 'values' if the zarr uses 'cell'.
    if "cell" in truth_da.dims:
        truth_da = truth_da.rename({"cell": "values"})
    truth_times = set(truth_da.time.values)  # keep as datetime64, tolist() yields ints for ns precision
    LOG.info("Truth opened lazily: %s", truth_da)

    if args.baseline_root:
        init_items = [(rt, None) for rt in iter_baseline_init_times(args.baseline_zarrs, args.step)]
        LOG.info("Found %d baseline init times", len(init_items))
    else:
        init_items = iter_init_dirs(args.run_root)
        LOG.info("Found %d init time directories", len(init_items))

    # Restrict to the experiment's configured init times if provided.
    # Without this, baseline zarrs (which contain a continuous archive) would
    # cause the script to process every init time in the file rather than
    # only those in the user's hindcast period.
    if args.reftimes:
        wanted = {datetime.strptime(s, DATETIME_FMT) for s in args.reftimes}
        init_items = [(rt, d) for rt, d in init_items if rt in wanted]
        LOG.info("Filtered to %d init times matching --reftimes", len(init_items))

    step_td = timedelta(hours=args.step)

    # Running accumulators per season – initialised on the first successfully
    # processed sample so that we can infer the spatial shape from the data.
    # Each entry is a numpy array over the spatial dimension(s).
    accum_n:      dict[str, np.ndarray | None] = {s: None for s in SEASONS}
    accum_sum_e:  dict[str, np.ndarray | None] = {s: None for s in SEASONS}
    accum_sum_se: dict[str, np.ndarray | None] = {s: None for s in SEASONS}
    accum_sum_ae: dict[str, np.ndarray | None] = {s: None for s in SEASONS}
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
        fct_params = list(_DERIVED[args.param]) if args.param in _DERIVED else [args.param]

        try:
            if args.baseline_root:
                zarr_path = args.baseline_root / f"FCST{reftime.strftime('%y')}.zarr"
                fcst = load_baseline_from_zarr(
                    root=zarr_path,
                    reftime=reftime,
                    steps=[args.step],
                    params=fct_params,
                )
            else:
                # Cumulative params (e.g. TOT_PREC) are disaggregated via diff inside
                # load_fct_data_from_grib, so we need to also load the preceding step.
                if args.param in _CUMULATIVE_PARAMS:
                    prev_step = _preceding_step(grib_dir, args.step)
                    if prev_step is None:
                        LOG.warning("No preceding step for cumulative param %s at step %d, skipping %s",
                                    args.param, args.step, reftime)
                        n_skip += 1
                        continue
                    load_steps = [prev_step, args.step]
                else:
                    load_steps = [args.step]
                fcst = load_fct_data_from_grib(
                    root=grib_dir,
                    reftime=reftime,
                    steps=load_steps,
                    params=fct_params,
                )
        except Exception as exc:
            LOG.warning("Could not load forecast for %s: %s", reftime, exc)
            n_skip += 1
            continue

        # Drop lead_time dimension (select only the requested step).
        if "lead_time" in fcst.dims:
            fcst = fcst.sel(lead_time=np.timedelta64(args.step, "h"))

        # Compute derived variable if needed.
        if args.param in _DERIVED:
            fcst = fcst.assign({args.param: _compute_derived(fcst, args.param)})

        if first_iter:
            LOG.info("fcst (after step selection): %s", fcst)
            fcst_raw = fcst[args.param].values if args.param in fcst else None
            if fcst_raw is not None:
                n_nan_fcst = int(np.isnan(fcst_raw).sum())
                LOG.info("fcst[%s]: shape=%s, min=%.4g, max=%.4g, n_nan=%d",
                         args.param, fcst_raw.shape,
                         float(np.nanmin(fcst_raw)) if n_nan_fcst < fcst_raw.size else float("nan"),
                         float(np.nanmax(fcst_raw)) if n_nan_fcst < fcst_raw.size else float("nan"),
                         n_nan_fcst)

        # --- load truth slice ---
        truth_slice = truth_da.sel(time=valid_time).compute()
        # For derived variables truth_da is already the derived DataArray,
        # so wrap it in a Dataset for map_forecast_to_truth compatibility.
        truth_ds = truth_slice.to_dataset(name=args.param) if isinstance(truth_slice, xr.DataArray) else truth_slice

        if first_iter:
            truth_raw = truth_slice.values
            n_nan_truth = int(np.isnan(truth_raw).sum())
            LOG.info("truth_slice[%s]: shape=%s, min=%.4g, max=%.4g, n_nan=%d",
                     args.param, truth_raw.shape,
                     float(np.nanmin(truth_raw)) if n_nan_truth < truth_raw.size else float("nan"),
                     float(np.nanmax(truth_raw)) if n_nan_truth < truth_raw.size else float("nan"),
                     n_nan_truth)

        # --- map forecast onto truth grid ---
        try:
            fcst_mapped = map_forecast_to_truth(fcst, truth_ds)
        except Exception as exc:
            LOG.warning("Spatial mapping failed for %s: %s", reftime, exc)
            n_skip += 1
            continue

        fcst_param = fcst_mapped[args.param]
        # Squeeze any ensemble/eps dimension (deterministic run stored with size-1 eps).
        for dim in ["eps", "ensemble", "number"]:
            if dim in fcst_param.dims and fcst_param.sizes[dim] == 1:
                fcst_param = fcst_param.squeeze(dim, drop=True)
        fcst_vals = fcst_param.values
        truth_vals = truth_slice.values
        error = fcst_vals - truth_vals  # shape: spatial dims of truth

        if first_iter:
            n_nan_mapped = int(np.isnan(fcst_vals).sum())
            LOG.info("fcst_mapped[%s]: shape=%s, min=%.4g, max=%.4g, n_nan=%d",
                     args.param, fcst_vals.shape,
                     float(np.nanmin(fcst_vals)) if n_nan_mapped < fcst_vals.size else float("nan"),
                     float(np.nanmax(fcst_vals)) if n_nan_mapped < fcst_vals.size else float("nan"),
                     n_nan_mapped)
            n_nan_err = int(np.isnan(error).sum())
            LOG.info("error: shape=%s, min=%.4g, max=%.4g, n_nan=%d / %d",
                     error.shape,
                     float(np.nanmin(error)) if n_nan_err < error.size else float("nan"),
                     float(np.nanmax(error)) if n_nan_err < error.size else float("nan"),
                     n_nan_err, error.size)

        n_nan_error = int(np.isnan(error).sum())
        if n_nan_error == error.size:
            LOG.warning("reftime=%s: error is all-NaN (%d points) — nothing accumulated.",
                        reftime.strftime(DATETIME_FMT), error.size)

        # --- initialise accumulators on first valid sample ---
        if accum_n["all"] is None:
            for s in SEASONS:
                accum_n[s]      = np.zeros(error.shape, dtype=np.int64)
                accum_sum_e[s]  = np.zeros(error.shape, dtype=np.float64)
                accum_sum_se[s] = np.zeros(error.shape, dtype=np.float64)
                accum_sum_ae[s] = np.zeros(error.shape, dtype=np.float64)
            ref_truth_slice = truth_slice

        # --- accumulate into the matching season bucket and "all" (NaN-safe) ---
        season = _season_of(reftime)
        valid = ~np.isnan(error)
        for s in [season, "all"]:
            accum_n[s][valid]      += 1
            accum_sum_e[s][valid]  += error[valid]
            accum_sum_se[s][valid] += error[valid] ** 2
            accum_sum_ae[s][valid] += np.abs(error[valid])
        n_ok += 1

    LOG.info("Finished: %d init times processed, %d skipped", n_ok, n_skip)

    if n_ok == 0:
        LOG.error("No data could be processed – no output written.")
        return

    # --- compute aggregate maps per season, then stack into a season dimension ---
    spatial_coords = {
        c: ref_truth_slice[c]
        for c in ref_truth_slice.coords
        if set(ref_truth_slice[c].dims).issubset(set(ref_truth_slice.dims))
        and c != "time"
    }
    spatial_dims = list(ref_truth_slice.dims)
    out_dims = ["season"] + spatial_dims
    out_coords = {"season": SEASONS, **spatial_coords}

    def _seasonal_da(compute_fn) -> xr.DataArray:
        """Stack per-season arrays into a (season, *spatial) DataArray."""
        slices = []
        for s in SEASONS:
            n = accum_n[s]
            with np.errstate(invalid="ignore", divide="ignore"):
                slices.append(compute_fn(n, s).astype(np.float32))
        return xr.DataArray(np.stack(slices), dims=out_dims, coords=out_coords)

    out = xr.Dataset(
        {
            f"{args.param}.BIAS": _seasonal_da(
                lambda n, s: np.where(n > 0, accum_sum_e[s] / n, np.nan)
            ),
            f"{args.param}.RMSE": _seasonal_da(
                lambda n, s: np.where(n > 0, np.sqrt(accum_sum_se[s] / n), np.nan)
            ),
            f"{args.param}.MAE": _seasonal_da(
                lambda n, s: np.where(n > 0, accum_sum_ae[s] / n, np.nan)
            ),
            f"{args.param}.N": _seasonal_da(
                lambda n, s: np.where(n > 0, n, np.nan)
            ),
        },
        attrs={
            "param": args.param,
            "step_h": args.step,
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
            "Supports both model runs (GRIB) and baselines (zarr). "
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
        help="Root directory of a baseline (e.g. /path/to/ICON-CH1-EPS), containing FCST<YY>.zarr files.",
    )
    parser.add_argument(
        "--baseline_zarrs",
        type=Path,
        nargs="+",
        default=None,
        help="Explicit list of baseline zarr paths (used by Snakemake for dependency tracking).",
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
        "--reftimes",
        nargs="+",
        default=None,
        help=(
            "Optional list of init times (YYYYMMDDHHMM) to restrict processing to. "
            "Required for baselines whose zarr contains a continuous archive."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output NetCDF file.",
    )
    args = parser.parse_args()

    if bool(args.run_root) == bool(args.baseline_root):
        parser.error("Exactly one of --run_root or --baseline_root must be provided.")

    if args.output is None:
        source = args.run_root or args.baseline_root
        args.output = source / f"verif_spatial_{args.param}_step{args.step:03d}h.nc"

    main(args)
