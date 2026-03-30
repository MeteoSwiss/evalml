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

from data_input import load_fct_data_from_grib
from verification.spatial import map_forecast_to_truth

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

DATETIME_FMT = "%Y%m%d%H%M"

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

    init_dirs = iter_init_dirs(args.run_root)
    LOG.info("Found %d init time directories", len(init_dirs))

    step_td = timedelta(hours=args.step)

    # Running accumulators – initialised on the first successfully processed
    # sample so that we can infer the spatial shape from the data.
    accum_n: np.ndarray | None = None
    accum_sum_e: np.ndarray | None = None
    accum_sum_se: np.ndarray | None = None
    accum_sum_ae: np.ndarray | None = None
    ref_truth_slice: xr.DataArray | None = None  # kept for output coordinates

    n_ok = 0
    n_skip = 0

    for reftime, grib_dir in init_dirs:
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

        # --- load forecast ---
        grib_params = list(_DERIVED[args.param]) if args.param in _DERIVED else [args.param]
        try:
            fcst = load_fct_data_from_grib(
                root=grib_dir,
                reftime=reftime,
                steps=[args.step],
                params=grib_params,
            )
        except Exception as exc:
            LOG.warning("Could not load GRIB for %s: %s", reftime, exc)
            n_skip += 1
            continue

        # Drop lead_time dimension (single step requested).
        if "lead_time" in fcst.dims:
            fcst = fcst.sel(lead_time=np.timedelta64(args.step, "h"))

        # Compute derived variable if needed.
        if args.param in _DERIVED:
            fcst = fcst.assign({args.param: _compute_derived(fcst, args.param)})

        # --- load truth slice ---
        truth_slice = truth_da.sel(time=valid_time).compute()
        # For derived variables truth_da is already the derived DataArray,
        # so wrap it in a Dataset for map_forecast_to_truth compatibility.
        truth_ds = truth_slice.to_dataset(name=args.param) if isinstance(truth_slice, xr.DataArray) else truth_slice

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

        # --- initialise accumulators ---
        if accum_n is None:
            accum_n = np.zeros(error.shape, dtype=np.int64)
            accum_sum_e = np.zeros(error.shape, dtype=np.float64)
            accum_sum_se = np.zeros(error.shape, dtype=np.float64)
            accum_sum_ae = np.zeros(error.shape, dtype=np.float64)
            ref_truth_slice = truth_slice

        # --- accumulate (NaN-safe) ---
        valid = ~np.isnan(error)
        accum_n[valid] += 1
        accum_sum_e[valid] += error[valid]
        accum_sum_se[valid] += error[valid] ** 2
        accum_sum_ae[valid] += np.abs(error[valid])
        n_ok += 1

    LOG.info("Finished: %d init times processed, %d skipped", n_ok, n_skip)

    if n_ok == 0:
        LOG.error("No data could be processed – no output written.")
        return

    # --- compute aggregate maps ---
    with np.errstate(invalid="ignore", divide="ignore"):
        bias = np.where(accum_n > 0, accum_sum_e / accum_n, np.nan).astype(np.float32)
        rmse = np.where(accum_n > 0, np.sqrt(accum_sum_se / accum_n), np.nan).astype(
            np.float32
        )
        mae = np.where(accum_n > 0, accum_sum_ae / accum_n, np.nan).astype(np.float32)
        count = np.where(accum_n > 0, accum_n, np.nan).astype(np.float32)

    # Build a spatial template: keep only spatial dims and lat/lon coords,
    # dropping scalar coords like time (added by .sel).
    spatial_coords = {
        c: ref_truth_slice[c]
        for c in ref_truth_slice.coords
        if set(ref_truth_slice[c].dims).issubset(set(ref_truth_slice.dims))
        and c != "time"
    }

    def _da(data: np.ndarray) -> xr.DataArray:
        return xr.DataArray(data, dims=ref_truth_slice.dims, coords=spatial_coords)

    out = xr.Dataset(
        {
            f"{args.param}.BIAS": _da(bias),
            f"{args.param}.RMSE": _da(rmse),
            f"{args.param}.MAE": _da(mae),
            f"{args.param}.N": _da(count),
        },
        attrs={
            "param": args.param,
            "step_h": args.step,
            "run_root": str(args.run_root),
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
            "Compute spatial maps of temporally-aggregated forecast errors "
            "by streaming over GRIB files from a model run."
        )
    )
    parser.add_argument(
        "--run_root",
        type=Path,
        help="Root directory of a model run (e.g. output/data/runs/<run_id>).",
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
        "--output",
        type=Path,
        default=None,
        help=(
            "Output NetCDF file. "
            "Default: <run_root>/verif_spatial_<param>_step<NNN>h.nc"
        ),
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = (
            args.run_root / f"verif_spatial_{args.param}_step{args.step:03d}h.nc"
        )

    main(args)
