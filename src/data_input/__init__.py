import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Any

import earthkit.data as ekd
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

LOG = logging.getLogger(__name__)

# IFS shortNames that differ from ICON parameter names.
# Used when loading GRIB output from global models (e.g. AIFS-single) that write IFS names.
_IFS_TO_ICON = {
    "tp": "TOT_PREC",
    "msl": "PMSL",
    "10u": "U_10M",
    "10v": "V_10M",
    "2t": "T_2M",
    "2d": "TD_2M",
    "sp": "PS",
    "lsm": "FR_LAND",
}
_ICON_TO_IFS = {v: k for k, v in _IFS_TO_ICON.items()}

XARRAY_ENGINE_PROFILE = {
    "ensure_dims": ["z", "number", "step", "forecast_reference_time"],
    "add_valid_time_coord": True,
    "global_attrs": [{"institution": "MeteoSwiss"}, {"Conventions": "CF-1.8"}],
}

ZERO_KELVIN = -273.15  # °C

# Base params whose GRIB representation is cumulative-from-start.
# Extend this frozenset to enable new accumulatable variables.
_ACCUMULATABLE_PARAMS: frozenset[str] = frozenset({"TOT_PREC"})

# Spatial/multi-field derivations: output param → required input params.
DERIVED_PARAMS: dict[str, tuple[str, ...]] = {
    "SP_10M": ("U_10M", "V_10M"),
}


def parse_aggregated_param(param: str) -> tuple[str, int | None]:
    """Decompose a param name into (base_param, agg_hours).

    Always returns a tuple; agg_hours is None for non-aggregated params.

    'TOT_PREC6' → ('TOT_PREC', 6)
    'T_2M'      → ('T_2M', None)
    'SP_10M'    → ('SP_10M', None)

    A param is aggregated when its name ends with a plain integer AND its prefix
    belongs to _ACCUMULATABLE_PARAMS. Extending _ACCUMULATABLE_PARAMS adds
    support for new base params; new aggregation periods need no code change
    (only a DWH_PARAM_MAP entry in load_obs_data_from_jretrieve).
    """
    m = re.fullmatch(r"([A-Z_]+?)(\d+)", param)
    if m and m.group(1) in _ACCUMULATABLE_PARAMS:
        return m.group(1), int(m.group(2))
    return param, None


def compute_derived(ds: xr.Dataset, param: str) -> xr.DataArray:
    """Compute a spatial derived variable from its components already in ds."""
    if param == "SP_10M":
        return (ds["U_10M"] ** 2 + ds["V_10M"] ** 2) ** 0.5
    raise ValueError(f"No recipe for derived variable '{param}'")


def get_base_params(params: list[str]) -> list[str]:
    """Expand derived/aggregated params to the native params needed for loading.

    'SP_10M'    → {'U_10M', 'V_10M'}
    'TOT_PREC6' → {'TOT_PREC'}
    'T_2M'      → {'T_2M'}

    Two-step: resolve aggregated param names to base params, then resolve
    spatial derived params to their components. Returns a deduplicated list.
    """
    bases = {parse_aggregated_param(p)[0] for p in params}
    return list({c for p in bases for c in DERIVED_PARAMS.get(p, (p,))})


def get_steps(steps: list[int], params: list[str]) -> list[int]:
    """Compute all time steps needed to satisfy all aggregation windows.

    For each aggregated param (e.g. TOT_PREC6, agg_hours=6) and each requested
    step s, adds s-agg_hours if s-agg_hours >= 0 (needed as the lower bound of
    the accumulation diff window). Non-aggregated and spatial derived params
    require no extra steps.

    'TOT_PREC6', steps=[6,12,18]  → [0, 6, 12, 18]
    'TOT_PREC6', steps=[3,6,12]   → [0, 3, 6, 12]  # step 3: 3-6<0, not added
    'T_2M',      steps=[6,12]     → [6, 12]
    """
    extra = set()
    for p in params:
        _, n = parse_aggregated_param(p)
        if n is not None:
            for s in steps:
                if s - n >= 0:
                    extra.add(s - n)
    return sorted(set(steps) | extra)


def _select_valid_times(ds, times: np.datetime64, strict: bool = False):
    # (handle special case where some valid times are not in the dataset, e.g. at the end)
    times_np = np.asarray(times, dtype="datetime64[ns]")
    times_included = np.isin(times_np, ds.time.values)
    if times_included.all():
        return ds.sel(time=times_np)
    elif times_included.any():
        missing = times_np[~times_included]
        if strict:
            raise ValueError(
                f"Some valid times are not included in the dataset:\n{missing}"
            )
        LOG.warning(
            "Some valid times are not included in the dataset: \n%s",
            missing,
        )
        return ds.sel(time=times_np[times_included])
    else:
        raise ValueError(
            "Valid times are not included in the dataset. "
            "Please check the valid times and the dataset."
        )


def parse_steps(steps: str) -> list[int]:
    # check that steps is in the format "start/stop/step"
    if "/" not in steps:
        raise ValueError(f"Expected steps in format 'start/stop/step', got '{steps}'")
    if len(steps.split("/")) != 3:
        raise ValueError(f"Expected steps in format 'start/stop/step', got '{steps}'")
    start, end, step = map(int, steps.split("/"))
    return list(range(start, end + 1, step))


def _load_analysis_data_from_zarr(
    root: Path, reftime: datetime, steps: list[int], params: list[str]
) -> xr.Dataset:
    """Load analysis data from an anemoi-generated Zarr dataset

    This function loads analysis data from a Zarr dataset, processing it to make it more
    xarray-friendly. It renames variables, sets the time index, and pivots the dataset.
    """
    USE_IFS_NAMES = {"-co2-", "-ea-", "ifsnames"}
    if any(tag in root.name for tag in USE_IFS_NAMES):
        # Zarr stores IFS shortNames; map ICON param names to IFS for selection
        zarr_names = {p: _ICON_TO_IFS.get(p, p) for p in params}
    else:
        # Zarr stores ICON param names; TOT_PREC has a time-resolution suffix
        tot_prec_key = "TOT_PREC_6H" if min(np.diff(steps)) == 6 else "TOT_PREC_1H"
        zarr_names = {p: p.replace("TOT_PREC", tot_prec_key) for p in params}

    ds = xr.open_zarr(root, consolidated=False)

    # rename "dates" to "time" and set it as index
    ds = ds.set_index(time="dates")

    # set 'variables' attr as dimension coordinate
    ds = ds.assign_coords({"variable": ds.attrs["variables"]})

    # select variables and valid time, squeeze ensemble dimension
    ds = ds.sel(variable=[zarr_names[p] for p in params]).squeeze("ensemble", drop=True)

    # recover original 2D shape (not present for global Gaussian grids)
    if "field_shape" in ds.attrs and len(ds.attrs["field_shape"]) == 2:
        ny, nx = ds.attrs["field_shape"]
        y_idx, x_idx = np.unravel_index(np.arange(ny * nx), shape=(ny, nx))
        ds = ds.assign_coords({"y": ("cell", y_idx), "x": ("cell", x_idx)})
        ds = ds.set_index(cell=("y", "x"))
        ds = ds.unstack("cell")

    # set lat lon as coords (optional)
    if "latitudes" in ds and "longitudes" in ds:
        ds = ds.rename({"latitudes": "latitude", "longitudes": "longitude"})
    if "latitude" in ds and "longitude" in ds:
        ds = ds.set_coords(["latitude", "longitude"])
    ds = (
        ds["data"]
        .to_dataset("variable")
        .rename({v: k for k, v in zarr_names.items() if v in ds["variable"].values})
    )

    # change precipitation units from m to kg m-2
    for prec_key in ("TOT_PREC_6H", "TOT_PREC_1H", "TOT_PREC"):
        if prec_key in ds:
            ds[prec_key] = (
                ds[prec_key] * 1000
            )  # convert precipitation units from m to mm

    # rename 'cell' dimension to 'values' (it's earthkit-data default for flattened spatial dim)
    if "cell" in ds.dims:
        ds = ds.rename({"cell": "values"})

    times = np.datetime64(reftime) + np.asarray(steps, dtype="timedelta64[h]")
    return _select_valid_times(ds, times)


def _collect_ml_grib_files(root: Path, steps: list[int] | None = None) -> list[Path]:
    """Return GRIB files for an ML inference run (flat directory layout).

    When `steps` is provided, the discovered files are filtered to those whose
    name ends with ``_{step:03d}.grib``.
    """
    # TODO: this glob pattern is a dirty fix for anemoi-inference writing outputs
    # with wrong formatting. Eventually we will either have to have a fix upstream
    # or write a single output file.
    files = sorted(root.glob("20*.grib"))
    if steps is None:
        return files

    # again, two different patterns might be used for step formatting
    suffixes = {f"_{step:03d}.grib" for step in steps}
    suffixes |= {f"_{step}.grib" for step in steps}
    return [f for f in files if any(f.name.endswith(s) for s in suffixes)]


def _collect_icon_archive_files(
    root: Path, reftime: datetime, steps: list[int], member_id: str = "000"
) -> list[Path]:
    """Return surface GRIB files for one member of an ICON operational archive.

    `root` is the top-level ICON directory, e.g.
    ``/store_new/mch/msopr/osm/ICON-CH1-EPS``. The FCST<year> subdirectory
    is derived automatically from `reftime`.
    """
    fcst_root = root / f"FCST{reftime:%y}"
    reftime_dirs = sorted(fcst_root.glob(f"{reftime:%y%m%d%H}_*"))
    if not reftime_dirs:
        raise ValueError(
            f"No archive subdirectory found for {reftime:%y%m%d%H} in {fcst_root}"
        )
    reftime_dir = reftime_dirs[-1]
    LOG.info("Reading member %s from %s", member_id, reftime_dir)

    if "ICON-CH1-EPS" in root.parts:
        gribname = "i1eff"
    elif "ICON-CH2-EPS" in root.parts:
        gribname = "i2eff"
    else:
        raise ValueError(
            f"Cannot determine model from path (expected ICON-CH1-EPS or "
            f"ICON-CH2-EPS): {root}"
        )

    return [
        reftime_dir / "grib" / f"{gribname}{lt // 24:02}{lt % 24:02}0000_{member_id}"
        for lt in steps
    ]


def _discover_icon_member_ids(
    root: Path, reftime: datetime, steps: list[int]
) -> list[str]:
    """Return sorted list of numeric member IDs present in the ICON archive for `reftime`."""
    first_file = _collect_icon_archive_files(root, reftime, [steps[0]])[0]
    prefix = first_file.name.rsplit("_", 1)[0]
    return sorted(
        p.name.rsplit("_", 1)[1] for p in first_file.parent.glob(f"{prefix}_???")
    )


def load_from_grib_file(file: str | list[str], sel_kwargs):
    # Coerce Path objects to str: earthkit-data unwraps a single-element list
    # into one File source without converting, and then fails on non-str paths.
    if isinstance(file, (list, tuple)):
        file = [str(f) for f in file]
    else:
        file = str(file)
    fieldlist = ekd.from_source("file", file, lazily=True).to_fieldlist()
    return fieldlist_to_xarray(fieldlist.sel(**sel_kwargs))


def variable_name_profile(
    level_type: Literal["height_above_ground_level", "mean_sea", "surface", "pressure"],
) -> dict[str, Any]:
    """Resolve variable name profile based on the level type."""
    if level_type in ["height_above_ground_level", "mean_sea", "surface"]:
        return {}
    elif level_type == "pressure":
        return {
            "variable_key": "p_l",
            "remapping": {"p_l": "{parameter.variable}_{vertical.level}"},
        }
    else:
        raise ValueError(f"Unsupported level type: {level_type}")


def fieldlist_to_xarray(fieldlist) -> xr.Dataset:
    ds = xr.Dataset()
    if len(fieldlist) == 0:
        return ds
    for level_type_group in fieldlist.group_by("vertical.level_type"):
        # earthkit-data should return the group key...TODO: open issue?
        level_type = level_type_group.get("vertical.level_type")[0]
        profile = XARRAY_ENGINE_PROFILE | variable_name_profile(level_type)
        _ds = level_type_group.to_xarray(**profile, allow_holes=True)
        ds = ds.merge(_ds, compat="no_conflicts", combine_attrs="no_conflicts")
    return ds


def _ensure_accum_ic(cumul: xr.DataArray, steps: list[int]) -> xr.DataArray:
    """Ensure a zero initial condition at step 0 in a cumulative-from-start field.

    Should be called before any disaggregation when 0 is in load_steps.
    Handles two modes:
      (a) step 0 is absent entirely → prepend a zero-filled slice
      (b) step 0 is present but all-NaN → fill with zeros
    Only acts when 0 is in steps; other steps are genuine data and
    must not be zero-filled.
    """
    if 0 not in steps:
        return cumul
    step0_td = np.timedelta64(0, "ns")
    step0_idx = np.where(cumul["step"].values == step0_td)[0]
    if step0_idx.size == 0:
        LOG.info("Step 0 of cumulative field missing; prepending zeros (IC = 0).")
        zero = xr.zeros_like(cumul.isel(step=[0])).assign_coords(step=[step0_td])
        cumul = xr.concat([zero, cumul], dim="step")
    elif cumul[{"step": int(step0_idx[0])}].isnull().all():
        LOG.info("Step 0 of cumulative field is all-NaN; filling with zeros (IC = 0).")
        cumul[{"step": int(step0_idx[0])}] = 0.0
    return cumul


def _disaggregate_accum(cumul: xr.DataArray, steps: list[int], n: int) -> xr.DataArray:
    """Compute n-hour period accumulations from a cumulative-from-start field.

    For each step s in steps where s >= n: result[s] = cumul[s] - cumul[s-n],
    clipped to 0. For steps s < n (window extends before model start), result is NaN.
    Raises ValueError if any valid window gives significantly negative values (data is
    not actually cumulative from start).
    """
    step_coords = [np.timedelta64(s, "h") for s in steps]
    result = xr.full_like(cumul.sel(step=step_coords), fill_value=np.nan)

    valid_steps = [s for s in steps if s >= n]
    for s in valid_steps:
        result.loc[{"step": np.timedelta64(s, "h")}] = cumul.sel(
            step=np.timedelta64(s, "h")
        ) - cumul.sel(step=np.timedelta64(s - n, "h"))

    if valid_steps:
        valid_min = float(
            result.sel(step=[np.timedelta64(s, "h") for s in valid_steps])
            .min()
            .compute()
        )
        if valid_min < -0.1:
            raise ValueError(
                f"Disaggregated field has significantly negative values "
                f"(min={valid_min:.3e}); the source field may already be "
                "period-accumulated rather than cumulative-from-start."
            )
    return result.clip(min=0)  # clip leaves NaN intact


def _load_forecast_data_from_grib(files: list[Path], params: list[str]) -> xr.Dataset:
    """Load forecast data from a list of GRIB files. Returns raw fields as-is.

    External callers should use :func:`load_forecast_data`, which handles
    param expansion, step expansion, IC synthesis, and disaggregation.
    Cumulative-from-start fields (e.g. TOT_PREC) are returned without
    disaggregation — that is the caller's responsibility.
    """
    # Extend param selection to include IFS aliases so that global-model GRIB files
    # (which use IFS shortNames like "tp", "2t") are also matched.
    params_extended = list(
        {p for p in params} | {_ICON_TO_IFS[p] for p in params if p in _ICON_TO_IFS}
    )
    ds = load_from_grib_file(files, {"parameter.variable": params_extended})

    # Rename any IFS shortNames back to ICON names
    ifs_rename = {
        ifs: icon for ifs, icon in _IFS_TO_ICON.items() if ifs in ds.data_vars
    }
    if ifs_rename:
        ds = ds.rename(ifs_rename)

    return ds


def _jretrieve_df_to_xarray(df, short_names, catalog) -> xr.Dataset:
    """Pivot long-form jretrieve obs into a (time, values) cube aligned to the
    catalog, NaN-filled for missing cells."""
    station_to_idx = {sid: i for i, sid in enumerate(catalog.station_id)}
    if df.empty:
        time_index = pd.DatetimeIndex([])
    else:
        df = df.copy()
        df["time"] = pd.to_datetime(df["termin"].astype(str), format="%Y%m%d%H%M%S")
        time_index = pd.DatetimeIndex(sorted(df["time"].unique()))
    n_t, n_s = len(time_index), catalog.n
    coords = {
        "time": ("time", time_index.values.astype("datetime64[ns]")),
        "values": ("values", catalog.nat_abbr),
        "latitude": ("values", catalog.latitude),
        "longitude": ("values", catalog.longitude),
    }
    data_vars: dict[str, tuple] = {}
    if df.empty:
        for p in short_names:
            data_vars[p] = (("time", "values"), np.full((n_t, n_s), np.nan, np.float32))
    else:
        time_to_idx = {t: i for i, t in enumerate(time_index)}
        df["_si"] = df["station"].map(station_to_idx)
        df["_ti"] = df["time"].map(time_to_idx)
        df = df.dropna(subset=["_si", "_ti"])
        df["_si"] = df["_si"].astype(int)
        df["_ti"] = df["_ti"].astype(int)
        for p in short_names:
            arr = np.full((n_t, n_s), np.nan, dtype=np.float32)
            if p in df.columns:
                arr[df["_ti"].to_numpy(), df["_si"].to_numpy()] = df[p].to_numpy(
                    dtype=np.float32
                )
            data_vars[p] = (("time", "values"), arr)
    return xr.Dataset(data_vars=data_vars, coords=coords)


def load_obs_data_from_jretrieve(
    root, reftime: datetime, steps: list[int], params: list[str]
) -> xr.Dataset:
    """Load SwissMetNet (SMN) surface observations from the DWH via jretrievedwh.

    ``root`` is a marker string selecting stations, e.g. ``jretrievedwh:SwissMetNet``
    (default group), ``jretrievedwh:locations=ARO,KLO``, or
    ``jretrievedwh:bbox=45.8,47.8,5.9,10.5`` (optionally ``;stage=devt``). Returns
    a Dataset with dims (time, values), values=nat_abbr, latitude/longitude coords,
    variables renamed to ICON names in SI units (T/TD in Kelvin, pressure in Pa).
    Only the requested hourly valid times are kept.
    """
    DWH_PARAM_MAP = {
        "T_2M": "tre200s0",
        "TD_2M": "tde200s0",
        "PS": "prestas0",
        "PMSL": "pp0qffs0",
        "TOT_PREC": "rre150h0",
        "FF_10M": "fkl010z0",
        "SP_10M": "fkl010z0",
        "DD_10M": "dkl010z0",
        "VMAX_10M": "fkl010z1",
    }
    DWH_WIND_SPEED = "fkl010z0"
    DWH_WIND_DIR = "dkl010z0"
    DWH_CELSIUS_TO_KELVIN = {"tre200s0", "tde200s0"}
    DWH_HPA_TO_PA = {"prestas0", "pp0qffs0"}

    from data_input import jretrieve as jr

    stations, stage, seq_type = jr.parse_selection(root)
    jr.check_prerequisites(stage)

    want_uv = "U_10M" in params or "V_10M" in params
    short_names: list[str] = [DWH_PARAM_MAP[p] for p in params if p in DWH_PARAM_MAP]
    if want_uv:
        short_names += [DWH_WIND_SPEED, DWH_WIND_DIR]
    short_names = list(dict.fromkeys(short_names))
    if not short_names:
        raise ValueError(f"No DWH parameter mapping for requested params: {params}")

    start = reftime
    end = start + timedelta(hours=max(steps))
    if len(steps) > 1:
        end += timedelta(hours=steps[-1] - steps[-2])

    catalog = jr.StationCatalog.from_meta(
        jr.fetch_meta(
            stations=stations, params=short_names, seq_type=seq_type, stage=stage
        )
    )
    step_hours = (steps[1] - steps[0]) if len(steps) > 1 else 1
    df = jr.fetch_data(
        stations=stations,
        params=short_names,
        start=start,
        end=end,
        increment_minutes=step_hours * 60,
        seq_type=seq_type,
        stage=stage,
    )
    raw = _jretrieve_df_to_xarray(df, short_names, catalog)

    out = xr.Dataset(coords=raw.coords)
    for icon, short in DWH_PARAM_MAP.items():
        if icon in params and short in raw:
            var = raw[short]
            if short in DWH_CELSIUS_TO_KELVIN:
                var = var - ZERO_KELVIN
            elif short in DWH_HPA_TO_PA:
                var = var * 100.0
            out[icon] = var
    if want_uv and DWH_WIND_SPEED in raw and DWH_WIND_DIR in raw:
        ff = raw[DWH_WIND_SPEED]
        dd_rad = np.deg2rad(raw[DWH_WIND_DIR])
        if "U_10M" in params:
            out["U_10M"] = -ff * np.sin(dd_rad)
        if "V_10M" in params:
            out["V_10M"] = -ff * np.cos(dd_rad)

    out = out.dropna("values", how="all")
    times = np.datetime64(reftime) + np.asarray(steps, dtype="timedelta64[h]")
    return _select_valid_times(out, times, strict=True)


def load_truth_data(
    root, reftime: datetime, steps: list[int], params: list[str]
) -> xr.Dataset:
    """Load truth data from an analysis Zarr dataset or DWH observations via jretrieve."""
    if root.suffix == ".zarr":
        LOG.info("Loading ground truth from an analysis zarr dataset...")
        truth = _load_analysis_data_from_zarr(
            root=root,
            reftime=reftime,
            steps=steps,
            params=params,
        )
        truth = truth.compute().chunk(
            {"time": 1, "y": -1, "x": -1}
            if "y" in truth.dims and "x" in truth.dims
            else {"time": 1, "values": -1}
        )
    elif "jretrieve" in str(root):
        LOG.info("Loading ground truth from JRetrieve...")
        truth = load_obs_data_from_jretrieve(
            root=root,
            reftime=reftime,
            steps=steps,
            params=params,
        )
    else:
        raise ValueError(f"Unsupported truth root: {root}")
    return truth


def _load_INCA_baseline_from_netcdf(
    root: Path,
    reftime: datetime,
    steps: list[int],
    params: list[str],
    freq: str = "1h",
    fill_missing_files: bool = True,
) -> xr.Dataset:
    """Load INCA analysis/nowcast data from per-variable NetCDF files.

    Files are read from {root}/{year}/{month}/{VAR}_INCA_{YYYYmmddHHMM}.nc.
    Each INCA variable lives in a separate file and covers 6 hours from reftime.

    Args:
        root:    Base directory of the INCA archive, e.g.
                 Path("/store_new/mch/msclim/INCA"). Year and month
                 subdirectories are appended automatically.
        reftime: Reference time (forecast initialisation time). Used to locate
                 the source files and to build the output time coordinate.
        steps:   List of step indices interpreted as multiples of freq.
                 freq='1h'  : integers 0–6  (hours from reftime).
                 freq='10min': integers 0–36 (× 10 min from reftime).
                 freq='5min' : integers 0–72 (× 5 min from reftime).
        params:  List of output variable names. Supported values:

                   param     description               unit     freq       source    native   src unit  avail.from
                   --------  ------------------------  -------  ---------  --------  -------  --------  -------
                   T_2M      2 m temperature           K        1h/10min   TT        1h       °C        2022
                   TD_2M     2 m dewpoint temperature  K        1h/10min   TD        1h       °C        2022
                   TOT_PREC  total precipitation rate  kg m-2   1h/10min   RR        10min    mm/h      2022
                   TOT_PREC  total precipitation rate  kg m-2   5min       RP        5min     mm/h      2025-05
                   FF_10M    10 m wind speed           m/s      1h         FF        1h       m/s       2022
                   FF_10M    10 m wind speed           m/s      10min      FF_10min  10min    m/s       2025-05
                   DD_10M    10 m wind direction       °        1h         DD        1h       °         2022
                   DD_10M    10 m wind direction       °        10min      DD_10min  10min    °         2025-05
                   VMAX_10M  10 m wind gust            m/s      1h         WG        1h       m/s       2022
                   VMAX_10M  10 m wind gust            m/s      10min      WG_10min  10min    m/s       2025-05
                   CLCT      total cloud cover         %        1h/10min   CT        10min    %         2022
                   U_10M     10 m zonal wind           m/s      1h/10min   derived from DD_10M, FF_10M
                   V_10M     10 m meridional wind      m/s      1h/10min   derived from DD_10M, FF_10M

                 U_10M and V_10M use the meteorological convention: DD is
                 the direction the wind blows FROM, clockwise from North.
                 freq='5min' only supports TOT_PREC.
        freq:    Output time granularity: '1h' (default), '10min', or '5min'.
                 steps are interpreted as multiples of this interval.
                 Max step: 6 for '1h', 36 for '10min', 72 for '5min'.
                 freq='5min' only supports TOT_PREC (from RP, avail. since 2025-05-14).
                 At freq='10min', T_2M and TD_2M (hourly native) have NaN at
                 non-hourly timestamps.
        fill_missing_files: If True, missing files are filled with NaN arrays instead
                 of raising. Defaults to True.
    Returns:
        xr.Dataset with dimensions (step, y, x) and coordinates:
          x, y                     – Swiss CH1903 (EPSG:21781) easting/northing [m]
          latitude, longitude      – WGS84 latitude/longitude [°], shape (y, x),
                                     derived from CH1903 via pyproj
          step                     – forecast lead time (timedelta64[ns])
          valid_time               – absolute timestamps (datetime64[ns])
          forecast_reference_time  – scalar reference time (datetime64[ns])
        in case one or more variables are missing return array(s) filled with NaNs
    """
    # INCA grid in CH1903/LV03 (EPSG:21781): 1 km spacing, 710 × 640 points
    # Used only as fallback dimensions for NaN-fill arrays when a file is missing.
    _INCA_CHX = np.arange(255500, 965500, 1000, dtype=np.float64)
    _INCA_CHY = np.arange(-159500, 480500, 1000, dtype=np.float64)

    def _chxy_to_latlon(x_1d, y_1d) -> dict:
        x_2d, y_2d = np.meshgrid(x_1d, y_1d)
        lon_2d, lat_2d = Transformer.from_crs(
            "EPSG:21781", "EPSG:4326", always_xy=True
        ).transform(x_2d, y_2d)
        return {
            "latitude": (
                ("y", "x"),
                lat_2d,
                {"units": "degrees_north", "long_name": "latitude"},
            ),
            "longitude": (
                ("y", "x"),
                lon_2d,
                {"units": "degrees_east", "long_name": "longitude"},
            ),
        }

    def _nan_array(units: str) -> xr.DataArray:
        return xr.DataArray(
            np.full(
                (len(valid_times), len(_INCA_CHY), len(_INCA_CHX)),
                np.nan,
                dtype=np.float32,
            ),
            dims=["valid_time", "y", "x"],
            coords={
                "valid_time": valid_times,
                "y": _INCA_CHY,
                "x": _INCA_CHX,
                **_chxy_to_latlon(_INCA_CHX, _INCA_CHY),
            },
            attrs={"units": units},
        )

    def _open_convert(rt: datetime, pfx: str) -> tuple[Path, xr.DataArray | None]:
        """Open an INCA file and apply unit conversion.

        Returns (path, DataArray) on success, (path, None) when the file is missing.
        """
        fp = (
            root
            / f"{rt.year:04d}"
            / f"{rt.month:02d}"
            / f"{pfx}_INCA_{rt:%Y%m%d%H%M}.nc"
        )
        try:
            d = xr.open_dataset(fp, drop_variables=["grid_mapping"]).rename(
                {"chx": "x", "chy": "y", "time": "valid_time"}
            )
        except FileNotFoundError:
            return fp, None
        LOG.info("Reading %s", fp)
        da = d[pfx]
        u = da.attrs.get("units", "")
        if u == "degrees C":
            da = (da - ZERO_KELVIN).assign_attrs({**da.attrs, "units": "K"})
        elif u == "mm/h":
            da = da.assign_attrs({**da.attrs, "units": "kg m-2"})
        return fp, da

    def _load_shifted(param: str, prefix: str) -> xr.DataArray:
        """Load T_2M / TD_2M working around the INCA full-hour bug.

        For reftimes at full hours (HH:00), steps 1-N in the current run file are
        affected by a known INCA bug.  Step 0 is taken from the current reftime;
        steps 1+ are taken from the run 10 min earlier (HH-1:50), which is unaffected.
        """
        prev_rf = reftime - timedelta(minutes=10)
        LOG.info(
            "Applying INCA shifted-run workaround for %s: step 0 from %s, steps 1+ from %s",
            param,
            reftime,
            prev_rf,
        )
        parts: list[xr.DataArray] = []

        # Step 0 from current reftime file
        zero_idx = [i for i, s in enumerate(steps) if s == 0]
        if zero_idx:
            fp, da_raw = _open_convert(reftime, prefix)
            if da_raw is None:
                if not fill_missing_files:
                    raise FileNotFoundError(
                        f"INCA file not found for parameter {param!r}: {fp}"
                    )
                LOG.warning("INCA file not found, filling %s with NaN: %s", param, fp)
                parts.append(_nan_array(PARAM_UNITS[param]).isel(valid_time=zero_idx))
            else:
                parts.append(
                    da_raw.isel(valid_time=zero_idx).assign_coords(
                        valid_time=valid_times[zero_idx]
                    )
                )

        # Steps 1+ from previous reftime file (positional index = step value)
        nz_idx = [i for i, s in enumerate(steps) if s != 0]
        if nz_idx:
            nz_steps = [steps[i] for i in nz_idx]
            fp, da_raw = _open_convert(prev_rf, prefix)
            if da_raw is None:
                if not fill_missing_files:
                    raise FileNotFoundError(
                        f"INCA file not found for parameter {param!r}: {fp}"
                    )
                LOG.warning("INCA file not found, filling %s with NaN: %s", param, fp)
                parts.append(_nan_array(PARAM_UNITS[param]).isel(valid_time=nz_idx))
            else:
                parts.append(
                    da_raw.isel(valid_time=nz_steps).assign_coords(
                        valid_time=valid_times[nz_idx]
                    )
                )

        da = xr.concat(parts, dim="valid_time") if len(parts) > 1 else parts[0]
        return da.rename(param)

    # Maps output variable name -> INCA file prefix, per freq.
    # File prefix == variable name inside the NetCDF file.
    PARAM_TO_PREFIX: dict[str, dict[str, str]] = {
        "1h": {
            "T_2M": "TT",
            "TD_2M": "TD",
            "TOT_PREC": "RR",
            "FF_10M": "FF",
            "DD_10M": "DD",
            "CLCT": "CT",
            "VMAX_10M": "WG",
        },
        "10min": {
            "T_2M": "TT",
            "TD_2M": "TD",
            "TOT_PREC": "RR",
            "FF_10M": "FF_10min",
            "DD_10M": "DD_10min",
            "CLCT": "CT",
            "VMAX_10M": "WG_10min",
        },
        "5min": {
            "TOT_PREC": "RP",
        },
    }
    DERIVED_DEPS = {"U_10M": ["DD_10M", "FF_10M"], "V_10M": ["DD_10M", "FF_10M"]}
    PARAM_UNITS = {
        "T_2M": "K",
        "TD_2M": "K",
        "TOT_PREC": "kg m-2",
        "FF_10M": "m/s",
        "DD_10M": "degrees",
        "CLCT": "%",
        "VMAX_10M": "m/s",
        "U_10M": "m/s",
        "V_10M": "m/s",
    }
    FREQ_TO_TD = {
        "1h": np.timedelta64(1, "h"),
        "10min": np.timedelta64(10, "m"),
        "5min": np.timedelta64(5, "m"),
    }

    if freq not in FREQ_TO_TD:
        raise ValueError(f"freq must be '1h', '10min', or '5min', got {freq!r}")
    MAX_STEPS = {"1h": 6, "10min": 36, "5min": 72}
    if max(steps) > MAX_STEPS[freq]:
        raise ValueError(
            f"max step for freq={freq!r} is {MAX_STEPS[freq]}, got {max(steps)}"
        )
    step_td = FREQ_TO_TD[freq]
    valid_times = (np.datetime64(reftime) + np.array(steps) * step_td).astype(
        "datetime64[ns]"
    )

    prefix_map = PARAM_TO_PREFIX[freq]

    datasets: dict[str, xr.DataArray] = {}

    # Determine which native INCA variables to load
    to_load: set[str] = set()
    for param in params:
        if param in prefix_map:
            to_load.add(param)
        elif param in DERIVED_DEPS:
            deps = DERIVED_DEPS[param]
            missing = [d for d in deps if d not in prefix_map]
            if missing:
                raise ValueError(
                    f"Parameter {param!r} requires {missing} which are not "
                    f"available at freq={freq!r}"
                )
            to_load.update(deps)
        else:
            LOG.warning("INCA does not support parameter %r, filling with NaN", param)
            datasets[param] = _nan_array("unknown").rename(param)
            continue

    # T_2M and TD_2M are affected by a known INCA bug at full-hour reftimes (steps 1-6).
    # Use the run from 10 min earlier for those steps; see _load_shifted().
    _SHIFTED_PARAMS = {"T_2M", "TD_2M"}

    for param in to_load:
        prefix = prefix_map[param]

        if param in _SHIFTED_PARAMS and freq == "1h":
            datasets[param] = _load_shifted(param, prefix)
            continue

        filepath = (
            root
            / f"{reftime.year:04d}"
            / f"{reftime.month:02d}"
            / f"{prefix}_INCA_{reftime:%Y%m%d%H%M}.nc"
        )
        try:
            ds_var = xr.open_dataset(filepath, drop_variables=["grid_mapping"])
            ds_var = ds_var.rename({"chx": "x", "chy": "y", "time": "valid_time"})
        except FileNotFoundError:
            if not fill_missing_files:
                raise FileNotFoundError(
                    f"INCA file not found for parameter {param!r}: {filepath}"
                )
            LOG.warning("INCA file not found, filling %s with NaN: %s", param, filepath)
            datasets[param] = _nan_array(PARAM_UNITS[param]).rename(param)
            continue
        LOG.info("Reading %s", filepath)
        # Convert units if necessary
        da = ds_var[prefix]
        units = da.attrs.get("units", "")
        if units == "degrees C":
            da = (da - ZERO_KELVIN).assign_attrs({**da.attrs, "units": "K"})
        elif units == "mm/h":
            da = da.assign_attrs({**da.attrs, "units": "kg m-2"})
        # Reindex to the target time grid; variables coarser than freq get NaN
        # at timestamps absent from their native resolution
        datasets[param] = da.rename(param).reindex(valid_time=valid_times)

    merged = xr.merge(list(datasets.values()), join="override", compat="override")

    # Add lat/lon derived from the x/y coordinates in the loaded NetCDF files
    merged = merged.assign_coords(**_chxy_to_latlon(merged.x.values, merged.y.values))

    # Derive wind components (meteorological convention: direction wind blows FROM)
    if "U_10M" in params or "V_10M" in params:
        dd_rad = np.deg2rad(merged["DD_10M"])
        ff = merged["FF_10M"]
        if "U_10M" in params:
            merged["U_10M"] = (-ff * np.sin(dd_rad)).assign_attrs(units="m/s")
        if "V_10M" in params:
            merged["V_10M"] = (-ff * np.cos(dd_rad)).assign_attrs(units="m/s")

    # Restructure to match the earthkit GRIB engine profile: `step` is the
    # lead-time dimension, `valid_time` and `forecast_reference_time` are coords.
    ref_time_np = np.datetime64(reftime, "ns")
    lead_times = (np.array(steps) * step_td).astype("timedelta64[ns]")
    merged = merged.assign_coords(step=("valid_time", lead_times))
    merged = merged.swap_dims({"valid_time": "step"})
    merged = merged.assign_coords(forecast_reference_time=ref_time_np)

    return merged[list(params)]


def _load_icon_baseline_from_grib(
    root: Path,
    reftime: datetime,
    steps: list[int],
    params: list[str],
    member: str = "000",
) -> xr.Dataset:
    """Load an ICON-CH1-EPS or ICON-CH2-EPS baseline from the operational GRIB archive.

    `member` selects which data to load:
    - ``"mean"``: compute the average over all available ensemble members
    - ``"median"``: load the pre-computed median member file from the archive
    - ``"control"`` or ``"000"``: load the control member
    - any 3-digit string (e.g. ``"001"``…): load that specific member
    """
    if member == "control":
        member = "000"
    if member == "mean":
        member_ids = _discover_icon_member_ids(root, reftime, steps)
        LOG.info(
            "Computing ensemble mean over %d members: %s", len(member_ids), member_ids
        )
        acc = None
        n_loaded = 0
        for mid in member_ids:
            try:
                ds = _load_forecast_data_from_grib(
                    files=_collect_icon_archive_files(
                        root, reftime, steps, member_id=mid
                    ),
                    params=params,
                )
                if "number" in ds.dims:
                    ds = ds.isel(number=0, drop=True)
                acc = ds if acc is None else acc + ds
                n_loaded += 1
            except Exception as exc:
                LOG.warning("Skipping member %s: %s", mid, exc)
        if acc is None:
            raise ValueError(
                f"No ensemble members could be loaded for {reftime} from {root}"
            )
        LOG.info("Ensemble mean computed over %d members.", n_loaded)
        return acc / n_loaded
    else:
        return _load_forecast_data_from_grib(
            files=_collect_icon_archive_files(root, reftime, steps, member_id=member),
            params=params,
        )


def _disaggregated_and_derived_params(
    ds: xr.Dataset, steps: list[int], params: list[str]
) -> xr.Dataset:
    """Apply disaggregation and derived-param computation to a raw-loaded dataset.

    Expects ds to contain cumulative-from-start fields for any accumulatable base
    params (e.g. TOT_PREC) and all component fields for spatial derived params.
    Returns a dataset with only the originally requested params at the originally
    requested steps.
    """
    load_steps = get_steps(steps, params)

    # IC synthesis + disaggregation for suffixed aggregated params (TOT_PREC6 etc.)
    aggregated = {
        p: parse_aggregated_param(p)
        for p in params
        if parse_aggregated_param(p)[1] is not None
    }
    base_params_seen: set[str] = set()
    for agg_param, (base, n) in aggregated.items():
        if base in ds.data_vars:
            if base not in base_params_seen:
                ds[base] = _ensure_accum_ic(ds[base], load_steps)
                base_params_seen.add(base)
            ds[agg_param] = _disaggregate_accum(ds[base], steps, n)

    # Select only the originally requested steps (drop preceding helper steps)
    if load_steps != list(steps) and "step" in ds.dims:
        ds = ds.sel(step=[np.timedelta64(s, "h") for s in steps])

    # Spatial derived params
    for p in params:
        if p in DERIVED_PARAMS:
            ds = ds.assign({p: compute_derived(ds, p)})

    # Drop base params not explicitly requested by the caller
    for base, _ in aggregated.values():
        if base not in params and base in ds.data_vars:
            ds = ds.drop_vars(base)
    for p in params:
        if p in DERIVED_PARAMS:
            for comp in DERIVED_PARAMS[p]:
                if comp not in params:
                    ds = ds.drop_vars(comp, errors="ignore")

    return ds


def load_forecast_data(
    root, reftime: datetime, steps: list[int], params: list[str], member: str = "000"
) -> xr.Dataset:
    """Load forecast data from GRIB files or an ICON archive.

    Handles derived and aggregated params transparently:
    - ``SP_10M`` is computed from ``U_10M`` / ``V_10M``
    - ``TOT_PREC6`` is disaggregated from the cumulative ``TOT_PREC`` field
    - Plain ``TOT_PREC`` is returned as cumulative-from-start without disaggregation

    Routing (in order):
    1. ``*.grib`` files present in *root* → :func:`_load_forecast_data_from_grib`
       (ML inference output)
    2. ``INCA`` in path parts → :func:`_load_INCA_baseline_from_netcdf`
    3. Otherwise → ICON operational archive (via :func:`_load_icon_baseline_from_grib`)
    """
    root = Path(root)
    load_params = get_base_params(params)
    load_steps = get_steps(steps, params)
    if any(root.glob("*.grib")):
        LOG.info("Loading forecasts from GRIB files...")
        ds = _load_forecast_data_from_grib(
            files=_collect_ml_grib_files(root, load_steps),
            params=load_params,
        )
    elif "INCA" in root.parts:
        LOG.info("Loading INCA baseline from NetCDF files...")
        ds = _load_INCA_baseline_from_netcdf(root, reftime, load_steps, load_params)
    else:
        LOG.info("Loading baseline forecasts from ICON GRIB archive...")
        ds = _load_icon_baseline_from_grib(
            root, reftime, load_steps, load_params, member=member
        )
    return _disaggregated_and_derived_params(ds, steps, params)
