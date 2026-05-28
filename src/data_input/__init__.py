import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

eccodes_definition_path = Path(sys.prefix) / "share/eccodes-cosmo-resources/definitions"
os.environ["ECCODES_DEFINITION_PATH"] = str(eccodes_definition_path)

from pyproj import Transformer  # noqa: E402
from meteodatalab import data_source, grib_decoder  # noqa: E402

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

LOG = logging.getLogger(__name__)

ZERO_KELVIN = -273.15  # °C


def _select_valid_times(ds, times: np.datetime64):
    # (handle special case where some valid times are not in the dataset, e.g. at the end)
    times_np = np.asarray(times, dtype="datetime64[ns]")
    times_included = np.isin(times_np, ds.time.values)
    if times_included.all():
        return ds.sel(time=times_np)
    elif times_included.any():
        LOG.warning(
            "Some valid times are not included in the dataset: \n%s",
            times_np[~times_included],
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


def load_analysis_data_from_zarr(
    root: Path, reftime: datetime, steps: list[int], params: list[str]
) -> xr.Dataset:
    """Load analysis data from an anemoi-generated Zarr dataset

    This function loads analysis data from a Zarr dataset, processing it to make it more
    xarray-friendly. It renames variables, sets the time index, and pivots the dataset.
    """
    PARAMS_MAP_COSMO2 = {
        "T_2M": "2t",
        "TD_2M": "2d",
        "U_10M": "10u",
        "V_10M": "10v",
        "PS": "sp",
        "PMSL": "msl",
        "TOT_PREC": "tp",
    }
    tot_prec_string = "TOT_PREC_6H" if min(np.diff(steps)) == 6 else "TOT_PREC_1H"
    PARAMS_MAP_COSMO1 = {
        v: v.replace("TOT_PREC", tot_prec_string) for v in PARAMS_MAP_COSMO2.keys()
    }
    PARAMS_MAP = PARAMS_MAP_COSMO2 if "co2" in root.name else PARAMS_MAP_COSMO1

    ds = xr.open_zarr(root, consolidated=False)

    # rename "dates" to "time" and set it as index
    ds = ds.set_index(time="dates")

    # set 'variables' attr as dimension coordinate
    ds = ds.assign_coords({"variable": ds.attrs["variables"]})

    # select variables and valid time, squeeze ensemble dimension
    ds = ds.sel(variable=[PARAMS_MAP[p] for p in params]).squeeze("ensemble", drop=True)

    # recover original 2D shape
    if len(ds.attrs["field_shape"]) == 2:
        ny, nx = ds.attrs["field_shape"]
        y_idx, x_idx = np.unravel_index(np.arange(ny * nx), shape=(ny, nx))
        ds = ds.assign_coords({"y": ("cell", y_idx), "x": ("cell", x_idx)})
        ds = ds.set_index(cell=("y", "x"))
        ds = ds.unstack("cell")

    # set lat lon as coords (optional)
    if "latitudes" in ds and "longitudes" in ds:
        ds = ds.rename({"latitudes": "lat", "longitudes": "lon"})
    ds = ds.set_coords(["lat", "lon"])
    ds = (
        ds["data"]
        .to_dataset("variable")
        .rename({v: k for k, v in PARAMS_MAP.items() if v in ds["variable"].values})
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


def _collect_ml_grib_files(
    root: Path, reftime: datetime, steps: list[int] | None = None
) -> list[Path]:
    """Return GRIB files for an ML inference run (flat directory layout).

    When `steps` is provided, the discovered files are filtered to those whose
    name ends with ``_{step:03d}.grib``.
    """
    files = sorted(root.glob(f"{reftime:%Y%m%d%H%M}*.grib"))
    if steps is None:
        return files
    suffixes = {f"_{step:03d}.grib" for step in steps}
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
    LOG.info("Reading ICON archive from %s", reftime_dir)

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


def load_forecast_data_from_grib(files: list[Path], params: list[str]) -> xr.Dataset:
    """Load forecast data from a list of GRIB files."""
    fds = data_source.FileDataSource(datafiles=files)
    ds = grib_decoder.load(fds, {"param": params})
    for var, da in ds.items():
        if "z" in da.dims and da.sizes["z"] == 1:
            ds[var] = da.squeeze("z", drop=True)
        elif "z" in da.dims and da.sizes["z"] > 1:
            ds[var] = da.rename({"z": da.attrs["vcoord_type"]})
    ds = xr.merge([ds[p].rename(p) for p in ds], compat="no_conflicts")
    lead_times = ds.lead_time.values
    if "TOT_PREC" in ds.data_vars:
        ## Disaggregate TOT_PREC from cumulative-from-start (expected when the
        ## accumulate_from_start_of_forecast post-processor is enabled in
        ## anemoi-inference) to per-step accumulations.
        ##
        ## anemoi-inference sometimes omits step 0 from the GRIB even with
        ## accumulate_from_start_of_forecast enabled. After the outer-join
        ## merge above, lead_time=0 of TOT_PREC is then NaN, which would
        ## propagate through .diff() and wipe out the first period
        ## accumulation. Set it explicitly to 0 (cumulative-from-start has
        ## nothing accumulated at the forecast initial time by definition).
        ## Restricting to lead_time=0 leaves any other NaNs (e.g. from
        ## boundary-trim masks) untouched.
        ds = ds.assign(
            TOT_PREC=xr.where(
                ds.lead_time == np.timedelta64(0, "h"),
                0.0,
                ds.TOT_PREC,
            )
        )
        ## Sanity-check that the incoming data is actually cumulative: if
        ## .diff() produces significantly negative values, TOT_PREC is already
        ## period-accumulated and a second disaggregation would produce
        ## garbage. In that case raise — we always expect cumulative-from-
        ## start precipitation here.
        diff = ds.TOT_PREC.diff("lead_time")
        min_diff = float(diff.min().compute())
        if min_diff < -0.1:  # TOT_PREC canonical units are mm
            raise ValueError(
                f"TOT_PREC in the GRIB appears to already be "
                f"period-accumulated (min(.diff()) = {min_diff:.3e} m). "
                f"Check that the accumulate_from_start_of_forecast "
                f"post-processor is enabled in the anemoi-inference config "
                f"for this source."
            )
        ## .diff() drops lead_time=0; .reindex() restores it as NaN (no
        ## accumulation period exists at the forecast initial time). Clip
        ## small float-noise negatives to zero (anything below -0.1 mm has
        ## already been caught by the check above).
        ds = ds.assign(TOT_PREC=diff.clip(min=0.0).reindex(lead_time=lead_times))
    # make sure time coordinate is available, and valid_time is not
    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    if "time" not in ds.coords:
        ds = ds.assign_coords(time=ds.ref_time + ds.lead_time)
    ds = ds.squeeze("ref_time", drop=False)

    # rename 'cell' dimension to 'values' (it's earthkit-data default for flattened spatial dim)
    if "cell" in ds.dims:
        ds = ds.rename({"cell": "values"})
    return ds


def load_obs_data_from_peakweather(
    root, reftime: datetime, steps: list[int], params: list[str], freq: str = "1h"
) -> xr.Dataset:
    """Load PeakWeather station observations into an xarray Dataset.

    Returns a Dataset with dimensions `time` and `values`, values coordinates
    (`lat`, `lon`), and variables renamed to ICON parameter names.
    Temperatures are converted to Kelvin when present.
    """
    from peakweather.dataset import PeakWeatherDataset

    param_names = {
        "temperature": "T_2M",
        "wind_u": "U_10M",
        "wind_v": "V_10M",
        "precipitation": "TOT_PREC",
        "pressure": "PS",
        "wind_gust": "VMAX_10M",
    }
    param_names = {k: v for k, v in param_names.items() if v in params}

    start = reftime
    end = start + timedelta(hours=max(steps))
    if len(steps) > 1:
        end += timedelta(hours=steps[-1] - steps[-2])  # extend by 1 extra step
    years = list(set([start.year, end.year]))
    pw = PeakWeatherDataset(root=root, years=years, freq=freq)
    ds, mask = pw.get_observations(
        parameters=[k for k in param_names.keys()],
        first_date=f"{start:%Y-%m-%d %H:%M}",
        last_date=f"{end:%Y-%m-%d %H:%M}",
        return_mask=True,
    )
    ds = (
        ds.stack(["nat_abbr", "name"], future_stack=True)
        .to_xarray()
        .to_dataset(dim="name")
    )
    mask = (
        mask.stack(["nat_abbr", "name"], future_stack=True)
        .to_xarray()
        .to_dataset(dim="name")
    )
    ds = ds.where(mask)
    ds = ds.rename({"datetime": "time", "nat_abbr": "values"})
    ds = ds.rename(param_names)
    ds = ds.assign_coords(time=ds.indexes["time"].tz_convert("UTC").tz_localize(None))
    ds = ds.assign_coords(values=ds.indexes["values"])
    ds = ds.assign_coords(lon=("values", pw.stations_table["longitude"]))
    ds = ds.assign_coords(lat=("values", pw.stations_table["latitude"]))
    if "T_2M" in ds:
        ds["T_2M"] = ds["T_2M"] - ZERO_KELVIN  # convert to Kelvin
    ds = ds.dropna("values", how="all")

    times = np.datetime64(reftime) + np.asarray(steps, dtype="timedelta64[h]")
    return _select_valid_times(ds, times)


def load_truth_data(
    root, reftime: datetime, steps: list[int], params: list[str]
) -> xr.Dataset:
    """Load truth data from analysis Zarr or PeakWeather observations."""
    if root.suffix == ".zarr":
        LOG.info("Loading ground truth from an analysis zarr dataset...")
        truth = load_analysis_data_from_zarr(
            root=root,
            reftime=reftime,
            steps=steps,
            params=params,
        )
        truth = truth.compute().chunk(
            {"y": -1, "x": -1}
            if "y" in truth.dims and "x" in truth.dims
            else {"values": -1}
        )
    elif "peakweather" in str(root):
        LOG.info("Loading ground truth from PeakWeather observations...")
        truth = load_obs_data_from_peakweather(
            root=root,
            reftime=reftime,
            steps=steps,
            params=params,
        )
    else:
        raise ValueError(f"Unsupported truth root: {root}")
    return truth


def load_INCA_baseline_from_netcdf(
    root: Path,
    reftime: datetime,
    steps: list[int],
    params: list[str],
    freq: str = "1h",
    fill_missing_files: bool = True,
) -> xr.Dataset:
    """Load INCA analysis/nowcast data from per-variable NetCDF files.

    Files are read from {root}/{year}/{month}/{VAR}_INCA_{YYYYmmddHHMM}.nc.
    Each INCA variable lives in a separate file and covers 6 hours from ref_time.

    Args:
        root:    Base directory of the INCA archive, e.g.
                 Path("/store_new/mch/msclim/INCA"). Year and month
                 subdirectories are appended automatically.
        reftime: Reference time (forecast initialisation time). Used to locate
                 the source files and to build the output time coordinate.
        steps:   List of step indices interpreted as multiples of freq.
                 freq='1h'  : integers 0–6  (hours from ref_time).
                 freq='10min': integers 0–36 (× 10 min from ref_time).
                 freq='5min' : integers 0–72 (× 5 min from ref_time).
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
        xr.Dataset with dimensions (time, y, x) and coordinates:
          x, y  – Swiss CH1903 (EPSG:21781) easting/northing [m]
          lat, lon  – WGS84 latitude/longitude [°], shape (y, x),
                      derived from CH1903 via pyproj
          time      – absolute timestamps (datetime64[ns]).
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
            "lat": (
                ("y", "x"),
                lat_2d,
                {"units": "degrees_north", "long_name": "latitude"},
            ),
            "lon": (
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
            dims=["time", "y", "x"],
            coords={
                "time": valid_times,
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
                {"chx": "x", "chy": "y"}
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
                parts.append(_nan_array(PARAM_UNITS[param]).isel(time=zero_idx))
            else:
                parts.append(
                    da_raw.isel(time=zero_idx).assign_coords(time=valid_times[zero_idx])
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
                parts.append(_nan_array(PARAM_UNITS[param]).isel(time=nz_idx))
            else:
                parts.append(
                    da_raw.isel(time=nz_steps).assign_coords(time=valid_times[nz_idx])
                )

        da = xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]
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
            ds_var = ds_var.rename({"chx": "x", "chy": "y"})
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
        datasets[param] = da.rename(param).reindex(time=valid_times)

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

    # Restructure to match load_forecast_data_from_grib: lead_time is the dimension,
    # time (valid_time) and ref_time (scalar) are coordinates.
    ref_time_np = np.datetime64(reftime, "ns")
    lead_times = (np.array(steps) * step_td).astype("timedelta64[ns]")
    merged = merged.assign_coords(lead_time=("time", lead_times))
    merged = merged.swap_dims({"time": "lead_time"})
    merged = merged.assign_coords(ref_time=ref_time_np)

    return merged[list(params)]


def load_icon_baseline_from_grib(
    root: Path,
    reftime: datetime,
    steps: list[int],
    params: list[str],
) -> xr.Dataset:
    """Load an ICON-CH1-EPS or ICON-CH2-EPS baseline from the operational GRIB archive."""
    return load_forecast_data_from_grib(
        files=_collect_icon_archive_files(root, reftime, steps),
        params=params,
    )


def load_forecast_data(
    root, reftime: datetime, steps: list[int], params: list[str]
) -> xr.Dataset:
    """Load forecast data from GRIB files or an ICON archive.

    Routing (in order):
    1. ``*.grib`` files present in *root* → :func:`load_forecast_data_from_grib`
       (ML inference output)
    2. ``INCA`` in path parts → :func:`load_INCA_baseline_from_netcdf`
    3. Otherwise → ICON operational archive (via :func:`load_icon_baseline_from_grib`)
    """
    root = Path(root)
    if any(root.glob("*.grib")):
        LOG.info("Loading forecasts from GRIB files...")
        return load_forecast_data_from_grib(
            files=_collect_ml_grib_files(root, reftime, steps),
            params=params,
        )
    if "INCA" in root.parts:
        LOG.info("Loading INCA baseline from NetCDF files...")
        return load_INCA_baseline_from_netcdf(root, reftime, steps, params)
    LOG.info("Loading baseline forecasts from ICON GRIB archive...")
    return load_icon_baseline_from_grib(root, reftime, steps, params)
