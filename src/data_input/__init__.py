import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

eccodes_definition_path = Path(sys.prefix) / "share/eccodes-cosmo-resources/definitions"
os.environ["ECCODES_DEFINITION_PATH"] = str(eccodes_definition_path)

from meteodatalab import data_source, grib_decoder  # noqa: E402

import numpy as np  # noqa: E402
import pyproj  # noqa: E402
import xarray as xr  # noqa: E402

LOG = logging.getLogger(__name__)

# INCA grid in CH1903/LV03 (EPSG:21781): 1 km spacing, 710 × 640 points
_INCA_CHX = np.arange(255500, 965500, 1000, dtype=np.float64)
_INCA_CHY = np.arange(-159500, 480500, 1000, dtype=np.float64)


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
    ds["TOT_PREC"] = ds["TOT_PREC"] * 1000  # convert precipitation units from m to mm

    # rename 'cell' dimension to 'values' (it's earthkit-data default for flattened spatial dim)
    if "cell" in ds.dims:
        ds = ds.rename({"cell": "values"})

    times = np.datetime64(reftime) + np.asarray(steps, dtype="timedelta64[h]")
    return _select_valid_times(ds, times)


def load_fct_data_from_grib(
    root: Path, reftime: datetime, steps: list[int], params: list[str]
) -> xr.Dataset:
    """Load forecast data from GRIB files for a specific valid time."""
    files = sorted(root.glob(f"{reftime:%Y%m%d%H%M}*.grib"))
    fds = data_source.FileDataSource(datafiles=files)
    ds = grib_decoder.load(fds, {"param": params, "step": steps})
    for var, da in ds.items():
        if "z" in da.dims and da.sizes["z"] == 1:
            ds[var] = da.squeeze("z", drop=True)
        elif "z" in da.dims and da.sizes["z"] > 1:
            ds[var] = da.rename({"z": da.attrs["vcoord_type"]})
    ds = xr.merge([ds[p].rename(p) for p in ds], compat="no_conflicts")
    lead_times = np.array(steps, dtype="timedelta64[h]")
    # Restrict to the requested lead times so that the TOT_PREC disaggregation
    # below operates on the correct step interval even if the GRIB contains
    # extra (e.g. hourly) steps beyond those requested — e.g. when consuming
    # output from an interpolator emulator or a baseline with sub-step output.
    ds = ds.sel(lead_time=lead_times)
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
    ds = ds.sel(ref_time=reftime)

    # rename 'cell' dimension to 'values' (it's earthkit-data default for flattened spatial dim)
    if "cell" in ds.dims:
        ds = ds.rename({"cell": "values"})
    return ds


def load_baseline_from_zarr(
    root: Path, reftime: datetime, steps: list[int], params: list[str]
) -> xr.Dataset:
    """Load forecast data from a Zarr dataset."""
    try:
        baseline = xr.open_zarr(root, consolidated=True, decode_timedelta=True)
    except ValueError:
        raise ValueError(f"Could not open baseline zarr at {root}")

    baseline = baseline.rename(
        {"forecast_reference_time": "ref_time", "step": "lead_time"}
    ).sortby("lead_time")
    lead_times = np.array(steps, dtype="timedelta64[h]")
    # Restrict to the requested lead times up-front so that the TOT_PREC
    # disaggregation below operates on the correct step interval, and so that
    # all other variables avoid loading unused hourly steps from the zarr.
    baseline = baseline[params].sel(ref_time=reftime, lead_time=lead_times)
    if "TOT_PREC" in baseline.data_vars:
        if baseline.TOT_PREC.units == "m":
            baseline = baseline.assign(TOT_PREC=lambda x: x.TOT_PREC * 1000)
            baseline.TOT_PREC.attrs["units"] = "kg m-2"
        ## Disaggregate TOT_PREC from cumulative-from-start (the expected zarr
        ## convention for processed NWP output) to per-step accumulations.
        ##
        ## Sanity-check that the incoming data is actually cumulative: if
        ## .diff() produces significantly negative values, TOT_PREC is already
        ## period-accumulated and a second disaggregation would produce
        ## garbage. In that case raise — we always expect cumulative-from-
        ## start precipitation here.
        diff = baseline.TOT_PREC.diff("lead_time")
        min_diff = float(diff.min().compute())
        if min_diff < -0.1:  # TOT_PREC canonical units are mm
            raise ValueError(
                f"TOT_PREC in the baseline zarr appears to already be "
                f"period-accumulated (min(.diff()) = {min_diff:.3e} m)."
            )
        ## .diff() drops lead_time=0; .reindex() restores it as NaN (no
        ## accumulation period exists at the forecast initial time). Clip
        ## small float-noise negatives to zero (anything below -0.1 mm has
        ## already been caught by the check above).
        baseline = baseline.assign(
            TOT_PREC=diff.clip(min=0.0).reindex(lead_time=lead_times)
        )
    baseline = baseline.assign_coords(time=baseline.ref_time + baseline.lead_time)
    if "latitude" in baseline.coords and "longitude" in baseline:
        baseline = baseline.rename({"latitude": "lat", "longitude": "lon"})
    return baseline


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
        ds["T_2M"] = ds["T_2M"] + 273.15  # convert to Kelvin
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


def load_forecast_data(
    root, reftime: datetime, steps: list[int], params: list[str]
) -> xr.Dataset:
    """Load forecast data from GRIB files or a baseline Zarr dataset."""

    if any(root.glob("*.grib")):
        LOG.info("Loading forecasts from GRIB files...")
        fcst = load_fct_data_from_grib(
            root=root,
            reftime=reftime,
            steps=steps,
            params=params,
        )
    else:
        LOG.info("Loading baseline forecasts from zarr dataset...")
        fcst = load_baseline_from_zarr(
            root=root,
            reftime=reftime,
            steps=steps,
            params=params,
        )

    return fcst


def load_INCA_baseline_from_netcdf(
    root: Path,
    reftime: datetime,
    steps: list[int],
    params: list[str],
    freq: str = "1h",
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

    Returns:
        xr.Dataset with dimensions (time, chy, chx) and coordinates:
          chx, chy  – Swiss CH1903 (EPSG:21781) easting/northing [m]
          lat, lon  – WGS84 latitude/longitude [°], shape (chy, chx),
                      derived from CH1903 via pyproj
          time      – absolute timestamps (datetime64[ns]).
        in case one or more variables are missing return array(s) filled with NaNs
    """
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
            raise ValueError(f"INCA baseline does not support parameter: {param}")

    # Precompute lat/lon from the canonical INCA grid (no native file required)
    transformer = pyproj.Transformer.from_crs("EPSG:21781", "EPSG:4326", always_xy=True)
    _chx_2d, _chy_2d = np.meshgrid(_INCA_CHX, _INCA_CHY)
    _lon_2d, _lat_2d = transformer.transform(_chx_2d, _chy_2d)
    latlon_coords = {
        "lat": (
            ("chy", "chx"),
            _lat_2d,
            {"units": "degrees_north", "long_name": "latitude"},
        ),
        "lon": (
            ("chy", "chx"),
            _lon_2d,
            {"units": "degrees_east", "long_name": "longitude"},
        ),
    }

    # load parameter by parameter
    datasets: dict[str, xr.DataArray] = {}
    for param in to_load:
        prefix = prefix_map[param]
        filepath = (
            root
            / f"{reftime.year:04d}"
            / f"{reftime.month:02d}"
            / f"{prefix}_INCA_{reftime:%Y%m%d%H%M}.nc"
        )
        try:
            ds_var = xr.open_dataset(filepath, drop_variables=["grid_mapping"])
        except FileNotFoundError:
            LOG.warning("INCA file not found, filling %s with NaN: %s", param, filepath)
            datasets[param] = xr.DataArray(
                np.full(
                    (len(valid_times), len(_INCA_CHY), len(_INCA_CHX)),
                    np.nan,
                    dtype=np.float32,
                ),
                dims=["time", "chy", "chx"],
                coords={
                    "time": valid_times,
                    "chy": _INCA_CHY,
                    "chx": _INCA_CHX,
                    **latlon_coords,
                },
                attrs={"units": PARAM_UNITS[param]},
                name=param,
            )
            continue
        # Convert units if necessary
        da = ds_var[prefix]
        units = da.attrs.get("units", "")
        if units == "degrees C":
            da = (da + 273.15).assign_attrs({**da.attrs, "units": "K"})
        elif units == "mm/h":
            da = da.assign_attrs({**da.attrs, "units": "kg m-2"})
        # Reindex to the target time grid; variables coarser than freq get NaN
        # at timestamps absent from their native resolution
        datasets[param] = da.rename(param).reindex(time=valid_times)

    merged = xr.merge(list(datasets.values()), join="override")

    # Add lat/lon derived from the canonical INCA grid
    merged = merged.assign_coords(**latlon_coords)

    # Derive wind components (meteorological convention: direction wind blows FROM)
    if "U_10M" in params or "V_10M" in params:
        dd_rad = np.deg2rad(merged["DD_10M"])
        ff = merged["FF_10M"]
        if "U_10M" in params:
            merged["U_10M"] = (-ff * np.sin(dd_rad)).assign_attrs(units="m/s")
        if "V_10M" in params:
            merged["V_10M"] = (-ff * np.cos(dd_rad)).assign_attrs(units="m/s")

    return merged[list(params)]
