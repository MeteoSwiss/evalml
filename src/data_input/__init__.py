import yaml
import logging
from datetime import datetime, timedelta
from pathlib import Path
from functools import lru_cache

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
import earthkit.data as ekd  # noqa: E402

LOG = logging.getLogger(__name__)

# IFS shortNames that COSMO eccodes definitions don't remap to COSMO names.
# These need explicit aliasing so callers can use COSMO names consistently.
_IFS_TO_COSMO = {
    "tp": "TOT_PREC",
    "msl": "PMSL",
    "10u": "U_10M",
    "10v": "V_10M",
    "2t": "T_2M",
    "2d": "TD_2M",
    "sp": "PS",
    "lsm": "FR_LAND",
    "z": "FIS",
}
_COSMO_TO_IFS = {v: k for k, v in _IFS_TO_COSMO.items()}


@lru_cache(maxsize=1)
def earthkit_xarray_engine_profile() -> dict:
    fn = Path(__file__).parent / "profile.yaml"
    with open(fn) as f:
        profile = yaml.safe_load(f)
    return profile


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
    USE_IFS_NAMES = {"-co2-", "-ea-"}
    PARAMS_MAP = (
        PARAMS_MAP_COSMO2
        if any(tag in root.name for tag in USE_IFS_NAMES)
        else PARAMS_MAP_COSMO1
    )

    ds = xr.open_zarr(root, consolidated=False)

    # rename "dates" to "time" and set it as index
    ds = ds.set_index(time="dates")

    # set 'variables' attr as dimension coordinate
    ds = ds.assign_coords({"variable": ds.attrs["variables"]})

    # select variables and valid time, squeeze ensemble dimension
    ds = ds.sel(variable=[PARAMS_MAP[p] for p in params]).squeeze("ensemble", drop=True)

    # recover original 2D shape
    if "field_shape" in ds.attrs and len(ds.attrs["field_shape"]) == 2:
        ny, nx = ds.attrs["field_shape"]
        y_idx, x_idx = np.unravel_index(np.arange(ny * nx), shape=(ny, nx))
        ds = ds.assign_coords({"y": ("cell", y_idx), "x": ("cell", x_idx)})
        ds = ds.set_index(cell=("y", "x"))
        ds = ds.unstack("cell")

    # set lat lon as coords (optional)
    if "latitudes" in ds and "longitudes" in ds:
        ds = ds.rename({"latitudes": "lat", "longitudes": "lon"})
    if "latitude" in ds and "longitude" in ds:
        ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    if "lat" in ds and "lon" in ds:
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

    profile = earthkit_xarray_engine_profile()
    LOG.debug(f"loading GRIB for params {params} and steps {steps} from {root}")
    # Extend param selection to include IFS aliases (e.g. "tp" for "TOT_PREC") so
    # that both COSMO-named and IFS-named GRIB files (global models) are handled.
    params_sel = list(
        {p for p in params} | {_COSMO_TO_IFS[p] for p in params if p in _COSMO_TO_IFS}
    )
    # Precipitation params don't have a step=0 field (accumulation is zero at
    # analysis time and is often not written), so loading them together with
    # other variables causes an inconsistent-step error in earthkit-data.
    # Load them separately with step>0, then merge.
    _PREC_PARAMS = {"tp", "TOT_PREC"}
    prec_params = [p for p in params_sel if p in _PREC_PARAMS]
    other_params = [p for p in params_sel if p not in _PREC_PARAMS]
    fieldlist = ekd.from_source("file", files)
    datasets = []
    if other_params:
        datasets.append(
            fieldlist.sel(param=other_params, step=steps).to_xarray(profile=profile)
        )
    if prec_params:
        prec_steps = [s for s in steps if s > 0]
        datasets.append(
            fieldlist.sel(param=prec_params, step=prec_steps).to_xarray(profile=profile)
        )
    ds: xr.Dataset = (
        xr.merge(datasets, join="outer") if len(datasets) > 1 else datasets[0]
    )
    # Rename any IFS names back to COSMO names
    ifs_rename = {
        ifs: cosmo for ifs, cosmo in _IFS_TO_COSMO.items() if ifs in ds.data_vars
    }
    if ifs_rename:
        ds = ds.rename(ifs_rename)

    for var, da in ds.data_vars.items():
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

    # set lat lon as coords (optional)
    if "latitudes" in ds and "longitudes" in ds:
        ds = ds.rename({"latitudes": "lat", "longitudes": "lon"})
    if "latitude" in ds and "longitude" in ds:
        ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    if "lat" in ds and "lon" in ds:
        ds = ds.set_coords(["lat", "lon"])
        
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
        LOG.info(f"Loading ground truth from {root}")
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
        LOG.info(f"Loading ground truth from {root}")
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
        LOG.info(f"Loading forecasts from GRIB files from {root}")
        fcst = load_fct_data_from_grib(
            root=root,
            reftime=reftime,
            steps=steps,
            params=params,
        )
    else:
        LOG.info(f"Loading baseline forecasts from zarr dataset from {root}")
        fcst = load_baseline_from_zarr(
            root=root,
            reftime=reftime,
            steps=steps,
            params=params,
        )

    return fcst
