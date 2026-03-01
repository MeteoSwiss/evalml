import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

eccodes_definition_path = Path(sys.prefix) / "share/eccodes-cosmo-resources/definitions"
os.environ["ECCODES_DEFINITION_PATH"] = str(eccodes_definition_path)

from meteodatalab import data_source, grib_decoder  # noqa: E402

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

LOG = logging.getLogger(__name__)


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
    PARAMS_MAP_COSMO1 = {
        v: v.replace("TOT_PREC", "TOT_PREC_6H") for v in PARAMS_MAP_COSMO2.keys()
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
    if "TOT_PREC" in ds.data_vars:
        LOG.info("Disaggregating precipitation")
        ds = ds.assign(
            TOT_PREC=lambda x: (
                x.TOT_PREC.fillna(0)
                .diff("lead_time")
                .pad(lead_time=(1, 0), constant_value=None)
                .clip(min=0.0)
            )
        )
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
    if "TOT_PREC" in baseline.data_vars:
        if baseline.TOT_PREC.units == "kg m-2":
            baseline = baseline.assign(TOT_PREC=lambda x: x.TOT_PREC / 1000)
            baseline.TOT_PREC.attrs["units"] = "m"
        ## disaggregate precipitation
        baseline = baseline.assign(
            TOT_PREC=lambda x: (
                x.TOT_PREC.fillna(0)
                .diff("lead_time")
                .pad(lead_time=(1, 0), constant_value=None)
                .clip(min=0.0)
            )
        )
    baseline = baseline[params].sel(
        ref_time=reftime,
        lead_time=np.array(steps, dtype="timedelta64[h]"),
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
