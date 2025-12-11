import yaml
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable
from functools import lru_cache

eccodes_definition_path = Path(sys.prefix) / "share/eccodes-cosmo-resources/definitions"
os.environ["ECCODES_DEFINITION_PATH"] = str(eccodes_definition_path)

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
import earthkit.data as ekd  # noqa: E402

LOG = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def earthkit_xarray_engine_profile() -> dict:
    fn = Path(__file__).parent / "profile.yaml"
    with open(fn) as f:
        profile = yaml.safe_load(f)
    return profile


def load_analysis_data_from_zarr(
    analysis_zarr: Path, times: Iterable[datetime], params: list[str]
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
    PARAMS_MAP = PARAMS_MAP_COSMO2 if "co2" in analysis_zarr.name else PARAMS_MAP_COSMO1

    ds = xr.open_zarr(analysis_zarr, consolidated=False)

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
        ds = ds.rename({"latitudes": "latitude", "longitudes": "longitude"})
    ds = ds.set_coords(["latitude", "longitude"])
    ds = (
        ds["data"]
        .to_dataset("variable")
        .rename({v: k for k, v in PARAMS_MAP.items() if v in ds["variable"].values})
    )

    # select valid times
    # (handle special case where some valid times are not in the dataset, e.g. at the end)
    times_included = times.isin(ds.time.values).values
    if all(times_included):
        ds = ds.sel(time=times)
    elif np.sum(times_included) < len(times_included):
        LOG.warning(
            "Some valid times are not included in the dataset: \n%s",
            times[~times_included].values,
        )
        ds = ds.sel(time=times[times_included])
    else:
        raise ValueError(
            "Valid times are not included in the dataset. "
            "Please check the valid times and the dataset."
        )
    return ds


def load_fct_data_from_grib(
    grib_output_dir: Path, reftime: datetime, steps: list[int], params: list[str]
) -> xr.Dataset:
    """Load forecast data from GRIB files for a specific valid time."""
    files = sorted(grib_output_dir.glob("20*.grib"))

    profile = earthkit_xarray_engine_profile()
    ds: xr.Dataset = (
        ekd.from_source("file", files)
        .sel(param=params, step=steps)
        .to_xarray(profile=profile)
    )
    # fds = data_source.FileDataSource(datafiles=files)
    # ds = grib_decoder.load(fds, {"param": params, "step": steps})
    for var, da in ds.data_vars.items():
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
    return ds


def load_baseline_from_zarr(
    zarr_path: Path, reftime: datetime, steps: list[int], params: list[str]
) -> xr.Dataset:
    """Load forecast data from a Zarr dataset."""
    try:
        baseline = xr.open_zarr(zarr_path, consolidated=True, decode_timedelta=True)
    except ValueError:
        raise ValueError(f"Could not open baseline zarr at {zarr_path}")

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
    return baseline
