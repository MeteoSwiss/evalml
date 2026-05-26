import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Any

import earthkit.data as ekd
import numpy as np
import xarray as xr

LOG = logging.getLogger(__name__)

XARRAY_ENGINE_PROFILE = {
    "ensure_dims": ["z", "number", "step", "forecast_reference_time"],
    "add_valid_time_coord": True,
    "global_attrs": [{"institution": "MeteoSwiss"}, {"Conventions": "CF-1.8"}],
}


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
        ds = ds.rename({"latitudes": "latitude", "longitudes": "longitude"})
    ds = ds.set_coords(["latitude", "longitude"])
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

    `root` is the FCST<year> directory, e.g.
    ``/store_new/mch/msopr/osm/ICON-CH1-EPS/FCST25``.
    """
    reftime_dirs = sorted(root.glob(f"{reftime:%y%m%d%H}_*"))
    if not reftime_dirs:
        raise ValueError(
            f"No archive subdirectory found for {reftime:%y%m%d%H} in {root}"
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


def load_from_grib_file(file: str | list[str], sel_kwargs):
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
    for level_type_group in fieldlist.group_by("vertical.level_type"):
        # earthkit-data should return the group key...TODO: open issue?
        level_type = level_type_group.get("vertical.level_type")[0]
        profile = XARRAY_ENGINE_PROFILE | variable_name_profile(level_type)
        _ds = level_type_group.to_xarray(**profile, allow_holes=True)
        ds = ds.merge(_ds, compat="no_conflicts", combine_attrs="no_conflicts")
    return ds


def _tot_prec_handling(tp: xr.DataArray) -> xr.DataArray:
    _full_step_coord = tp["step"]  # step coordinate before .diff()

    # anemoi-inference sometimes omits step 0 from the GRIB even with
    # accumulate_from_start_of_forecast enabled. If missing, earthkit-data
    # will fill it with NaNs following the `allow_holes=True` flag.
    if tp[{"step": 0}].isnull().all():
        LOG.warning(
            "Step 0 of TOT_PREC is missing, filling with zeroes "
            "assuming accumulate_from_start_of_forecast is enabled."
        )
        tp[{"step": 0}] = 0.0

    # Disaggregate TOT_PREC from cumulative-from-start (expected when the
    # accumulate_from_start_of_forecast post-processor is enabled in
    # anemoi-inference) to per-step accumulations.
    LOG.info(
        "Disaggregating TOT_PREC from cumulative-from-start to per-step accumulations."
    )
    tp = tp.diff("step")

    # Sanity-check that the incoming data is actually cumulative. If
    # some values are significantly negative, it indicates that the data
    # is already period-accumulated.
    min_diff = float(tp.min().compute())
    if min_diff < -0.1:  # NOTE: TOT_PREC canonical units are mm
        raise ValueError(
            "TOT_PREC in the GRIB appears to already be "
            f"period-accumulated (min(.diff()) = {min_diff:.3e} m). "
            "Check that the accumulate_from_start_of_forecast post-processor "
            "is enabled in the anemoi-inference config for this source."
        )

    # Clip remaining small negative values to zero
    tp = tp.clip(min=0.0)

    # Reindex to match the original lead_time coordinate
    tp = tp.reindex(step=_full_step_coord)

    return tp


def load_fct_data_from_grib(files: list[Path], params: list[str]) -> xr.Dataset:
    """Load forecast data from a list of GRIB files."""
    ds = load_from_grib_file(files, {"parameter.variable": params})

    if "TOT_PREC" in ds.data_vars:
        ds["TOT_PREC"] = _tot_prec_handling(ds["TOT_PREC"])

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
    ds = ds.assign_coords(longitude=("values", pw.stations_table["longitude"]))
    ds = ds.assign_coords(latitude=("values", pw.stations_table["latitude"]))
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
    """Load forecast data from GRIB files or an ICON archive.

    Routing (in order):
    1. ``*.grib`` files present in *root* → :func:`load_fct_data_from_grib`
       (ML inference output)
    2. Otherwise → ICON operational archive
    """
    root = Path(root)
    if any(root.glob("*.grib")):
        LOG.info("Loading forecasts from GRIB files...")
        return load_fct_data_from_grib(
            # NOTE: root is already for a specific reftime
            files=_collect_ml_grib_files(root, steps),
            params=params,
        )
    LOG.info("Loading baseline forecasts from ICON GRIB archive...")
    return load_fct_data_from_grib(
        files=_collect_icon_archive_files(root, reftime, steps),
        params=params,
    )
