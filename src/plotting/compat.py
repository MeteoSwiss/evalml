from datetime import datetime
from pathlib import Path

import earthkit.data as ekd
import geopandas as gpd
import numpy as np
import pandas as pd
from meteodatalab import data_source
from meteodatalab import grib_decoder
from shapely.geometry import MultiPoint


def load_state_from_grib(
    file: Path, paramlist: list[str] | None = None
) -> dict[str, np.ndarray | dict[str, np.ndarray] | gpd.GeoSeries]:
    reftime = datetime.strptime(file.parents[1].name, "%Y%m%d%H%M")
    lead_time_hours = int(file.stem.split("_")[-1])
    fds = data_source.FileDataSource(datafiles=[str(file)])
    ds = grib_decoder.load(fds, {"param": paramlist})
    state = {}
    lats = ds[paramlist[0]].lat.data.flatten()
    lons = ds[paramlist[0]].lon.data.flatten()
    state["forecast_reference_time"] = reftime
    state["valid_time"] = reftime + pd.to_timedelta(lead_time_hours, unit="h")
    state["longitudes"] = lons
    state["latitudes"] = lats
    # Add the limited-area model envelope polygon (convex hull) before global coords are added
    lam_hull = MultiPoint(list(zip(lons.tolist(), lats.tolist()))).convex_hull
    state["lam_envelope"] = gpd.GeoSeries([lam_hull], crs="EPSG:4326")
    state["fields"] = {}
    for param in paramlist or []:
        if param in ds:
            state["fields"][param] = ds[param].values.flatten()
        else:
            # initialize with NaNs to keep consistent length
            state["fields"][param] = np.full(lats.size, np.nan, dtype=float)
    global_file = str(file.parent / f"ifs-{file.stem}.grib")
    if Path(global_file).exists():
        global_file = str(file.parent / f"ifs-{file.stem}.grib")
        fds_global = ekd.from_source("file", global_file)
        ds_global = {
            u.metadata("param"): u.values
            for u in fds_global
            if u.metadata("param") in paramlist
        }
        # Use first key from ds_global instead of paramlist[0]
        ref_key = next(iter(ds_global), None)
        if ref_key is not None:
            global_lats = fds_global.metadata("latitudes")[0]
            global_lons = fds_global.metadata("longitudes")[0]
            if max(global_lons) > 180:
                global_lons = ((global_lons + 180) % 360) - 180
            mask = np.where(~np.isnan(ds_global[ref_key]))[0]
            n_add = int(mask.size)
            state["longitudes"] = np.concatenate(
                [state["longitudes"], global_lons[mask]]
            )
            state["latitudes"] = np.concatenate([state["latitudes"], global_lats[mask]])
            for param in paramlist or state["fields"].keys():
                add = (
                    ds_global[param][mask]
                    if param in ds_global
                    else np.full(n_add, np.nan, dtype=float)
                )
                # ensure base array exists (in case param wasn't in local ds)
                base = state["fields"].get(
                    param, np.full(lats.size, np.nan, dtype=float)
                )
                state["fields"][param] = np.concatenate([base, add])
    return state


def load_state_from_raw(
    file: Path, paramlist: list[str] | None = None
) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
    _state: dict[str, np.ndarray] = np.load(file, allow_pickle=True)
    reftime = datetime.strptime(file.parents[1].name, "%Y%m%d%H%M")
    validtime = datetime.strptime(file.stem, "%Y%m%d%H%M%S")
    state = {}
    lons = _state["longitudes"]
    if max(lons) > 180:
        lons = ((lons + 180) % 360) - 180
    state["longitudes"] = lons
    state["latitudes"] = _state["latitudes"]
    state["valid_time"] = validtime
    state["lead_time"] = state["valid_time"] - reftime
    state["forecast_reference_time"] = reftime
    state["fields"] = {}
    for key, value in _state.items():
        if key.startswith("field_"):
            state["fields"][key.removeprefix("field_")] = value
    return state
