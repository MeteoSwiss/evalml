from datetime import datetime
from pathlib import Path

import earthkit.data as ekd
import numpy as np
import pandas as pd
from anemoi.datasets.grids import cutout_mask
from meteodatalab import data_source
from meteodatalab import grib_decoder


def load_state_from_grib(
    file: Path, paramlist: list[str] | None = None
) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
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
    state["fields"] = {}
    for param in paramlist:
        if param in ds:
            state["fields"][param] = ds[param].values.flatten()
    global_file = str(file.parent / f"ifs-{file.stem}.grib")
    if Path(global_file).exists():
        global_file = str(file.parent / f"ifs-{file.stem}.grib")
        fds_global = ekd.from_source("file", global_file)
        ds_global = {
            u.metadata("param"): u.values
            for u in fds_global
            if u.metadata("param") in paramlist
        }
        global_lats = fds_global.metadata("latitudes")[0]
        global_lons = fds_global.metadata("longitudes")[0]
        if max(global_lons) > 180:
            global_lons = ((global_lons + 180) % 360) - 180
        mask = cutout_mask(lats, lons, global_lats, global_lons, cropping_distance=0)
        state["longitudes"] = np.concatenate([state["longitudes"], global_lons[mask]])
        state["latitudes"] = np.concatenate([state["latitudes"], global_lats[mask]])
        for param in paramlist:
            if param in ds and param in ds_global:
                state["fields"][param] = np.concatenate(
                    [state["fields"][param], ds_global[param][mask]]
                )
    return state


def load_state_from_raw(
    file: Path, paramlist: list[str] | None = None
) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
    _state: dict[str, np.ndarray] = np.load(file, allow_pickle=True)
    reftime = datetime.strptime(file.parents[1].name, "%Y%m%d%H%M")
    validtime = datetime.strptime(file.stem, "%Y%m%d%H%M%S")
    state = {}
    state["longitudes"] = _state["longitudes"]
    state["latitudes"] = _state["latitudes"]
    state["valid_time"] = validtime
    state["lead_time"] = state["valid_time"] - reftime
    state["forecast_reference_time"] = reftime
    state["fields"] = {}
    for key, value in _state.items():
        if key.startswith("field_"):
            state["fields"][key.removeprefix("field_")] = value
    return state
