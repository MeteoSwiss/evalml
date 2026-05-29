from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import MultiPoint
from data_input import load_from_grib_file


PARAMS_MAP = {
    "T_2M": "2t",
    "TD_2M": "2d",
    "U_10M": "10u",
    "V_10M": "10v",
    "PS": "sp",
    "PMSL": "msl",
    "TOT_PREC": "tp",
}

PARAMS_MAP_INV = {v: k for k, v in PARAMS_MAP.items()}


def load_state_from_grib(
    file: Path, paramlist: list[str] | None = None
) -> dict[str, np.ndarray | dict[str, np.ndarray] | gpd.GeoSeries]:
    ds = load_from_grib_file(file, {"parameter.variable": paramlist})
    state = {}
    ref_param = next((p for p in (paramlist or []) if p in ds), None)
    if ref_param is None:
        raise ValueError(
            f"None of the requested params {paramlist} found in {file}. "
            "The GRIB file may not contain these fields at this lead time "
            "(e.g. accumulated fields like TOT_PREC are undefined at step 0)."
        )
    state["forecast_reference_time"] = datetime.fromtimestamp(
        ds["forecast_reference_time"].values.item() / 1e9
    )
    state["valid_time"] = datetime.fromtimestamp(ds["valid_time"].values.item() / 1e9)
    state["longitudes"] = ds["longitude"].values.flatten()
    state["latitudes"] = ds["latitude"].values.flatten()
    # Add the limited-area model envelope polygon (convex hull) before global coords are added
    lam_hull = MultiPoint(
        list(zip(state["longitudes"].tolist(), state["latitudes"].tolist()))
    ).convex_hull
    state["lam_envelope"] = gpd.GeoSeries([lam_hull], crs="EPSG:4326")
    state["fields"] = {}
    for param in paramlist or []:
        if param in ds:
            state["fields"][param] = ds[param].values.flatten()
        else:
            # initialize with NaNs to keep consistent length
            state["fields"][param] = np.full(
                ds["longitude"].values.size, np.nan, dtype=float
            )
    global_file = str(file.parent / f"ifs-{file.stem}.grib")
    if Path(global_file).exists():
        _paramlist_ecmwf = [PARAMS_MAP[p] for p in paramlist]
        ds = load_from_grib_file(global_file, {"parameter.variable": _paramlist_ecmwf})
        mask = ~np.isnan(ds[_paramlist_ecmwf[0]].values.squeeze())
        global_lons = ds["longitude"].values.flatten()
        if np.max(global_lons) > 180:
            global_lons = ((global_lons + 180) % 360) - 180
        state["longitudes"] = np.concatenate([state["longitudes"], global_lons[mask]])
        state["latitudes"] = np.concatenate(
            [state["latitudes"], ds["latitude"].values.flatten()[mask]]
        )
        for param in _paramlist_ecmwf:
            if param in ds:
                state["fields"][PARAMS_MAP_INV[param]] = np.concatenate(
                    [
                        state["fields"][PARAMS_MAP_INV[param]],
                        ds[param].values.flatten()[mask],
                    ]
                )
            else:
                state["fields"][PARAMS_MAP_INV[param]] = np.concatenate(
                    [
                        state["fields"][PARAMS_MAP_INV[param]],
                        np.full(mask.size, np.nan, dtype=float),
                    ]
                )
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
