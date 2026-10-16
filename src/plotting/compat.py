from datetime import datetime
from pathlib import Path

import tempfile

import geopandas as gpd
import numpy as np
from shapely.geometry import MultiPoint
from data_input import load_from_grib_file, _fetch_fdb_fieldlist, _FDB_MODEL, _FDB_GLOBAL_MODEL


PARAMS_MAP = {
    "T_2M": "2t",
    "TD_2M": "2d",
    "U_10M": "10u",
    "V_10M": "10v",
    "PS": "sp",
    "PMSL": "msl",
    "TOT_PREC": "tp",
    "CLCT": "tcc",
    "CLCL": "lcc",
    "SSRD": "ssrd",
}

PARAMS_MAP_INV = {v: k for k, v in PARAMS_MAP.items()}


def load_state_from_grib(
    file: Path, paramlist: list[str] | None = None
) -> dict[str, np.ndarray | dict[str, np.ndarray] | gpd.GeoSeries]:
    # Also request IFS shortname aliases
    paramlist_extended = list(
        {p for p in (paramlist or [])}
        | {PARAMS_MAP[p] for p in (paramlist or []) if p in PARAMS_MAP}
    )
    ds = load_from_grib_file(file, {"parameter.variable": paramlist_extended})
    # Rename any IFS shortnames back to COSMO names
    ifs_rename = {
        ifs: cosmo for ifs, cosmo in PARAMS_MAP_INV.items() if ifs in ds.data_vars
    }
    if ifs_rename:
        ds = ds.rename(ifs_rename)
        # TODO check if needed
    # if "tp" in ifs_rename and "TOT_PREC" in ds:
    #     # IFS/ECMWF convention: "tp" is accumulated precip in meters.
    #     # Convert to kg m-2 (mm) to match the ICON-native convention.
    #     ds["TOT_PREC"] = ds["TOT_PREC"] * 1000
    state = {}
    ref_param = next((p for p in (paramlist or []) if p in ds), None)
    if ref_param is None:
        raise ValueError(
            f"None of the requested params {paramlist} found in {file}. "
            "The GRIB file may not contain these fields at this lead time "
            "(e.g. accumulated fields like TOT_PREC are undefined at step 0)."
        )
    # Cast datetime64 directly: datetime.fromtimestamp() would read the epoch
    # value in the host's local zone and relabel these UTC times (+2h on CEST).
    state["forecast_reference_time"] = (
        ds["forecast_reference_time"].values.astype("datetime64[s]").item()
    )
    state["valid_time"] = ds["valid_time"].values.astype("datetime64[s]").item()
    state["longitudes"] = ds["longitude"].values.flatten()
    state["latitudes"] = ds["latitude"].values.flatten()
    # Add the limited-area model envelope polygon (convex hull) before global coords are added.
    # Skip for global grids (lon range ~360°) — the hull would cover the whole globe.
    lon_range = state["longitudes"].max() - state["longitudes"].min()
    if lon_range > 350:
        state["lam_envelope"] = gpd.GeoSeries([], crs="EPSG:4326")
    else:
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


def load_state_from_fdb(
    fdb_root: str, reftime: datetime, step: int, paramlist: list[str] | None = None
) -> dict[str, np.ndarray | dict[str, np.ndarray] | gpd.GeoSeries]:
    """FDB equivalent of :func:`load_state_from_grib` for one lead time.

    Fetches the regional (`_FDB_MODEL`) and global (`_FDB_GLOBAL_MODEL`) grids
    from FDB -- exactly the requested params, via `_fetch_fdb_fieldlist` --
    and writes them to temp files using the same naming convention
    `load_state_from_grib` already expects locally (a primary file plus an
    `ifs-<stem>.grib` companion in the same directory), then delegates to it
    unchanged -- reusing its state-dict construction and global-grid
    concatenation (it works with flat point lists, not an aligned xarray grid,
    so the regional/global point-count mismatch that :func:`data_input.
    load_forecast_data` sidesteps by dropping the global grid entirely isn't a
    problem here).
    """
    regional = _fetch_fdb_fieldlist(fdb_root, reftime, [step], paramlist, model=_FDB_MODEL)
    if regional is None:
        raise FileNotFoundError(
            f"No FDB data found for step={step} model={_FDB_MODEL} in {fdb_root}"
        )
    try:
        global_ = _fetch_fdb_fieldlist(fdb_root, reftime, [step], paramlist, model=_FDB_GLOBAL_MODEL)
    except ValueError:
        # Some requested params may have no known id for the global grid
        # (_FDB_GLOBAL_PARAM_IDS only covers what's actually been needed from
        # it) -- tolerate that the same way as "nothing archived" below,
        # rather than failing the whole frame over a companion grid.
        global_ = None

    with tempfile.TemporaryDirectory() as tmp_dir:
        primary = Path(tmp_dir) / "frame.grib"
        with open(primary, "wb") as f:
            for fld in regional:
                f.write(fld.message())
        if global_ is not None:
            with open(Path(tmp_dir) / "ifs-frame.grib", "wb") as f:
                for fld in global_:
                    f.write(fld.message())
        return load_state_from_grib(primary, paramlist=paramlist)


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
