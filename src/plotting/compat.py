from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import MultiPoint
from data_input import (
    load_fieldlist_from_grib_file,
    fieldlist_to_xarray,
    rename_ifs_to_icon,
    _fetch_fdb_fieldlist,
    _FDB_MODEL,
    _FDB_GLOBAL_MODEL,
)


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
    "CLCM": "mcc",
    "CLCH": "hcc",
    "SKT": "skt",
    "SSRD": "ssrd",
}

PARAMS_MAP_INV = {v: k for k, v in PARAMS_MAP.items()}


def _decode_state(
    regional_fieldlist,
    global_fieldlist,
    paramlist: list[str] | None,
    source: str,
) -> dict[str, np.ndarray | dict[str, np.ndarray] | gpd.GeoSeries]:
    """Build a plotting state dict from the regional (lam_0) and optional
    global (IFS) fieldlists for one lead time.

    This is the common decoding step shared by :func:`load_state_from_grib`
    and :func:`load_state_from_fdb`, which differ only in how they fetch the
    two fieldlists (local files vs FDB retrieve) -- once both have an
    earthkit FieldList in hand, the xarray conversion, renaming, and
    flat-point-list state assembly are identical. `source` is only used to
    make the "nothing found" error message useful.
    """
    ds = fieldlist_to_xarray(regional_fieldlist)
    # Rename any IFS shortnames back to COSMO names
    ds = rename_ifs_to_icon(ds, PARAMS_MAP_INV)
    state = {}
    ref_param = next((p for p in (paramlist or []) if p in ds), None)
    if ref_param is None:
        raise ValueError(
            f"None of the requested params {paramlist} found in {source}. "
            "The source may not contain these fields at this lead time "
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

    if global_fieldlist is not None:
        _paramlist_ecmwf = [PARAMS_MAP[p] for p in paramlist]
        gds = fieldlist_to_xarray(global_fieldlist)
        mask = ~np.isnan(gds[_paramlist_ecmwf[0]].values.squeeze())
        global_lons = gds["longitude"].values.flatten()
        if np.max(global_lons) > 180:
            global_lons = ((global_lons + 180) % 360) - 180
        state["longitudes"] = np.concatenate([state["longitudes"], global_lons[mask]])
        state["latitudes"] = np.concatenate(
            [state["latitudes"], gds["latitude"].values.flatten()[mask]]
        )
        for param in _paramlist_ecmwf:
            if param in gds:
                state["fields"][PARAMS_MAP_INV[param]] = np.concatenate(
                    [
                        state["fields"][PARAMS_MAP_INV[param]],
                        gds[param].values.flatten()[mask],
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


def load_state_from_grib(
    file: Path, paramlist: list[str] | None = None
) -> dict[str, np.ndarray | dict[str, np.ndarray] | gpd.GeoSeries]:
    # Also request IFS shortname aliases
    paramlist_extended = list(
        {p for p in (paramlist or [])}
        | {PARAMS_MAP[p] for p in (paramlist or []) if p in PARAMS_MAP}
    )
    regional_fieldlist = load_fieldlist_from_grib_file(
        file, {"parameter.variable": paramlist_extended}
    )

    global_fieldlist = None
    global_file = file.parent / f"ifs-{file.stem}.grib"
    if global_file.exists():
        _paramlist_ecmwf = [PARAMS_MAP[p] for p in paramlist]
        global_fieldlist = load_fieldlist_from_grib_file(
            global_file, {"parameter.variable": _paramlist_ecmwf}
        )

    return _decode_state(regional_fieldlist, global_fieldlist, paramlist, source=str(file))


def load_state_from_fdb(
    fdb_root: str, reftime: datetime, step: int, paramlist: list[str] | None = None
) -> dict[str, np.ndarray | dict[str, np.ndarray] | gpd.GeoSeries]:
    """FDB equivalent of :func:`load_state_from_grib` for one lead time.

    Fetches the regional (`_FDB_MODEL`) and global (`_FDB_GLOBAL_MODEL`) grids
    from FDB -- exactly the requested params, via `_fetch_fdb_fieldlist` --
    and decodes them directly through the same :func:`_decode_state` shared
    with the local-file path (it works with flat point lists, not an aligned
    xarray grid, so the regional/global point-count mismatch that
    :func:`data_input.load_forecast_data` sidesteps by dropping the global
    grid entirely isn't a problem here).
    """
    regional_fieldlist = _fetch_fdb_fieldlist(
        fdb_root, reftime, [step], paramlist, model=_FDB_MODEL
    )
    if regional_fieldlist is None:
        raise FileNotFoundError(
            f"No FDB data found for step={step} model={_FDB_MODEL} in {fdb_root}"
        )
    try:
        global_fieldlist = _fetch_fdb_fieldlist(
            fdb_root, reftime, [step], paramlist, model=_FDB_GLOBAL_MODEL
        )
    except ValueError:
        # Some requested params may have no known id for the global grid
        # (_FDB_GLOBAL_PARAM_IDS only covers what's actually been needed from
        # it) -- tolerate that the same way as "nothing archived" below,
        # rather than failing the whole frame over a companion grid.
        global_fieldlist = None

    return _decode_state(
        regional_fieldlist, global_fieldlist, paramlist, source=f"{fdb_root} (step={step})"
    )


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
