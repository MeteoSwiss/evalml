"""Canonical in-memory data schema for evalml.

Every *gridded* loader returns an :class:`xarray.Dataset` shaped like the output
of opening GRIB with earthkit-data and applying :data:`XARRAY_ENGINE_PROFILE`:

Dims
    ``forecast_reference_time`` (size 1), ``step`` (lead time, ``timedelta64``),
    optional ``number`` (ensemble) / ``z`` (level); spatial dims ``(y, x)`` for
    gridded data or ``values`` for flattened / unstructured data.
Coords
    ``valid_time`` (= reftime + step), ``latitude`` and ``longitude`` on the
    spatial dims, ``step``, ``forecast_reference_time``.
Variables
    ICON canonical names (see :mod:`data.naming`).
Attrs
    ``institution = "MeteoSwiss"``, ``Conventions = "CF-1.8"``; longitudes in
    ``[-180, 180]``.

Station observations (PeakWeather) are intentionally exempt: they keep their
station-native ``time``/``values`` shape and are broadcast onto the forecast grid
only at error-computation time.
"""

import numpy as np
import xarray as xr

# earthkit-data xarray engine profile: the single source of truth for the
# canonical GRIB-derived representation.
XARRAY_ENGINE_PROFILE = {
    "ensure_dims": ["z", "number", "step", "forecast_reference_time"],
    "add_valid_time_coord": True,
    "global_attrs": [{"institution": "MeteoSwiss"}, {"Conventions": "CF-1.8"}],
}

REQUIRED_COORDS = (
    "valid_time",
    "latitude",
    "longitude",
    "step",
    "forecast_reference_time",
)


def wrap_longitude(lon):
    """Wrap longitudes to the canonical ``[-180, 180]`` range.

    A no-op when all values are already <= 180. Accepts and returns a numpy
    array. Mirrors the previous inline behaviour (``np.max`` guard).
    """
    lon = np.asarray(lon)
    if lon.size and np.max(lon) > 180:
        return ((lon + 180) % 360) - 180
    return lon


def validate_canonical(ds: xr.Dataset) -> xr.Dataset:
    """Assert *ds* conforms to the canonical gridded schema; return it unchanged.

    Raises ``ValueError`` naming what is missing. Does not mutate or normalize -
    active normalization of non-GRIB sources is wired into their loaders in T4.
    """
    missing = [c for c in REQUIRED_COORDS if c not in ds.coords]
    if missing:
        raise ValueError(f"Dataset missing canonical coords: {missing}")
    if "step" not in ds.dims:
        raise ValueError("Dataset missing canonical 'step' dimension")
    if not (("y" in ds.dims and "x" in ds.dims) or "values" in ds.dims):
        raise ValueError("Dataset must have spatial dims (y, x) or 'values'")
    return ds
