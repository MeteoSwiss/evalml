"""Derived-variable computations over the canonical schema.

Single home for parameters computed from raw fields: wind speed, wind
components from speed/direction, and de-accumulation of from-start-of-forecast
accumulated fields. Functions are type-preserving (numpy in → numpy out,
xarray in → xarray out) so the gridded loaders and the plotting paths share
them. Prefers earthkit-meteo primitives where they exist.
"""

import logging

import earthkit.meteo.wind as ekm_wind
import numpy as np

LOG = logging.getLogger(__name__)


def wind_speed(u, v):
    """Horizontal wind speed sqrt(u**2 + v**2), via earthkit-meteo.

    Type-preserving: numpy in → ndarray, xarray in → DataArray (coords kept).
    """
    return ekm_wind.speed(u, v)


def uv_components(speed, direction_deg):
    """Zonal (u) and meridional (v) wind components from speed and direction.

    Meteorological convention: ``direction_deg`` is the direction the wind
    blows FROM, degrees clockwise from north. Returns ``(u, v)``,
    type-preserving.
    """
    direction_rad = np.deg2rad(direction_deg)
    u = -speed * np.sin(direction_rad)
    v = -speed * np.cos(direction_rad)
    return u, v


def deaccumulate(da, dim="step"):
    """Disaggregate a from-start-of-forecast accumulated field to per-step.

    Fills a missing first step with 0 (anemoi-inference sometimes omits step 0
    even with accumulate_from_start_of_forecast), diffs along ``dim``,
    sanity-checks the input is cumulative (raises if it looks already
    period-accumulated), clips small negatives, and reindexes to the original
    ``dim`` coordinate (so the first step becomes NaN).
    """
    name = da.name or "field"
    full_coord = da[dim]

    if da[{dim: 0}].isnull().all():
        LOG.warning(
            "Step 0 of %s is missing, filling with zeroes assuming "
            "accumulate_from_start_of_forecast is enabled.",
            name,
        )
        da[{dim: 0}] = 0.0

    LOG.info("Disaggregating %s from cumulative-from-start to per-step.", name)
    da = da.diff(dim)

    min_diff = float(da.min().compute())
    if min_diff < -0.1:
        raise ValueError(
            f"{name} appears to already be period-accumulated "
            f"(min(.diff()) = {min_diff:.3e}). Check that the "
            "accumulate_from_start_of_forecast post-processor is enabled."
        )

    da = da.clip(min=0.0)
    return da.reindex({dim: full_coord})
