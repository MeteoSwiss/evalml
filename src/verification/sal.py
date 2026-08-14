"""SAL (Structure–Amplitude–Location) spatial precipitation verification.

Wernli et al. (2008) SAL is an object-based score that compares a forecast
precipitation field against a reference over a fixed domain, returning three
signed, dimensionless components:

  S  structure  — bias in object size/shape (negative: too peaked/small,
                  positive: too widespread/flat), range roughly (-2, 2)
  A  amplitude  — bias in domain-mean precipitation (normalised ratio),
                  range (-2, 2)
  L  location   — displacement of the precipitation field, 0 (perfect) → 2

All three are normalised ratios, so they are invariant to a constant rescaling
of the precipitation unit (mm vs kg m-2, etc.); only the object-detection
threshold and any reported domain means depend on the unit.

SAL requires both fields on a common 2-D raster with near-square pixels: the
Location term measures centroid displacement in grid cells normalised by the
domain diagonal, and pysteps assumes square pixels. Native model/analysis fields
(ICON unstructured, KENDA ``(y, x)``, ...) are therefore remapped onto a regular
lat–lon raster that is chosen to be metrically near-isotropic over the domain of
interest before scoring — see :func:`build_regular_grid`.

The heavy lifting (object detection + the three components) is delegated to
``pysteps.verification.salscores.sal``; this module only adds the raster
construction, the nearest-neighbour remap (reusing
``verification.spatial.spherical_nearest_neighbor_indices``), and a thin wrapper
that gates dry windows.
"""

from __future__ import annotations

import numpy as np
from pysteps.verification.salscores import sal as _pysteps_sal

from verification.spatial import spherical_nearest_neighbor_indices

# pysteps' own defaults (Wernli et al. 2008, eq. 1): the detection threshold is
# ``thr_factor * thr_quantile-percentile`` of the wet precipitation. Kept
# identical to the pysteps defaults so results match a bare ``sal(pred, obs)``
# call.
DEFAULT_THR_FACTOR = 0.067
DEFAULT_THR_QUANTILE = 0.95

# Minimum truth point density (points per km^2) for SAL to treat the truth as a
# resolved precipitation field. A gridded analysis packs the domain densely
# (~1 point/km^2 at 1 km spacing, even on an unstructured mesh); a station
# network is ~0.005 points/km^2 and cannot form a field once remapped. The cut
# at 0.05 corresponds to a mean spacing of ~4.5 km — coarser than any analysis
# used as truth, far denser than any observation network.
MIN_TRUTH_POINT_DENSITY = 0.05


def build_regular_grid(
    extent: tuple[float, float, float, float],
    step_lat: float,
    step_lon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a regular lat–lon raster covering *extent*.

    Parameters
    ----------
    extent
        ``(lon_min, lon_max, lat_min, lat_max)`` in degrees (PlateCarree),
        matching the ordering of ``DomainConfig.extent``.
    step_lat, step_lon
        Grid spacing in degrees. Choose them so the pixels are metrically
        near-square at the domain's central latitude (e.g. 0.01° lat ×
        0.0145° lon at ~46.5°N), honouring pysteps' square-pixel assumption.

    Returns
    -------
    lats, lons, lat2d, lon2d
        The 1-D axes and the two 2-D meshgrids, each of shape
        ``(len(lats), len(lons))``. The upper bounds are included.
    """
    lon_min, lon_max, lat_min, lat_max = extent
    lons = np.arange(lon_min, lon_max + step_lon / 2, step_lon)
    lats = np.arange(lat_min, lat_max + step_lat / 2, step_lat)
    lon2d, lat2d = np.meshgrid(lons, lats)
    return lats, lons, lat2d, lon2d


def remap_indices(
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    tgt_lat2d: np.ndarray,
    tgt_lon2d: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbour indices mapping a source grid onto the target raster.

    Returns a flat index array (length ``tgt_lat2d.size``) into the flattened
    source points, so it can be reused across many time steps that share the
    same source grid — build it once per (source grid, target raster).
    """
    return spherical_nearest_neighbor_indices(
        np.asarray(src_lat).ravel(),
        np.asarray(src_lon).ravel(),
        np.asarray(tgt_lat2d).ravel(),
        np.asarray(tgt_lon2d).ravel(),
    )


def remap_field(
    field: np.ndarray,
    indices: np.ndarray,
    shape: tuple[int, int],
    fill: float = 0.0,
) -> np.ndarray:
    """Remap a native field onto the target raster using precomputed *indices*.

    NaNs (e.g. off-domain cells) are replaced by *fill* (0 by default), matching
    the convention that missing precipitation reads as no precipitation.
    """
    flat = np.asarray(field, dtype=float).ravel()
    out = flat[indices].reshape(shape)
    return np.nan_to_num(out, nan=fill)


def point_density_per_km2(lat: np.ndarray, lon: np.ndarray) -> float:
    """Approximate density (points per km²) of scattered lat/lon points.

    Uses an equirectangular area approximation over the points' bounding box
    (longitude spacing scaled by cos of the mean latitude). This is a coarse
    but robust discriminator between a resolved analysis field (dense, ~1/km²)
    and sparse station observations (~0.005/km²) — it does not assume the truth
    is on a structured ``(y, x)`` grid, which matters because analyses such as
    KENDA-CH1 are stored on an unstructured mesh. Degenerate inputs (< 2 points,
    or all collinear so the box has zero area) return 0.0.
    """
    lat = np.asarray(lat, dtype=float).ravel()
    lon = np.asarray(lon, dtype=float).ravel()
    if lat.size < 2:
        return 0.0
    lat_span = float(lat.max() - lat.min())
    lon_span = float(lon.max() - lon.min())
    if lat_span <= 0.0 or lon_span <= 0.0:
        return 0.0
    km_per_deg = 111.32
    mean_lat_rad = np.deg2rad((float(lat.max()) + float(lat.min())) / 2.0)
    height_km = lat_span * km_per_deg
    width_km = lon_span * km_per_deg * float(np.cos(mean_lat_rad))
    area_km2 = height_km * width_km
    if area_km2 <= 0.0:
        return 0.0
    return lat.size / area_km2


def compute_sal(
    prediction: np.ndarray,
    observation: np.ndarray,
    thr_factor: float = DEFAULT_THR_FACTOR,
    thr_quantile: float = DEFAULT_THR_QUANTILE,
) -> tuple[float, float, float]:
    """Compute the SAL triple for two co-located 2-D fields.

    Returns ``(S, A, L)`` as floats. A window in which either field is
    everywhere dry (max ≤ 0) has no detectable objects, so ``(nan, nan, nan)``
    is returned rather than raising — the caller decides how to treat dry
    windows (typically drop them via a wet-case filter downstream).
    """
    pred = np.asarray(prediction, dtype=float)
    obs = np.asarray(observation, dtype=float)
    # Use the finite values only, so an all-NaN or empty field is a dry window
    # rather than a RuntimeWarning from np.nanmax over an all-NaN slice.
    pred_finite = pred[np.isfinite(pred)]
    obs_finite = obs[np.isfinite(obs)]
    if pred_finite.size == 0 or obs_finite.size == 0:
        return (np.nan, np.nan, np.nan)
    if not (pred_finite.max() > 0 and obs_finite.max() > 0):
        return (np.nan, np.nan, np.nan)
    s, a, ell = _pysteps_sal(
        pred, obs, thr_factor=thr_factor, thr_quantile=thr_quantile
    )
    return (float(s), float(a), float(ell))
