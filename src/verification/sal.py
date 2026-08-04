"""SAL (Structure–Amplitude–Location) spatial precipitation verification.

Wernli et al. (2008) SAL is an object-based score comparing a forecast precip
field to a reference over a fixed domain, returning three signed, dimensionless
components: S (structure, object size/shape bias), A (amplitude, domain-mean
bias) and L (location, field displacement). All are normalised ratios, hence
invariant to a constant unit rescaling. Both fields must share a common raster
with near-square pixels (pysteps' Location term assumes square pixels), so native
fields are remapped onto a regular lat–lon raster (see build_regular_grid). The
object detection + components are delegated to
pysteps.verification.salscores.sal; this module adds the raster, the
nearest-neighbour remap, and a dry-window gate.
"""

from __future__ import annotations

import numpy as np
from pysteps.verification.salscores import sal as _pysteps_sal

from verification.spatial import spherical_nearest_neighbor_indices

# pysteps defaults (Wernli et al. 2008, eq. 1): detection threshold is
# thr_factor * the thr_quantile-percentile of the wet precipitation.
DEFAULT_THR_FACTOR = 0.067
DEFAULT_THR_QUANTILE = 0.95

# Below this truth point count SAL warns: a gridded analysis has millions of
# points, a station network ~150, so the exact cut is not critical.
MIN_TRUTH_POINTS = 10_000

# Fixed SAL scoring raster: greater-Alpine domain, ~1.1 km near-square cells at
# ~46.5°N (step_lon/step_lat ≈ 1/cos(46.5°) keeps pixels metrically square).
DEFAULT_GRID_EXTENT = (-1.0, 18.0, 42.0, 50.5)  # lon_min, lon_max, lat_min, lat_max
DEFAULT_GRID_STEP_LAT = 0.01
DEFAULT_GRID_STEP_LON = 0.0145


def build_regular_grid(
    extent: tuple[float, float, float, float],
    step_lat: float,
    step_lon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a regular lat–lon raster covering *extent* (lon_min, lon_max,
    lat_min, lat_max, degrees) at the given degree spacings. Returns the 1-D
    axes and 2-D meshgrids (lats, lons, lat2d, lon2d), upper bounds included.
    Choose the steps so pixels are near-square at the domain centre (pysteps
    assumes square pixels)."""
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
    """Nearest-neighbour flat indices (length ``tgt_lat2d.size``) into the
    flattened source points; reusable across time steps sharing the source grid."""
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
    """Remap a native field onto the target raster via precomputed *indices*;
    NaNs (e.g. off-domain cells) become *fill* (0 = no precipitation)."""
    flat = np.asarray(field, dtype=float).ravel()
    out = flat[indices].reshape(shape)
    return np.nan_to_num(out, nan=fill)


def compute_sal(
    prediction: np.ndarray,
    observation: np.ndarray,
    thr_factor: float = DEFAULT_THR_FACTOR,
    thr_quantile: float = DEFAULT_THR_QUANTILE,
) -> tuple[float, float, float]:
    """Compute the SAL triple ``(S, A, L)`` for two co-located 2-D fields.
    A window where either field is everywhere dry (max ≤ 0) has no detectable
    objects, so ``(nan, nan, nan)`` is returned rather than raising."""
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
