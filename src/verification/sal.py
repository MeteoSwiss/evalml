"""SAL (Structure–Amplitude–Location) spatial precipitation verification.

Wernli et al. (2008) SAL compares a forecast precipitation field to a reference
over a fixed domain, returning three signed, dimensionless components: S
(structure), A (amplitude) and L (location). Object detection and the components
are delegated to pysteps.verification.salscores.sal; this module adds the
scoring raster, the nearest-neighbour remap onto it, and a dry-window gate.
"""

from __future__ import annotations

import numpy as np
from pysteps.verification.salscores import sal as _pysteps_sal

from verification.spatial import spherical_nearest_neighbor_indices

# pysteps defaults (Wernli et al. 2008, eq. 1): detection threshold is
# thr_factor * the thr_quantile-percentile of the wet precipitation.
DEFAULT_THR_FACTOR = 0.067
DEFAULT_THR_QUANTILE = 0.95

# Defaults for the scoring raster, overridable per experiment via the config
# (SalConfig.grid_extent / grid_step_lat / grid_step_lon, which carry the same
# values). The extent is the bounding box of the ICON-CH1 analysis domain, the
# SAL truth: KENDA-CH1 spans lon -0.8171..17.7106, lat 42.0279..50.5005, so this
# rounds outwards by <=0.29 deg. Scoring much beyond the truth's support is not
# free -- the remap has no distance cutoff, so cells outside it repeat the
# nearest border value, and a larger raster also deflates pysteps' L term via
# its longer diagonal.
DEFAULT_GRID_EXTENT = (-1.0, 18.0, 42.0, 50.5)  # lon_min, lon_max, lat_min, lat_max
DEFAULT_GRID_STEP_LAT = 0.01
DEFAULT_GRID_STEP_LON = 0.0145


def sal_raster(
    extent: tuple[float, float, float, float] = DEFAULT_GRID_EXTENT,
    step_lat: float = DEFAULT_GRID_STEP_LAT,
    step_lon: float = DEFAULT_GRID_STEP_LON,
) -> tuple[np.ndarray, np.ndarray]:
    """The SAL scoring raster over *extent* as ``(lat2d, lon2d)`` meshgrids,
    latitude varying along axis 0. Cells stay inside *extent*: an upper bound is
    included when the span is a whole multiple of the spacing, else the last cell
    falls short of it by up to one step. Pick *step_lat* and *step_lon* so cells
    are metrically near-square at the extent's central latitude (pysteps' L term
    assumes square pixels); the defaults give ~1.1 km cells, exactly square at
    46.4°N (0.0145/0.01 = 1/cos(46.4°))."""
    lon_min, lon_max, lat_min, lat_max = extent
    # The epsilon admits an exact upper bound despite float drift (~1e-11 over
    # ~1e3 steps) while staying ~6 orders below one step, so it cannot overshoot.
    lons = np.arange(lon_min, lon_max + step_lon * 1e-6, step_lon)
    lats = np.arange(lat_min, lat_max + step_lat * 1e-6, step_lat)
    lon2d, lat2d = np.meshgrid(lons, lats)
    return lat2d, lon2d


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
    """Compute the SAL triple ``(S, A, L)`` for two co-located 2-D fields; NaNs
    count as dry. A window where either field is everywhere dry (max ≤ 0) has no
    detectable objects, so ``(nan, nan, nan)`` is returned rather than raising."""
    pred = np.nan_to_num(np.asarray(prediction, dtype=float))
    obs = np.nan_to_num(np.asarray(observation, dtype=float))
    if not (pred.size and obs.size and pred.max() > 0 and obs.max() > 0):
        return (np.nan, np.nan, np.nan)
    s, a, ell = _pysteps_sal(
        pred, obs, thr_factor=thr_factor, thr_quantile=thr_quantile
    )
    return (float(s), float(a), float(ell))
