"""Spatial mapping helpers for aligning forecasts and references.

This module contains reusable nearest-neighbor utilities used by verification
and plotting scripts to map data between different spatial supports.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree


def spherical_nearest_neighbor_indices(
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> np.ndarray:
    """Return indices of nearest source points for each target point.

    Distances are computed in 3D Cartesian space after projecting latitude and
    longitude (degrees) onto the unit sphere. This avoids distortions from
    Euclidean distance in degree space.

    Parameters
    ----------
    source_lat, source_lon
        Latitude and longitude of source points in degrees.
    target_lat, target_lon
        Latitude and longitude of target points in degrees.

    Returns
    -------
    np.ndarray
        Integer indices into source points, one index per target point.
    """

    source_lat = np.asarray(source_lat).ravel()
    source_lon = np.asarray(source_lon).ravel()
    target_lat = np.asarray(target_lat).ravel()
    target_lon = np.asarray(target_lon).ravel()

    source_lat_rad = np.deg2rad(source_lat)
    source_lon_rad = np.deg2rad(source_lon)
    target_lat_rad = np.deg2rad(target_lat)
    target_lon_rad = np.deg2rad(target_lon)

    source_xyz = np.c_[
        np.cos(source_lat_rad) * np.cos(source_lon_rad),
        np.cos(source_lat_rad) * np.sin(source_lon_rad),
        np.sin(source_lat_rad),
    ]
    target_xyz = np.c_[
        np.cos(target_lat_rad) * np.cos(target_lon_rad),
        np.cos(target_lat_rad) * np.sin(target_lon_rad),
        np.sin(target_lat_rad),
    ]

    tree = cKDTree(source_xyz)
    _, nearest_idx = tree.query(target_xyz, k=1)
    return np.asarray(nearest_idx, dtype=int)


def nearest_grid_yx_indices(
    grid: xr.Dataset | xr.DataArray, target_lat: np.ndarray, target_lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Find nearest `(y, x)` grid indices for target coordinates.

    Parameters
    ----------
    grid
        Dataset or DataArray with `lat` and `lon` coordinates defined on a
        `(y, x)` grid.
    target_lat, target_lon
        Target coordinates in degrees.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays of `y` and `x` indices for each target location.
    """

    if "lat" not in grid or "lon" not in grid:
        raise ValueError("Input must provide 'lat' and 'lon' coordinates")

    lat2d = np.asarray(grid["lat"].values)
    lon2d = np.asarray(grid["lon"].values)
    if lat2d.ndim != 2 or lon2d.ndim != 2:
        raise ValueError("'lat' and 'lon' must be 2D on (y, x) for y/x indexing")

    flat_idx = spherical_nearest_neighbor_indices(
        source_lat=lat2d.ravel(),
        source_lon=lon2d.ravel(),
        target_lat=target_lat,
        target_lon=target_lon,
    )
    y_idx, x_idx = np.unravel_index(flat_idx, lat2d.shape)
    return np.asarray(y_idx, dtype=int), np.asarray(x_idx, dtype=int)


def map_forecast_to_truth(fcst: xr.Dataset, truth: xr.Dataset) -> xr.Dataset:
    """Map forecast points to truth locations using nearest-neighbor matching.

    The forecast is flattened to a single spatial `values` dimension (when
    provided as `(y, x)`), then sampled at the nearest points to each truth
    location. Returned forecast coordinates are overwritten with truth station
    coordinates to make subsequent verification align naturally.

    Parameters
    ----------
    fcst
        Forecast dataset with `lat` and `lon` coordinates on either `(y, x)` or
        `values`.
    truth
        Reference dataset with `lat` and `lon` coordinates on either `(y, x)` or
        `values`.

    Returns
    -------
    xr.Dataset
        Mapped forecast dataset.
    """
    # TODO: return fcst unchanged when forecast and truth are already aligned

    truth_is_grid = "y" in truth.dims and "x" in truth.dims

    if "y" in fcst.dims and "x" in fcst.dims:
        fcst = fcst.stack(values=("y", "x"))
    if truth_is_grid:
        truth = truth.stack(values=("y", "x"))

    nearest_idx = spherical_nearest_neighbor_indices(
        source_lat=fcst["lat"].values,
        source_lon=fcst["lon"].values,
        target_lat=truth["lat"].values,
        target_lon=truth["lon"].values,
    )

    fcst = fcst.isel(values=nearest_idx)
    fcst = fcst.drop_vars(["x", "y", "values"], errors="ignore")
    fcst = fcst.assign_coords(lon=("values", truth.lon.data))
    fcst = fcst.assign_coords(lat=("values", truth.lat.data))
    fcst = fcst.assign_coords(values=truth["values"])

    if truth_is_grid:
        fcst = fcst.unstack("values")

    return fcst
