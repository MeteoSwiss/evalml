"""Spatial mapping helpers for aligning forecasts and references.

This module contains reusable nearest-neighbor utilities used by verification
and plotting scripts to map data between different spatial supports.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree


def spherical_nearest_neighbor_indices(
    source_latitude: np.ndarray,
    source_longitude: np.ndarray,
    target_latitude: np.ndarray,
    target_longitude: np.ndarray,
) -> np.ndarray:
    """Return indices of nearest source points for each target point.

    Distances are computed in 3D Cartesian space after projecting latitude and
    longitude (degrees) onto the unit sphere. This avoids distortions from
    Euclidean distance in degree space.

    Parameters
    ----------
    source_latitude, source_longitude
        Latitude and longitude of source points in degrees.
    target_latitude, target_longitude
        Latitude and longitude of target points in degrees.

    Returns
    -------
    np.ndarray
        Integer indices into source points, one index per target point.
    """

    source_latitude = np.asarray(source_latitude).ravel()
    source_longitude = np.asarray(source_longitude).ravel()
    target_latitude = np.asarray(target_latitude).ravel()
    target_longitude = np.asarray(target_longitude).ravel()

    source_latitude_rad = np.deg2rad(source_latitude)
    source_longitude_rad = np.deg2rad(source_longitude)
    target_latitude_rad = np.deg2rad(target_latitude)
    target_longitude_rad = np.deg2rad(target_longitude)

    source_xyz = np.c_[
        np.cos(source_latitude_rad) * np.cos(source_longitude_rad),
        np.cos(source_latitude_rad) * np.sin(source_longitude_rad),
        np.sin(source_latitude_rad),
    ]
    target_xyz = np.c_[
        np.cos(target_latitude_rad) * np.cos(target_longitude_rad),
        np.cos(target_latitude_rad) * np.sin(target_longitude_rad),
        np.sin(target_latitude_rad),
    ]

    tree = cKDTree(source_xyz)
    _, nearest_idx = tree.query(target_xyz, k=1)
    return np.asarray(nearest_idx, dtype=int)


def nearest_grid_yx_indices(
    grid: xr.Dataset | xr.DataArray,
    target_latitude: np.ndarray,
    target_longitude: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Find nearest `(y, x)` grid indices for target coordinates.

    Parameters
    ----------
    grid
        Dataset or DataArray with `lat` and `lon` coordinates defined on a
        `(y, x)` grid.
    target_latitude, target_longitude
        Target coordinates in degrees.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays of `y` and `x` indices for each target location.
    """

    if "latitude" not in grid or "longitude" not in grid:
        raise ValueError("Input must provide 'latitude' and 'longitude' coordinates")

    lat2d = np.asarray(grid["latitude"].values)
    lon2d = np.asarray(grid["longitude"].values)
    if lat2d.ndim != 2 or lon2d.ndim != 2:
        raise ValueError(
            "'latitude' and 'longitude' must be 2D on (y, x) for y/x indexing"
        )

    flat_idx = spherical_nearest_neighbor_indices(
        source_latitude=lat2d.ravel(),
        source_longitude=lon2d.ravel(),
        target_latitude=target_latitude,
        target_longitude=target_longitude,
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
        Forecast dataset with `latitude` and `longitude` coordinates on either
        `(y, x)` or `values`.
    truth
        Reference dataset with `latitude` and `longitude` coordinates on either
        `(y, x)` or `values`.

    Returns
    -------
    xr.Dataset
        Mapped forecast dataset.
    """
    fcst_lat = fcst["latitude"].values
    fcst_lon = fcst["longitude"].values
    truth_lat = truth["latitude"].values
    truth_lon = truth["longitude"].values
    if (
        fcst_lat.shape == truth_lat.shape
        and fcst_lon.shape == truth_lon.shape
        and np.max(np.abs(fcst_lat - truth_lat)) < 0.0003
        and np.max(np.abs(fcst_lon - truth_lon)) < 0.0003
    ):
        if np.array_equal(fcst_lat, truth_lat) and np.array_equal(fcst_lon, truth_lon):
            return fcst
        coords = {
            "latitude": (fcst["latitude"].dims, truth["latitude"].data),
            "longitude": (fcst["longitude"].dims, truth["longitude"].data),
        }
        if "values" in fcst.dims and "values" in truth.dims:
            coords["values"] = truth["values"].data
        return fcst.assign_coords(coords)

    truth_is_grid = "y" in truth.dims and "x" in truth.dims

    if "y" in fcst.dims and "x" in fcst.dims:
        fcst = fcst.stack(values=("y", "x"))
    if truth_is_grid:
        truth = truth.stack(values=("y", "x"))

    nearest_idx = spherical_nearest_neighbor_indices(
        source_latitude=fcst["latitude"].values,
        source_longitude=fcst["longitude"].values,
        target_latitude=truth["latitude"].values,
        target_longitude=truth["longitude"].values,
    )

    fcst = fcst.isel(values=nearest_idx)
    fcst = fcst.drop_vars(["x", "y", "values"], errors="ignore")
    fcst = fcst.assign_coords(longitude=("values", truth["longitude"].data))
    fcst = fcst.assign_coords(latitude=("values", truth["latitude"].data))
    fcst = fcst.assign_coords(values=truth["values"])

    if truth_is_grid:
        fcst = fcst.unstack("values")

    return fcst
