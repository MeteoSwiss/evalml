import logging
import time

from pathlib import Path

import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader

import numpy as np
from shapely import contains_xy
from shapely.ops import transform
import pyproj
import xarray as xr

import abc
from shapely.geometry import Polygon

LOG = logging.getLogger(__name__)


class AggregationMasks(abc.ABC):
    @abc.abstractmethod
    def get_masks(self, *args, **kwargs) -> xr.DataArray:
        pass


class SpatialAggregationMasks(AggregationMasks):
    @abc.abstractmethod
    def get_masks(self, lat: xr.DataArray, lon: xr.DataArray) -> xr.DataArray:
        pass


class ShapefileSpatialAggregationMasks(SpatialAggregationMasks):
    regions: dict[str, list[Polygon]]

    def __init__(
        self, shp: str | list[str], src_crs=ccrs.epsg(2056), dst_crs=ccrs.PlateCarree()
    ):
        proj = pyproj.Transformer.from_crs(
            src_crs.proj4_init, dst_crs.proj4_init, always_xy=True
        ).transform

        regions = {}
        # add inner region for ML evaluation
        regions["all"] = [
            Polygon(list(zip([1.5, 16, 16, 1.5, 1.5], [43, 43, 49.5, 49.5, 43])))
        ]
        shp = [shp] if isinstance(shp, str) else shp
        for shapefile in shp:
            region_name = Path(shapefile).stem
            reader = Reader(shapefile)
            regions[region_name] = [
                transform(proj, record.geometry) for record in reader.records()
            ]
        self.regions = regions

    def get_masks(self, lat: xr.DataArray, lon: xr.DataArray) -> xr.DataArray:
        masks = []
        for region_name, polygons in self.regions.items():
            mask = self._mask_from_polygons(polygons, lat, lon)
            masks.append(mask.assign_coords(region=region_name))
        return xr.concat(masks, dim="region")

    @staticmethod
    def _mask_from_polygons(
        polygons: list[Polygon], lat: xr.DataArray, lon: xr.DataArray
    ) -> xr.DataArray:
        mask = np.zeros(lon.shape, dtype=bool)
        for poly in polygons:
            mask |= contains_xy(poly, lon.values, lat.values)
        return xr.DataArray(mask, coords=lon.coords, dims=lon.dims)


def _compute_scores(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    dim=["x", "y"],
    prefix="",
    suffix="",
    source="",
) -> xr.Dataset:
    """
    Compute basic verification metrics between two xarray DataArrays (fcst and obs).
    Returns a xarray Dataset with the computed metrics.
    """
    error = fcst - obs
    scores = xr.Dataset(
        {
            f"{prefix}BIAS{suffix}": error.mean(dim=dim, skipna=True),
            f"{prefix}MSE{suffix}": (error**2).mean(dim=dim, skipna=True),
            f"{prefix}MAE{suffix}": abs(error).mean(dim=dim, skipna=True),
            f"{prefix}VAR{suffix}": error.var(dim=dim, skipna=True),
            f"{prefix}CORR{suffix}": xr.corr(fcst, obs, dim=dim),
            f"{prefix}R2{suffix}": xr.corr(fcst, obs, dim=dim) ** 2,
        }
    )
    scores = scores.expand_dims({"source": [source]})
    return scores


def _compute_statistics(
    data: xr.DataArray,
    dim=["x", "y"],
    prefix="",
    suffix="",
    source="",
) -> xr.Dataset:
    """
    Compute basic statistics of a xarray DataArray (data).
    Returns a xarray Dataset with the computed statistics.
    """
    stats = xr.Dataset(
        {
            f"{prefix}mean{suffix}": data.mean(dim=dim, skipna=True),
            f"{prefix}var{suffix}": data.var(dim=dim, skipna=True),
            f"{prefix}min{suffix}": data.min(dim=dim, skipna=True),
            f"{prefix}max{suffix}": data.max(dim=dim, skipna=True),
        }
    )
    stats = stats.expand_dims({"source": [source]})
    return stats


def _merge_metrics(ds: xr.Dataset) -> xr.Dataset:
    out = xr.merge(ds, compat="no_conflicts")
    if "ref_time" not in out.dims:
        out = out.expand_dims("ref_time").set_coords("ref_time")
    out = out.compute(num_workers=4, scheduler="threads")
    return out


def _compute_masks(ds: xr.Dataset) -> xr.Dataset:
    # extract first data_var from ds and only retain x and y dimensions
    darr = ds[list(ds.data_vars)[0]].isel(
        **{dim: 0 for dim in ds[list(ds.data_vars)[0]].dims if dim not in ["x", "y"]}
    )
    # compile list of masks to use with data arrays in ds
    mask = xr.ones_like(darr, dtype=bool).expand_dims(region=["all"])
    return mask


def verify(
    fcst: xr.Dataset,
    obs: xr.Dataset,
    fcst_label: str,
    obs_label: str,
    regions: list[str] | None = None,
) -> xr.Dataset:
    """
    Compare two xarray Datasets (fcst and obs) and return pandas DataFrame with
    basic verification metrics and statistics for both fcst and obs.
    """
    start = time.time()

    # rewrite the verification to use dask and xarray
    # chunk the data to avoid memory issues
    # compute the metrics in parallel
    # return the results as a xarray Dataset
    fcst_aligned, obs_aligned = xr.align(fcst, obs, join="inner", copy=False)
    region_polygons = ShapefileSpatialAggregationMasks(shp=regions)
    masks = region_polygons.get_masks(
        lon=obs_aligned["longitude"], lat=obs_aligned["latitude"]
    )

    scores = []
    statistics = []
    for param in fcst_aligned.data_vars:
        if param not in obs_aligned.data_vars:
            LOG.warning("Parameter %s not in obs, skipping", param)
            continue
        score = []
        fcst_statistics = []
        obs_statistics = []
        for region in masks.region.values:
            LOG.info("Verifying parameter %s for region %s", param, region)
            fcst_param = fcst_aligned[param].where(masks.sel(region=region))
            obs_param = obs_aligned[param].where(masks.sel(region=region))

            # scores vs time (reduce spatially)
            score.append(
                _compute_scores(
                    fcst_param, obs_param, prefix=param + ".", source=fcst_label
                ).expand_dims(region=[region])
            )

            # statistics vs time (reduce spatially)
            fcst_statistics.append(
                _compute_statistics(
                    fcst_param, prefix=param + ".", source=fcst_label
                ).expand_dims(region=[region])
            )
            obs_statistics.append(
                _compute_statistics(
                    obs_param, prefix=param + ".", source=obs_label
                ).expand_dims(region=[region])
            )

        score = xr.concat(score, dim="region")
        fcst_statistics = xr.concat(fcst_statistics, dim="region")
        obs_statistics = xr.concat(obs_statistics, dim="region")
        statistics.append(xr.concat([fcst_statistics, obs_statistics], dim="source"))
        scores.append(score)

    scores = _merge_metrics(scores)
    statistics = _merge_metrics(statistics)
    out = xr.merge([scores, statistics], join="outer", compat="no_conflicts")
    LOG.info("Computed metrics in %.2f seconds", time.time() - start)
    LOG.info("Metrics dataset: \n%s", out)
    return out
