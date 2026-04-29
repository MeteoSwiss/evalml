import logging
import os
import time

from pathlib import Path

import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader

import numpy as np
from shapely import contains_xy
from shapely.ops import transform
import pyproj
import xarray as xr
import scores
import operator as op

import abc
from shapely.geometry import Polygon

LOG = logging.getLogger(__name__)

OPS = {
    ">": op.gt,
    ">=": op.ge,
    "<": op.lt,
    "<=": op.le,
    "==": op.eq,
    "!=": op.ne,
}


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
        if shp and shp != [""]:
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


# deconstruct the threshold string into the value and the operator


def _threshold_value_and_operator(threshold: str):
    """
    Parse a threshold string like '> 10.0' into (operator.gt, 10.0).
    Supported operators: >, >=, <, <=, ==, !=
    """
    splits = " ".join(threshold.split()).split(" ")  # remove multiple whitespaces
    if len(splits) == 2 and splits[0] in OPS:
        try:
            value = float(splits[1])
        except ValueError:
            raise ValueError(f"Invalid threshold value: '{splits[1]}'")
        return OPS[splits[0]], value
    raise ValueError(f"Invalid threshold string: '{threshold}'")


def _binary_confusion_matrix(
    fcst: xr.DataArray, obs: xr.DataArray, threshold_array: xr.DataArray, dim: list[str]
) -> xr.DataArray:
    """
    Compute counts of the confusion matrix (contingency table, e.g. hits, misses, ...)

    Return an xarray.DataArray with the definition of the events in the dimension as given by
    `threshold_array` and the elements of the confusion matrix in the dimension `contingency`.
    """
    contingency_table = []
    for threshold in threshold_array.values.flat:
        threshold_operator, threshold_value = _threshold_value_and_operator(threshold)
        event_operator = scores.categorical.ThresholdEventOperator(
            default_event_threshold=threshold_value,
            default_op_fn=threshold_operator,
        )
        contingency_manager = event_operator.make_contingency_manager(fcst, obs)
        contingency_table.append(
            contingency_manager.transform(reduce_dims=dim).get_table()
        )
    return xr.concat(contingency_table, dim=threshold_array)


def _compute_scores(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    dim: list[str],
    prefix="",
    suffix="",
    source="",
    thresholds: list[float] | None = None,
) -> xr.Dataset:
    """
    Compute basic verification metrics between two xarray DataArrays (fcst and obs).
    Returns a xarray Dataset with the computed metrics.
    Computation of scores for continuous and categorical forecasts are supported.
    Categorical forecasts are specified with a list of events via the thresholds argument
    (e.g. [">= 10.0", "< 0.0"]).
    """
    result = xr.Dataset(
        {
            f"{prefix}BIAS{suffix}": scores.continuous.additive_bias(
                fcst, obs, reduce_dims=dim
            ),
            f"{prefix}MSE{suffix}": scores.continuous.mse(fcst, obs, reduce_dims=dim),
            f"{prefix}MAE{suffix}": scores.continuous.mae(fcst, obs, reduce_dims=dim),
            f"{prefix}CORR{suffix}": scores.continuous.correlation.pearsonr(
                fcst,
                obs,
                reduce_dims=dim,
            ),
        }
    )
    LOG.info(f"Compute scores for {prefix} {suffix}")
    LOG.info(f"compute thresholds for {thresholds}")
    if thresholds is not None:
        # check if thresholds is a list, if not convert to list
        if not isinstance(thresholds, list):
            thresholds = [thresholds]
        threshold_array = xr.DataArray(
            data=thresholds, dims=f"{prefix}threshold{suffix}"
        )
        result[f"{prefix}contingency_table{suffix}"] = _binary_confusion_matrix(
            fcst, obs, threshold_array, dim
        )

    result = result.expand_dims({"source": [source]})
    return result


def _compute_statistics(
    data: xr.DataArray,
    dim: list[str],
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


def _merge_metrics(ds: xr.Dataset, num_workers: int = 4) -> xr.Dataset:
    out = xr.merge(ds, compat="no_conflicts")
    if "ref_time" not in out.dims:
        out = out.expand_dims("ref_time").set_coords("ref_time")
    out = out.compute(num_workers=num_workers, scheduler="threads")
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
    dim: list[str] | None = None,
    threshold_dict: dict[str, list[float]] | None = None,
    num_workers: int | None = None,
) -> xr.Dataset:
    """
    Compute verification metrics and statistics comparing forecast and observation datasets.

    This function aligns the forecast (fcst) and observation (obs) xarray Datasets, applies spatial region masks,
    and computes standard verification metrics (e.g., BIAS, MSE, MAE, CORR) and basic statistics (mean, var, min, max)
    for each parameter and region. Optionally, categorical metrics using thresholds can be computed.

    Parameters
    ----------
    fcst : xr.Dataset
        Forecast dataset with named data variables (parameters) and spatial/temporal coordinates.
    obs : xr.Dataset
        Observation (truth) dataset, aligned with fcst.
    fcst_label : str
        Label for the forecast source (used in output dataset).
    obs_label : str
        Label for the observation source (used in output dataset).
    regions : list[str] or None, optional
        List of shapefile paths or region names to use for spatial aggregation. If None, uses default region ('all').
    dim : list[str] or None, optional
        List of dimension names to reduce over when computing metrics/statistics. If None, tries to infer from fcst.
    threshold_dict : dict[str, list[float]] or None, optional
        Dictionary mapping parameter names to threshold lists for categorical metrics. If None, no thresholds used.
    num_workers : int or None, optional
        Number of parallel workers for computation. If None, uses available CPU cores minus 2.

    Returns
    -------
    xr.Dataset
        Dataset containing computed verification metrics and statistics for each parameter, region, and source.
        Dimensions typically include region, source, and any non-reduced dimensions from the input datasets.
    """
    start = time.time()

    if num_workers is None:
        try:
            num_workers = len(os.sched_getaffinity(0))
        except AttributeError:
            num_workers = max((os.cpu_count() or 6) - 2, 1)

    if dim is None:
        if "x" in fcst.dims and "y" in fcst.dims:
            dim = ["x", "y"]
        elif "values" in fcst.dims:
            dim = ["values"]
        else:
            dim = ["values"]

    # rewrite the verification to use dask and xarray
    # chunk the data to avoid memory issues
    # compute the metrics in parallel
    # return the results as a xarray Dataset
    fcst_aligned, obs_aligned = xr.align(fcst, obs, join="inner", copy=False)
    region_polygons = ShapefileSpatialAggregationMasks(shp=regions)
    masks = region_polygons.get_masks(lon=obs_aligned["lon"], lat=obs_aligned["lat"])

    scores = []
    statistics = []
    for param in fcst_aligned.data_vars:
        if param not in obs_aligned.data_vars:
            LOG.warning("Parameter %s not in obs, skipping", param)
            continue
        score = []
        fcst_statistics = []
        obs_statistics = []
        thresholds = (
            threshold_dict.get(param, None)
            if isinstance(threshold_dict, dict)
            else None
        )
        LOG.info(f"Thresholds for {param}: {thresholds}")
        for region in masks.region.values:
            LOG.info("Verifying parameter %s for region %s", param, region)
            fcst_param = fcst_aligned[param].where(masks.sel(region=region))
            obs_param = obs_aligned[param].where(masks.sel(region=region))

            # scores vs time (reduce spatially)
            score.append(
                _compute_scores(
                    fcst_param,
                    obs_param,
                    prefix=param + ".",
                    source=fcst_label,
                    dim=dim,
                    thresholds=thresholds,
                ).expand_dims(region=[region])
            )

            # statistics vs time (reduce spatially)
            fcst_statistics.append(
                _compute_statistics(
                    fcst_param,
                    prefix=param + ".",
                    source=fcst_label,
                    dim=dim,
                ).expand_dims(region=[region])
            )
            obs_statistics.append(
                _compute_statistics(
                    obs_param,
                    prefix=param + ".",
                    source=obs_label,
                    dim=dim,
                ).expand_dims(region=[region])
            )

        score = xr.concat(score, dim="region")
        fcst_statistics = xr.concat(fcst_statistics, dim="region")
        obs_statistics = xr.concat(obs_statistics, dim="region")
        param_statistics = xr.concat([fcst_statistics, obs_statistics], dim="source")
        # Compute eagerly per parameter to prevent dask graph bloat
        scores.append(_merge_metrics([score], num_workers=num_workers))
        statistics.append(_merge_metrics([param_statistics], num_workers=num_workers))

    out = xr.merge(scores + statistics, join="outer", compat="no_conflicts")
    LOG.info("Computed metrics in %.2f seconds", time.time() - start)
    LOG.info("Metrics dataset: \n%s", out)
    return out
