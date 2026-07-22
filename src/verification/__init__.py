import logging
import os
import re
import time

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

_T_LAPSE_RATE = 0.0065  # K/m — ICAO standard atmosphere
_LAPSE_RATE_PARAMS: dict[str, float] = {"T_2M": _T_LAPSE_RATE}


def apply_lapse_rate_correction_inplace(
    fcst: xr.Dataset,
    obs: xr.Dataset,
    params: list[str],
) -> xr.Dataset:
    """Correct T_2M and TD_2M in *fcst* to the elevation of *obs*.

    Requires both *fcst* and *obs* to carry an ``elevation`` coordinate (metres).
    For forecasts this is the model orography from the ICON external parameter
    file; for observations it comes from station metadata or FIS geopotential.
    The function silently returns *fcst* unchanged when either coordinate is
    absent so that pipelines without elevation data are not broken.

    Formula applied per parameter:
        T_corrected = T_forecast − Γ × (elevation_obs − elevation_fcst)

    A positive height difference (obs higher than forecast grid cell) lowers the
    corrected value, consistent with the standard atmospheric lapse rate.
    """
    missing = [
        name
        for name, ds in (("forecast", fcst), ("observations", obs))
        if "elevation" not in ds.coords
    ]
    if missing:
        raise ValueError(
            f"Lapse-rate correction requested but elevation coordinate is missing "
            f"from: {', '.join(missing)}."
        )
    dz = obs["elevation"] - fcst["elevation"]

    dz_vals = np.asarray(dz).ravel()
    n_missing = int(np.sum(~np.isfinite(dz_vals)))
    if n_missing > 0:
        raise ValueError(
            f"Lapse-rate correction: {n_missing} missing elevation value(s) in dz; "
            "both forecast and observation elevation coordinates must be fully defined."
        )

    max_abs_dz = float(np.abs(dz_vals).max())
    if max_abs_dz < 1.0:
        LOG.info(
            "Lapse-rate correction: forecast and truth altitudes agree within rounding "
            "(max |Δz| = %.2f m); correction is negligible.",
            max_abs_dz,
        )
    else:
        LOG.info(
            "Lapse-rate correction: Δz range [%.1f, %.1f] m, mean %.1f m.",
            float(dz_vals.min()),
            float(dz_vals.max()),
            float(dz_vals.mean()),
        )

    for param, rate in _LAPSE_RATE_PARAMS.items():
        if param in params and param in fcst.data_vars:
            correction = rate * dz
            if max_abs_dz >= 1.0:
                c_vals = np.asarray(correction).ravel()
                LOG.info(
                    "Lapse-rate correction for %s (Γ=%.4f K/m): "
                    "correction range [%.3f, %.3f] K, mean %.3f K.",
                    param,
                    rate,
                    float(c_vals.min()),
                    float(c_vals.max()),
                    float(c_vals.mean()),
                )
            fcst[param] = fcst[param] - correction


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
        self,
        regions: list[dict],
        src_crs=ccrs.epsg(2056),
        dst_crs=ccrs.PlateCarree(),
    ):
        proj = pyproj.Transformer.from_crs(
            src_crs.proj4_init, dst_crs.proj4_init, always_xy=True
        ).transform

        self.regions = {}
        for spec in regions:
            name = spec["name"]
            if spec["type"] == "bbox":
                lon_min, lon_max, lat_min, lat_max = spec["bbox"]
                self.regions[name] = [_bbox_polygon(lon_min, lon_max, lat_min, lat_max)]
            elif spec["type"] == "shp":
                reader = Reader(spec["path"])
                self.regions[name] = [
                    transform(proj, record.geometry) for record in reader.records()
                ]

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


def decode_metric(label: str) -> str:
    OP_DICT = {
        "_gt_": " > ",
        "_ge_": " >= ",
        "_lt_": " < ",
        "_le_": " <= ",
        "_eq_": " == ",
        "_ne_": " != ",
    }
    for k, v in OP_DICT.items():
        label = label.replace(k, v)
    label = re.sub(r"(?<=\d)p(?=\d)", ".", label)
    return label


def _binary_confusion_matrix(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    thresholds: list[tuple],
    dim: list[str],
) -> xr.DataArray:
    """
    Compute confusion matrix counts (tp, fp, fn, tn, total) for all thresholds at once.

    Thresholds sharing the same operator are broadcast as a single extra dimension so
    the spatial reduction happens in one dask pass instead of one pass per threshold.
    """
    from collections import defaultdict

    op_groups: dict[str, list[float]] = defaultdict(list)
    for op_txt, val in thresholds:
        op_groups[op_txt].append(val)

    # Points where both fcst and obs carry valid (non-masked) data
    valid = fcst.notnull() & obs.notnull()
    # Fill NaN so comparisons return False rather than NaN for masked points
    fcst_filled = fcst.where(valid, 0)
    obs_filled = obs.where(valid, 0)

    contingency_dim = xr.DataArray(
        ["tp_count", "fp_count", "fn_count", "tn_count", "total_count"],
        dims="contingency",
    )

    tables = []
    for op_txt, values in op_groups.items():
        try:
            op_fn = getattr(op, op_txt)
        except AttributeError:
            raise AttributeError(f"operator {op_txt} is not available")

        threshold_labels = [f"{op_txt}_{str(v).replace('.', 'p')}" for v in values]
        # Broadcast comparison across all threshold values simultaneously
        vals_da = xr.DataArray(
            values, dims="threshold", coords={"threshold": threshold_labels}
        )
        fcst_ev = op_fn(fcst_filled, vals_da)  # (...spatial/time..., threshold) bool
        obs_ev = op_fn(obs_filled, vals_da)

        tp = (fcst_ev & obs_ev & valid).sum(dim)
        fp = (fcst_ev & ~obs_ev & valid).sum(dim)
        fn = (~fcst_ev & obs_ev & valid).sum(dim)
        tn = (~fcst_ev & ~obs_ev & valid).sum(dim)
        total = valid.sum(dim).broadcast_like(tp)

        tables.append(xr.concat([tp, fp, fn, tn, total], dim=contingency_dim))

    return xr.concat(tables, dim="threshold")


def _compute_scores(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    dim: list[str],
    prefix="",
    suffix="",
    source="",
    thresholds: dict[str, list[float]] | None = None,
) -> xr.Dataset:
    """
    Compute basic verification metrics between two xarray DataArrays (fcst and obs).
    Returns a xarray Dataset with the computed metrics.
    Computation of scores for continuous and categorical forecasts are supported.
    Categorical forecasts are specified via a dict mapping operator keys (gt, ge, lt, le, eq, ne)
    to lists of threshold values (e.g. {"gt": [10.0], "lt": [0.0]}).
    """
    LOG.info(f"Compute scores for {prefix} {suffix}")
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
    if thresholds is not None:
        LOG.info(f"compute thresholds for {thresholds}")
        threshold_pairs = [
            (key, val) for key, values in thresholds.items() for val in values
        ]
        confusion_matrix = (
            _binary_confusion_matrix(fcst, obs, threshold_pairs, dim).rename(
                {"threshold": f"{prefix}threshold{suffix}"}
            )  # make threshold dimension unique
        )
        result[f"{prefix}contingency_table{suffix}"] = confusion_matrix

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
    if "forecast_reference_time" not in out.dims:
        out = out.expand_dims("forecast_reference_time").set_coords(
            "forecast_reference_time"
        )
    out = out.compute(num_workers=num_workers, scheduler="threads")
    return out


def _bbox_polygon(lon_min, lon_max, lat_min, lat_max) -> Polygon:
    return Polygon(
        [
            (lon_min, lat_min),
            (lon_max, lat_min),
            (lon_max, lat_max),
            (lon_min, lat_max),
            (lon_min, lat_min),
        ]
    )


def verify(
    fcst: xr.Dataset,
    obs: xr.Dataset,
    fcst_label: str,
    obs_label: str,
    regions: list[dict] | None = None,
    dim: list[str] | None = None,
    threshold_dict: dict[str, dict[str, list[float]]] | None = None,
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
    regions : list[dict] or None, optional
        Ordered list of region specs. Each entry is either
        ``{"type": "bbox", "name": ..., "bbox": [lon_min, lon_max, lat_min, lat_max]}`` or
        ``{"type": "shp", "name": ..., "path": ...}``. The list order is preserved in the
        output NetCDF region coordinate; the first entry is the domain region used by
        dashboards and scorecards when region stratification is not active.
    dim : list[str] or None, optional
        List of dimension names to reduce over when computing metrics/statistics. If None, tries to infer from fcst.
    threshold_dict : dict[str, dict[str, list[float]]] or None, optional
        Dictionary mapping parameter names to threshold dicts for categorical metrics.
        Each threshold dict maps operator keys (gt, ge, lt, le, eq, ne) to lists of values.
        If None, no thresholds used.
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

    if not regions:
        raise ValueError(
            "At least one region must be specified. "
            "Provide an ordered list of region specs via the 'regions' argument."
        )

    fcst_aligned, obs_aligned = xr.align(fcst, obs, join="inner", copy=False)
    region_polygons = ShapefileSpatialAggregationMasks(regions=regions)
    masks = region_polygons.get_masks(
        lon=obs_aligned["longitude"], lat=obs_aligned["latitude"]
    )

    scores = []
    statistics = []
    for param in fcst_aligned.data_vars:
        if param not in obs_aligned.data_vars:
            LOG.warning("Parameter %s not in obs, skipping", param)
            continue
        thresholds = (
            threshold_dict.get(param, None)
            if isinstance(threshold_dict, dict)
            else None
        )
        LOG.info(
            "Verifying parameter %s for %d regions, thresholds: %s",
            param,
            len(masks.region),
            thresholds,
        )

        # Apply all region masks at once via broadcast — adds "region" as leading dim
        fcst_param = fcst_aligned[param].where(masks)
        obs_param = obs_aligned[param].where(masks)

        score = _compute_scores(
            fcst_param,
            obs_param,
            prefix=param + ".",
            source=fcst_label,
            dim=dim,
            thresholds=thresholds,
        )
        fcst_stats = _compute_statistics(
            fcst_param,
            prefix=param + ".",
            source=fcst_label,
            dim=dim,
        )
        obs_stats = _compute_statistics(
            obs_param,
            prefix=param + ".",
            source=obs_label,
            dim=dim,
        )
        param_statistics = xr.concat([fcst_stats, obs_stats], dim="source")
        # Compute eagerly per parameter to prevent dask graph bloat
        scores.append(_merge_metrics([score], num_workers=num_workers))
        statistics.append(_merge_metrics([param_statistics], num_workers=num_workers))

    out = xr.merge(scores + statistics, join="outer", compat="no_conflicts")
    LOG.info("Computed metrics in %.2f seconds", time.time() - start)
    LOG.info("Metrics dataset: \n%s", out)
    return out
