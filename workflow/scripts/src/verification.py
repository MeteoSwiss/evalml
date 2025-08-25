import xarray as xr
import time
import logging

LOG = logging.getLogger(__name__)


def _compute_scores(
    fcst: xr.DataArray, obs: xr.DataArray, dim=["x", "y"], prefix="", suffix="", source=""
) -> xr.Dataset:
    """
    Compute basic verification metrics between two xarray DataArrays (fcst and obs).
    Returns a xarray Dataset with the computed metrics.
    """
    error = fcst - obs
    scores = xr.Dataset(
        {
            f"{prefix}BIAS{suffix}": error.mean(dim=dim),
            f"{prefix}MSE{suffix}": (error**2).mean(dim=dim),
            f"{prefix}MAE{suffix}": abs(error).mean(dim=dim),
            f"{prefix}VAR{suffix}": error.var(dim=dim, skipna=True),
            f"{prefix}CORR{suffix}": xr.corr(fcst, obs, dim=dim),
            f"{prefix}R2{suffix}": xr.corr(fcst, obs, dim=dim) ** 2,
        }
    )
    scores = scores.expand_dims({"source": [source]})
    return scores


def _compute_statistics(
    data: xr.DataArray, dim=["x", "y"], prefix="", suffix="", source="",
) -> xr.Dataset:
    """
    Compute basic statistics of a xarray DataArray (data).
    Returns a xarray Dataset with the computed statistics.
    """
    stats = xr.Dataset(
        {
            f"{prefix}mean{suffix}": data.mean(dim=dim),
            f"{prefix}var{suffix}": data.var(dim=dim, skipna=True),
            f"{prefix}min{suffix}": data.min(dim=dim),
            f"{prefix}max{suffix}": data.max(dim=dim),
        }
    )
    stats = stats.expand_dims({"source": [source]})
    return stats

def _merge_metrics(ds = xr.Dataset) -> xr.Dataset:
    out = xr.merge(ds)
    if "ref_time" not in out.dims:
        out = out.expand_dims("ref_time").set_coords("ref_time")
    out = out.compute(num_workers=4, scheduler="threads")
    return out

def verify(
    fcst: xr.Dataset, obs: xr.Dataset, fcst_label: str, obs_label: str
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
    scores = []
    statistics = []
    for param in fcst.data_vars:
        if param not in obs.data_vars:
            LOG.warning("Parameter %s not in obs, skipping", param)
            continue
        LOG.info("Verifying parameter %s", param)
        fcst_param, obs_param = xr.align(fcst[param], obs[param], join="inner")
        score = _compute_scores(fcst_param, obs_param, prefix=param + ".", source=fcst_label)
        scores.append(score)
        score = _compute_scores(
            fcst_param,
            obs_param,
            prefix=param + ".",
            suffix=".spatial",
            dim="lead_time",
            source=fcst_label
        )
        scores.append(score)
        fcst_statistics = _compute_statistics(fcst_param, prefix=param + ".", source=fcst_label)
        obs_statistics = _compute_statistics(obs_param, prefix=param + ".", source=obs_label)
        statistics.append(xr.merge([fcst_statistics, obs_statistics]))

    scores = _merge_metrics(scores)
    statistics = _merge_metrics(statistics)
    out = xr.merge([scores, statistics])
    LOG.info("Computed metrics in %.2f seconds", time.time() - start)
    LOG.info("Metrics dataset: \n%s", out)
    return out
