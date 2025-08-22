import xarray as xr
import pandas as pd
import time
import logging

LOG = logging.getLogger(__name__)

def _scores(fcst: xr.DataArray, obs: xr.DataArray, suffix: str = "") -> xr.Dataset:
    """
    Compute basic verification metrics between two xarray DataArrays (fcst and obs).
    Returns an xarray Dataset with the computed metrics.
    """
    error = fcst - obs
    scores = xr.Dataset(
        {
            f"BIAS{suffix}": error.mean(dim=["y", "x"]),
            f"MSE{suffix}": (error**2).mean(dim=["y", "x"]),
            f"MAE{suffix}": abs(error).mean(dim=["y", "x"]),
            f"VAR{suffix}": error.var(dim=["y", "x"]),
            f"CORR{suffix}": xr.corr(fcst, obs, dim=["y", "x"]),
            f"R2{suffix}": xr.corr(fcst, obs, dim=["y", "x"]) ** 2,
        }
    )
    return scores

def _statistics(data: xr.DataArray, suffix: str = "") -> xr.Dataset:
    """
    Compute basic statistics for an xarray DataArray (data).
    Returns an xarray Dataset with the computed statistics.
    """
    stats = xr.Dataset(
        {
            f"mean{suffix}": data.mean(dim=["y", "x"]),
            f"var{suffix}": data.var(dim=["y", "x"]),
            f"min{suffix}": data.min(dim=["y", "x"]),
            f"max{suffix}": data.max(dim=["y", "x"]),
        }
    )
    return stats

def verify(
    fcst: xr.Dataset, obs: xr.Dataset, fcst_label: str, obs_label: str
) -> pd.DataFrame:
    """
    Compare two xarray Datasets (fcst and obs) and return pandas DataFrame with
    basic verification metrics and statistics for both fcst and obs.
    """
    start = time.time()
    fcst_arr = fcst.to_array("param").chunk("auto")
    obs_arr = obs.to_array("param").chunk("auto")
    scores = _scores(fcst_arr, obs_arr)
    fcst_stats = _statistics(fcst_arr)
    obs_stats = _statistics(obs_arr)
    
    fcst_anom = fcst_arr - fcst_arr.mean(dim=["lead_time"])
    obs_anom = obs_arr - obs_arr.mean(dim=["lead_time"])
    scores = xr.merge([scores, _scores(fcst_anom, obs_anom, suffix="_anom")])
    fcst_stats = xr.merge([fcst_stats, _statistics(fcst_anom, suffix="_anom")])
    obs_stats = xr.merge([obs_stats, _statistics(obs_anom, suffix="_anom")])

    for i in [3,5,7,9,11,15]:
        fcst_upscale = fcst_arr.coarsen(x=i, y=i, boundary="trim").mean()
        obs_upscale = obs_arr.coarsen(x=i, y=i, boundary="trim").mean()
        scores = xr.merge([scores, _scores(fcst_upscale, obs_upscale, suffix=f"_upscale{i}")])
        fcst_stats = xr.merge([fcst_stats, _statistics(fcst_upscale, suffix=f"_upscale{i}")])
        obs_stats = xr.merge([obs_stats, _statistics(obs_upscale, suffix=f"_upscale{i}")])

    out_fcst = xr.merge([scores, fcst_stats]).expand_dims({"source": [fcst_label]})
    out_obs = obs_stats.expand_dims({"source": [obs_label]})
    out = xr.merge([out_fcst, out_obs]).compute(num_workers=4, scheduler="threads")
    LOG.info("Computed statistics in %.2f seconds", time.time() - start)
    LOG.info("Statistics dataset: \n%s", out)
    out = (
        out
        .to_array("metric")
        .to_dataframe(name="value")
        .dropna()
        .reset_index()
    )
    return out
