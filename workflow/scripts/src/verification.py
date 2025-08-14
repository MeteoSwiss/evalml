import xarray as xr
import pandas as pd
import time
import logging

LOG = logging.getLogger(__name__)


def verify(fcst: xr.Dataset, obs: xr.Dataset) -> pd.DataFrame:
    """
    Compare two xarray Datasets (fcst and obs) and return an xarray Dataset with
    basic verification metrics and statistics for both fcst and obs.
    """
    start = time.time()
    fcst_arr = fcst.to_array("param").chunk("auto")
    obs_arr = obs.to_array("param")
    error_arr = fcst_arr - obs_arr

    out = xr.Dataset(
        {
            "BIAS": error_arr.mean(dim=["y", "x"]),
            "MSE": (error_arr**2).mean(dim=["y", "x"]),
            "MAE": abs(error_arr).mean(dim=["y", "x"]),
            "VAR": error_arr.var(dim=["y", "x"]),
            "CORR": xr.corr(fcst_arr, obs_arr, dim=["y", "x"]),
            "fcst_mean": fcst_arr.mean(dim=["y", "x"]),
            "fcst_var": fcst_arr.var(dim=["y", "x"]),
            "fcst_min": fcst_arr.min(dim=["y", "x"]),
            "fcst_max": fcst_arr.max(dim=["y", "x"]),
            "obs_mean": obs_arr.mean(dim=["y", "x"]),
            "obs_var": obs_arr.var(dim=["y", "x"]),
            "obs_min": obs_arr.min(dim=["y", "x"]),
            "obs_max": obs_arr.max(dim=["y", "x"]),
        }
    )
    out = out.compute(num_workers=4, scheduler="threads")
    LOG.info("Computed statistics in %.2f seconds", time.time() - start)
    LOG.info("Statistics dataset: \n%s", out)
    # adopt wide format for now, where each column contains a metric / statistic
    out = out.assign(R2=lambda ds: ds.CORR**2).to_dataframe().reset_index()
    return out
