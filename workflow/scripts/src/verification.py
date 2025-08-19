import xarray as xr
import pandas as pd
import time
import logging

LOG = logging.getLogger(__name__)


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
    error_arr = fcst_arr - obs_arr

    out_fcst = xr.Dataset(
        {
            "BIAS": error_arr.mean(dim=["y", "x"]),
            "MSE": (error_arr**2).mean(dim=["y", "x"]),
            "MAE": abs(error_arr).mean(dim=["y", "x"]),
            "VAR": error_arr.var(dim=["y", "x"]),
            "CORR": xr.corr(fcst_arr, obs_arr, dim=["y", "x"]),
            "mean": fcst_arr.mean(dim=["y", "x"]),
            "var": fcst_arr.var(dim=["y", "x"]),
            "min": fcst_arr.min(dim=["y", "x"]),
            "max": fcst_arr.max(dim=["y", "x"]),
        },
    ).expand_dims({"label": [fcst_label]})

    out_obs = xr.Dataset(
        {
            "mean": obs_arr.mean(dim=["y", "x"]),
            "var": obs_arr.var(dim=["y", "x"]),
            "min": obs_arr.min(dim=["y", "x"]),
            "max": obs_arr.max(dim=["y", "x"]),
        }
    ).expand_dims({"label": [obs_label]})
    out = xr.merge([out_fcst, out_obs]).compute(num_workers=4, scheduler="threads")
    LOG.info("Computed statistics in %.2f seconds", time.time() - start)
    LOG.info("Statistics dataset: \n%s", out)
    out = (
        out.assign(R2=lambda ds: ds.CORR**2)
        .to_array("metric")
        .to_dataframe(name="value")
        .dropna()
        .reset_index()
    )
    return out
