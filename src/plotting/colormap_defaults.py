"""Mapping of field names to good default plotting parameters."""

from collections import defaultdict
from matplotlib import pyplot as plt
import warnings
from .colormap_loader import load_ncl_colormap
from matplotlib.colors import BoundaryNorm
import numpy as np

def _fallback():
    warnings.warn("No colormap found for this parameter, using fallback.", UserWarning)
    return {"cmap": plt.get_cmap("viridis"), "norm": None, "units": ""}

_CMAP_DEFAULTS = {
    "SP": {"cmap": plt.get_cmap("coolwarm", 11), "vmin": 800 * 100, "vmax": 1100 * 100},
    "TD_2M": load_ncl_colormap("t2m_29lev.ct"),
    "T_2M": load_ncl_colormap("t2m_29lev.ct") | {"units": "degC"},
    "V_10M": load_ncl_colormap("modified_uv_17lev.ct") | {"units": "m/s"},
    "U_10M": load_ncl_colormap("modified_uv_17lev.ct") | {"units": "m/s"},
    "SP_10M": load_ncl_colormap("modified_uv_17lev.ct") | {"units": "m/s"},
    # "10si": {"cmap": plt.get_cmap("GnBu", 11), "vmin": 0, "vmax": 25},
    "T_850": {"cmap": plt.get_cmap("inferno", 11), "vmin": 220, "vmax": 310},
    "FI_850": {"cmap": plt.get_cmap("coolwarm", 11), "vmin": 8000, "vmax": 17000},
    "QV_925": load_ncl_colormap("RH_6lev.ct"),
    "TOT_PREC": {
        "colors": [
            "#ffffff",
            "#04e9e7",
            "#019ff4",
            "#0300f4",
            "#02fd02",
            "#01c501",
            "#008e00",
            "#fdf802",
            "#e5bc00",
            "#fd9500",
            "#fd0000",
            "#d40000",
            "#bc0000",
            "#f800fd",
        ],
        "vmin": 0,
        "vmax": 100,
        "units": "mm",
        "levels": [0, 0.05, 0.1, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 100],
    },

    # hard-code this for the moment, can still make smarter later on:
    # RMSE and MAE first (is all the same). Sequential colour map to reflect the nature of the data (error, all positive).
    # Red is suggestive of 'bad' (high error).
    # Use a limited number of levels so that absolute values of error can be read from the map. 
    # always start at 0 so that the saturation of the colour corresponds to the error magnitude.

    # RMSE:
    "U_10M.RMSE.spatial":    {"cmap": plt.get_cmap("Reds", 6), "vmin": 0} | {"units": "m/s"},
    "V_10M.RMSE.spatial":    {"cmap": plt.get_cmap("Reds", 6), "vmin": 0} | {"units": "m/s"},
    "SP_10M.RMSE.spatial":   {"cmap": plt.get_cmap("Reds", 6), "vmin": 0} | {"units": "m/s"},
    "TD_2M.RMSE.spatial":    {"cmap": plt.get_cmap("Reds", 6), "levels": [0, 0.5, 1, 1.5, 2, 2.5, 3]} | {"units": "°C"},
    "T_2M.RMSE.spatial":     {"cmap": plt.get_cmap("Reds", 6), "levels": [0, 0.5, 1, 1.5, 2, 2.5, 3]} | {"units": "°C"},
    "PMSL.RMSE.spatial":     {"cmap": plt.get_cmap("Reds", 6), "vmin": 0} | {"units": "Pa"},
    "PS.RMSE.spatial":       {"cmap": plt.get_cmap("Reds", 6), "vmin": 0} | {"units": "Pa"},
    "TOT_PREC.RMSE.spatial": {"cmap": plt.get_cmap("Reds", 6), "levels": [0, 0.001, 0.0015, 0.002, 0.003, 0.004]} | {"units": "mm"},
    # would ideally want a 6th colour on the high end of the colour scale, but somehow 
    # matplotlib does not do that -> ?

    # MAE:
    "U_10M.MAE.spatial":    {"cmap": plt.get_cmap("Reds", 6), "vmin": 0} | {"units": "m/s"},
    "V_10M.MAE.spatial":    {"cmap": plt.get_cmap("Reds", 6), "vmin": 0} | {"units": "m/s"},
    "SP_10M.MAE.spatial":   {"cmap": plt.get_cmap("Reds", 6), "vmin": 0} | {"units": "m/s"},
    "TD_2M.MAE.spatial":    {"cmap": plt.get_cmap("Reds", 6), "levels": [0, 0.5, 1, 1.5, 2, 2.5, 3]} | {"units": "°C"},
    "T_2M.MAE.spatial":     {"cmap": plt.get_cmap("Reds", 6), "levels": [0, 0.5, 1, 1.5, 2, 2.5, 3]} | {"units": "°C"},
    "PMSL.MAE.spatial":     {"cmap": plt.get_cmap("Reds", 6), "vmin": 0} | {"units": "Pa"},
    "PS.MAE.spatial":       {"cmap": plt.get_cmap("Reds", 6), "vmin": 0} | {"units": "Pa"},
    "TOT_PREC.MAE.spatial": {"cmap": plt.get_cmap("Reds", 6), "levels": [0, 0.001, 0.0015, 0.002, 0.003, 0.004]} | {"units": "mm"}, 
    # the levels for precip are a bit on the bright side, but still worth keeping consistent with RMSE. 

    # Bias:
    # diverging colour scheme for the Bias to reflect the nature of the data (can be positive or negative, symmetric).
    # Red-Blue colour scheme for all variables except precipitation, where a Brown-Green scheme is more suggestive.
    "U_10M.BIAS.spatial":    {"cmap": plt.get_cmap("RdBu_r", 11)} | {"units": "m/s"}, 
    "V_10M.BIAS.spatial":    {"cmap": plt.get_cmap("RdBu_r", 11)} | {"units": "m/s"},
    "SP_10M.BIAS.spatial":   {"cmap": plt.get_cmap("RdBu_r", 11)} | {"units": "m/s"},
    "TD_2M.BIAS.spatial":    {"cmap": plt.get_cmap("RdBu_r", 11), "levels" : np.arange(start = -2.75, stop = 2.76, step = 0.5)} | {"units": "°C"},
    "T_2M.BIAS.spatial":     {"cmap": plt.get_cmap("RdBu_r", 11), "levels" : np.arange(start = -2.75, stop = 2.76, step = 0.5)} | {"units": "°C"},
    "PMSL.BIAS.spatial":     {"cmap": plt.get_cmap("RdBu_r", 11)} | {"units": "Pa"},
    "PS.BIAS.spatial":       {"cmap": plt.get_cmap("RdBu_r", 11)} | {"units": "Pa"},
    "TOT_PREC.BIAS.spatial": {"cmap": plt.get_cmap("BrBG", 9), "levels": [-0.001, -0.0005, -0.00025, -0.0001, 0.0001, 0.00025, 0.0005, 0.001]} | {"units": "mm"}
}

CMAP_DEFAULTS = defaultdict(_fallback, _CMAP_DEFAULTS)
