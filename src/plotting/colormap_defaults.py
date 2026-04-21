"""Mapping of field names to good default plotting parameters."""

from collections import defaultdict
from matplotlib import pyplot as plt
import warnings
from .colormap_loader import load_ncl_colormap


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
    # Error-field colormaps: diverging, centered at zero. Used when plotting
    # (forecast - truth) or (baseline - truth) differences.
    "SP_error": {
        "cmap": plt.get_cmap("RdBu_r", 21),
        "vmin": -500,
        "vmax": 500,
        "units": "Pa",
    },
    "TD_2M_error": {
        "cmap": plt.get_cmap("RdBu_r", 21),
        "vmin": -10,
        "vmax": 10,
        "units": "degC",
    },
    "T_2M_error": {
        "cmap": plt.get_cmap("RdBu_r", 21),
        "vmin": -10,
        "vmax": 10,
        "units": "degC",
    },
    "U_10M_error": {
        "cmap": plt.get_cmap("RdBu_r", 17),
        "vmin": -8,
        "vmax": 8,
        "units": "m/s",
    },
    "V_10M_error": {
        "cmap": plt.get_cmap("RdBu_r", 17),
        "vmin": -8,
        "vmax": 8,
        "units": "m/s",
    },
    "SP_10M_error": {
        "cmap": plt.get_cmap("RdBu_r", 17),
        "vmin": -8,
        "vmax": 8,
        "units": "m/s",
    },
    "T_850_error": {
        "cmap": plt.get_cmap("RdBu_r", 21),
        "vmin": -10,
        "vmax": 10,
        "units": "K",
    },
    "FI_850_error": {
        "cmap": plt.get_cmap("RdBu_r", 21),
        "vmin": -500,
        "vmax": 500,
        "units": "m^2/s^2",
    },
    "QV_925_error": {
        "cmap": plt.get_cmap("BrBG", 21),
        "vmin": -0.002,
        "vmax": 0.002,
        "units": "kg/kg",
    },
    "TOT_PREC_error": {
        "cmap": plt.get_cmap("BrBG", 21),
        "vmin": -20,
        "vmax": 20,
        "units": "mm",
    },
}

CMAP_DEFAULTS = defaultdict(_fallback, _CMAP_DEFAULTS)
