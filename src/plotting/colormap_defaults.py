"""Mapping of field names to good default plotting parameters."""

from collections import defaultdict
from matplotlib import pyplot as plt
import warnings
from .colormap_loader import load_ncl_colormap


def _fallback():
    warnings.warn("No colormap found for this parameter, using fallback.", UserWarning)
    return {"cmap": plt.get_cmap("viridis"), "norm": None, "units": ""}


_CMAP_DEFAULTS = {
    "sp": {"cmap": plt.get_cmap("coolwarm", 11), "vmin": 800 * 100, "vmax": 1100 * 100},
    "2d": load_ncl_colormap("t2m_29lev.ct"),
    "2t": load_ncl_colormap("t2m_29lev.ct") | {"units": "degC"},
    "10v": load_ncl_colormap("modified_uv_17lev.ct") | {"units": "m/s"},
    "10u": load_ncl_colormap("modified_uv_17lev.ct") | {"units": "m/s"},
    "10sp": load_ncl_colormap("modified_uv_17lev.ct") | {"units": "m/s"},
    "10si": {"cmap": plt.get_cmap("GnBu", 11), "vmin": 0, "vmax": 25},
    "t_850": {"cmap": plt.get_cmap("inferno", 11), "vmin": 220, "vmax": 310},
    "z_850": {"cmap": plt.get_cmap("coolwarm", 11), "vmin": 8000, "vmax": 17000},
    "q_925": load_ncl_colormap("RH_6lev.ct"),
}

CMAP_DEFAULTS = defaultdict(_fallback, _CMAP_DEFAULTS)
