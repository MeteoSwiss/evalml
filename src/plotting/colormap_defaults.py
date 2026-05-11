"""Mapping of field names to good default plotting parameters."""

from collections import defaultdict
from matplotlib import pyplot as plt
import warnings
from .colormap_loader import load_ncl_colormap
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
    "TOT_PREC_1H": {
        "colors": [
            "#ffffff",
            "#ebf6ff",
            "#d6e2ff",
            "#b5c9ff",
            "#8eb2ff",
            "#7f96ff",
            "#7285f8",
            "#6370f8",
            "#009e1e",
            "#3cbc3d",
            "#b3d16e",
            "#b9f96e",
            "#fff913",
            "#ffa309",
            "#e50000",
            "#bd0000",
            "#810000",
            "#000000",
        ],
        "vmin": 0,
        "vmax": 160,
        "units": "mm",
        "levels": [
            0,
            0.10,
            0.16,
            0.25,
            0.40,
            0.63,
            1.00,
            1.60,
            2.50,
            4.00,
            6.30,
            10.00,
            16.00,
            25.00,
            40.00,
            63.00,
            100.00,
            160.00,
        ],
    },
    "TOT_PREC_6H": {
        "colors": [
            "#ffffff",
            "#d6e2ff",
            "#b5c9ff",
            "#8eb2ff",
            "#7f96ff",
            "#6370f7",
            "#0063ff",
            "#009696",
            "#00c633",
            "#63ff00",
            "#96ff00",
            "#c6ff33",
            "#ffff00",
            "#ffc600",
            "#ffa000",
            "#ff7c00",
            "#ff1900",
        ],
        "vmin": 0,
        "vmax": 120,
        "units": "mm",
        "levels": [
            0,
            0.5,
            1.0,
            2.0,
            5.0,
            10.0,
            15.0,
            20.0,
            25.0,
            30.0,
            40.0,
            50.0,
            60.0,
            70.0,
            80.0,
            100.0,
            120.0,
        ],
    },
    # Sequential Reds for RMSE and MAE: error is non-negative, larger ⇒ darker.
    # Levels start at 0 so saturation maps directly to error magnitude;
    # discrete levels make absolute values readable from the colour bar.
    # RMSE:
    "U_10M.RMSE.map": {
        "cmap": plt.get_cmap("Reds", 6),
        "levels": [0, 0.5, 1, 1.5, 2, 2.5, 3],
    }
    | {"units": "m/s"},
    "V_10M.RMSE.map": {
        "cmap": plt.get_cmap("Reds", 6),
        "levels": [0, 0.5, 1, 1.5, 2, 2.5, 3],
    }
    | {"units": "m/s"},
    "SP_10M.RMSE.map": {
        "cmap": plt.get_cmap("Reds", 6),
        "levels": [0, 0.5, 1, 1.5, 2, 2.5, 3],
    }
    | {"units": "m/s"},
    "TD_2M.RMSE.map": {
        "cmap": plt.get_cmap("Reds", 6),
        "levels": [0, 0.5, 1, 1.5, 2, 2.5, 3],
    }
    | {"units": "°C"},
    "T_2M.RMSE.map": {
        "cmap": plt.get_cmap("Reds", 6),
        "levels": [0, 0.5, 1, 1.5, 2, 2.5, 3],
    }
    | {"units": "°C"},
    "PMSL.RMSE.map": {
        "cmap": plt.get_cmap("Reds", 7),
        "levels": [0, 50, 100, 150, 200, 250, 300, 350],
    }
    | {"units": "Pa"},
    "PS.RMSE.map": {
        "cmap": plt.get_cmap("Reds", 7),
        "levels": [0, 50, 100, 150, 200, 250, 300, 350],
    }
    | {"units": "Pa"},
    "TOT_PREC.RMSE.map": {
        "cmap": plt.get_cmap("Reds", 6),
        "levels": [0, 1, 1.5, 2, 3, 4],
    }
    | {"units": "mm"},
    # MAE:
    "U_10M.MAE.map": {
        "cmap": plt.get_cmap("Reds", 6),
        "levels": [0, 0.5, 1, 1.5, 2, 2.5, 3],
    }
    | {"units": "m/s"},
    "V_10M.MAE.map": {
        "cmap": plt.get_cmap("Reds", 6),
        "levels": [0, 0.5, 1, 1.5, 2, 2.5, 3],
    }
    | {"units": "m/s"},
    "SP_10M.MAE.map": {
        "cmap": plt.get_cmap("Reds", 6),
        "levels": [0, 0.5, 1, 1.5, 2, 2.5, 3],
    }
    | {"units": "m/s"},
    "TD_2M.MAE.map": {
        "cmap": plt.get_cmap("Reds", 6),
        "levels": [0, 0.5, 1, 1.5, 2, 2.5, 3],
    }
    | {"units": "°C"},
    "T_2M.MAE.map": {
        "cmap": plt.get_cmap("Reds", 6),
        "levels": [0, 0.5, 1, 1.5, 2, 2.5, 3],
    }
    | {"units": "°C"},
    "PMSL.MAE.map": {
        "cmap": plt.get_cmap("Reds", 7),
        "levels": [0, 50, 100, 150, 200, 250, 300, 350],
    }
    | {"units": "Pa"},
    "PS.MAE.map": {
        "cmap": plt.get_cmap("Reds", 7),
        "levels": [0, 50, 100, 150, 200, 250, 300, 350],
    }
    | {"units": "Pa"},
    "TOT_PREC.MAE.map": {
        "cmap": plt.get_cmap("Reds", 6),
        "levels": [0, 1, 1.5, 2, 3, 4],
    }
    | {"units": "mm"},
    # the levels for precip are a bit on the bright side, but still worth keeping consistent with RMSE.
    # Bias:
    # diverging colour scheme for the Bias to reflect the nature of the data (can be positive or negative, symmetric).
    # Red-Blue colour scheme for all variables except precipitation, where a Brown-Green scheme is more suggestive.
    "U_10M.BIAS.map": {
        "cmap": plt.get_cmap("RdBu_r", 9),
        "levels": np.arange(start=-2.25, stop=2.26, step=0.5),
    }
    | {"units": "m/s"},
    "V_10M.BIAS.map": {
        "cmap": plt.get_cmap("RdBu_r", 9),
        "levels": np.arange(start=-2.25, stop=2.26, step=0.5),
    }
    | {"units": "m/s"},
    "SP_10M.BIAS.map": {
        "cmap": plt.get_cmap("RdBu_r", 9),
        "levels": np.arange(start=-2.25, stop=2.26, step=0.5),
    }
    | {"units": "m/s"},
    "TD_2M.BIAS.map": {
        "cmap": plt.get_cmap("RdBu_r", 11),
        "levels": np.arange(start=-2.75, stop=2.76, step=0.5),
    }
    | {"units": "°C"},
    "T_2M.BIAS.map": {
        "cmap": plt.get_cmap("RdBu_r", 11),
        "levels": np.arange(start=-2.75, stop=2.76, step=0.5),
    }
    | {"units": "°C"},
    "PMSL.BIAS.map": {
        "cmap": plt.get_cmap("RdBu_r", 11),
        "levels": np.arange(start=-110, stop=111, step=20),
    }
    | {"units": "Pa"},
    "PS.BIAS.map": {
        "cmap": plt.get_cmap("RdBu_r", 11),
        "levels": np.arange(start=-110, stop=111, step=20),
    }
    | {"units": "Pa"},
    "TOT_PREC.BIAS.map": {
        "cmap": plt.get_cmap("BrBG", 9),
        "levels": [-1, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1],
    }
    | {"units": "mm"},
}

CMAP_DEFAULTS = defaultdict(_fallback, _CMAP_DEFAULTS)
