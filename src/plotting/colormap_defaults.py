"""Mapping of field names to good default plotting parameters."""

from collections import defaultdict
from matplotlib import pyplot as plt
import warnings
from .colormap_loader import load_ncl_colormap
import numpy as np


def _fallback():
    warnings.warn("No colormap found for this parameter, using fallback.", UserWarning)
    return {"cmap": plt.get_cmap("viridis"), "norm": None, "units": ""}


# Sequential Reds shared by the error-magnitude scores (RMSE, MAE, STDE), which
# are all non-negative. Defined once per parameter under the generic
# "{param}.score.map" key; the score-map lookup falls back to it for any score
# without a dedicated "{param}.{score}.map" entry. To give a single score its
# own colours, add that explicit key — it takes precedence over this fallback.
_SCORE_REDS = {"cmap": plt.get_cmap("Reds", 6), "levels": [0, 0.5, 1, 1.5, 2, 2.5, 3]}
_SCORE_REDS_PA = {
    "cmap": plt.get_cmap("Reds", 7),
    "levels": [0, 50, 100, 150, 200, 250, 300, 350],
}
_SCORE_REDS_PRECIP = {"cmap": plt.get_cmap("Reds", 6), "levels": [0, 1, 1.5, 2, 3, 4]}


def _precip_score_map(accum_h: int) -> dict:
    """Score-map config for period-accumulated precip, levels scaled by accum_h / 2."""
    scale = accum_h / 2
    return {
        "cmap": plt.get_cmap("Reds", 6),
        "levels": [lev * scale for lev in [0, 1, 1.5, 2, 3, 4]],
        "units": "mm",
    }


def _precip_bias_map(accum_h: int) -> dict:
    """BIAS-map config for period-accumulated precip, levels scaled by accum_h / 2."""
    scale = accum_h / 2
    return {
        "cmap": plt.get_cmap("BrBG", 9),
        "levels": [lev * scale for lev in [-1, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1]],
        "units": "mm",
    }


_CMAP_DEFAULTS = {
    "SP": {
        "cmap": plt.get_cmap("coolwarm", 11),
        "vmin": 800 * 100,
        "vmax": 1100 * 100,
        "extend": "both",
    },
    "TD_2M": load_ncl_colormap("t2m_29lev.ct") | {"extend": "both"},
    "T_2M": load_ncl_colormap("t2m_29lev.ct") | {"units": "degC", "extend": "both"},
    "V_10M": load_ncl_colormap("modified_uv_17lev.ct")
    | {"units": "m/s", "extend": "both"},
    "U_10M": load_ncl_colormap("modified_uv_17lev.ct")
    | {"units": "m/s", "extend": "both"},
    "SP_10M": load_ncl_colormap("modified_uv_17lev.ct")
    | {"units": "m/s", "extend": "max"},
    "T_850": {
        "cmap": plt.get_cmap("inferno", 11),
        "vmin": 220,
        "vmax": 310,
        "extend": "both",
    },
    "FI_850": {
        "cmap": plt.get_cmap("coolwarm", 11),
        "vmin": 8000,
        "vmax": 17000,
        "extend": "both",
    },
    "QV_925": load_ncl_colormap("RH_6lev.ct") | {"extend": "both"},
    "CLCT": {
        # extend="neither" relies on preprocess_field() clipping away from
        # exact 0/1 (see plot_forecast_frame.py) to avoid a tricontourf bug
        # on orthographic projections.
        "cmap": plt.get_cmap("Blues_r"),
        "vmin": 0,
        "vmax": 1,
        "extend": "neither",
        "units": "",
        "levels": list(np.linspace(0, 1, 21)),
    },
    "CLCL": {
        "cmap": plt.get_cmap("Blues_r"),
        "vmin": 0,
        "vmax": 1,
        "extend": "neither",
        "units": "",
        "levels": list(np.linspace(0, 1, 21)),
    },
    "SSRD": {
        # tricontourf always bands regardless of "levels" being set (it falls
        # back to an auto locator with ~7 bands otherwise) — use a fine level
        # set here to approximate a smooth gradient instead. extend="max"
        # only (not "both") since preprocess_field() already clips away from
        # exact 0 — see CLCT.
        "cmap": plt.get_cmap("YlOrRd"),
        "vmin": 0,
        "vmax": 4e6,
        "extend": "max",
        "units": "J m-2",
        "levels": list(np.linspace(0, 4e6, 21)),
    },
    "TOT_PREC_1H": {
        "extend": "max",
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
        "extend": "max",
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
    # Sequential Reds for error-magnitude scores (RMSE, MAE, STDE): error is
    # non-negative, larger ⇒ darker. Levels start at 0 so saturation maps
    # directly to error magnitude; discrete levels make absolute values readable
    # from the colour bar. Defined once per parameter under "{param}.score.map";
    # the lookup uses these for any score lacking a dedicated entry. The precip
    # levels are a bit on the bright side, but kept consistent with the rest.
    "U_10M.score.map": _SCORE_REDS | {"units": "m/s"},
    "V_10M.score.map": _SCORE_REDS | {"units": "m/s"},
    "SP_10M.score.map": _SCORE_REDS | {"units": "m/s"},
    "TD_2M.score.map": _SCORE_REDS | {"units": "°C"},
    "T_2M.score.map": _SCORE_REDS | {"units": "°C"},
    "PMSL.score.map": _SCORE_REDS_PA | {"units": "Pa"},
    "PS.score.map": _SCORE_REDS_PA | {"units": "Pa"},
    "TOT_PREC.score.map": _SCORE_REDS_PRECIP | {"units": "mm"},
    "TOT_PREC1.score.map": _precip_score_map(1),
    "TOT_PREC6.score.map": _precip_score_map(6),
    "TOT_PREC24.score.map": _precip_score_map(24),
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
    "TOT_PREC1.BIAS.map": _precip_bias_map(1),
    "TOT_PREC6.BIAS.map": _precip_bias_map(6),
    "TOT_PREC24.BIAS.map": _precip_bias_map(24),
}

CMAP_DEFAULTS = defaultdict(_fallback, _CMAP_DEFAULTS)
