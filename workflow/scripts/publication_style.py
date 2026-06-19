"""Shared visual style for the publication figures.

Single source of truth for colors, markers, line styles, font sizes, and the
human-readable parameter labels used by both ``publication_figures.py``
(lead-time figure) and ``publication_meteogram.py``. Tweak the look of the
paper figures here.
"""

# Font sizes (points)
FS_AXES = 13   # tick labels, axis labels, in-panel annotations
FS_TITLE = 16  # panel titles, y-axis labels, legend, suptitle

# Label used for the station-observations source (overlaid in the meteogram).
OBS_LABEL = "Observations"

# Source colors
COLOR_OBS = "#4ecb8d"
COLOR_CH1 = "#008dff"
COLOR_CH2 = "#003a7d"
COLOR_VARDA = "#d83034"

# Human-readable variable names (used for panel titles / labels).
PARAM_LABELS = {
    "T_2M": "2m temperature",
    "TD_2M": "2m dew point temperature",
    "TOT_PREC": "total precipitation",
    "SP_10M": "wind speed",
    "DD_10M": "wind direction",
    "U_10M": "10m u-component of wind",
    "V_10M": "10m v-component of wind",
}


def param_label(param: str) -> str:
    """Full variable name for a parameter code (falls back to the code)."""
    return PARAM_LABELS.get(param, param)


def line_style(src: str) -> dict:
    """Return matplotlib plot kwargs (color/marker/line) for a source label.

    Change colors, markers, and line styles for every figure here.
    """
    if src == OBS_LABEL:
        return dict(
            color=COLOR_OBS,
            linestyle="none",
            marker="o",
            markersize=3.5,
        )
    color = (
        COLOR_CH1 if "CH1" in src else
        COLOR_CH2 if "CH2" in src else
        COLOR_VARDA if "Varda" in src else
        "gray"
    )
    linestyle = "--" if "EPS mean" in src else "-"
    linewidth = 2.25 if "Varda" in src else 1.5
    return dict(color=color, linestyle=linestyle, linewidth=linewidth)
