"""Shared visual style for the publication figures.

Source of truth for colors, markers, line styles, and human-readable parameter
labels used by the publication notebooks (``notebooks/publication/*.py``) and
by ``plot_meteogram_region.py``.  Font sizes and layout defaults live in the
packaged ``publication.mplstyle``; apply it with::

    import matplotlib.pyplot as plt
    from evalml.publication import style
    plt.style.use(style.mplstyle_path())

Tweak the look of the paper figures here.
"""

from importlib import resources
from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap

# Print widths (inches). Figures are placed in the paper at 100% (no resizing),
# so set a figure's WIDTH to one of these by its column span; keep the height
# proportional. Font sizes are absolute (in publication.mplstyle), so text
# appears the same size on every figure regardless of width.
FIG_WIDTH_1COL = 3.35   # single column
FIG_WIDTH_2COL = 5.7    # spans both columns (page width)


def figure_width(columns: int) -> float:
    """Print width in inches for a 1- or 2-column figure."""
    return FIG_WIDTH_2COL if columns >= 2 else FIG_WIDTH_1COL


# Label used for the station-observations source (overlaid in the meteogram).
OBS_LABEL = "Observations"

# Source colors
COLOR_OBS = "#4ecb8d"
COLOR_CH1 = "#008dff"
COLOR_CH2 = "#003a7d"
COLOR_VARDA = "#d83034"
COLOR_AIFS = "#78399C"

# Skill score colormap: red = baseline better, blue = model better.
# ColorBrewer RdBu palette. The deep RdBu ends (#b2182b/#2166ac) are reserved for
# the out-of-range extremes (colorbar arrows); the in-range bins use the lighter
# tiers so the extremes stand out (see _build_skill_artifacts).
COLOR_SKILL_MODEL_BETTER = "#2166ac"  # RdBu blue (extreme)
COLOR_SKILL_BASELINE_BETTER = "#b2182b"  # RdBu red (extreme)

SKILL_CMAP = LinearSegmentedColormap.from_list(
    "skill",
    [
        "#d6604d",  # red: strongest in-range bin
        "#f4a582",
        "#fddbc7",
        "#ffffff",  # neutral
        "#d1e5f0",
        "#92c5de",
        "#4393c3",  # blue: strongest in-range bin
    ],
)
# Grey colour for the neutral band (|skill| < 0.05).
SKILL_GREY = "#ffffff"
# Levels capped at ±0.55 (0.10 spacing); the central bin [−0.05, 0.05] is SKILL_GREY.
SKILL_LEVELS = [
    -1.5,
    -0.4,
    -0.15,
    -0.05,
    0.05,
    0.15,
    0.3,
    0.6,
]

# Human-readable score names used in panel titles / labels.
SCORE_LABELS = {
    "RMSE": "RMSE",
    "STDE": "STDE",
    "BIAS": "Bias",
    "MAE": "MAE",
    "MSE_SKILL": "MSE skill",
    "BIAS_CONTRIB": "Contribution of bias to MSE skill",
}

REGION_LABELS = {
    "icon": "Switzerland",
    "jura": "Jura",
    "mittelland": "Swiss Plateau",
    "innerealpentaeler": "Inneralpine Valleys",
    "alpennordhang": "Northern Slopes",
    "alpensuedseite": "Southern Slopes",
    "alpen": "Alps"
}

# Human-readable variable names (used for panel titles / labels).
PARAM_LABELS = {
    "T_2M": "2m Temperature",
    "TD_2M": "2m Dew Point Temperature",
    "TOT_PREC1": "Total Precipitation (hourly)",
    "TOT_PREC6": "Total Precipitation (6-hourly)",
    "SP_10M": "Wind Speed",
    "PMSL": "Sea Level Pressure",
    "DD_10M": "Wind Direction",
    "U_10M": "Eastward Wind",
    "V_10M": "Northward Wind",
}


def param_label(param: str) -> str:
    """Full variable name for a parameter code (falls back to the code)."""
    return PARAM_LABELS.get(param, param)


def region_label(region: str) -> str:
    """Region label for region"""
    return REGION_LABELS.get(region, region)


def line_style(src: str) -> dict:
    """Return matplotlib plot kwargs (color/marker/line) for a source label.

    Change colors, markers, and line styles for every figure here.
    """
    if src == OBS_LABEL:
        return dict(
            color=COLOR_OBS,
            linestyle="none",
            marker="o",
            markersize=2.5,
        )
    color = (
        COLOR_CH1
        if "CH1" in src
        else COLOR_CH2
        if "CH2" in src
        else COLOR_VARDA
        if "Varda" in src
        else COLOR_AIFS
        if "AIFS" in src
        else "gray"
    )
    linestyle = "--" if "EPS mean" in src else "-"
    linewidth = 1.3 if "Varda" in src else 0.9
    return dict(color=color, linestyle=linestyle, linewidth=linewidth)


def mplstyle_path() -> Path:
    """Filesystem path to the packaged publication matplotlib style.

    Apply with ``plt.style.use(mplstyle_path())``. Kept as a function (not a
    module constant) so ``importlib.resources`` resolves it lazily and works
    both from the editable checkout and an installed wheel.
    """
    return Path(resources.files("evalml.publication") / "publication.mplstyle")
