"""Shared visual style for the publication figures.

Source of truth for colors, markers, line styles, and human-readable parameter
labels used by ``publication_figures.py``, ``publication_meteogram.py``, and
``publication_scoremaps.py``.  Font sizes and layout defaults live in
``publication.mplstyle``; apply it with::

    plt.style.use(Path(__file__).parent / "publication.mplstyle")

Tweak the look of the paper figures here.
"""

from matplotlib.colors import LinearSegmentedColormap

# Label used for the station-observations source (overlaid in the meteogram).
OBS_LABEL = "Observations"

# Source colors
COLOR_OBS = "#4ecb8d"
COLOR_CH1 = "#008dff"
COLOR_CH2 = "#003a7d"
COLOR_VARDA = "#d83034"

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
    -0.55,
    -0.45,
    -0.35,
    -0.25,
    -0.15,
    -0.05,
    0.05,
    0.15,
    0.25,
    0.35,
    0.45,
    0.55,
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

# Human-readable variable names (used for panel titles / labels).
PARAM_LABELS = {
    "T_2M": "2m Temperature",
    "TD_2M": "2m Dew Point Temperature",
    "TOT_PREC": "Total Precipitation",
    "SP_10M": "Wind Speed",
    "DD_10M": "Wind Direction",
    "U_10M": "Eastward Wind",
    "V_10M": "Northward Wind",
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
        COLOR_CH1
        if "CH1" in src
        else COLOR_CH2
        if "CH2" in src
        else COLOR_VARDA
        if "Varda" in src
        else "gray"
    )
    linestyle = "--" if "EPS mean" in src else "-"
    linewidth = 2.25 if "Varda" in src else 1.5
    return dict(color=color, linestyle=linestyle, linewidth=linewidth)
