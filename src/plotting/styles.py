"""Shared plotting styles and field preprocessing for map plots."""

from earthkit.meteo.utils.convert import kelvin_to_celsius
import earthkit.meteo.wind as ekm_wind
import earthkit.plots as ekp
from matplotlib.colors import Colormap
import numpy as np

from plotting.colormap_defaults import CMAP_DEFAULTS


def get_style(param, units_override=None, accu=1):
    """Get style and colormap settings for the plot."""
    lookup = f"{param}_{accu}H" if param == "TOT_PREC" else param
    cfg = CMAP_DEFAULTS[lookup]
    units = units_override if units_override is not None else cfg.get("units", "")

    bounds = cfg.get("bounds", cfg.get("levels", None))
    prebuilt_cmap = cfg.get("cmap", None)

    # When the config provides a pre-built matplotlib Colormap (e.g. a
    # ListedColormap), we must use the earthkit Style's vmin/vmax path, which
    # handles isinstance(colors, Colormap) correctly.  The levels path calls
    # cmap_and_norm → len(Colormap) → TypeError.
    #
    # earthkit's configure_style intercepts _STYLE_KWARGS (cmap/colors/levels/vmin/vmax)
    # from tricontourf kwargs before matplotlib sees them.  To work around this:
    #  - embed the Colormap directly in the Style via colors=
    #  - inject bounds as 'levels' into style._kwargs so they survive to matplotlib
    #  - pass only norm= as a kwarg (not in _STYLE_KWARGS, so not intercepted)
    extend = cfg.get("extend", "both")

    if isinstance(prebuilt_cmap, Colormap) and bounds is not None:
        style = ekp.styles.Style(
            colors=prebuilt_cmap,
            vmin=bounds[0],
            vmax=bounds[-1],
            extend=extend,
            units=units,
        )
        style._kwargs["levels"] = list(bounds)
        return {
            "style": style,
            "norm": cfg.get("norm", None),
        }

    return {
        "style": ekp.styles.Style(
            levels=bounds,
            extend=extend,
            units=units,
            colors=cfg.get("colors", None),
        ),
        "norm": cfg.get("norm", None),
        "cmap": prebuilt_cmap,
        "vmin": cfg.get("vmin", None),
        "vmax": cfg.get("vmax", None),
    }


def preprocess_field(param: str, state: dict):
    """
    - Temperatures: K -> °C
    - Wind speed: sqrt(u^2 + v^2)
    - Precipitation: m -> mm
    Returns: (field_array, units_override or None)
    """
    fields = state["fields"]
    if param in ("T_2M", "TD_2M", "T", "TD"):
        return kelvin_to_celsius(fields[param]), "°C"
    if param == "SP_10M":
        return ekm_wind.speed(fields["U_10M"], fields["V_10M"]), "m/s"
    if param == "SP":
        return ekm_wind.speed(fields["U"], fields["V"]), "m/s"
    if param == "TOT_PREC":
        return np.maximum(fields[param], 0), "mm"
    if param in ("CLCT", "CLCL"):
        # Avoid exact 0/1 plateaus breaking tricontourf on orthographic
        # projections (tmp/reproduce_clct_bug.py). Pair with extend="neither".
        # Any new bounded field with silent-blank or GeometryCollection-crash
        # globe frames likely needs the same clip-away-from-boundary fix.
        return np.clip(fields[param], 1e-6, 1 - 1e-6), None
    if param == "SSRD":
        # Same issue, bottom boundary only (night-side plateau).
        return np.maximum(fields[param], 1e-6), None
    return fields[param], None
