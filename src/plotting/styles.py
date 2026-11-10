"""Shared plotting styles and field preprocessing for map plots."""

from earthkit.meteo.utils.convert import kelvin_to_celsius
import earthkit.meteo.wind as ekm_wind
import earthkit.plots as ekp
from matplotlib.colors import Colormap
import numpy as np

from plotting.colormap_defaults import CMAP_DEFAULTS


def _precip_lookup(accu):
    """Key of the TOT_PREC palette for an accumulation window of ``accu`` hours.

    Palettes exist only for some windows (1h, 6h); others use the longest
    defined window not exceeding ``accu`` (e.g. 3h -> 1h, 24h -> 6h).
    """
    windows = sorted(
        int(key[len("TOT_PREC_") : -1])
        for key in CMAP_DEFAULTS
        if key.startswith("TOT_PREC_") and key.endswith("H")
    )
    window = max((w for w in windows if w <= accu), default=windows[0])
    return f"TOT_PREC_{window}H"


def get_style(param, units_override=None, accu=1):
    """Get style and colormap settings for the plot."""
    lookup = _precip_lookup(accu) if param == "TOT_PREC" else param
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
    - Wind speed: sqrt(u^2 + v^2)
    - Everything else: see :func:`prepare_values`
    Returns: (field_array, units_override or None)
    """
    fields = state["fields"]
    if param == "SP_10M":
        values = ekm_wind.speed(fields["U_10M"], fields["V_10M"])
    elif param == "SP":
        values = ekm_wind.speed(fields["U"], fields["V"])
    else:
        values = fields[param]
    return prepare_values(param, values)


def prepare_values(param: str, values):
    """Unit conversion and clipping of an already derived ``param`` field.

    - Temperatures: K -> °C
    - Precipitation: m -> mm
    Used directly by callers that load derived params (e.g. ``SP_10M``).
    Returns: (field_array, units_override or None)
    """
    if param in ("T_2M", "TD_2M", "TW_2M", "SKT", "T", "TD"):
        return kelvin_to_celsius(values), "°C"
    if param in ("SP_10M", "SP"):
        return values, "m/s"
    if param == "TOT_PREC":
        return np.maximum(values, 0), "mm"
    if param in ("CLCT", "CLCL", "CLCM", "CLCH"):
        # Cloud cover is in percent. Avoid exact 0/100 plateaus breaking
        # tricontourf on orthographic projections (tmp/reproduce_clct_bug.py).
        # Pair with extend="neither". Any new bounded field with silent-blank
        # or GeometryCollection-crash globe frames likely needs the same
        # clip-away-from-boundary fix.
        return np.clip(values, 1e-4, 100 - 1e-4), None
    if param == "SSRD":
        # Same issue, bottom boundary only (night-side plateau).
        return np.maximum(values, 1e-6), None
    return values, None
