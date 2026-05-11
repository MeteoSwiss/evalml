"""Canonical units for verification parameters and metrics.

Storage units in the verification netCDFs (BIAS, RMSE, MAE, STDE, ... all
inherit these). Update the dict if a parameter's internal representation
changes.
"""

PARAM_UNITS: dict[str, str] = {
    "T_2M": "K",
    "TD_2M": "K",
    "PMSL": "Pa",
    "PS": "Pa",
    "TOT_PREC": "mm",
    "U_10M": "m/s",
    "V_10M": "m/s",
    "SP_10M": "m/s",
}

UNITLESS_METRICS: set[str] = {"CORR", "R2"}


def metric_units(metric: str, param: str) -> str:
    """Return the canonical units of (metric, param), or '' if unitless/unknown."""
    if metric.upper() in UNITLESS_METRICS:
        return ""
    return PARAM_UNITS.get(param, "")
