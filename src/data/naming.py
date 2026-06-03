"""Canonical parameter-name mappings shared across data sources.

`PARAMS_MAP` maps evalml's canonical ICON short names to the ECMWF / anemoi
short names used as GRIB ``shortName`` and as anemoi-datasets variable names.
This is the single source of truth; loaders and plotting import it rather than
re-declaring their own copies.
"""

# ICON canonical name -> ECMWF / anemoi short name
PARAMS_MAP = {
    "T_2M": "2t",
    "TD_2M": "2d",
    "U_10M": "10u",
    "V_10M": "10v",
    "PS": "sp",
    "PMSL": "msl",
    "TOT_PREC": "tp",
}

PARAMS_MAP_INV = {v: k for k, v in PARAMS_MAP.items()}
