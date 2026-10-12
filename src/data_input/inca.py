"""INCA-CH analyses as ground truth.

Only step 0 (the analysis) of each INCA run is used: the truth at valid time
``t`` comes from the run started at ``t``. Archive layout::

    {root}/{YYYY}/{MM}/{PREFIX}_INCA_{YYYYmmddHHMM}.nc           # most variables
    {root}/{YYYY}/analysis/{MM}/{PREFIX}_INCA_{YYYYmmddHHMM}.nc  # CAPE and CIN
    {root}/INCA1km_topography_parameters.nc                      # elevation (DEM)

Variables get ICON names (see ``INCA_VARIABLES``). ``load_analysis`` returns the
same structure as the KENDA zarr loader (``step`` dimension, accumulations
cumulative from the reference time), so the shared post-processing in
``data_input`` applies unchanged. Missing files become NaN and are logged.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import Transformer

from data_input.accumulation import _accumulate_from_hourly

LOG = logging.getLogger(__name__)

TOPOGRAPHY_FILE = "INCA1km_topography_parameters.nc"

# INCA x/y coordinates are Swiss CH1903 / LV03.
_INCA_CRS = "EPSG:21781"

_CELSIUS_TO_KELVIN = 273.15


@dataclass(frozen=True)
class IncaVariable:
    """How one evalml variable is read from the INCA archive.

    Attributes
    ----------
    prefix
        INCA file prefix, also the variable name inside the file.
    description
        Human-readable description, for logs.
    subdir
        Directory between year and month ("analysis" for CAPE/CIN).
    rate_interval_min
        For precipitation rates (mm/h): minutes between runs. The rates are
        summed into amounts (kg m-2). ``None`` for all other variables.
    """

    prefix: str
    description: str
    subdir: str = ""
    rate_interval_min: int | None = None

    @property
    def is_rate(self) -> bool:
        return self.rate_interval_min is not None


# evalml (ICON) variable name -> INCA source.
INCA_VARIABLES: dict[str, IncaVariable] = {
    # temperature
    "T_2M": IncaVariable("TT", "2 m temperature"),
    "TD_2M": IncaVariable("TD", "2 m dew point temperature"),
    "TW_2M": IncaVariable("TW", "2 m wet-bulb temperature"),
    "T_G": IncaVariable("TG", "surface temperature"),
    # wind: 10-min products, the closest to the instantaneous model wind
    "FF_10M": IncaVariable("FF_10min", "10 m wind speed, 10-min mean"),
    "DD_10M": IncaVariable("DD_10min", "10 m wind direction, 10-min mean"),
    "VMAX_10M": IncaVariable("WG_10min", "10 m wind gust, 10-min"),
    # precipitation, accumulated from rates
    "TOT_PREC": IncaVariable(
        "RR", "precipitation (CombiPrecip based)", rate_interval_min=10
    ),
    "TOT_SNOW": IncaVariable(
        "RS", "snowfall (CombiPrecip based)", rate_interval_min=10
    ),
    "TOT_PREC_RADAR": IncaVariable(
        "RP", "precipitation (radar based)", rate_interval_min=5
    ),
    "TOT_SNOW_RADAR": IncaVariable("PN", "snowfall (radar based)", rate_interval_min=5),
    "PREC_TYPE": IncaVariable("PT", "precipitation type (CombiPrecip based)"),
    "PREC_TYPE_RADAR": IncaVariable("NT", "precipitation type (radar based)"),
    # clouds, sunshine, lightning
    "CLCT": IncaVariable("CT", "total cloud cover"),
    "SUNSHINE_REL": IncaVariable("SU", "relative sunshine duration"),
    "LIGHTNING_PROB": IncaVariable("LP", "probability of lightning within 4 km"),
    # freezing level
    "HZEROCL": IncaVariable("Z0", "height of the 0 degC isotherm"),
    "SNOWLMT": IncaVariable("ZS", "snowfall limit"),
    "MELTLAYER_TOP": IncaVariable("ZM", "upper limit of the melting layer"),
    # convection
    "CAPE_MU": IncaVariable("CP", "most unstable CAPE", subdir="analysis"),
    "CIN_MU": IncaVariable("CN", "most unstable CIN", subdir="analysis"),
}

# Wind components are derived from the 10-min mean speed and direction.
_WIND_SPEED, _WIND_DIRECTION = "FF_10M", "DD_10M"
WIND_COMPONENTS = ("U_10M", "V_10M")

SUPPORTED_PARAMS = frozenset(INCA_VARIABLES) | frozenset(WIND_COMPONENTS)


# ---------------------------------------------------------------------------
# Archive access
# ---------------------------------------------------------------------------


def topography_file(root: Path | str) -> Path:
    """Path of the static INCA topography file (used for elevation)."""
    return Path(root) / TOPOGRAPHY_FILE


def load_dem(topo_nc: Path, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Elevation [m] from the INCA topography file, interpolated to the (y, x) grid."""
    with xr.open_dataset(topo_nc) as ds:
        dem = (
            ds["DEM"]
            .assign_coords(
                p_j=("p_j", ds["x"].values.astype(np.float64)),
                p_i=("p_i", ds["y"].values.astype(np.float64)),
            )
            .rename({"p_j": "x", "p_i": "y"})
        )
        return dem.interp(x=x, y=y, method="linear").values.astype(np.float32)


def check_prerequisites(root: Path | str) -> None:
    """Fail early when the INCA archive or its topography file is unavailable."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"INCA archive not found: {root}")
    if not topography_file(root).is_file():
        raise FileNotFoundError(
            f"INCA topography file not found: {topography_file(root)}"
        )


def analysis_file(root: Path | str, var: IncaVariable, time: datetime) -> Path:
    """Path of the INCA run initialised at ``time``, whose step 0 is the analysis."""
    month_dir = Path(root) / f"{time:%Y}" / var.subdir / f"{time:%m}"
    return month_dir / f"{var.prefix}_INCA_{time:%Y%m%d%H%M}.nc"


def _read_analysis(path: Path, prefix: str) -> xr.DataArray | None:
    """Step 0 of an INCA file as a (y, x) field in SI units; None if the file is missing.

    Step 0 is selected by position; its valid time is the time in the file name.
    """
    if not path.exists():
        return None
    with xr.open_dataset(
        path, decode_times=False, drop_variables=["grid_mapping"]
    ) as ds:
        da = ds[prefix].isel(time=0, drop=True).load()
    da = da.rename({"chx": "x", "chy": "y"})
    units = da.attrs.get("units", "").strip()
    if units == "degrees C":
        return (da + _CELSIUS_TO_KELVIN).assign_attrs(units="K")
    return da.assign_attrs(units=units)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _sources_to_load(params: list[str]) -> list[str]:
    """INCA variables to read for ``params``; unsupported params are skipped."""
    unsupported = [p for p in params if p not in SUPPORTED_PARAMS]
    if unsupported:
        LOG.warning(
            "INCA does not provide %s; these parameters are skipped. Supported: %s",
            unsupported,
            sorted(SUPPORTED_PARAMS),
        )
    sources = []
    for p in params:
        if p in WIND_COMPONENTS:
            sources += [_WIND_SPEED, _WIND_DIRECTION]
        elif p in INCA_VARIABLES:
            sources.append(p)
    return list(dict.fromkeys(sources))


def _read_fields(
    root: Path, var: IncaVariable, times: list[datetime]
) -> list[xr.DataArray | None]:
    """Read the analyses valid at ``times``; log how many files are missing."""
    paths = [analysis_file(root, var, t) for t in times]
    fields = [_read_analysis(p, var.prefix) for p in paths]
    missing = [p for p, f in zip(paths, fields) if f is None]
    if missing:
        LOG.warning(
            "INCA: %d of %d %s files missing, filled with NaN (first: %s)",
            len(missing),
            len(paths),
            var.prefix,
            missing[0],
        )
    return fields


def _hourly_amounts(
    root: Path, var: IncaVariable, reftime: datetime, n_hours: int
) -> list[xr.DataArray | None]:
    """Precipitation amounts (kg m-2) for each hour from reftime to reftime + n_hours.

    The rate stamped ``t`` is taken as the mean over ``(t - interval, t]``. An hourly
    amount is the mean of the available rates in that hour (times 1 h), so single
    missing runs are tolerated; an hour without any run is NaN.
    """
    # TODO: confirm with the INCA team that the rate stamped t covers the interval
    # ending at t (and not the one starting at t).
    slots_per_hour = 60 // var.rate_interval_min
    interval = timedelta(minutes=var.rate_interval_min)
    hour_ends = [reftime + timedelta(hours=h) for h in range(1, n_hours + 1)]
    slot_times = [
        end - k * interval for end in hour_ends for k in range(slots_per_hour)
    ]
    rates = _read_fields(root, var, slot_times)

    amounts = []
    for i in range(n_hours):
        available = [
            r
            for r in rates[i * slots_per_hour : (i + 1) * slots_per_hour]
            if r is not None
        ]
        if not available:
            amounts.append(None)
            continue
        mean_rate = xr.concat(available, dim="slot").mean("slot")  # mm/h
        amounts.append(mean_rate.assign_attrs(units="kg m-2"))  # x 1 h
    return amounts


def _stack(
    fields: list[xr.DataArray | None], steps_h: list[int], template: xr.DataArray
) -> xr.DataArray:
    """Stack (y, x) fields along ``step`` (hours), NaN where a field is missing."""
    nan_field = xr.DataArray(
        np.full(template.shape, np.nan, dtype=template.dtype),
        dims=template.dims,
        coords=template.coords,
    )
    filled = [f if f is not None else nan_field for f in fields]
    step = np.asarray(steps_h, dtype="timedelta64[h]").astype("timedelta64[ns]")
    return xr.concat(
        filled, dim="step", coords="minimal", compat="override"
    ).assign_coords(step=step)


def _grid_coords(root: Path, x: np.ndarray, y: np.ndarray) -> dict:
    """Latitude, longitude and elevation coordinates on the INCA (y, x) grid."""
    x_2d, y_2d = np.meshgrid(x, y)
    lon, lat = Transformer.from_crs(_INCA_CRS, "EPSG:4326", always_xy=True).transform(
        x_2d, y_2d
    )
    elevation = load_dem(topography_file(root), x, y)
    return {
        "latitude": (("y", "x"), lat, {"units": "degrees_north"}),
        "longitude": (("y", "x"), lon, {"units": "degrees_east"}),
        "elevation": (("y", "x"), elevation, {"units": "m"}),
    }


def load_analysis(
    root: Path | str, reftime: datetime, steps: list[int], params: list[str]
) -> xr.Dataset:
    """Load INCA analyses valid at ``reftime + steps`` (hours).

    Same contract as the KENDA zarr loader: returns a Dataset with dimensions
    ``(step, y, x)`` and latitude/longitude/elevation coordinates. Accumulated
    variables are cumulative from ``reftime``. Parameters that INCA does not
    provide are skipped with a warning.
    """
    root = Path(root)
    valid_times = [reftime + timedelta(hours=s) for s in steps]

    raw: dict[str, list[xr.DataArray | None]] = {}
    hourly: dict[str, list[xr.DataArray | None]] = {}
    for name in _sources_to_load(params):
        var = INCA_VARIABLES[name]
        LOG.info("Loading INCA %s (%s) for %s", var.prefix, var.description, name)
        if var.is_rate:
            hourly[name] = _hourly_amounts(root, var, reftime, max(steps))
        else:
            raw[name] = _read_fields(root, var, valid_times)

    # The grid is taken from the files themselves; missing files become NaN on it.
    found = [
        f
        for fields in (*raw.values(), *hourly.values())
        for f in fields
        if f is not None
    ]
    if not found:
        if not raw and not hourly:
            return xr.Dataset()
        raise FileNotFoundError(
            f"No INCA analyses found under {root} for reftime {reftime:%Y-%m-%d %H:%M}"
        )
    template = found[0]

    ds = xr.Dataset(
        {name: _stack(fields, steps, template) for name, fields in raw.items()}
    )
    hours = list(range(1, max(steps) + 1))
    for name, amounts in hourly.items():
        if hours:
            # skipna=False: a missing hour makes later totals NaN, not zero.
            ds[name] = _accumulate_from_hourly(
                _stack(amounts, hours, template), steps, skipna=False
            )
        else:  # only step 0 requested: nothing has accumulated yet
            ds[name] = _stack([xr.zeros_like(template)], steps, template)

    if any(p in WIND_COMPONENTS for p in params):
        speed, direction = ds[_WIND_SPEED], np.deg2rad(ds[_WIND_DIRECTION])
        # meteorological convention: direction the wind blows from, clockwise from north
        ds["U_10M"] = (-speed * np.sin(direction)).assign_attrs(units="m/s")
        ds["V_10M"] = (-speed * np.cos(direction)).assign_attrs(units="m/s")

    ds = ds[[p for p in params if p in ds]]
    return ds.assign_coords(_grid_coords(root, ds["x"].values, ds["y"].values))
