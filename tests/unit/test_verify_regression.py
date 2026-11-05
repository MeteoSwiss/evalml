"""verify() output must not change when its internals are refactored.

The golden files under ``tests/unit/data/`` were written by verify() before the
region/station-group mask construction and the missing-forecast rule were factored out
into helpers shared with verify_ensemble() (commit be3dc5c). Golden files rather than
loading the old code from git: the test does not depend on the git history, and any
later change of verify()'s output shows up here too.

The inputs are built from closed-form expressions (no random numbers), so they do not
depend on the numpy version. To regenerate the files after an intended change of
verify()'s output: ``uv run python tests/unit/test_verify_regression.py``.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from verification import verify

DATA = Path(__file__).parent / "data"
_FRT = np.datetime64("2024-01-01T00", "ns")
_REGIONS = [
    {"type": "bbox", "name": "all", "bbox": [1.5, 16, 43, 49.5]},
    {"type": "bbox", "name": "west", "bbox": [1.5, 7.5, 43, 49.5]},
]


def station_inputs() -> tuple[xr.Dataset, xr.Dataset]:
    """12 stations, 3 steps; gaps in forecast and truth; circular, categorical,
    truth-only and forecast-only parameters."""
    n, steps = 12, np.array([0, 6, 12])
    i = np.arange(n)[None, :]
    s = np.arange(steps.size)[:, None]
    coords = {
        "values": [f"S{k:02d}" for k in range(n)],
        "longitude": ("values", np.linspace(5.0, 10.0, n)),
        "latitude": ("values", np.linspace(46.0, 47.0, n)),
        "forecast_reference_time": _FRT,
        "step": steps,
    }
    dims = ("step", "values")
    t_obs = 280.0 + 3.0 * np.sin(0.7 * i + 0.3 * s)
    t_fcst = t_obs + 0.8 * np.cos(1.3 * i - 0.5 * s) + 0.2
    t_fcst[1, 4] = np.nan  # a forecast gap at one step
    t_obs[2, 7] = np.nan  # a truth gap
    tp_obs = np.maximum(0.0, 2.0 * np.sin(0.9 * i + s))
    tp_fcst = np.maximum(0.0, 2.0 * np.sin(0.9 * i + s + 0.4) + 0.3)
    dd_obs = np.mod(40.0 * i + 25.0 * s, 360.0)
    dd_fcst = np.mod(dd_obs + 30.0 * np.sin(i + s) + 5.0, 360.0)
    pt_obs = np.array([0, 1, 5, 7])[(i + s) % 4] * np.ones_like(t_obs)
    pt_fcst = np.array([0, 1, 5, 7])[(i + 2 * s + 1) % 4] * np.ones_like(t_obs)
    fcst = xr.Dataset(
        {
            "T_2M": (dims, t_fcst),
            "TOT_PREC": (dims, tp_fcst),
            "DD_10M": (dims, dd_fcst),
            "PTYPE": (dims, pt_fcst),
            "FCST_ONLY": (dims, t_fcst),
        },
        coords=coords,
    )
    obs = xr.Dataset(
        {
            "T_2M": (dims, t_obs),
            "TOT_PREC": (dims, tp_obs),
            "DD_10M": (dims, dd_obs),
            "PTYPE": (dims, pt_obs),
            "CAPE_MU": (dims, 100.0 + 10.0 * i + s * np.ones_like(t_obs)),
        },
        coords=coords,
    )
    return fcst, obs


STATION_THRESHOLDS = {
    "T_2M": {"lt": [280.0], "gt": [278.5, 282.0]},
    "TOT_PREC": {"gt": [0.0, 1.0]},
    "DD_10M": {"gt": [180.0]},
    "PTYPE": {"eq": [0, 1, 5, 7]},
}


def gridded_inputs() -> tuple[xr.Dataset, xr.Dataset]:
    """INCA-like truth on a regular (y, x) grid, with gaps."""
    ny, nx, steps = 5, 6, np.array([0, 6])
    y = np.arange(ny)[None, :, None]
    x = np.arange(nx)[None, None, :]
    s = np.arange(steps.size)[:, None, None]
    lat, lon = np.meshgrid(
        np.linspace(46.0, 47.0, ny), np.linspace(5.0, 10.0, nx), indexing="ij"
    )
    coords = {
        "y": np.arange(ny),
        "x": np.arange(nx),
        "latitude": (("y", "x"), lat),
        "longitude": (("y", "x"), lon),
        "forecast_reference_time": _FRT,
        "step": steps,
    }
    dims = ("step", "y", "x")
    t_obs = 275.0 + np.sin(0.5 * y + 0.8 * x + s)
    t_fcst = t_obs + 0.5 * np.cos(x - y + s)
    t_fcst[1, 0, 0] = np.nan
    t_obs[0, 4, 5] = np.nan
    tp_obs = np.maximum(0.0, np.sin(y * x + s))
    tp_fcst = np.maximum(0.0, np.sin(y * x + s + 0.2))
    fcst = xr.Dataset(
        {"T_2M": (dims, t_fcst), "TOT_PREC": (dims, tp_fcst)}, coords=coords
    )
    obs = xr.Dataset({"T_2M": (dims, t_obs), "TOT_PREC": (dims, tp_obs)}, coords=coords)
    return fcst, obs


def cases() -> dict:
    """name -> verify() keyword arguments."""
    st_fcst, st_obs = station_inputs()
    gr_fcst, gr_obs = gridded_inputs()
    common = dict(fcst_label="fcst", obs_label="obs", regions=_REGIONS, num_workers=1)
    return {
        "stations": dict(
            fcst=st_fcst.chunk({"step": 1}),
            obs=st_obs,
            threshold_dict=STATION_THRESHOLDS,
            holdout_stations=["S00", "S03", "S07", "S11"],
            **common,
        ),
        "stations_tolerant": dict(
            fcst=st_fcst,
            obs=st_obs,
            threshold_dict=STATION_THRESHOLDS,
            max_missing_fraction=0.5,
            **common,
        ),
        "gridded": dict(
            fcst=gr_fcst.chunk({"step": 1}),
            obs=gr_obs,
            threshold_dict={"TOT_PREC": {"gt": [0.1, 0.5]}},
            **common,
        ),
    }


@pytest.mark.parametrize("name", list(cases()))
def test_verify_output_unchanged(name):
    result = verify(**cases()[name])
    expected = xr.load_dataset(DATA / f"verify_golden_{name}.nc")

    xr.testing.assert_identical(result, expected)
    for var in expected.variables:
        assert result[var].dims == expected[var].dims, var
        if expected[var].dtype.kind in "fiub":
            assert result[var].dtype == expected[var].dtype, var
            # Bit for bit, NaN at the same places
            np.testing.assert_array_equal(result[var].values, expected[var].values)


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    for name, kwargs in cases().items():
        verify(**kwargs).to_netcdf(DATA / f"verify_golden_{name}.nc")
