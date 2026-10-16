"""Unit tests for the INCA ground truth (data_input.inca) on a tiny synthetic archive."""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from data_input import _accumulate_from_hourly, inca, load_truth_data, truth_source

REFTIME = datetime(2025, 11, 1, 0, 0)
# LV03 grid around Bern; (600000, 200000) is the LV03 reference point.
X = np.array([599000.0, 600000.0, 601000.0])
Y = np.array([199000.0, 200000.0])
# Value written at the nowcast steps (>= 1) of every file: must never be read.
NOWCAST_SENTINEL = -999.0


@pytest.fixture
def inca_root(tmp_path) -> Path:
    """An empty INCA archive containing only the topography file."""
    root = tmp_path / "INCA"
    root.mkdir()
    xr.Dataset(
        {
            "DEM": (("p_i", "p_j"), np.full((Y.size, X.size), 500.0)),
            "x": ("p_j", X),
            "y": ("p_i", Y),
        }
    ).to_netcdf(root / inca.TOPOGRAPHY_FILE)
    return root


def write_run(
    root: Path, prefix: str, time: datetime, value: float, units: str, subdir=""
):
    """Write one INCA run whose step 0 (the analysis) holds ``value`` everywhere."""
    data = np.full((3, Y.size, X.size), NOWCAST_SENTINEL, dtype=np.float32)
    data[0] = value
    ds = xr.Dataset(
        {
            prefix: (("time", "chy", "chx"), data, {"units": units}),
            "grid_mapping": ((), 0.0),
        },
        coords={
            "time": ("time", [0.0, 600.0, 1200.0], {"units": f"seconds since {time}"}),
            "chx": X,
            "chy": Y,
        },
    )
    path = inca.analysis_file(root, inca.IncaVariable(prefix, "", subdir=subdir), time)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def write_rates(root: Path, prefix: str, hours: int, rate: float, interval_min=10):
    """Write a constant precipitation rate (mm/h) for every run in (REFTIME, REFTIME + hours]."""
    n_slots = hours * 60 // interval_min
    for k in range(1, n_slots + 1):
        write_run(
            root, prefix, REFTIME + timedelta(minutes=k * interval_min), rate, "mm/h"
        )


def at(ds: xr.Dataset, param: str, step_h: int) -> np.ndarray:
    return ds[param].sel(time=np.datetime64(REFTIME + timedelta(hours=step_h))).values


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "root, expected",
    [
        ("/store/kenda-ch1.zarr", "zarr"),
        ("jretrievedwh:1,2", "jretrieve"),
        ("jretrievedwh:bbox=45.8,47.8,5.9,10.5", "jretrieve"),
        ("/store_new/mch/msclim/INCA", "inca"),
    ],
)
def test_truth_source(root, expected):
    assert truth_source(root) == expected
    assert truth_source(Path(root)) == expected


def test_truth_source_rejects_unknown_root():
    with pytest.raises(ValueError, match="Unsupported truth root"):
        truth_source("/store/somewhere/else")


def test_check_prerequisites(inca_root, tmp_path):
    inca.check_prerequisites(inca_root)
    with pytest.raises(FileNotFoundError):
        inca.check_prerequisites(tmp_path / "missing" / "INCA")


# ---------------------------------------------------------------------------
# Instantaneous variables
# ---------------------------------------------------------------------------


def test_temperature_from_step_zero_in_kelvin(inca_root):
    for h, celsius in [(0, 10.0), (6, 12.0)]:
        write_run(inca_root, "TT", REFTIME + timedelta(hours=h), celsius, "degrees C")

    truth = load_truth_data(inca_root, REFTIME, [0, 6], ["T_2M"])

    assert set(truth.dims) == {"time", "y", "x"}
    np.testing.assert_allclose(at(truth, "T_2M", 0), 283.15, rtol=1e-6)
    np.testing.assert_allclose(at(truth, "T_2M", 6), 285.15, rtol=1e-6)
    for coord in ("latitude", "longitude", "elevation"):
        assert truth[coord].dims == ("y", "x")
    np.testing.assert_allclose(truth["elevation"], 500.0)
    bern = truth.sel(x=600000.0, y=200000.0)
    np.testing.assert_allclose(bern["latitude"], 46.9511, atol=1e-3)
    np.testing.assert_allclose(bern["longitude"], 7.4386, atol=1e-3)


def test_cape_is_read_from_analysis_subdirectory(inca_root):
    write_run(inca_root, "CP", REFTIME, 250.0, "J/kg", subdir="analysis")

    truth = load_truth_data(inca_root, REFTIME, [0], ["CAPE_MU"])

    np.testing.assert_allclose(at(truth, "CAPE_MU", 0), 250.0)


def test_wind_uses_10min_products(inca_root):
    # Westerly wind of 5 m/s (blowing from 270 degrees) with gusts of 12 m/s.
    write_run(inca_root, "FF_10min", REFTIME, 5.0, "m/s")
    write_run(inca_root, "DD_10min", REFTIME, 270.0, "degrees")
    write_run(inca_root, "WG_10min", REFTIME, 12.0, "m/s")
    # Hourly products with different values: must not be read.
    write_run(inca_root, "FF", REFTIME, 1.0, "m/s")
    write_run(inca_root, "DD", REFTIME, 0.0, "degrees")
    write_run(inca_root, "WG", REFTIME, 2.0, "m/s")

    truth = load_truth_data(
        inca_root, REFTIME, [0], ["SP_10M", "U_10M", "V_10M", "VMAX_10M"]
    )

    np.testing.assert_allclose(at(truth, "SP_10M", 0), 5.0, rtol=1e-6)
    np.testing.assert_allclose(at(truth, "U_10M", 0), 5.0, rtol=1e-6)
    np.testing.assert_allclose(at(truth, "V_10M", 0), 0.0, atol=1e-5)
    np.testing.assert_allclose(at(truth, "VMAX_10M", 0), 12.0)
    assert "FF_10M" not in truth and "DD_10M" not in truth


def test_missing_file_gives_nan(inca_root, caplog):
    write_run(inca_root, "TT", REFTIME, 10.0, "degrees C")

    with caplog.at_level(logging.WARNING):
        truth = load_truth_data(inca_root, REFTIME, [0, 1], ["T_2M"])

    assert np.isfinite(at(truth, "T_2M", 0)).all()
    assert np.isnan(at(truth, "T_2M", 1)).all()
    assert "1 of 2 TT files missing" in caplog.text


def test_unsupported_params_are_skipped(inca_root, caplog):
    write_run(inca_root, "TT", REFTIME, 10.0, "degrees C")

    with caplog.at_level(logging.WARNING):
        truth = load_truth_data(inca_root, REFTIME, [0], ["T_2M", "PMSL"])

    assert list(truth.data_vars) == ["T_2M"]
    assert "PMSL" in caplog.text


def test_no_files_at_all_raises(inca_root):
    with pytest.raises(FileNotFoundError, match="No INCA analyses found"):
        load_truth_data(inca_root, REFTIME, [0], ["T_2M"])


# ---------------------------------------------------------------------------
# Precipitation (accumulated from rates)
# ---------------------------------------------------------------------------


def test_six_hour_precipitation_from_constant_rate(inca_root):
    write_rates(inca_root, "RR", hours=6, rate=2.0)

    truth = load_truth_data(inca_root, REFTIME, [6], ["TOT_PREC6"])

    np.testing.assert_allclose(at(truth, "TOT_PREC6", 6), 12.0, rtol=1e-6)


def test_radar_precipitation_uses_5min_runs(inca_root):
    write_rates(inca_root, "RP", hours=1, rate=3.0, interval_min=5)

    truth = load_truth_data(inca_root, REFTIME, [1], ["TOT_PREC_RADAR1"])

    np.testing.assert_allclose(at(truth, "TOT_PREC_RADAR1", 1), 3.0, rtol=1e-6)


def test_rate_stamped_at_hour_start_belongs_to_previous_hour(inca_root):
    # The run at REFTIME covers (REFTIME - 10 min, REFTIME]: it is not part of hour 1.
    write_run(inca_root, "RR", REFTIME, 600.0, "mm/h")
    write_rates(inca_root, "RR", hours=1, rate=6.0)

    truth = load_truth_data(inca_root, REFTIME, [1], ["TOT_PREC1"])

    np.testing.assert_allclose(at(truth, "TOT_PREC1", 1), 6.0, rtol=1e-6)


def test_single_missing_run_does_not_bias_hourly_amount(inca_root):
    write_rates(inca_root, "RR", hours=1, rate=6.0)
    inca.analysis_file(
        inca_root, inca.INCA_VARIABLES["TOT_PREC"], REFTIME + timedelta(minutes=30)
    ).unlink()

    truth = load_truth_data(inca_root, REFTIME, [1], ["TOT_PREC1"])

    np.testing.assert_allclose(at(truth, "TOT_PREC1", 1), 6.0, rtol=1e-6)


def test_hour_without_runs_is_nan_not_zero(inca_root):
    write_rates(inca_root, "RR", hours=1, rate=6.0)  # hour 2 has no files

    truth = load_truth_data(inca_root, REFTIME, [1, 2], ["TOT_PREC1"])

    np.testing.assert_allclose(at(truth, "TOT_PREC1", 1), 6.0, rtol=1e-6)
    assert np.isnan(at(truth, "TOT_PREC1", 2)).all()


def test_accumulate_from_hourly_default_still_skips_nan():
    # The KENDA path relies on the default (skipna=True); INCA opts out.
    hourly = xr.DataArray(
        [1.0, np.nan, 2.0],
        dims="step",
        coords={
            "step": np.array([1, 2, 3], dtype="timedelta64[h]").astype(
                "timedelta64[ns]"
            )
        },
    )
    np.testing.assert_allclose(_accumulate_from_hourly(hourly, [3]).values, [3.0])
    assert np.isnan(_accumulate_from_hourly(hourly, [3], skipna=False).values).all()
