import numpy as np
import pytest
import xarray as xr

from meteogram_derivations import (
    add_derived,
    expand_to_base_params,
    station_timeseries_to_long,
    wind_direction_deg,
    wind_speed,
)


def test_expand_expands_derived_and_dedupes():
    assert expand_to_base_params(["T_2M", "TOT_PREC", "SP_10M", "DD_10M"]) == [
        "T_2M",
        "TOT_PREC",
        "U_10M",
        "V_10M",
    ]


def test_expand_passes_through_non_derived():
    assert expand_to_base_params(["T_2M", "U_10M"]) == ["T_2M", "U_10M"]


def test_wind_speed_pythagoras():
    assert wind_speed(3.0, 4.0) == 5.0


def test_wind_direction_cardinals():
    # meteorological: direction wind comes FROM
    assert np.isclose(wind_direction_deg(0.0, -1.0), 0.0)   # from north
    assert np.isclose(wind_direction_deg(-1.0, 0.0), 90.0)  # from east
    assert np.isclose(wind_direction_deg(0.0, 1.0), 180.0)  # from south
    assert np.isclose(wind_direction_deg(1.0, 0.0), 270.0)  # from west


def test_wind_direction_in_range():
    rng = np.linspace(-10, 10, 50)
    u, v = np.meshgrid(rng, rng)
    dd = wind_direction_deg(u, v)
    assert np.all((dd >= 0) & (dd < 360))


def test_add_derived_adds_requested_only():
    ds = xr.Dataset({"U_10M": ("t", [3.0]), "V_10M": ("t", [4.0])})
    out = add_derived(ds, ["SP_10M"])
    assert np.isclose(out["SP_10M"].values[0], 5.0)
    assert "DD_10M" not in out


def test_add_derived_raises_without_uv():
    ds = xr.Dataset({"T_2M": ("t", [280.0])})
    with pytest.raises(ValueError, match="U_10M and V_10M"):
        add_derived(ds, ["SP_10M"])


def test_station_timeseries_to_long_shape_and_columns():
    times = np.array(["2025-04-01T00", "2025-04-01T01"], dtype="datetime64[h]")
    ds = xr.Dataset(
        {"T_2M": ("valid_time", [280.0, 281.0]),
         "TOT_PREC": ("valid_time", [0.0, 0.1])},
        coords={"valid_time": times},
    )
    df = station_timeseries_to_long(ds, "Varda-Single", ["T_2M", "TOT_PREC"])
    assert list(df.columns) == ["source", "valid_time", "param", "value"]
    assert set(df["param"]) == {"T_2M", "TOT_PREC"}
    assert len(df) == 4
    assert df["source"].unique().tolist() == ["Varda-Single"]


def test_station_timeseries_to_long_raises_on_absent_param():
    times = np.array(["2025-04-01T00"], dtype="datetime64[h]")
    ds = xr.Dataset({"T_2M": ("valid_time", [280.0])}, coords={"valid_time": times})
    with pytest.raises(KeyError, match="absent"):
        station_timeseries_to_long(ds, "Varda-Single", ["T_2M", "TOT_PREC"])


def test_station_timeseries_to_long_raises_without_time_coord():
    ds = xr.Dataset({"T_2M": ("x", [280.0])}, coords={"x": [0]})
    with pytest.raises(KeyError, match="valid_time' or 'time'"):
        station_timeseries_to_long(ds, "Varda-Single", ["T_2M"])
