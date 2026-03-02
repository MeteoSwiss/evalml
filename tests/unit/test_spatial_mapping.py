import numpy as np
import xarray as xr

from verification.spatial import (
    map_forecast_to_truth,
    nearest_grid_yx_indices,
    spherical_nearest_neighbor_indices,
)


def test_spherical_nearest_neighbor_indices_returns_expected_points():
    source_lat = np.array([46.0, 46.0, 47.0, 47.0])
    source_lon = np.array([7.0, 8.0, 7.0, 8.0])
    target_lat = np.array([46.1, 46.9])
    target_lon = np.array([7.1, 7.9])

    idx = spherical_nearest_neighbor_indices(
        source_lat=source_lat,
        source_lon=source_lon,
        target_lat=target_lat,
        target_lon=target_lon,
    )

    assert np.array_equal(idx, np.array([0, 3]))


def test_nearest_grid_yx_indices_returns_grid_indices():
    lat = xr.DataArray([[46.0, 46.0], [47.0, 47.0]], dims=("y", "x"))
    lon = xr.DataArray([[7.0, 8.0], [7.0, 8.0]], dims=("y", "x"))
    grid = xr.Dataset(coords={"lat": lat, "lon": lon})

    y_idx, x_idx = nearest_grid_yx_indices(
        grid=grid,
        target_lat=np.array([46.1, 46.9]),
        target_lon=np.array([7.1, 7.9]),
    )

    assert np.array_equal(y_idx, np.array([0, 1]))
    assert np.array_equal(x_idx, np.array([0, 1]))


def test_map_forecast_to_truth_maps_and_aligns_time():
    fcst_time = np.array(
        ["2024-01-01T00:00", "2024-01-01T01:00"], dtype="datetime64[ns]"
    )
    truth_time = np.array(
        ["2024-01-01T00:00", "2024-01-01T01:00", "2024-01-01T02:00"],
        dtype="datetime64[ns]",
    )

    fcst = xr.Dataset(
        data_vars={
            "T_2M": (
                ("time", "y", "x"),
                np.array(
                    [
                        [[1.0, 2.0], [3.0, 4.0]],
                        [[10.0, 20.0], [30.0, 40.0]],
                    ]
                ),
            )
        },
        coords={
            "time": fcst_time,
            "y": [0, 1],
            "x": [0, 1],
            "lat": (("y", "x"), np.array([[46.0, 46.0], [47.0, 47.0]])),
            "lon": (("y", "x"), np.array([[7.0, 8.0], [7.0, 8.0]])),
        },
    )
    truth = xr.Dataset(
        data_vars={"T_2M": (("time", "values"), np.zeros((3, 2)))},
        coords={
            "time": truth_time,
            "values": ["STA1", "STA2"],
            "lat": ("values", np.array([46.1, 46.9])),
            "lon": ("values", np.array([7.1, 7.9])),
        },
    )

    mapped_fcst, mapped_truth = map_forecast_to_truth(fcst, truth)

    assert mapped_fcst["T_2M"].dims == ("time", "values")
    assert np.array_equal(mapped_truth["time"].values, fcst_time)
    assert np.array_equal(mapped_fcst["values"].values, np.array(["STA1", "STA2"]))
    assert np.allclose(mapped_fcst["lat"].values, np.array([46.1, 46.9]))
    assert np.allclose(mapped_fcst["lon"].values, np.array([7.1, 7.9]))
    assert np.allclose(mapped_fcst["T_2M"].values, np.array([[1.0, 4.0], [10.0, 40.0]]))
