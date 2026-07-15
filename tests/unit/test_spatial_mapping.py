import numpy as np
import xarray as xr

from verification.spatial import (
    map_forecast_to_truth,
    nearest_grid_yx_indices,
    spherical_nearest_neighbor_indices,
)


def test_spherical_nearest_neighbor_indices_returns_expected_points():
    source_latitude = np.array([46.0, 46.0, 47.0, 47.0])
    source_longitude = np.array([7.0, 8.0, 7.0, 8.0])
    target_latitude = np.array([46.1, 46.9])
    target_longitude = np.array([7.1, 7.9])

    idx = spherical_nearest_neighbor_indices(
        source_latitude=source_latitude,
        source_longitude=source_longitude,
        target_latitude=target_latitude,
        target_longitude=target_longitude,
    )

    assert np.array_equal(idx, np.array([0, 3]))


def test_nearest_grid_yx_indices_returns_grid_indices():
    latitude = xr.DataArray([[46.0, 46.0], [47.0, 47.0]], dims=("y", "x"))
    longitude = xr.DataArray([[7.0, 8.0], [7.0, 8.0]], dims=("y", "x"))
    grid = xr.Dataset(coords={"latitude": latitude, "longitude": longitude})

    y_idx, x_idx = nearest_grid_yx_indices(
        grid=grid,
        target_latitude=np.array([46.1, 46.9]),
        target_longitude=np.array([7.1, 7.9]),
    )

    assert np.array_equal(y_idx, np.array([0, 1]))
    assert np.array_equal(x_idx, np.array([0, 1]))


def test_map_forecast_to_truth_maps_forecast_to_truth_locations():
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
            "latitude": (("y", "x"), np.array([[46.0, 46.0], [47.0, 47.0]])),
            "longitude": (("y", "x"), np.array([[7.0, 8.0], [7.0, 8.0]])),
        },
    )
    truth = xr.Dataset(
        data_vars={"T_2M": (("time", "values"), np.zeros((3, 2)))},
        coords={
            "time": truth_time,
            "values": ["STA1", "STA2"],
            "latitude": ("values", np.array([46.1, 46.9])),
            "longitude": ("values", np.array([7.1, 7.9])),
        },
    )

    mapped_fcst = map_forecast_to_truth(fcst, truth)

    assert mapped_fcst["T_2M"].dims == ("time", "values")
    assert np.array_equal(mapped_fcst["time"].values, fcst_time)
    assert np.array_equal(mapped_fcst["values"].values, np.array(["STA1", "STA2"]))
    assert np.allclose(mapped_fcst["latitude"].values, np.array([46.1, 46.9]))
    assert np.allclose(mapped_fcst["longitude"].values, np.array([7.1, 7.9]))
    assert np.allclose(
        mapped_fcst["T_2M"].values,
        np.array([[1.0, 4.0], [10.0, 40.0]]),
    )


def test_map_forecast_to_truth_returns_fcst_unchanged_when_grids_are_aligned():
    fcst_time = np.array(["2024-01-01T00:00"], dtype="datetime64[ns]")
    latitude = np.array([[46.0, 46.0], [47.0, 47.0]])
    longitude = np.array([[7.0, 8.0], [7.0, 8.0]])

    fcst = xr.Dataset(
        data_vars={"T_2M": (("time", "y", "x"), np.array([[[1.0, 2.0], [3.0, 4.0]]]))},
        coords={
            "time": fcst_time,
            "y": [0, 1],
            "x": [0, 1],
            "latitude": (("y", "x"), latitude),
            "longitude": (("y", "x"), longitude),
        },
    )
    truth = xr.Dataset(
        data_vars={"T_2M": (("time", "y", "x"), np.zeros((1, 2, 2)))},
        coords={
            "time": fcst_time,
            "y": [0, 1],
            "x": [0, 1],
            "latitude": (("y", "x"), latitude),
            "longitude": (("y", "x"), longitude),
        },
    )

    result = map_forecast_to_truth(fcst, truth)
    _, result_aligned = xr.align(truth, result)

    assert result is fcst
    assert result["T_2M"].values is fcst["T_2M"].values
    assert np.array_equal(result["latitude"].values, truth["latitude"].values)
    assert np.array_equal(result["longitude"].values, truth["longitude"].values)
    assert np.array_equal(result_aligned["T_2M"].values, fcst["T_2M"].values)


def test_map_forecast_to_truth_returns_fcst_unchanged_when_grids_are_within_tolerance():
    fcst_time = np.array(["2024-01-01T00:00"], dtype="datetime64[ns]")
    latitude = np.array([[46.0, 46.0], [47.0, 47.0]])
    longitude = np.array([[7.0, 8.0], [7.0, 8.0]])

    fcst = xr.Dataset(
        data_vars={"T_2M": (("time", "y", "x"), np.array([[[1.0, 2.0], [3.0, 4.0]]]))},
        coords={
            "time": fcst_time,
            "y": [0, 1],
            "x": [0, 1],
            "latitude": (("y", "x"), latitude + 5e-8),
            "longitude": (("y", "x"), longitude - 5e-8),
        },
    )
    # Nudge coordinates by less than the 1e-6 tolerance — should still be treated as aligned.
    truth = xr.Dataset(
        data_vars={"T_2M": (("time", "y", "x"), np.zeros((1, 2, 2)))},
        coords={
            "time": fcst_time,
            "y": [0, 1],
            "x": [0, 1],
            "latitude": (("y", "x"), latitude),
            "longitude": (("y", "x"), longitude),
        },
    )

    result = map_forecast_to_truth(fcst, truth)
    _, result_aligned = xr.align(truth, result)

    assert result is not fcst
    assert result["T_2M"].values is fcst["T_2M"].values
    assert np.array_equal(result["latitude"].values, truth["latitude"].values)
    assert np.array_equal(result["longitude"].values, truth["longitude"].values)
    assert np.array_equal(result_aligned["T_2M"].values, fcst["T_2M"].values)


def test_map_forecast_to_truth_returns_fcst_unchanged_when_grids_are_within_tolerance_icon():
    fcst_time = np.array(["2024-01-01T00:00"], dtype="datetime64[ns]")
    latitude = np.array([[46.0, 46.0], [47.0, 47.0]]).flatten()
    longitude = np.array([[7.0, 8.0], [7.0, 8.0]]).flatten()

    fcst = xr.Dataset(
        data_vars={"T_2M": (("time", "values"), np.array([[1.0, 2.0, 3.0, 4.0]]))},
        coords={
            "time": fcst_time,
            "values": [0, 1, 2, 3],
            "latitude": (("values"), latitude + 5e-8),
            "longitude": (("values"), longitude - 5e-8),
        },
    )
    # Nudge coordinates by less than the 1e-6 tolerance — should still be treated as aligned.
    truth = xr.Dataset(
        data_vars={"T_2M": (("time", "values"), np.zeros((1, 4)))},
        coords={
            "time": fcst_time,
            "values": [3, 1, 2, 0],
            "latitude": (("values"), latitude),
            "longitude": (("values"), longitude),
        },
    )

    result = map_forecast_to_truth(fcst, truth)
    _, result_aligned = xr.align(truth, result)

    assert result is not fcst
    assert result["T_2M"].values is fcst["T_2M"].values
    assert np.array_equal(result["latitude"].values, truth["latitude"].values)
    assert np.array_equal(result["longitude"].values, truth["longitude"].values)
    assert np.array_equal(result["values"].values, truth["values"].values)
    assert np.array_equal(result_aligned["T_2M"].values, fcst["T_2M"].values)


def test_map_forecast_to_truth_restores_grid_when_truth_is_gridded():
    fcst_time = np.array(["2024-01-01T00:00"], dtype="datetime64[ns]")

    fcst = xr.Dataset(
        data_vars={
            "T_2M": (
                ("time", "y", "x"),
                np.array([[[1.0, 2.0], [3.0, 4.0]]]),
            )
        },
        coords={
            "time": fcst_time,
            "y": [0, 1],
            "x": [0, 1],
            "latitude": (("y", "x"), np.array([[46.0, 46.0], [47.0, 47.0]])),
            "longitude": (("y", "x"), np.array([[7.0, 8.0], [7.0, 8.0]])),
        },
    )
    truth = xr.Dataset(
        data_vars={"T_2M": (("time", "y", "x"), np.zeros((1, 2, 2)))},
        coords={
            "time": fcst_time,
            "y": [0, 1],
            "x": [0, 1],
            "latitude": (("y", "x"), np.array([[46.1, 46.1], [46.9, 46.9]])),
            "longitude": (("y", "x"), np.array([[7.1, 7.9], [7.1, 7.9]])),
        },
    )

    mapped_fcst = map_forecast_to_truth(fcst, truth)

    assert mapped_fcst["T_2M"].dims == ("time", "y", "x")
    assert np.array_equal(mapped_fcst["y"].values, np.array([0, 1]))
    assert np.array_equal(mapped_fcst["x"].values, np.array([0, 1]))
    assert np.allclose(mapped_fcst["latitude"].values, truth["latitude"].values)
    assert np.allclose(mapped_fcst["longitude"].values, truth["longitude"].values)
    assert np.allclose(
        mapped_fcst["T_2M"].values,
        np.array([[[1.0, 2.0], [3.0, 4.0]]]),
    )
