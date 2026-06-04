from datetime import datetime

import numpy as np
import xarray as xr

from data import load_analysis_data_from_zarr
from data.schema import validate_canonical


def _make_analysis_zarr(path):
    """Minimal anemoi-style zarr: dims (time, variable, ensemble, cell)."""
    reftime = datetime(2024, 1, 1)
    ntime, ncell = 3, 4
    dates = np.array(
        [np.datetime64(reftime) + np.timedelta64(h, "h") for h in range(ntime)]
    )
    variables = ["2t", "tp"]
    data = np.arange(ntime * 2 * 1 * ncell, dtype="float32").reshape(ntime, 2, 1, ncell)
    ds = xr.Dataset(
        {
            "data": (("time", "variable", "ensemble", "cell"), data),
            "dates": (("time",), dates),
            "latitudes": (("cell",), np.array([46.0, 46.5, 47.0, 47.5])),
            "longitudes": (("cell",), np.array([7.0, 7.5, 8.0, 8.5])),
        }
    )
    ds.attrs["variables"] = variables
    ds.attrs["field_shape"] = [ncell]  # 1-D / flattened -> 'values' dim
    ds.to_zarr(path, mode="w")
    return reftime


def test_analysis_zarr_returns_canonical_schema(tmp_path):
    # name must contain "co2" so the COSMO2 (short-name) param map is used
    zarr_path = tmp_path / "analysis_co2.zarr"
    reftime = _make_analysis_zarr(zarr_path)

    out = load_analysis_data_from_zarr(zarr_path, reftime, [0, 1, 2], ["T_2M", "TOT_PREC"])

    # canonical dims/coords (no bare 'time')
    assert "step" in out.dims and "values" in out.dims
    assert "time" not in out.dims and "time" not in out.coords
    assert {
        "step",
        "valid_time",
        "forecast_reference_time",
        "latitude",
        "longitude",
    } <= set(out.coords)
    assert list(out["step"].values.astype("timedelta64[h]").astype(int)) == [0, 1, 2]
    # variables renamed to ICON names; precip converted m -> mm (x1000)
    assert set(out.data_vars) == {"T_2M", "TOT_PREC"}
    # contract holds
    validate_canonical(out)
