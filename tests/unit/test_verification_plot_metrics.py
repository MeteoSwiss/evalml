"""Unit tests for merging the verification results of all participants."""

import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parents[2] / "workflow" / "scripts"))
from verification_plot_metrics import _concat_sources


def _participant(source: str, params: list[str]) -> xr.Dataset:
    """Aggregated verification of one participant, as written by the aggregation step."""
    ds = xr.Dataset(coords={"source": [source], "step": [6]})
    for p in params:
        ds[f"{p}.RMSE"] = (("source", "step"), [[1.0]])
        ds = ds.assign_coords({f"{p}.threshold": [f"gt_{p}"]})
    return ds


def test_concat_sources_with_different_thresholded_params():
    """A participant without CAPE_MU must not break the merge (CAPE_MU.threshold)."""
    icon = _participant("icon", ["T_2M", "CAPE_MU"])
    varda = _participant("varda", ["T_2M"])

    ds = _concat_sources([icon, varda])

    assert list(ds.source.values) == ["icon", "varda"]
    assert not any(c.endswith(".threshold") for c in ds.coords)
    np.testing.assert_allclose(ds["T_2M.RMSE"].values, [[1.0], [1.0]])
    assert np.isnan(ds["CAPE_MU.RMSE"].sel(source="varda")).all()
