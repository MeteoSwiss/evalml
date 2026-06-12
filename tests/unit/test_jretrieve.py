import numpy as np
import pandas as pd
import pytest

from data_input import jretrieve as jr


def test_stations_to_argv_group():
    assert jr._stations_to_argv({"group": "SwissMetNet"}) == ["-a", "stn_group,SwissMetNet"]


def test_stations_to_argv_locations_from_string():
    assert jr._stations_to_argv({"locations": "ARO,KLO"}) == ["-i", "nat_abbr,ARO,KLO"]


def test_stations_to_argv_bbox_from_string():
    assert jr._stations_to_argv({"bbox": "45.8,47.8,5.9,10.5"}) == [
        "-l", "45.8,47.8,5.9,10.5",
    ]


def test_stations_to_argv_rejects_ambiguous():
    with pytest.raises(ValueError, match="exactly one"):
        jr._stations_to_argv({"group": "x", "bbox": "1,2,3,4"})


def test_parse_selection_default_group():
    assert jr.parse_selection("jretrievedwh:") == ({"group": "SwissMetNet"}, "prod", "surface")
    assert jr.parse_selection("jretrievedwh:SwissMetNet") == (
        {"group": "SwissMetNet"}, "prod", "surface",
    )


def test_parse_selection_keyvalue_and_stage():
    assert jr.parse_selection("jretrievedwh:locations=ARO,KLO;stage=devt") == (
        {"locations": "ARO,KLO"}, "devt", "surface",
    )


def _sample_meta():
    return pd.DataFrame({
        "station": [2, 1, 1],
        "op_since": [19900101000000, 19800101000000, 19800101000000],
        "op_till": ["", "", ""],
        "parameter": ["tre200s0", "fkl010z0", "tre200s0"],
        "latitude": [47.48, 46.79, 46.79],
        "longitude": [8.54, 9.68, 9.68],
        "elev": [426.0, 1878.0, 1878.0],
        "stn_name": ["Zurich", "Arosa", "Arosa"],
        "nat_abbr": ["KLO", "ARO", "ARO"],
    })


def test_station_catalog_from_meta_collapses_and_sorts():
    cat = jr.StationCatalog.from_meta(_sample_meta())
    assert cat.n == 2
    assert list(cat.nat_abbr) == ["ARO", "KLO"]      # sorted by nat_abbr
    assert list(cat.station_id) == [1, 2]
    np.testing.assert_allclose(cat.latitude, [46.79, 47.48])
