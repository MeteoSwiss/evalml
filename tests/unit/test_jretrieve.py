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
