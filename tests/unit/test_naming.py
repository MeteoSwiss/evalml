from data.naming import PARAMS_MAP, PARAMS_MAP_INV


def test_params_map_icon_to_short_names():
    assert PARAMS_MAP == {
        "T_2M": "2t",
        "TD_2M": "2d",
        "U_10M": "10u",
        "V_10M": "10v",
        "PS": "sp",
        "PMSL": "msl",
        "TOT_PREC": "tp",
    }


def test_params_map_inverse_round_trips():
    for icon, short in PARAMS_MAP.items():
        assert PARAMS_MAP_INV[short] == icon
