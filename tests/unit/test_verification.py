import pytest

from verification import decode_metric


@pytest.mark.parametrize(
    "label, expected",
    [
        # happy path: no extra 'p' outside decimal markers, no operator substring in words
        ("ETS_gt_10p5", "ETS > 10.5"),
        ("ETS_ge_10p5", "ETS >= 10.5"),
        ("FAR_lt_5p0", "FAR < 5.0"),
        ("ETS_le_10p5", "ETS <= 10.5"),
        ("POD_eq_0p0", "POD == 0.0"),
        ("FAR_ne_5p5", "FAR != 5.5"),
        # edge cases: 'p' in variable name
        ("precip_gt_1p0", "precip > 1.0"),
        ("temp_lt_0p0", "temp < 0.0"),
        # edge case: operator abbreviation inside a word
        ("simple_label", "simple_label"),
    ],
)
def test_decode_metric(label, expected):
    assert decode_metric(label) == expected
