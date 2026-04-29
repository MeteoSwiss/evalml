import operator
import pytest
from verification import _threshold_value_and_operator


def test_threshold_value_and_operator():
    cases = [
        ("> 10.0", (operator.gt, 10.0)),
        (">= 5", (operator.ge, 5.0)),
        ("< 2.5", (operator.lt, 2.5)),
        ("<= 0", (operator.le, 0.0)),
        ("== 42", (operator.eq, 42.0)),
        ("!= -1.5", (operator.ne, -1.5)),
        (">=    3.14", (operator.ge, 3.14)),
        (" <  7 ", (operator.lt, 7.0)),
    ]
    for s, expected in cases:
        op_fn, value = _threshold_value_and_operator(s)
        assert op_fn == expected[0]
        assert value == expected[1]

    # Invalid cases
    with pytest.raises(ValueError):
        _threshold_value_and_operator("")
    with pytest.raises(ValueError):
        _threshold_value_and_operator("foo 10")
    with pytest.raises(ValueError):
        _threshold_value_and_operator("> abc")
    with pytest.raises(ValueError):
        _threshold_value_and_operator("10 >")
