import operator
import pytest
from verification import OPS


def test_ops_keys():
    assert set(OPS.keys()) == {"gt", "ge", "lt", "le", "eq", "ne"}


def test_ops_values():
    cases = [
        ("gt", operator.gt),
        ("ge", operator.ge),
        ("lt", operator.lt),
        ("le", operator.le),
        ("eq", operator.eq),
        ("ne", operator.ne),
    ]
    for key, expected in cases:
        assert OPS[key] == expected


def test_ops_unknown_key():
    with pytest.raises(KeyError):
        OPS[">"]
    with pytest.raises(KeyError):
        OPS["<="]
