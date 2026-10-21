import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "workflow" / "scripts"))
from compare_forecast_frames import parse_source  # noqa: E402


def test_parse_source():
    assert parse_source("ICON-CH1=/store/x=y") == ("ICON-CH1", "/store/x=y")


@pytest.mark.parametrize("text", ["no-separator", "=root", "label="])
def test_parse_source_rejects(text):
    with pytest.raises(ValueError):
        parse_source(text)
