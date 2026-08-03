from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def pytest_addoption(parser):
    parser.addoption(
        "--regenerate-expected",
        action="store_true",
        default=False,
        help=(
            "Integration metric tests: overwrite tests/integration/expected/<config>.yaml "
            "with the values this run produced instead of asserting against them. "
            "Regenerate one config at a time with -k, e.g. "
            "`pytest tests/integration/test_configs.py -m heavytest "
            "--regenerate-expected -k varda-single`. Off by default, so CI never "
            "regenerates."
        ),
    )


@pytest.fixture
def regenerate_expected(request):
    """True when --regenerate-expected was passed (see pytest_addoption)."""
    return request.config.getoption("--regenerate-expected")


@pytest.fixture
def example_config():
    configfile = PROJECT_ROOT / "tests/integration/configs/varda-single-1.0.yaml"
    with open(configfile, "r") as f:
        config = yaml.safe_load(f)
    return config
