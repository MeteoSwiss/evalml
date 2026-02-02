from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def example_forecasters_config():
    configfile = PROJECT_ROOT / "config/forecasters-co2.yaml"
    with open(configfile, "r") as f:
        config = yaml.safe_load(f)
    return config


@pytest.fixture
def example_interpolators_config():
    configfile = PROJECT_ROOT / "config/interpolators-co2.yaml"
    with open(configfile, "r") as f:
        config = yaml.safe_load(f)
    return config
