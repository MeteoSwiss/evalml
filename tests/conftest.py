from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def example_config():
    configfile = PROJECT_ROOT / "config/config.yaml"
    with open(configfile, "r") as f:
        config = yaml.safe_load(f)
    return config
