import pytest

from evalml.config import ConfigModel


def test_example_forecasters_config(example_forecasters_config):
    """Test that the example config loads correctly."""

    # this shoudd not raise an error
    _ = ConfigModel.model_validate(example_forecasters_config)

    # this should raise an error
    del example_forecasters_config["runs"]
    with pytest.raises(ValueError, match="Field required"):
        _ = ConfigModel.model_validate(example_forecasters_config)


def test_example_interpolators_config(example_interpolators_config):
    """Test that the example config loads correctly."""

    # this shoudd not raise an error
    _ = ConfigModel.model_validate(example_interpolators_config)

    # this should raise an error
    del example_interpolators_config["runs"]
    with pytest.raises(ValueError, match="Field required"):
        _ = ConfigModel.model_validate(example_interpolators_config)
