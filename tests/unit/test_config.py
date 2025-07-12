import pytest

from evalml.config import ExperimentConfig


def test_example_config(example_config):
    """Test that the example config loads correctly."""

    # this shoudd not raise an error
    _ = ExperimentConfig.model_validate(example_config)

    # this should raise an error
    del example_config["runs"]
    with pytest.raises(ValueError, match="Field required"):
        _ = ExperimentConfig.model_validate(example_config)
