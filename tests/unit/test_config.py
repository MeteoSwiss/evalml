import pytest

from evalml.config import ConfigModel


def test_example_config(example_config):
    """Test that the example config loads correctly."""

    # this shoudd not raise an error
    _ = ConfigModel.model_validate(example_config)

    # this should raise an error
    del example_config["runs"]
    with pytest.raises(ValueError, match="Field required"):
        _ = ConfigModel.model_validate(example_config)


def test_station_holdout(example_config):
    """station_holdout accepts a list of nat_abbr and rejects empty or duplicate lists."""

    example_config["experiment"]["station_holdout"] = ["CHM", "FRE"]
    cfg = ConfigModel.model_validate(example_config)
    assert cfg.experiment.station_holdout == ["CHM", "FRE"]

    example_config["experiment"]["station_holdout"] = []
    with pytest.raises(ValueError, match="must not be empty"):
        _ = ConfigModel.model_validate(example_config)

    example_config["experiment"]["station_holdout"] = ["CHM", "FRE", "CHM"]
    with pytest.raises(ValueError, match="duplicate stations"):
        _ = ConfigModel.model_validate(example_config)
