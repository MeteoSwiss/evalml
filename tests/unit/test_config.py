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


def test_tot_prec_units(example_config):
    run = example_config["runs"][0]["temporal_downscaler"]

    config = ConfigModel.model_validate(example_config)
    assert config.runs[0].temporal_downscaler.tot_prec_units == "kg m-2"

    run["tot_prec_units"] = "m"
    config = ConfigModel.model_validate(example_config)
    assert config.runs[0].temporal_downscaler.tot_prec_units == "m"

    run["tot_prec_units"] = "mm"
    with pytest.raises(ValueError, match="tot_prec_units"):
        ConfigModel.model_validate(example_config)
