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


def test_experiment_optional(example_config):
    """The experiment block can be omitted, e.g. for showcase or sandbox configs."""

    del example_config["experiment"]
    cfg = ConfigModel.model_validate(example_config)
    assert cfg.experiment is None


def _with_scorecard_reference(config, reference):
    config["experiment"]["scorecards"] = {
        "enabled": True,
        "sections": {
            "test": {
                "baseline": reference,
                "lead_times": "6/24/6",
                "stratification": "region",
                "variables": ["T_2M:RMSE"],
            }
        },
    }
    return config


def test_scorecard_reference_can_be_a_run(example_config):
    forecaster_label = next(
        item[k]["label"]
        for item in example_config["runs"]
        for k in item
        if k in ("forecaster", "temporal_downscaler")
    )
    ConfigModel.model_validate(
        _with_scorecard_reference(example_config, forecaster_label)
    )


def test_scorecard_reference_must_exist(example_config):
    with pytest.raises(ValueError, match="does not match the label"):
        ConfigModel.model_validate(
            _with_scorecard_reference(example_config, "no-such-label")
        )
