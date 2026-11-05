import copy

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


def test_score_on_common_points(example_config):
    """score_on_common_points accepts a subset of experiment.params only."""

    example_config["experiment"]["params"] = ["T_2M", "HZEROCL"]
    example_config["experiment"]["score_on_common_points"] = ["HZEROCL"]
    cfg = ConfigModel.model_validate(example_config)
    assert cfg.experiment.score_on_common_points == ["HZEROCL"]

    example_config["experiment"]["score_on_common_points"] = ["SNOWLMT"]
    with pytest.raises(ValueError, match="not in experiment.params"):
        _ = ConfigModel.model_validate(example_config)


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


def test_output_variables(example_config):
    """output_variables is optional, takes a list of names and rejects an empty list."""

    downscaler = example_config["runs"][0]["temporal_downscaler"]
    cfg = ConfigModel.model_validate(example_config)
    assert cfg.runs[0].temporal_downscaler.output_variables is None

    downscaler["output_variables"] = ["T_2M", "TOT_PREC", "*_850"]
    cfg = ConfigModel.model_validate(example_config)
    assert cfg.runs[0].temporal_downscaler.output_variables == [
        "T_2M",
        "TOT_PREC",
        "*_850",
    ]

    downscaler["output_variables"] = []
    with pytest.raises(ValueError, match="at least 1 item"):
        _ = ConfigModel.model_validate(example_config)


def test_output_variables_on_inline_forecaster(example_config):
    """The inline forecaster of a temporal downscaler accepts output_variables."""

    forecaster = example_config["runs"][0]["temporal_downscaler"]["forecaster"]
    forecaster["output_variables"] = ["T_2M", "tp"]
    cfg = ConfigModel.model_validate(example_config)
    assert cfg.runs[0].temporal_downscaler.forecaster.output_variables == [
        "T_2M",
        "tp",
    ]

    forecaster["output_variables"] = []
    with pytest.raises(ValueError, match="at least 1 item"):
        _ = ConfigModel.model_validate(example_config)


def _forecaster_item(example_config):
    """A top-level forecaster entry, built from the inline one of the example."""
    forecaster = copy.deepcopy(
        example_config["runs"][0]["temporal_downscaler"]["forecaster"]
    )
    example_config["runs"].append({"forecaster": forecaster})
    return forecaster


def test_ensemble(example_config):
    """ensemble is optional, needs at least 2 members and defaults the seed to 0."""

    forecaster = _forecaster_item(example_config)
    cfg = ConfigModel.model_validate(example_config)
    assert cfg.runs[-1].forecaster.ensemble is None
    assert cfg.model_dump(mode="json")["runs"][-1]["forecaster"]["ensemble"] is None

    forecaster["ensemble"] = {"members": 4}
    cfg = ConfigModel.model_validate(example_config)
    assert cfg.model_dump(mode="json")["runs"][-1]["forecaster"]["ensemble"] == {
        "members": 4,
        "seed": 0,
    }

    forecaster["ensemble"] = {"members": 2, "seed": 42}
    cfg = ConfigModel.model_validate(example_config)
    assert cfg.runs[-1].forecaster.ensemble.seed == 42

    forecaster["ensemble"] = {"members": 1}
    with pytest.raises(ValueError, match="greater than or equal to 2"):
        _ = ConfigModel.model_validate(example_config)

    forecaster["ensemble"] = {"seed": 1}
    with pytest.raises(ValueError, match="Field required"):
        _ = ConfigModel.model_validate(example_config)

    forecaster["ensemble"] = {"members": 4, "member": 3}
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        _ = ConfigModel.model_validate(example_config)


def test_ensemble_not_on_temporal_downscaler(example_config):
    """ensemble is rejected on the inline forecaster of a temporal downscaler, and
    is not a field of the temporal downscaler itself."""

    downscaler = example_config["runs"][0]["temporal_downscaler"]
    downscaler["forecaster"]["ensemble"] = {"members": 4}
    with pytest.raises(ValueError, match="cannot be an ensemble"):
        _ = ConfigModel.model_validate(example_config)

    del downscaler["forecaster"]["ensemble"]
    downscaler["ensemble"] = {"members": 4}
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        _ = ConfigModel.model_validate(example_config)
