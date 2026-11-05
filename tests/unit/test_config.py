import pytest
from pydantic import ValidationError

from evalml.config import ConfigModel, SpectraConfig


def test_example_config(example_config):
    """Test that the example config loads correctly."""

    # this shoudd not raise an error
    _ = ConfigModel.model_validate(example_config)

    # this should raise an error
    del example_config["runs"]
    with pytest.raises(ValueError, match="Field required"):
        _ = ConfigModel.model_validate(example_config)


def test_spectra_config_defaults():
    cfg = SpectraConfig()
    assert cfg.enabled is False
    assert cfg.method == "dct"
    assert cfg.variables == ["T_2M", "WIND_KE", "TOT_PREC1"]
    assert cfg.lead_times == []


def test_spectra_config_accepts_valid():
    cfg = SpectraConfig(
        enabled=True,
        method="fft",
        lead_times=[6, 48, 120],
        variables=["T_2M"],
        init_hours=[0, 12],
    )
    assert cfg.method == "fft"
    assert cfg.lead_times == [6, 48, 120]


def test_spectra_config_rejects_bad_method():
    with pytest.raises(ValidationError):
        SpectraConfig(method="wavelet")


def test_spectra_config_accepts_any_variable():
    cfg = SpectraConfig(variables=["SP_10M", "TOT_PREC6", "WIND_KE"])
    assert cfg.variables == ["SP_10M", "TOT_PREC6", "WIND_KE"]


def test_spectra_config_rejects_empty_variables():
    with pytest.raises(ValidationError):
        SpectraConfig(variables=[])


def test_spectra_config_forbids_extra():
    with pytest.raises(ValidationError):
        SpectraConfig(enabled=True, bogus=1)


def test_spectra_config_enabled_requires_lead_times():
    with pytest.raises(ValidationError):
        SpectraConfig(enabled=True, lead_times=[])


def test_spectra_config_disabled_allows_empty_lead_times():
    cfg = SpectraConfig(enabled=False)
    assert cfg.lead_times == []


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


def test_spectra_lead_times_must_be_produced_by_every_participant(example_config):
    # the INCA baseline of the example config only covers 0/6/1
    example_config["experiment"]["spectra"] = {"enabled": True, "lead_times": [3, 6]}
    ConfigModel.model_validate(example_config)

    example_config["experiment"]["spectra"]["lead_times"] = [6, 24]
    with pytest.raises(ValueError, match=r"spectra.lead_times contains \[24\]"):
        ConfigModel.model_validate(example_config)


def test_spectra_lead_times_unchecked_when_disabled(example_config):
    example_config["experiment"]["spectra"] = {"enabled": False, "lead_times": [999]}
    ConfigModel.model_validate(example_config)


def test_labels_must_be_unique(example_config):
    example_config["runs"][2]["baseline"]["label"] = "INCA"  # same as runs[1]
    with pytest.raises(ValueError, match=r"Labels must be unique.*\['INCA'\]"):
        ConfigModel.model_validate(example_config)


def test_inline_forecaster_label_must_be_unique(example_config):
    td = example_config["runs"][0]["temporal_downscaler"]
    td["forecaster"]["label"] = td["label"]
    with pytest.raises(ValueError, match="Labels must be unique"):
        ConfigModel.model_validate(example_config)


def test_inline_forecaster_may_repeat_a_listed_forecaster(example_config):
    """The same forecaster listed and used inline is one run: its label may repeat."""
    td = example_config["runs"][0]["temporal_downscaler"]
    td["forecaster"]["label"] = "Varda forecaster"
    example_config["runs"].append({"forecaster": dict(td["forecaster"])})
    ConfigModel.model_validate(example_config)
