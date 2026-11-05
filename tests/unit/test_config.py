import copy

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
    # the ICON-CH1-CTRL baseline of the example config only covers 0/33/1
    config = _spectra_ready(example_config, lead_times=[6, 24])
    ConfigModel.model_validate(config)

    config["experiment"]["spectra"]["lead_times"] = [6, 48]
    with pytest.raises(ValueError, match=r"spectra.lead_times contains \[48\]"):
        ConfigModel.model_validate(config)


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


def _spectra_ready(config, lead_times=(6,)):
    """Example config with spectra on: ICON participants and a zarr truth only."""
    config["runs"] = [
        r for r in config["runs"] if r.get("baseline", {}).get("label") != "INCA"
    ]
    config["experiment"].pop("scorecards", None)  # references the INCA baseline
    config["experiment"]["spectra"] = {"enabled": True, "lead_times": list(lead_times)}
    config["truth"] = {"label": "REA-L-CH1", "root": "/store/realch1.zarr"}
    return config


def test_spectra_accepts_icon_sources_and_zarr_truth(example_config):
    ConfigModel.model_validate(_spectra_ready(example_config))


def test_spectra_rejects_inca_baseline(example_config):
    config = _spectra_ready(example_config)
    config["runs"].append(
        {
            "baseline": {
                "label": "INCA",
                "root": "/store_new/mch/msclim/INCA",
                "steps": "0/6/1",
            }
        }
    )
    with pytest.raises(ValueError, match=r"native grid.*baseline 'INCA' \(INCA\)"):
        ConfigModel.model_validate(config)


@pytest.mark.parametrize(
    "root, kind",
    [
        ("jretrievedwh:SwissMetNet", "station observations"),
        ("/store_new/mch/msclim/INCA", "INCA"),
    ],
)
def test_spectra_rejects_station_and_inca_truth(example_config, root, kind):
    config = _spectra_ready(example_config)
    config["truth"]["root"] = root
    with pytest.raises(ValueError, match=rf"native grid.*truth 'REA-L-CH1' \({kind}\)"):
        ConfigModel.model_validate(config)


def test_inca_allowed_when_spectra_disabled(example_config):
    ConfigModel.model_validate(example_config)  # INCA baseline + station truth


def test_spectra_init_hours_must_not_be_empty():
    with pytest.raises(ValidationError):
        SpectraConfig(init_hours=[])
    assert SpectraConfig(init_hours=[0, 12]).init_hours == [0, 12]


def test_output_variables(example_config):
    """output_variables is optional, takes a list of names and rejects an empty list."""

    downscaler = example_config["runs"][0]["temporal_downscaler"]
    cfg = ConfigModel.model_validate(example_config)
    assert cfg.runs[0].temporal_downscaler.output_variables is None

    downscaler["output_variables"] = ["2t", "tp", "z", "*_850"]
    cfg = ConfigModel.model_validate(example_config)
    assert cfg.runs[0].temporal_downscaler.output_variables == [
        "2t",
        "tp",
        "z",
        "*_850",
    ]

    downscaler["output_variables"] = []
    with pytest.raises(ValueError, match="at least 1 item"):
        _ = ConfigModel.model_validate(example_config)


def test_output_variables_on_inline_forecaster(example_config):
    """The inline forecaster of a temporal downscaler accepts output_variables."""

    forecaster = example_config["runs"][0]["temporal_downscaler"]["forecaster"]
    forecaster["output_variables"] = ["2t", "tp"]
    cfg = ConfigModel.model_validate(example_config)
    assert cfg.runs[0].temporal_downscaler.forecaster.output_variables == ["2t", "tp"]

    forecaster["output_variables"] = []
    with pytest.raises(ValueError, match="at least 1 item"):
        _ = ConfigModel.model_validate(example_config)


def test_output_variables_keep_orography(example_config):
    """With the lapse-rate correction, a verified run's output_variables needs z."""

    downscaler = example_config["runs"][0]["temporal_downscaler"]
    downscaler["output_variables"] = ["2t", "tp"]
    with pytest.raises(ValueError, match="lacks z"):
        _ = ConfigModel.model_validate(example_config)

    for names in (["2t", "tp", "z"], ["2t", "*"], ["T_2M", "FIS"]):
        downscaler["output_variables"] = names
        _ = ConfigModel.model_validate(example_config)

    downscaler["output_variables"] = ["2t", "tp"]
    example_config["lapse_rate_correction"] = False
    _ = ConfigModel.model_validate(example_config)

    # without an experiment there is no verification, so no lapse-rate correction
    example_config["lapse_rate_correction"] = True
    del example_config["experiment"]
    _ = ConfigModel.model_validate(example_config)


def test_output_variables_inline_forecaster_needs_no_orography(example_config):
    """The inline forecaster of a downscaler is not verified as such: no z needed."""

    forecaster = example_config["runs"][0]["temporal_downscaler"]["forecaster"]
    forecaster["output_variables"] = ["2t", "tp"]
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
