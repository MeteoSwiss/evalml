import pytest
from pydantic import ValidationError

from evalml.config import ConfigModel, SpectraConfig


def test_example_forecasters_config(example_forecasters_config):
    """Test that the example config loads correctly."""

    # this shoudd not raise an error
    _ = ConfigModel.model_validate(example_forecasters_config)

    # this should raise an error
    del example_forecasters_config["runs"]
    with pytest.raises(ValueError, match="Field required"):
        _ = ConfigModel.model_validate(example_forecasters_config)


def test_example_temporal_downscalers_config(example_temporal_downscalers_config):
    """Test that the example config loads correctly."""

    # this shoudd not raise an error
    _ = ConfigModel.model_validate(example_temporal_downscalers_config)

    # this should raise an error
    del example_temporal_downscalers_config["runs"]
    with pytest.raises(ValueError, match="Field required"):
        _ = ConfigModel.model_validate(example_temporal_downscalers_config)


def test_spectra_config_defaults():
    cfg = SpectraConfig()
    assert cfg.enabled is False
    assert cfg.method == "dct"
    assert cfg.variables == ["T_2M", "WIND_KE", "TOT_PREC"]
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


def test_spectra_config_rejects_unknown_variable():
    with pytest.raises(ValidationError):
        SpectraConfig(variables=["GEOPOTENTIAL"])


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
