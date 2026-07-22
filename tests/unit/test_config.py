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


def test_publication_meteogram_block_validates():
    from evalml.config import ConfigModel

    import yaml
    from pathlib import Path

    # The publication block is opt-in and not shipped in the split paper configs;
    # attach one so this exercises the meteogram validation.
    cfg = yaml.safe_load(Path("config/varda-single_paper_analysis.yaml").read_text())
    cfg["publication"] = {
        "meteogram": {
            "enabled": False,
            "init_time": "202504010000",
            "station": "KLO",
            "params": ["T_2M", "TOT_PREC", "SP_10M", "DD_10M"],
        }
    }
    model = ConfigModel.model_validate(cfg)
    # Compare against the config rather than a hard-coded case so changing the
    # plotted meteogram case doesn't break this structural check.
    assert (
        model.publication.meteogram.init_time
        == cfg["publication"]["meteogram"]["init_time"]
    )
    assert "DD_10M" in model.publication.meteogram.params
