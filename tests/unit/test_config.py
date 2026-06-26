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

    cfg = yaml.safe_load(Path("config/varda-single_paper.yaml").read_text())
    model = ConfigModel.model_validate(cfg)
    assert model.publication.meteogram.init_time == "202504010000"
    assert "DD_10M" in model.publication.meteogram.params
