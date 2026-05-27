import pytest

from os.path import Path
from evalml.cli import _deep_merge, load_yaml
from evalml.config import ConfigModel


def test_deep_merge_override_wins():
    base = {"a": 1, "b": {"x": 1, "y": 2}}
    override = {"b": {"y": 99}, "c": 3}
    result = _deep_merge(base, override)
    assert result == {"a": 1, "b": {"x": 1, "y": 99}, "c": 3}


def test_deep_merge_non_dict_override_replaces():
    base = {"a": {"x": 1}}
    override = {"a": [1, 2, 3]}
    result = _deep_merge(base, override)
    assert result["a"] == [1, 2, 3]


def test_load_yaml_without_include(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("a: 1\n")
    assert load_yaml(f) == {"a": 1}


def test_load_yaml_include_merges_base(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("a: 1\nb:\n  x: 1\n  y: 2\n")

    child = tmp_path / "child.yaml"
    child.write_text("include: base.yaml\nb:\n  y: 99\nc: 3\n")

    result = load_yaml(child)
    assert result == {"a": 1, "b": {"x": 1, "y": 99}, "c": 3}


def test_load_yaml_include_validates_as_config_model():
    path = Path("config/showcase-interpolators-ich1.yaml")
    data = load_yaml(path)
    _ = ConfigModel.model_validate(data)


def test_example_forecasters_config(example_forecasters_config):
    """Test that the example config loads correctly."""

    # this shoudd not raise an error
    _ = ConfigModel.model_validate(example_forecasters_config)

    # this should raise an error
    del example_forecasters_config["runs"]
    with pytest.raises(ValueError, match="Field required"):
        _ = ConfigModel.model_validate(example_forecasters_config)


def test_example_interpolators_config(example_interpolators_config):
    """Test that the example config loads correctly."""

    # this shoudd not raise an error
    _ = ConfigModel.model_validate(example_interpolators_config)

    # this should raise an error
    del example_interpolators_config["runs"]
    with pytest.raises(ValueError, match="Field required"):
        _ = ConfigModel.model_validate(example_interpolators_config)


def test_legacy_top_level_baselines_still_supported(example_forecasters_config):
    """Top-level `baselines` remains accepted for backward compatibility."""

    cfg = {k: v for k, v in example_forecasters_config.items() if k != "runs"}
    cfg["runs"] = [
        run for run in example_forecasters_config["runs"] if "forecaster" in run
    ]
    cfg["baselines"] = [
        run for run in example_forecasters_config["runs"] if "baseline" in run
    ]

    _ = ConfigModel.model_validate(cfg)
