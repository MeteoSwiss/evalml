from pathlib import Path

import pytest

from evalml.config import ConfigModel


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


def test_workflow_parsing_excludes_baselines_from_run_configs(
    example_forecasters_config,
):
    """Baseline entries in `runs` should not be treated as ML run configs."""

    namespace = {
        "Path": Path,
        "config": example_forecasters_config,
    }
    common_rules = Path("workflow/rules/common.smk").read_text()

    exec(common_rules, namespace)

    run_configs = namespace["RUN_CONFIGS"]
    baseline_configs = namespace["BASELINE_CONFIGS"]

    assert all(
        run_config["model_type"] != "baseline" for run_config in run_configs.values()
    )
    assert baseline_configs == {
        "COSMO-E": {
            "label": "COSMO-E",
            "root": "/store_new/mch/msopr/ml/COSMO-E",
            "steps": "0/120/6",
        }
    }
