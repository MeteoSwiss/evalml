from pathlib import Path

import pytest

from evalml.config import ConfigModel, MultipanelPanelSpec, MultipanelPlotSpec


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


def test_workflow_derives_baseline_id_from_root_stem(example_interpolators_config):
    """Workflow baseline IDs should come from the baseline root path stem."""

    namespace = {
        "Path": Path,
        "config": example_interpolators_config,
    }
    common_rules = Path("workflow/rules/common.smk").read_text()

    exec(common_rules, namespace)

    baseline_configs = namespace["BASELINE_CONFIGS"]

    assert "COSMO-E_hourly" in baseline_configs
    assert "COSMO-E-1h" not in baseline_configs
    assert baseline_configs["COSMO-E_hourly"] == {
        "label": "COSMO-E",
        "root": "/store_new/mch/msopr/ml/COSMO-E_hourly",
        "steps": "0/120/1",
    }


def _spec(rows, cols, panel_count=None):
    n = panel_count if panel_count is not None else rows * cols
    return {
        "rows": rows,
        "cols": cols,
        "panels": [{"metric": "BIAS", "param": "T_2M"} for _ in range(n)],
    }


def test_multipanel_panel_defaults():
    panel = MultipanelPanelSpec.model_validate({"metric": "BIAS", "param": "T_2M"})
    assert panel.region == "all"
    assert panel.season == "all"
    assert panel.init_hour == -999
    assert panel.title is None
    assert panel.ylim is None


def test_multipanel_panel_forbids_extras():
    with pytest.raises(ValueError, match="Extra"):
        MultipanelPanelSpec.model_validate(
            {"metric": "BIAS", "param": "T_2M", "unknown": True}
        )


def test_multipanel_plot_accepts_matching_panel_count():
    spec = MultipanelPlotSpec.model_validate(_spec(2, 3))
    assert spec.rows == 2
    assert spec.cols == 3
    assert len(spec.panels) == 6


def test_multipanel_plot_rejects_mismatched_panel_count():
    with pytest.raises(ValueError, match=r"rows\*cols"):
        MultipanelPlotSpec.model_validate(_spec(2, 2, panel_count=3))


def test_multipanel_plot_forbids_extras():
    bad = _spec(1, 1)
    bad["unexpected"] = True
    with pytest.raises(ValueError, match="Extra"):
        MultipanelPlotSpec.model_validate(bad)


def test_multipanel_plot_rejects_zero_dim():
    with pytest.raises(ValueError):
        MultipanelPlotSpec.model_validate(_spec(0, 1, panel_count=0))


def test_configmodel_multipanel_plots_default(example_forecasters_config):
    """`multipanel_plots` is optional and defaults to an empty dict."""
    cfg = ConfigModel.model_validate(example_forecasters_config)
    assert cfg.multipanel_plots == {}


def test_configmodel_multipanel_plots_roundtrip(example_forecasters_config):
    example_forecasters_config["multipanel_plots"] = {
        "bias_overview": _spec(1, 2),
    }
    cfg = ConfigModel.model_validate(example_forecasters_config)
    assert "bias_overview" in cfg.multipanel_plots
    assert cfg.multipanel_plots["bias_overview"].rows == 1
    assert cfg.multipanel_plots["bias_overview"].cols == 2
