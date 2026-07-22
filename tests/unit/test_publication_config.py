"""Coherence-validation tests for the publication config block."""

import pytest
import yaml
from pathlib import Path

from evalml.config import ConfigModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def paper_config():
    # `varda-single_paper.yaml` was split into `_analysis` / `_stations`, and the
    # publication block (opt-in) was dropped from the shipped configs in that move.
    # Load the analysis config and attach a representative publication block so
    # these tests exercise the publication-config validation rules.
    cfg = yaml.safe_load(
        (PROJECT_ROOT / "config/varda-single_paper_analysis.yaml").read_text()
    )
    cfg["publication"] = {
        "leadtimes": {"enabled": True},
        "scoremaps": {
            "enabled": True,
            "steps": [6, 24],
            "params": ["T_2M", "SP_10M"],
            "scores": ["MSE_SKILL", "BIAS_CONTRIB"],
            "baseline_label": "ICON-CH1-CTRL",
            "season": "all",
            "region": "switzerland",
        },
        "meteogram": {
            "enabled": False,
            "init_time": "202504010000",
            "station": "KLO",
            "params": ["T_2M", "TOT_PREC", "SP_10M", "DD_10M"],
        },
    }
    return cfg


def test_paper_config_validates(paper_config):
    """The shipped publication config must still validate unchanged."""
    model = ConfigModel.model_validate(paper_config)
    # Per-task enable switches (paper-figures design).
    assert model.publication.leadtimes.enabled
    assert model.publication.scoremaps.enabled
    # Structural check against the config (not a fixed case) so the plotted
    # meteogram case can change without breaking this test.
    assert (
        model.publication.meteogram.init_time
        == paper_config["publication"]["meteogram"]["init_time"]
    )


def test_scoremaps_block_now_accepted(paper_config):
    """A scoremaps block is accepted by extra:forbid (needs zarr truth)."""
    paper_config["truth"] = {
        "label": "KENDA-CH1",
        "root": "/store/x.zarr",
    }
    paper_config["publication"]["scoremaps"] = {
        "enabled": True,
        "baseline_label": "ICON-CH1-CTRL",
        "steps": [24],
    }
    model = ConfigModel.model_validate(paper_config)
    assert model.publication.scoremaps.enabled
    assert model.publication.scoremaps.scores == ["MSE_SKILL", "BIAS_CONTRIB"]


def test_rule_a_scoremaps_require_zarr_truth(paper_config):
    """Rule (a): scoremaps against jretrieve/obs truth are rejected."""
    # Force jretrieve truth so the test doesn't depend on the config file's setting.
    paper_config["truth"] = {"label": "SwissMetNet", "root": "jretrieve:1,2"}
    paper_config["publication"]["scoremaps"] = {"enabled": True}
    with pytest.raises(ValueError, match="gridded.*zarr"):
        ConfigModel.model_validate(paper_config)


def test_scoremaps_steps_list_accepted(paper_config):
    """A `steps` list validates and is exposed on the model."""
    paper_config["truth"] = {"label": "KENDA-CH1", "root": "/store/x.zarr"}
    paper_config["publication"]["scoremaps"] = {
        "enabled": True,
        "baseline_label": "ICON-CH2-CTRL",  # steps 0/120/1 -> 6 and 24 producible
        "steps": [6, 24],
    }
    model = ConfigModel.model_validate(paper_config)
    assert model.publication.scoremaps.steps == [6, 24]


def test_rule_b_one_of_steps_not_producible(paper_config):
    """Rule (b): any lead time beyond the baseline's steps is rejected."""
    paper_config["truth"] = {"label": "KENDA-CH1", "root": "/store/x.zarr"}
    # ICON-CH1-CTRL steps 0/33/1 -> 120h not producible even though 24h is.
    paper_config["publication"]["scoremaps"] = {
        "enabled": True,
        "baseline_label": "ICON-CH1-CTRL",
        "steps": [24, 120],
    }
    with pytest.raises(ValueError, match="120h is not produced by baseline"):
        ConfigModel.model_validate(paper_config)


def test_rule_b_step_not_producible_by_baseline(paper_config):
    """Rule (b): a single step beyond a baseline's steps is rejected."""
    paper_config["truth"] = {"label": "KENDA-CH1", "root": "/store/x.zarr"}
    # ICON-CH1-CTRL has steps 0/33/1 -> 120h not producible.
    paper_config["publication"]["scoremaps"] = {
        "enabled": True,
        "baseline_label": "ICON-CH1-CTRL",
        "steps": [120],
    }
    with pytest.raises(ValueError, match="120h is not produced by baseline"):
        ConfigModel.model_validate(paper_config)


def test_rule_d_unknown_baseline_label(paper_config):
    """Rule (d): an unknown baseline label is rejected with the available list."""
    paper_config["truth"] = {"label": "KENDA-CH1", "root": "/store/x.zarr"}
    paper_config["publication"]["scoremaps"] = {
        "enabled": True,
        "baseline_label": "IFS",
    }
    with pytest.raises(ValueError, match="not found.*Available baseline labels"):
        ConfigModel.model_validate(paper_config)


def test_rule_c_meteogram_init_time_out_of_range(paper_config):
    """Rule (c): an enabled meteogram init_time outside the date range is rejected."""
    paper_config["publication"]["meteogram"]["enabled"] = True
    paper_config["publication"]["meteogram"]["init_time"] = "209901010000"
    with pytest.raises(ValueError, match="init_time.*not in the configured"):
        ConfigModel.model_validate(paper_config)


def test_meteogram_init_time_format_rejected(paper_config):
    """Field-local: malformed init_time (not YYYYMMDDHHMM) is rejected."""
    paper_config["publication"]["meteogram"]["init_time"] = "2025-04-01"
    with pytest.raises(ValueError):
        ConfigModel.model_validate(paper_config)


def test_disabled_meteogram_skips_init_time_check(paper_config):
    """An out-of-range meteogram init_time is ignored when the meteogram is disabled."""
    paper_config["publication"]["meteogram"]["enabled"] = False
    paper_config["publication"]["meteogram"]["init_time"] = "209901010000"
    ConfigModel.model_validate(paper_config)  # must not raise
