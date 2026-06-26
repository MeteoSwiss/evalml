"""Coherence-validation tests for the publication config block."""

import pytest
import yaml
from pathlib import Path

from evalml.config import ConfigModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def paper_config():
    cfg = yaml.safe_load((PROJECT_ROOT / "config/varda-single_paper.yaml").read_text())
    return cfg


def test_paper_config_validates(paper_config):
    """The shipped publication config must still validate unchanged."""
    model = ConfigModel.model_validate(paper_config)
    assert model.publication.enabled
    # Structural check against the config (not a fixed case) so the plotted
    # meteogram case can change without breaking this test.
    assert (
        model.publication.meteogram.init_time
        == paper_config["publication"]["meteogram"]["init_time"]
    )
    assert model.publication.scoremaps is None


def test_scoremaps_block_now_accepted(paper_config):
    """A scoremaps block is no longer rejected by extra:forbid (needs zarr truth)."""
    paper_config["truth"] = {
        "label": "KENDA-CH1",
        "root": "/store/x.zarr",
    }
    paper_config["publication"]["scoremaps"] = {
        "enabled": True,
        "baseline_label": "ICON-CH1-CTRL",
        "leadtime": 24,
    }
    model = ConfigModel.model_validate(paper_config)
    assert model.publication.scoremaps.enabled
    assert model.publication.scoremaps.scores == ["MSE_SKILL", "BIAS_CONTRIB"]


def test_rule_a_scoremaps_require_zarr_truth(paper_config):
    """Rule (a): scoremaps against jretrieve/obs truth are rejected."""
    paper_config["publication"]["scoremaps"] = {"enabled": True}
    with pytest.raises(ValueError, match="gridded.*zarr"):
        ConfigModel.model_validate(paper_config)


def test_scoremaps_leadtimes_list_accepted(paper_config):
    """A `leadtimes` list validates and is exposed via effective_leadtimes()."""
    paper_config["truth"] = {"label": "KENDA-CH1", "root": "/store/x.zarr"}
    paper_config["publication"]["scoremaps"] = {
        "enabled": True,
        "baseline_label": "ICON-CH2-CTRL",  # steps 0/120/1 -> 6 and 24 producible
        "leadtimes": [6, 24],
    }
    model = ConfigModel.model_validate(paper_config)
    assert model.publication.scoremaps.effective_leadtimes() == [6, 24]


def test_scoremaps_singular_leadtime_backward_compat(paper_config):
    """The singular `leadtime` still works as a one-element leadtimes list."""
    paper_config["truth"] = {"label": "KENDA-CH1", "root": "/store/x.zarr"}
    paper_config["publication"]["scoremaps"] = {
        "enabled": True,
        "baseline_label": "ICON-CH1-CTRL",
        "leadtime": 24,
    }
    model = ConfigModel.model_validate(paper_config)
    assert model.publication.scoremaps.effective_leadtimes() == [24]


def test_rule_b_one_of_leadtimes_not_producible(paper_config):
    """Rule (b): any lead time beyond the baseline's steps is rejected."""
    paper_config["truth"] = {"label": "KENDA-CH1", "root": "/store/x.zarr"}
    # ICON-CH1-CTRL steps 0/33/1 -> 120h not producible even though 24h is.
    paper_config["publication"]["scoremaps"] = {
        "enabled": True,
        "baseline_label": "ICON-CH1-CTRL",
        "leadtimes": [24, 120],
    }
    with pytest.raises(ValueError, match="120h is not produced by baseline"):
        ConfigModel.model_validate(paper_config)


def test_rule_b_leadtime_not_producible_by_baseline(paper_config):
    """Rule (b): leadtime beyond a baseline's steps is rejected."""
    paper_config["truth"] = {"label": "KENDA-CH1", "root": "/store/x.zarr"}
    # ICON-CH1-CTRL has steps 0/33/1 -> 120h not producible.
    paper_config["publication"]["scoremaps"] = {
        "enabled": True,
        "baseline_label": "ICON-CH1-CTRL",
        "leadtime": 120,
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
    """Rule (c): a meteogram init_time outside the date range is rejected."""
    paper_config["publication"]["meteogram"]["init_time"] = "209901010000"
    with pytest.raises(ValueError, match="init_time.*not in the configured"):
        ConfigModel.model_validate(paper_config)


def test_meteogram_init_time_format_rejected(paper_config):
    """Field-local: malformed init_time (not YYYYMMDDHHMM) is rejected."""
    paper_config["publication"]["meteogram"]["init_time"] = "2025-04-01"
    with pytest.raises(ValueError):
        ConfigModel.model_validate(paper_config)


def test_disabled_publication_skips_checks(paper_config):
    """An out-of-range meteogram is ignored when publication is disabled."""
    paper_config["publication"]["enabled"] = False
    paper_config["publication"]["meteogram"]["init_time"] = "209901010000"
    ConfigModel.model_validate(paper_config)  # must not raise
