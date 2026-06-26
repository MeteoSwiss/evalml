"""Tests for the publication manifest builder and resolver."""

from datetime import datetime

import pytest

from evalml.publication.manifest import build_manifest, load_manifest, write_manifest
from evalml.publication.resolver import Manifest, ResolutionError


def _globals(truth_root="jretrieve:1,2"):
    """RUN_CONFIGS/BASELINE_CONFIGS-shaped fixtures for the paper config."""
    run_configs = {
        "temporal_downscaler-f927-1ee3-on-forecaster-c304-23e7/495c": {
            "label": "Varda-Single",
            "steps": "0/120/1",
            "model_type": "temporal_downscaler",
            "_is_candidate": True,
        },
        # a nested forecaster registered as non-candidate must be excluded
        "forecaster-c304-23e7/abcd": {
            "label": "fc",
            "steps": "0/120/6",
            "model_type": "forecaster",
            "_is_candidate": False,
        },
    }
    baseline_configs = {
        "baseline-7342": {
            "label": "ICON-CH1-CTRL",
            "root": "/store/ICON-CH1-EPS",
            "steps": "0/33/1",
            "member": "control",
        },
        "baseline-mean1": {
            "label": "ICON-CH1-EPS mean",
            "root": "/store/ICON-CH1-EPS",
            "steps": "0/33/1",
            "member": "mean",
        },
    }
    return run_configs, baseline_configs, {"label": "T", "root": truth_root}


def _build(truth_root="jretrieve:1,2"):
    run_configs, baseline_configs, truth_cfg = _globals(truth_root)
    return build_manifest(
        run_configs=run_configs,
        baseline_configs=baseline_configs,
        truth_cfg=truth_cfg,
        truth_hash="2b83",
        reftimes=[datetime(2025, 4, 1, 0, 0), datetime(2025, 4, 3, 6, 0)],
        output_root="output/",
        publication_cfg={"enabled": True, "meteogram": {"init_time": "202504010000"}},
        master_hash="abcd",
    )


def test_build_manifest_participants_and_truth():
    m = _build()
    labels = {p["label"] for p in m["participants"]}
    assert "Varda-Single" in labels
    assert "fc" not in labels  # non-candidate run excluded
    assert m["truth"]["type"] == "jretrieve"
    assert m["truth"]["gridded"] is False
    assert m["dates"]["init_times"] == ["202504010000", "202504030600"]


def test_build_manifest_paths():
    m = _build()
    cand = next(p for p in m["participants"] if p["role"] == "candidate")
    assert cand["paths"]["verif_aggregated"] == (
        "output/data/runs/"
        "temporal_downscaler-f927-1ee3-on-forecaster-c304-23e7/495c/"
        "verif_aggregated_2b83.nc"
    )
    assert "{init_time}" in cand["paths"]["grib_dir_template"]
    assert "{param}" in cand["paths"]["scoremap_template"]


def test_resolver_get_candidate_single():
    m = Manifest(_build())
    cand = m.get_candidate()
    assert cand.label == "Varda-Single"


def test_resolver_resolve_baseline_and_paths():
    m = Manifest(_build())
    base = m.resolve_baseline("ICON-CH1-CTRL")
    sp = m.scoremap_path(base, "T_2M", 24)
    assert sp.endswith("data/baselines/baseline-7342/scoremaps/T_2M_24_2b83.nc")
    grib = m.grib_dir(m.get_candidate(), "202504010000")
    assert grib.endswith("495c/202504010000/grib")


def test_resolver_meteogram_baseline_specs_includes_all():
    m = Manifest(_build())
    specs = m.meteogram_baseline_specs()
    # Every configured baseline is overlaid; member is carried in the spec so the
    # script reads control as a single member and mean as the ensemble average.
    assert "ICON-CH1-EPS mean" in specs
    assert "ICON-CH1-CTRL" in specs
    assert "|control|" in specs and "|mean|" in specs


def test_resolver_verif_paths_all_participants():
    m = Manifest(_build())
    paths = m.verif_paths()
    assert len(paths) == 3  # 2 baselines + 1 candidate


def test_validate_scoremaps_requires_zarr():
    m = Manifest(_build(truth_root="jretrieve:1,2"))
    with pytest.raises(ResolutionError, match="gridded"):
        m.validate_request("scoremaps", baseline="ICON-CH1-CTRL", leadtime=24)


def test_validate_scoremaps_leadtime_not_producible():
    m = Manifest(_build(truth_root="/store/x.zarr"))
    with pytest.raises(ResolutionError, match="not produced"):
        m.validate_request("scoremaps", baseline="ICON-CH1-CTRL", leadtime=120)


def test_validate_meteogram_init_time():
    m = Manifest(_build())
    with pytest.raises(ResolutionError, match="not in the manifest"):
        m.validate_request("meteogram", init_time="209901010000")


def test_write_and_load_roundtrip(tmp_path):
    m = _build()
    path = tmp_path / "publication" / "manifest.json"
    write_manifest(path, m)
    loaded = load_manifest(path)
    assert loaded.get_candidate().label == "Varda-Single"
    assert loaded.master_hash == "abcd"
