import pytest
from pydantic import ValidationError

from evalml.config import (
    ConfigModel,
    GRIB_MODEL_TYPES,
    GRIBModelRunConfig,
    InferenceModelRunConfig,
    SpatialDownscalerConfig,
)


def _minimal_config(**overrides):
    cfg = {
        "description": "t",
        "dates": ["2024-01-01T00:00"],
        "runs": [
            {
                "spatial_downscaler": {
                    "root": "/store_new/x",
                    "label": "HiRAD-downscaling",
                    "steps": "0/12/1",
                }
            }
        ],
        "truth": {"label": "SwissMetNet", "root": "jretrievedwh:1,2"},
        "experiment": {
            "params": ["T_2M"],
            "stratification": {"regions": ["icon"]},
            "dashboard": {"stratification": []},
        },
        "locations": {"output_root": "output/"},
        "profile": {
            "executor": "slurm",
            "global_resources": {"gpus": 1},
            "default_resources": {
                "slurm_partition": "postproc",
                "cpus_per_task": 1,
                "mem_mb_per_cpu": 1800,
                "runtime": "1h",
            },
            "jobs": 1,
        },
    }
    cfg.update(overrides)
    return cfg


def test_spatial_downscaler_run_validates():
    model = ConfigModel(**_minimal_config())
    run = model.runs[0].spatial_downscaler
    assert run.root == "/store_new/x"
    assert run.label == "HiRAD-downscaling"
    assert run.steps == "0/12/1"


def test_spatial_downscaler_rejects_extra_keys():
    with pytest.raises(ValidationError):
        SpatialDownscalerConfig(root="/store_new/x", steps="0/12/1", bogus="nope")


def test_spatial_downscaler_requires_root():
    with pytest.raises(ValidationError):
        SpatialDownscalerConfig(steps="0/12/1")


def test_grib_model_types_is_spatial_downscaler_only():
    assert GRIB_MODEL_TYPES == frozenset({"spatial_downscaler"})


def test_grib_model_run_config_is_not_inference_model_run_config():
    grib_run = SpatialDownscalerConfig(root="/store_new/x", steps="0/12/1")
    assert not isinstance(grib_run, InferenceModelRunConfig)


def test_inference_model_run_config_is_not_grib_model_run_config():
    inference_run = InferenceModelRunConfig(
        checkpoint="/some/checkpoint.ckpt", steps="0/12/1", config={}
    )
    assert not isinstance(inference_run, GRIBModelRunConfig)
