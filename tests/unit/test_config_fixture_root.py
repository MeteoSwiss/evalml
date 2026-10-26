from pathlib import Path
from evalml.config import ConfigModel


def _minimal_config(**overrides):
    cfg = {
        "description": "t",
        "dates": ["2024-01-01T00:00"],
        "runs": [
            {
                "baseline": {
                    "label": "ICON-CH2-EPS",
                    "root": "/store_new/x",
                    "steps": "0/6/6",
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


def test_fixture_root_defaults_to_none():
    model = ConfigModel(**_minimal_config())
    assert model.fixture_root is None


def test_fixture_root_is_parsed_as_path():
    model = ConfigModel(**_minimal_config(fixture_root="/store_new/fx/meteogram-small"))
    assert model.fixture_root == Path("/store_new/fx/meteogram-small")
