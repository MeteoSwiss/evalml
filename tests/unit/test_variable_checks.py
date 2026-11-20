import json
import logging
import re
import shutil
import zipfile
from pathlib import Path

import pytest
import yaml

from evalml.variable_checks import (
    check_output_variables,
    checkpoint_variable_names,
    inference_config_variable_names,
    verified_params,
)

LOG = logging.getLogger(__name__)
RESOURCES = Path(__file__).resolve().parents[2] / "resources/inference"

VARIABLES = {"2t", "2d", "10u", "10v", "sp", "msl", "tp", "z", "t_850", "t_500"}
DEFAULT_PARAMS = ["T_2M", "TD_2M", "SP_10M", "PS", "PMSL", "TOT_PREC6"]


def _checkpoint(tmp_path, metadata, folder="inference-last"):
    path = tmp_path / "inference-last.ckpt"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{folder}/data.pkl", b"")
        z.writestr(f"{folder}/anemoi-metadata/anemoi.json", json.dumps(metadata))
    return path


def test_checkpoint_variable_names(tmp_path):
    """Variable names come from the inference metadata, GRIB names from the mars keys."""
    metadata = {
        "metadata_inference": {
            "seed": 0,
            "dataset_names": ["data"],
            "data": {
                "variable_types": {
                    "forcing": ["z", "cos_latitude"],
                    "prognostic": ["2t", "t_850"],
                    "diagnostic": ["tp"],
                }
            },
        },
        "dataset": {
            "data": {
                "variables": ["2t", "cos_latitude", "t_850", "tp", "z"],
                # the source datasets before selection list more variables
                "specific": {"variables": ["2t", "not_selected"]},
                "variables_metadata": {
                    "2t": {"mars": {"param": "2t", "levtype": "sfc"}},
                    "t_850": {"mars": {"param": "t", "levelist": 850}},
                    "tp": {"mars": {"param": "TOT_PREC", "levelist": None}},
                },
            }
        },
    }
    variables, grib_names = checkpoint_variable_names(_checkpoint(tmp_path, metadata))
    assert variables == {"z", "cos_latitude", "2t", "t_850", "tp"}
    # GRIB names equal to a variable name (2t, t_850) are not repeated
    assert grib_names == {"TOT_PREC"}


def test_checkpoint_variable_names_without_inference_metadata(tmp_path):
    """Older checkpoints: the dataset's variables, at the top level of `dataset`."""
    metadata = {"dataset": {"variables": ["2t", "tp"], "variables_metadata": {}}}
    variables, grib_names = checkpoint_variable_names(_checkpoint(tmp_path, metadata))
    assert variables == {"2t", "tp"}
    assert grib_names == set()


def test_checkpoint_without_metadata(tmp_path):
    path = tmp_path / "plain.ckpt"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("archive/data.pkl", b"")
    assert checkpoint_variable_names(path) == (set(), set())


def test_inference_config_variable_names_inline():
    config = {
        "typed_variables": {"SP_10M": {"param": "SP_10M"}},
        "patch_metadata": {
            "dataset": {
                "data": {
                    "variables_metadata": {
                        "PTYPE": {"mars": {"param": "PTYPE"}},
                        "slor": {"mars": {"param": "SSO_SIGMA"}},
                    }
                }
            }
        },
    }
    variables, grib_names = inference_config_variable_names(config, Path("."))
    assert variables == {"SP_10M", "PTYPE", "slor"}
    assert grib_names == {"PTYPE", "SSO_SIGMA"}


def test_inference_config_variable_names_patch_file(tmp_path):
    """The bundled downscaler template declares the derive-diagnostic fields in its patch."""
    template = RESOURCES / "configs/sgm-temporal-downscaler-global_trimedge_multi.yaml"
    config = yaml.safe_load(template.read_text())
    # prepare_workdir copies the metadata patches to the working directory's resources/
    shutil.copytree(RESOURCES / "metadata", tmp_path / "resources")
    variables, _ = inference_config_variable_names(config, tmp_path)
    assert {"SP_10M", "DD_10M", "PTYPE", "TOT_SNOW"} <= variables


def test_inference_config_without_declarations():
    assert inference_config_variable_names({"lead_time": 6}, Path(".")) == (
        set(),
        set(),
    )


def test_check_accepts_model_names_and_globs():
    check_output_variables(
        ["2t", "2d", "10u", "10v", "sp", "msl", "tp", "z", "t_*"],
        VARIABLES,
        set(),
        DEFAULT_PARAMS,
        LOG,
    )


@pytest.mark.parametrize(
    "names, unknown",
    [
        (["T_2M", "tp"], "['T_2M']"),  # GRIB short name
        (["2t", "2x"], "['2x']"),  # typo
        (["2t", "q_*"], "['q_*']"),  # glob matching nothing
    ],
)
def test_check_rejects_names_matching_nothing(names, unknown):
    with pytest.raises(ValueError, match=rf"entries {re.escape(unknown)} match no"):
        check_output_variables(names, VARIABLES, set(), [], LOG)


def test_check_error_lists_the_checkpoint_names():
    with pytest.raises(ValueError, match="10u, 10v, 2d, 2t, msl, sp, t_500"):
        check_output_variables(["T_2M"], VARIABLES, set(), [], LOG)


def test_check_grib_name_only_warns(caplog):
    """A name matching only the GRIB param name of a variable is accepted with a warning."""
    with caplog.at_level(logging.WARNING):
        check_output_variables(["TOT_PREC"], {"tp"}, {"TOT_PREC"}, ["TOT_PREC6"], LOG)
    assert "GRIB param name" in caplog.text


@pytest.mark.parametrize(
    "names, param, needs",
    [
        (["2t", "z"], "TOT_PREC6", "tp"),
        (["2t", "10u", "z"], "SP_10M", "10u, 10v"),
        (["10u", "10v", "z"], "T_2M", "2t"),
    ],
)
def test_check_rejects_missing_verified_params(names, param, needs):
    message = f"['{param}'] are not in output_variables ({param} needs {needs})"
    with pytest.raises(ValueError, match=re.escape(message)):
        check_output_variables(names, VARIABLES, set(), [param], LOG)


def test_check_skips_params_the_checkpoint_lacks():
    """Parameters the model cannot provide are left to the verification (skipped there)."""
    check_output_variables(["2t", "z"], {"2t", "z"}, set(), ["T_2M", "HZEROCL"], LOG)


def test_check_accepts_derived_params_written_natively():
    """A derived parameter written as such (e.g. by a post-processor) needs no components."""
    check_output_variables(
        ["SP_10M", "z"], {"SP_10M", "10u", "10v", "z"}, set(), ["SP_10M"], LOG
    )


def test_check_reports_both_errors():
    with pytest.raises(ValueError, match=r"match no.*Verified parameters"):
        check_output_variables(["T_2M", "z"], VARIABLES, set(), ["T_2M"], LOG)


def test_verified_params():
    assert verified_params({}) == []
    assert verified_params({"experiment": None}) == []
    base = {"stratification": {"regions": ["icon"]}}
    assert verified_params({"experiment": base}) == DEFAULT_PARAMS
    experiment = {
        **base,
        "params": ["T_2M"],
        "sal": {"enabled": True, "params": ["TOT_PREC1"]},
        "scoremaps": {"enabled": False, "params": ["PMSL"]},
    }
    assert verified_params({"experiment": experiment}) == ["T_2M", "TOT_PREC1"]
