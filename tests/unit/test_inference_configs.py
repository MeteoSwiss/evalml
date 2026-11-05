import copy
import importlib.util
import logging
from pathlib import Path

import pytest
import yaml

CONFIGS = sorted(
    (Path(__file__).resolve().parents[2] / "resources/inference/configs").glob("*.yaml")
)

SCRIPT = Path(__file__).resolve().parents[2] / "workflow/scripts/inference_prepare.py"
# inference_prepare.py loaded by path (workflow/scripts is not a package).
_spec = importlib.util.spec_from_file_location("inference_prepare", SCRIPT)
_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_script)
_LOG = logging.getLogger(__name__)


def set_output_variables(config, variables):
    return _script.set_output_variables(config, variables, _LOG)


def _rescales(node, out):
    """Collect (filter key, param, scale, offset) of every rescale transform filter."""
    if isinstance(node, list):
        for item in node:
            _rescales(item, out)
    elif isinstance(node, dict):
        for key, value in node.items():
            if key.endswith("_transform_filter") and isinstance(value, dict):
                f = (
                    value.get("rescale", value)
                    if value.get("filter", "rescale") == "rescale"
                    else {}
                )
                if "param" in f:
                    out.append((key, f["param"], f["scale"], f["offset"]))
            _rescales(value, out)
    return out


GRIB_NAMES = {
    "tp": "TOT_PREC",
    "tcc": "CLCT",
    "lcc": "CLCL",
    "mcc": "CLCM",
    "hcc": "CLCH",
}


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_rescale_has_grib_name_twin(path):
    """anemoi-inference <= 0.11.1 exposes the model name (e.g. tp) to output filters, >= 0.11.2
    and GRIB inputs the GRIB name (e.g. TOT_PREC); a filter on only one name silently does nothing."""
    rescales = _rescales(yaml.safe_load(path.read_text()), [])
    for key, param, scale, offset in rescales:
        if param in GRIB_NAMES:
            assert (key, GRIB_NAMES[param], scale, offset) in rescales, (
                path.name,
                param,
            )


VARIABLES = ["T_2M", "TOT_PREC", "tp", "*_850"]


def _grib_outputs(output, out):
    """Collect every grib output spec of an output block (bare grib or nested tee)."""
    for kind, spec in output.items():
        if kind == "grib":
            out.append(spec)
        elif kind == "tee":
            for sub in spec.get("outputs", []) if isinstance(spec, dict) else spec:
                _grib_outputs(sub, out)
    return out


def _without_variables(config):
    """Copy of a config with `variables` removed from every grib output."""
    config = copy.deepcopy(config)
    for spec in _grib_outputs(config["output"], []):
        spec.pop("variables", None)
    return config


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_set_output_variables_on_templates(path):
    """Every grib output of every bundled template gets the list; nothing else changes."""
    original = yaml.safe_load(path.read_text())
    config = set_output_variables(copy.deepcopy(original), VARIABLES)

    grib_outputs = _grib_outputs(config["output"], [])
    assert grib_outputs
    for spec in grib_outputs:
        assert spec["variables"] == VARIABLES
    assert _without_variables(config) == _without_variables(original)


def test_set_output_variables_bare_grib():
    config = {"output": {"grib": {"path": "grib/{step:03}.grib"}}, "lead_time": 6}
    set_output_variables(config, ["T_2M"])
    assert config == {
        "output": {"grib": {"path": "grib/{step:03}.grib", "variables": ["T_2M"]}},
        "lead_time": 6,
    }


def test_set_output_variables_short_form_grib():
    config = {"output": {"grib": "grib/{step:03}.grib"}}
    set_output_variables(config, ["T_2M"])
    assert config == {
        "output": {"grib": {"path": "grib/{step:03}.grib", "variables": ["T_2M"]}}
    }


def test_set_output_variables_nested_tee():
    config = {
        "output": {
            "tee": [
                {"grib": {"path": "a.grib"}},
                {"tee": [{"grib": {"path": "b.grib"}}, {"netcdf": {"path": "c.nc"}}]},
            ]
        }
    }
    set_output_variables(config, ["T_2M"])
    assert config == {
        "output": {
            "tee": [
                {"grib": {"path": "a.grib", "variables": ["T_2M"]}},
                {
                    "tee": [
                        {"grib": {"path": "b.grib", "variables": ["T_2M"]}},
                        {"netcdf": {"path": "c.nc"}},
                    ]
                },
            ]
        }
    }


def test_set_output_variables_no_grib_output(caplog):
    config = {"output": {"netcdf": {"path": "out.nc"}}}
    original = copy.deepcopy(config)
    with caplog.at_level(logging.WARNING):
        set_output_variables(config, ["T_2M"])
    assert config == original
    assert "no GRIB output" in caplog.text


def test_set_output_variables_overrides_template():
    config = {"output": {"grib": {"path": "a.grib", "variables": ["PS", "PMSL"]}}}
    set_output_variables(config, ["T_2M"])
    assert config["output"]["grib"]["variables"] == ["T_2M"]


def test_set_output_variables_one_list_per_output():
    """Each grib output gets its own list, so the yaml dump has no anchors."""
    config = {"output": {"tee": [{"grib": {"path": "a"}}, {"grib": {"path": "b"}}]}}
    set_output_variables(config, ["T_2M"])
    assert "&id" not in yaml.safe_dump(config)
