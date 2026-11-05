import copy
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace

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


VARIABLES = ["2t", "10u", "tp", "*_850"]


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
    set_output_variables(config, ["2t"])
    assert config == {
        "output": {"grib": {"path": "grib/{step:03}.grib", "variables": ["2t"]}},
        "lead_time": 6,
    }


def test_set_output_variables_short_form_grib():
    config = {"output": {"grib": "grib/{step:03}.grib"}}
    set_output_variables(config, ["2t"])
    assert config == {
        "output": {"grib": {"path": "grib/{step:03}.grib", "variables": ["2t"]}}
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
    set_output_variables(config, ["2t"])
    assert config == {
        "output": {
            "tee": [
                {"grib": {"path": "a.grib", "variables": ["2t"]}},
                {
                    "tee": [
                        {"grib": {"path": "b.grib", "variables": ["2t"]}},
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
        set_output_variables(config, ["2t"])
    assert config == original
    assert "no GRIB output" in caplog.text


def test_set_output_variables_overrides_template():
    config = {"output": {"grib": {"path": "a.grib", "variables": ["sp", "msl"]}}}
    set_output_variables(config, ["2t"])
    assert config["output"]["grib"]["variables"] == ["2t"]


def test_set_output_variables_one_list_per_output():
    """Each grib output gets its own list, so the yaml dump has no anchors."""
    config = {"output": {"tee": [{"grib": {"path": "a"}}, {"grib": {"path": "b"}}]}}
    set_output_variables(config, ["2t"])
    assert "&id" not in yaml.safe_dump(config)


ENSEMBLE_KEYS = {
    "productDefinitionTemplateNumber",
    "number",
    "numberOfForecastsInEnsemble",
    "dataType",
    "typeOfGeneratingProcess",
}


def _without_paths_and_encodings(config):
    """Copy of a config with `path`, `encoding` and `grib2_keys` removed from every
    grib output."""
    config = copy.deepcopy(config)
    for spec in _grib_outputs(config["output"], []):
        spec.pop("path", None)
        spec.pop("encoding", None)
        spec.pop("grib2_keys", None)
    return config


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_member_config_on_templates(path):
    """Every grib output of every bundled template writes to grib/{NNN}/ with the
    same file name and gets the ensemble keys; nothing else changes."""
    original = yaml.safe_load(path.read_text())
    config = _script.member_config(original, 7, 10)

    orig_specs = _grib_outputs(original["output"], [])
    specs = _grib_outputs(config["output"], [])
    assert specs and len(specs) == len(orig_specs)
    for spec, orig in zip(specs, orig_specs):
        head, name = orig["path"].rsplit("/", 1)
        assert head == "grib"
        assert spec["path"] == f"grib/007/{name}"
        assert spec["encoding"] == {
            **(orig.get("encoding") or {}),
            **_script.ensemble_encoding(7, 10),
        }
        assert spec["encoding"]["number"] == 7
        assert spec["encoding"]["numberOfForecastsInEnsemble"] == 10
        # eps in encoding reaches ecCodes, where the COSMO definitions lack it
        assert "eps" not in spec["encoding"]
        assert spec["grib2_keys"] == {**(orig.get("grib2_keys") or {}), "eps": 1}
    assert _without_paths_and_encodings(config) == _without_paths_and_encodings(
        original
    )
    # the input config is not modified
    assert original == yaml.safe_load(path.read_text())


def test_member_config_nested_tee_and_short_form():
    config = {
        "output": {
            "tee": {
                "outputs": [
                    {"grib": "grib/a_{step:03}.grib"},
                    {
                        "tee": [
                            {
                                "grib": {
                                    "path": "grib/b.grib",
                                    "encoding": {"centre": "lssw"},
                                }
                            },
                            {"netcdf": {"path": "c.nc"}},
                        ]
                    },
                ]
            }
        }
    }
    member = _script.member_config(config, 2, 3)
    encoding = _script.ensemble_encoding(2, 3)
    assert member == {
        "output": {
            "tee": {
                "outputs": [
                    {
                        "grib": {
                            "path": "grib/002/a_{step:03}.grib",
                            "encoding": encoding,
                            "grib2_keys": {"eps": 1},
                        }
                    },
                    {
                        "tee": [
                            {
                                "grib": {
                                    "path": "grib/002/b.grib",
                                    "encoding": {"centre": "lssw", **encoding},
                                    "grib2_keys": {"eps": 1},
                                }
                            },
                            {"netcdf": {"path": "c.nc"}},
                        ]
                    },
                ]
            }
        }
    }


def test_member_config_merges_encoding():
    """An existing encoding is merged, not replaced; the ensemble keys win."""
    config = {
        "output": {
            "grib": {
                "path": "grib/x.grib",
                "encoding": {"centre": "lssw", "typeOfGeneratingProcess": 2},
            }
        }
    }
    encoding = _script.member_config(config, 1, 2)["output"]["grib"]["encoding"]
    assert encoding["centre"] == "lssw"
    assert encoding["typeOfGeneratingProcess"] == 4
    assert set(encoding) == ENSEMBLE_KEYS | {"centre"}


def test_member_config_absolute_path():
    config = {"output": {"grib": {"path": "/abs/out/{step:03}.grib"}}}
    member = _script.member_config(config, 12, 20)
    assert member["output"]["grib"]["path"] == "/abs/out/012/{step:03}.grib"


def test_member_config_no_grib_output():
    with pytest.raises(ValueError, match="needs a GRIB output"):
        _script.member_config({"output": {"netcdf": {"path": "out.nc"}}}, 1, 2)


def test_write_member_configs(tmp_path):
    config_path = tmp_path / "config.yaml"
    base = yaml.safe_load(CONFIGS[0].read_text())
    config_path.write_text(yaml.safe_dump(base, sort_keys=False))

    paths = _script.write_member_configs(config_path, tmp_path, 3)

    assert [p.name for p in paths] == [f"config_00{m}.yaml" for m in (1, 2, 3)]
    assert yaml.safe_load(config_path.read_text()) == base  # base config untouched
    for m, p in enumerate(paths, start=1):
        assert (tmp_path / "grib" / f"{m:03d}").is_dir()
        text = p.read_text()
        assert "&id" not in text
        assert yaml.safe_load(text) == _script.member_config(base, m, 3)


FORECASTER = CONFIGS[0].parent / "sgm-forecaster-global-ich1.yaml"


def _fake_smk(tmp_path, ensemble, **extra_params):
    """Minimal stand-in for the snakemake object of inference_prepare_forecaster."""
    from snakemake.io import Namedlist

    tmp_path.mkdir(parents=True, exist_ok=True)
    run_id, init_time = "forecaster-abcd-1234/ef56", "202001010000"
    workdir = tmp_path / "data" / "runs" / run_id / init_time
    params = {
        "lead_time": "120h",
        "output_root": tmp_path / "data",
        "resources_root": Path(__file__).resolve().parents[2] / "resources/inference",
        "holdout_workdir_name": "holdout_stations.csv",
        "reftime_to_iso": "2020-01-01T00:00",
        "output_variables": None,
        **extra_params,
    }
    if ensemble is not ...:
        params["ensemble"] = ensemble
    return SimpleNamespace(
        rule="inference_prepare_forecaster",
        wildcards=SimpleNamespace(run_id=run_id, init_time=init_time),
        input=Namedlist(
            fromdict={"checkpoint": "/ckpt/inference-last.ckpt", "config": FORECASTER}
        ),
        output=Namedlist(
            fromdict={
                "config": str(workdir / "config.yaml"),
                "okfile": str(tmp_path / "logs" / "prepare.ok"),
            }
        ),
        log=[str(tmp_path / "prepare.log")],
        params=Namedlist(fromdict=params),
    ), workdir


@pytest.mark.parametrize("ensemble", [None, ...], ids=["none", "missing"])
def test_prepare_forecaster_deterministic(tmp_path, ensemble):
    """A deterministic run writes config.yaml only, and grib/ has no member dirs."""
    smk, workdir = _fake_smk(tmp_path, ensemble)
    _script.prepare_forecaster(smk)
    assert (workdir / "config.yaml").is_file()
    assert not list(workdir.glob("config_*.yaml"))
    assert not list((workdir / "grib").iterdir())


def test_prepare_forecaster_ensemble(tmp_path):
    """An ensemble run writes the same base config.yaml plus one config per member."""
    det_smk, det_workdir = _fake_smk(tmp_path / "det", None, output_variables=["T_2M"])
    _script.prepare_forecaster(det_smk)

    smk, workdir = _fake_smk(
        tmp_path / "ens", {"members": 3, "seed": 0}, output_variables=["T_2M"]
    )
    _script.prepare_forecaster(smk)

    assert (workdir / "config.yaml").read_bytes() == (
        det_workdir / "config.yaml"
    ).read_bytes()
    base = yaml.safe_load((workdir / "config.yaml").read_text())
    assert sorted(p.name for p in workdir.glob("config_*.yaml")) == [
        "config_001.yaml",
        "config_002.yaml",
        "config_003.yaml",
    ]
    for m in (1, 2, 3):
        member = yaml.safe_load((workdir / f"config_{m:03d}.yaml").read_text())
        assert member == _script.member_config(base, m, 3)
        for spec in _grib_outputs(member["output"], []):
            assert spec["variables"] == ["T_2M"]
        assert (workdir / "grib" / f"{m:03d}").is_dir()
