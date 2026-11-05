"""Tests for the run identity and DAG-build checks in workflow/rules/common.smk."""

import ast
import hashlib
import json
import logging
from pathlib import Path

import pytest
import yaml

COMMON = Path(__file__).resolve().parents[2] / "workflow/rules/common.smk"
FUNCTIONS = {
    "generate_json_hash",
    "run_specific_hash",
    "warn_ensembles_without_output_variables",
}


def _load_functions():
    """Load a few self-contained functions of common.smk.

    common.smk is plain Python, but executing it whole needs a snakemake config;
    only the function definitions (and HASH_LENGTH) are executed here.
    """
    tree = ast.parse(COMMON.read_text())
    body = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS)
        or (
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == "HASH_LENGTH" for t in node.targets)
        )
    ]
    namespace = {"yaml": yaml, "json": json, "hashlib": hashlib, "logging": logging}
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(COMMON), "exec"), namespace
    )
    return namespace


_common = _load_functions()
run_specific_hash = _common["run_specific_hash"]
warn_ensembles_without_output_variables = _common[
    "warn_ensembles_without_output_variables"
]


@pytest.fixture
def run_config(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "output:\n  grib:\n    path: grib/{date}{time:04}_{step:03}.grib\n"
        "lead_time: 120h\n"
    )
    return {"steps": "0/120/6", "config": str(config)}


def test_deterministic_hash_unchanged(run_config):
    """run_ids of deterministic runs are the same as before ensembles existed
    (values computed with the code of 4995a9a)."""
    assert run_specific_hash(run_config, "forecaster") == "8eb0"
    assert run_specific_hash({**run_config, "ensemble": None}, "forecaster") == "8eb0"
    with_vars = {**run_config, "output_variables": ["T_2M", "FIS"]}
    assert run_specific_hash(with_vars, "forecaster") == "beff"
    assert run_specific_hash({**with_vars, "ensemble": None}, "forecaster") == "beff"


def test_ensemble_hash(run_config):
    """members and seed each give a new run_id."""
    hashes = {
        run_specific_hash(run_config, "forecaster"),
        run_specific_hash(
            {**run_config, "ensemble": {"members": 4, "seed": 0}}, "forecaster"
        ),
        run_specific_hash(
            {**run_config, "ensemble": {"members": 5, "seed": 0}}, "forecaster"
        ),
        run_specific_hash(
            {**run_config, "ensemble": {"members": 4, "seed": 1}}, "forecaster"
        ),
    }
    assert len(hashes) == 4


def test_warn_ensembles_without_output_variables(caplog):
    run_configs = {
        "forecaster-a/0001": {"ensemble": None, "output_variables": None},
        "forecaster-a/0002": {
            "ensemble": {"members": 4, "seed": 0},
            "output_variables": ["T_2M"],
        },
        "forecaster-a/0003": {
            "ensemble": {"members": 10, "seed": 0},
            "output_variables": None,
        },
    }
    with caplog.at_level(logging.WARNING, logger="snakemake"):
        warn_ensembles_without_output_variables(run_configs)
    assert len(caplog.records) == 1
    assert "forecaster-a/0003" in caplog.text
    assert "10 members without output_variables" in caplog.text
