# ----------------------------------------------------- #
# INFERENCE WORKFLOW                                    #
# ----------------------------------------------------- #

import os
from pathlib import Path


rule create_inference_pyproject:
    input:
        toml="workflow/envs/anemoi_inference.toml",
    output:
        pyproject="resources/inference/{run_id}/pyproject.toml",
    log:
        "logs/create-inference-pyproject-{run_id}.log",
    conda:
        "../envs/anemoi_inference.yaml"
    localrule: True
    params:
        mlflow_uri="https://servicedepl.meteoswiss.ch/mlstore/",
    script:
        "../scripts/set_inference_pyproject.py"


rule create_inference_venv:
    input:
        pyproject="resources/inference/{run_id}/pyproject.toml",
    output:
        venv=temp(directory("resources/inference/{run_id}/.venv")),
    log:
        "logs/create-inference-venv-{run_id}.log",
    localrule: True
    params:
        py_version="3.10",  # TODO: parse this from mlflow too
    conda:
        "../envs/anemoi_inference.yaml"
    shell:
        "uv venv -p python{params.py_version} --relocatable --link-mode=copy {output.venv};"
        "cd $(dirname {input.pyproject}) && uv pip install ."


rule create_inference_uenv:
    input:
        rules.create_inference_venv.output.venv,
    output:
        Path(os.environ.get("SCRATCH")) / "{run_id}.squashfs",
    localrule: True
    conda:
        "../envs/anemoi_inference.yaml"
    shell:
        "mksquashfs resources/inference/{wildcards.run_id}/.venv $SCRATCH/tmp/{wildcards.run_id}.squashfs "
        "-no-recovery -noappend -Xcompression-level 3 "


rule run_inference:
    input:
        py_venv="$SCRATCH/{run_id}.squashfs",
        config="config/anemoi_inference.yaml",
        checkpoint=Path(config["locations"]["checkpoint_root"])
        / "{run_id}"
        / "inference-last.ckpt",
    output:
        "resources/inference/output/{run_id}/{init_time}",
    log:
        "logs/anemoi-inference-run-{run_id}-{init_time}.log",
    conda:
        "../envs/anemoi-inference.yaml"
    resources:
        partition="normal",
        cpus_per_task=8,
        time="20m",
        gres="gpu:4",
    shell:
        "uenv start {input.py_venv}:$PWD/.venv;"
        ". .venv/bin/activate;"
        "export TZ=UTC;"
        "anemoi-inference run {input.config} "
        " checkpoint={input.checkpoint}"
        " lead_time={config[experiment][lead_time]}"
        " > {log} 2>&1"
