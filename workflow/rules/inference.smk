# ----------------------------------------------------- #
# INFERENCE WORKFLOW                                    #
# ----------------------------------------------------- #

import os
from pathlib import Path


rule set_inference_requirements:
    input:
        "workflow/envs/anemoi_inference.txt",
    output:
        requirements="resources/inference/{run_id}/requirements.txt",
    log:
        "logs/anemoi-inference-requirements-{run_id}.log",
    conda:
        "../envs/anemoi_inference.yaml"
    localrule: True
    params:
        mlflow_uri="https://servicedepl.meteoswiss.ch/mlstore/",
    script:
        "../scripts/set_inference_requirements.py"


rule create_inference_venv:
    input:
        requirements=rules.set_inference_requirements.output.requirements,
    output:
        venv=temp("resources/inference/{run_id}/.venv"),
    localrule: True
    conda:
        "../envs/anemoi_inference.yaml"
    shell:
        "uv venv -p python --relocatable --link-mode=copy resources/inference/{wildcards.run_id}/.venv;"
        "resources/inference/{wildcards.run_id}/.venv/bin/python -m ensurepip --upgrade;"
        "resources/inference/{wildcards.run_id}/.venv/bin/pip3 install -r {input.requirements}"


rule create_inference_uenv:
    input:
        rules.create_inference_venv.output.venv,
    output:
        Path(os.environ.get("SCRATCH")) / "{run_id}.squashfs",
    localrule: True
    conda:
        "../envs/anemoi_inference.yaml"
    shell:
        "mksquashfs resources/inference/{wildcards.run_id}/.venv $SCRATCH/{wildcards.run_id}.squashfs "
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
        "uenv start [--view=modules mch/v8:rc4] {input.py_venv}:$PWD/.venv;"
        ".venv/bin/activate;"
        "export TZ=UTC;"
        "anemoi-inference run {input.config} "
        " checkpoint={input.checkpoint}"
        " lead_time={config[experiment][lead_time]}"
        " > {log} 2>&1"
