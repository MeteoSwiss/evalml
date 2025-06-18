# ----------------------------------------------------- #
# INFERENCE WORKFLOW                                    #
# ----------------------------------------------------- #

import os
from pathlib import Pathsnake


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
    localrule: True
    params:
        py_version=parse_input(
            input.pyproject, parse_toml, key="project.requires-python"
        ),
    conda:
        "../envs/anemoi_inference.yaml"
    shell:
        "uv venv -p {params.py_version} --relocatable --link-mode=copy {output.venv};"
        "source {output.venv}/bin/activate;"
        "cd $(dirname {input.pyproject}) && uv sync;"
        "python -m compileall -j 8 -o 0 -o 1 -o 2 .venv/lib/python{params.py_version}/site-packages >/dev/null"
        # optionally, precompile to bytecode to reduce the import times
        # "find {output.venv} -exec stat --format='%i' {} + | sort -u | wc -l"  # optionally, how many files did I create?


rule make_squashfs_image:
    input:
        venv=rules.create_inference_venv.output.venv,
    output:
        image=Path(os.environ.get("SCRATCH")) / "sqfs-images" / "{run_id}.squashfs",
    localrule: True
    log:
        "logs/make-squashfs-image-{run_id}.log",
    shell:
        "mksquashfs {input.venv} {output.image}"
        " -no-recovery -noappend -Xcompression-level 3"
        " > {log} 2>/dev/null"
        # we can safely ignore the many warnings "Unrecognised xattr prefix..."


rule run_inference:
    input:
        image=rules.make_squashfs_image.output.image,
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
        "uenv start {input.image};"
        "source /user-environment/bin/activate;"
        "export TZ=UTC;"
        "anemoi-inference run {input.config} "
        " checkpoint={input.checkpoint}"
        " lead_time={config[experiment][lead_time]}"
        " > {log} 2>&1"
