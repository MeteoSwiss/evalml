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
    log:
        "logs/make-squashfs-image-{run_id}.log",
    shell:
        "mksquashfs {input.venv} {output.image}"
        " -no-recovery -noappend -Xcompression-level 3"
        " > {log} 2>/dev/null"
        # we can safely ignore the many warnings "Unrecognised xattr prefix..."


rule run_inference_group:
    input:
        pyproject=rules.create_inference_pyproject.output.pyproject,
        image=rules.make_squashfs_image.output.image,
        config="config/anemoi_inference.yaml",
    output:
        okfile=temp("resources/inference/{run_id}/output/group-{group_index}.ok"),
    params:
        checkpoints_path=parse_input(
            input.pyproject, parse_toml, key="tool.anemoi.checkpoints_path"
        ),
        group_reftimes=lambda wc: REFTIMES_GROUPS[int(wc.group_index)],
        group_size=config["execution"]["run_group_size"],
        leadtime="120h",  # lead time in hours
    log:
        "logs/anemoi-inference-run-{run_id}-{group_index}.log",
    conda:
        "../envs/anemoi_inference.yaml"
    resources:
        slurm_partition="normal",
        cpus_per_task=8,
        runtime="20m",
        gres="gpu:4",
        slurm_extra=lambda wc, input: f"--uenv={input.image}:/user-environment",
    script:
        "../scripts/run_inference_group.py"


rule map_init_time_to_inference_group:
    input:
        lambda wc: f"resources/inference/{wc.run_id}/output/group-{REFTIME_TO_GROUP[wc.init_time]}.ok",
    output:
        "resources/inference/{run_id}/output/{init_time}/raw",
