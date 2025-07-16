# ----------------------------------------------------- #
# INFERENCE WORKFLOW                                    #
# ----------------------------------------------------- #

import os
from pathlib import Path
from datetime import datetime


configfile: "config/config.yaml"


rule create_inference_pyproject:
    input:
        toml="workflow/envs/anemoi_inference.toml",
    output:
        pyproject=OUT_ROOT / "data/runs/{run_id}/pyproject.toml",
    log:
        OUT_ROOT / "logs/create_inference_pyproject/{run_id}.log",
    localrule: True
    script:
        "../scripts/set_inference_pyproject.py"


rule create_inference_venv:
    input:
        pyproject=rules.create_inference_pyproject.output.pyproject,
    output:
        venv=temp(directory(OUT_ROOT / "data/runs/{run_id}/.venv")),
    params:
        py_version=parse_input(
            input.pyproject, parse_toml, key="project.requires-python"
        ),
    localrule: True
    log:
        OUT_ROOT / "logs/create_inference_venv/{run_id}.log",
    shell:
        """(
        PROJECT_ROOT=$(dirname {input.pyproject})
        uv venv --project $PROJECT_ROOT --relocatable --link-mode=copy {output.venv}
        source {output.venv}/bin/activate
        cd $(dirname {input.pyproject})
        uv sync
        python -m compileall -j 8 -o 0 -o 1 -o 2 .venv/lib/python*/site-packages
        echo 'Testing that eccodes is working...'
        if ! python -c "import eccodes" &>/dev/null; then
            echo 'ERROR: eccodes is not installed correctly in the virtual environment.'
            echo 'Please check the installation and try again.'
            exit 1
        fi
        echo 'Inference virutal environment successfully created at {output.venv}'
        ) > {log} 2>&1
        """


# optionally, precompile to bytecode to reduce the import times
# find {output.venv} -exec stat --format='%i' {} + | sort -u | wc -l  # optionally, how many files did I create?


rule make_squashfs_image:
    input:
        venv=rules.create_inference_venv.output.venv,
    output:
        image=OUT_ROOT / "data/runs/{run_id}/venv.squashfs",
    log:
        OUT_ROOT / "logs/make_squashfs_image/{run_id}.log",
    localrule: True
    shell:
        # we can safely ignore the many warnings "Unrecognised xattr prefix..."
        "mksquashfs {input.venv} {output.image}"
        " -no-recovery -noappend -Xcompression-level 3"
        " > {log} 2>/dev/null"


rule run_inference:
    input:
        pyproject=rules.create_inference_pyproject.output.pyproject,
        image=rules.make_squashfs_image.output.image,
        config=str(Path("config/anemoi_inference.yaml").resolve()),
    output:
        directory(OUT_ROOT / "data/runs/{run_id}/{init_time}/grib"),
        directory(OUT_ROOT / "data/runs/{run_id}/{init_time}/raw"),
    params:
        checkpoints_path=parse_input(
            input.pyproject, parse_toml, key="tool.anemoi.checkpoints_path"
        ),
        lead_time=config["lead_time"],
        reftime_to_iso=lambda wc: datetime.strptime(
            wc.init_time, "%Y%m%d%H%M"
        ).strftime("%Y-%m-%dT%H:%M"),
        output_root=OUT_ROOT / "data",
    log:
        OUT_ROOT / "logs/inference_run/{run_id}-{init_time}.log",
    resources:
        slurm_partition="debug",
        cpus_per_task=32,
        mem_mb_per_cpu=8000,
        runtime="20m",
        gres="gpu:1",
        slurm_extra=lambda wc, input: f"--uenv={input.image}:/user-environment",
    shell:
        """
        (
        export TZ=UTC
        source /user-environment/bin/activate
        export ECCODES_DEFINITION_PATH=/user-environment/share/eccodes-cosmo-resources/definitions

        # prepare the working directory
        WORKDIR={params.output_root}/runs/{wildcards.run_id}/{wildcards.init_time}
        mkdir -p $WORKDIR && cd $WORKDIR && mkdir -p grib raw
        cp {input.config} config.yaml

        CMD_ARGS=(
            date={params.reftime_to_iso}
            checkpoint={params.checkpoints_path}/inference-last.ckpt
            lead_time={params.lead_time}
        )
        echo "=========================================================="
        echo "SLURM JOB ID: $SLURM_JOB_ID"
        echo "HOSTNAME: $(hostname)"
        echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
        echo "=========================================================="

        anemoi-inference run config.yaml "${{CMD_ARGS[@]}}"
        ) > {log} 2>&1
        """
