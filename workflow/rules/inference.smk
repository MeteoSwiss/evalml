# ----------------------------------------------------- #
# INFERENCE WORKFLOW                                    #
# ----------------------------------------------------- #

import os
from pathlib import Path
from datetime import datetime


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
        lockfile=OUT_ROOT / "data/runs/{run_id}/uv.lock",
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
        "mksquashfs $(realpath {input.venv}) {output.image}"
        " -no-recovery -noappend -Xcompression-level 3"
        " > {log} 2>/dev/null"


rule create_inference_sandbox:
    """Generate a zipped directory that can be used as a sandbox for running inference jobs.

    TO use this sandbox, unzip it to a target directory.

    ```bash
    unzip sandbox.zip -d /path/to/target/directory
    ```
    """
    input:
        script="workflow/scripts/inference_create_sandbox.py",
        image=rules.make_squashfs_image.output.image,
        pyproject=rules.create_inference_pyproject.output.pyproject,
        lockfile=rules.create_inference_venv.output.lockfile,
        config=str(Path("config/anemoi_inference.yaml").resolve()),
    output:
        sandbox=OUT_ROOT / "data/runs/{run_id}/sandbox.zip",
    log:
        OUT_ROOT / "logs/create_inference_sandbox/{run_id}.log",
    localrule: True
    shell:
        """
        uv run {input.script} \
            --pyproject {input.pyproject} \
            --lockfile {input.lockfile} \
            --output {output.sandbox} \
            > {log} 2>&1
        """


rule run_inference_group:
    input:
        pyproject=rules.create_inference_pyproject.output.pyproject,
        image=rules.make_squashfs_image.output.image,
        config=str(Path("config/anemoi_inference.yaml").resolve()),
    output:
        okfile=temp(touch(OUT_ROOT / "data/runs/{run_id}/group-{group_index}.ok")),
    params:
        checkpoints_path=parse_input(
            input.pyproject, parse_toml, key="tool.anemoi.checkpoints_path"
        ),
        reftimes=lambda wc: [
            t.strftime("%Y-%m-%dT%H:%M") for t in REFTIMES_GROUPS[int(wc.group_index)]
        ],
        lead_time=config["lead_time"],
        output_root=(OUT_ROOT / "data").resolve(),
    # TODO: we can have named logs for each reftime
    log:
        OUT_ROOT / "logs/inference_run/{run_id}-{group_index}.log",
    resources:
        slurm_partition="short",
        cpus_per_task=32,
        mem_mb_per_cpu=8000,
        runtime="20m",
        gres="gpu:1",  # because we use --exclusive, this will be 1 GPU per run (--ntasks-per-gpus is automatically set to 1)
        # see https://github.com/MeteoSwiss/mch-anemoi-evaluation/pull/3#issuecomment-2998997104
        slurm_extra=lambda wc, input: f"--uenv={Path(input.image).resolve()}:/user-environment --exclusive",
    shell:
        """
        export TZ=UTC
        source /user-environment/bin/activate
        export ECCODES_DEFINITION_PATH=/user-environment/share/eccodes-cosmo-resources/definitions
        i=0
        for reftime in {params.reftimes}; do

            # prepare the working directory
            _reftime_str=$(date -d "$reftime" +%Y%m%d%H%M)
            WORKDIR={params.output_root}/runs/{wildcards.run_id}/$_reftime_str
            mkdir -p $WORKDIR && cd $WORKDIR && mkdir -p grib raw
            cp {input.config} config.yaml


            CMD_ARGS=(
                date=$reftime
                checkpoint={params.checkpoints_path}/inference-last.ckpt
                lead_time={params.lead_time}
            )

            CUDA_VISIBLE_DEVICES=$i anemoi-inference run config.yaml "${{CMD_ARGS[@]}}" > inference.log 2>&1 &
            echo "Started inference for reftime $reftime in $WORKDIR"
            echo "CUDA_VISIBLE_DEVICES=$i"
            i=$((i + 1))

        done
        wait
        """


rule map_init_time_to_inference_group:
    localrule: True
    input:
        lambda wc: OUT_ROOT
        / f"data/runs/{wc.run_id}/group-{REFTIME_TO_GROUP[wc.init_time]}.ok",
    output:
        directory(OUT_ROOT / "data/runs/{run_id}/{init_time}/grib"),
        directory(OUT_ROOT / "data/runs/{run_id}/{init_time}/raw"),
