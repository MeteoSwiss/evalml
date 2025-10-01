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
    params:
        extra_dependencies=lambda wc: RUN_CONFIGS[wc.run_id].get(
            "extra_dependencies", []
        ),
        mlflow_id=lambda wc: RUN_CONFIGS[wc.run_id].get("mlflow_id"),
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
        config=lambda wc: Path(RUN_CONFIGS[wc.run_id]["config"]).resolve(),
        readme_template="resources/inference/sandbox/readme.md.jinja2",
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
            --readme-template {input.readme_template} \
            --output {output.sandbox} \
            > {log} 2>&1
        """


def get_resource(wc, field: str, default):
    """Fetch a resource field from the run config, or return the default."""
    rc = RUN_CONFIGS[wc.run_id]
    if rc["inference_resources"] is None:
        return default
    return getattr(rc["inference_resources"], field) or default


rule inference_forecaster:
    input:
        pyproject=rules.create_inference_pyproject.output.pyproject,
        image=rules.make_squashfs_image.output.image,
        config=lambda wc: Path(RUN_CONFIGS[wc.run_id]["config"]).resolve(),
    output:
        okfile=touch(OUT_ROOT / "logs/inference_forecaster/{run_id}-{init_time}.ok"),
    params:
        checkpoints_path=parse_input(
            input.pyproject, parse_toml, key="tool.anemoi.checkpoints_path"
        ),
        lead_time=config["lead_time"],
        output_root=(OUT_ROOT / "data").resolve(),
        resources_root=Path("resources/inference").resolve(),
        reftime_to_iso=lambda wc: datetime.strptime(
            wc.init_time, "%Y%m%d%H%M"
        ).strftime("%Y-%m-%dT%H:%M"),
    log:
        OUT_ROOT / "logs/inference_forecaster/{run_id}-{init_time}.log",
    resources:
        slurm_partition=lambda wc: get_resource(wc, "slurm_partition", "short-shared"),
        cpus_per_task=lambda wc: get_resource(wc, "cpus_per_task", 24),
        mem_mb_per_cpu=lambda wc: get_resource(wc, "mem_mb_per_cpu", 8000),
        runtime=lambda wc: get_resource(wc, "runtime", "20m"),
        gres=lambda wc: f"gpu:{get_resource(wc, 'gpu', 1)}",
        ntasks=lambda wc: get_resource(wc, "tasks", 1),
        slurm_extra=lambda wc, input: f"--uenv={Path(input.image).resolve()}:/user-environment",
        gpus=lambda wc: get_resource(wc, "gpu", 1),
    shell:
        """
        (
        export TZ=UTC
        source /user-environment/bin/activate
        export ECCODES_DEFINITION_PATH=/user-environment/share/eccodes-cosmo-resources/definitions

        # prepare the working directory
        WORKDIR={params.output_root}/runs/{wildcards.run_id}/{wildcards.init_time}
        mkdir -p $WORKDIR && cd $WORKDIR && mkdir -p grib raw _resources
        cp {input.config} config.yaml && cp -r {params.resources_root}/templates/* _resources/

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


def _get_forecaster_run_id(run_id):
    """Get the forecaster run ID from the RUN_CONFIGS."""
    return RUN_CONFIGS[run_id]["forecaster"]["mlflow_id"][0:9]


rule inference_interpolator:
    """Run the interpolator for a specific run ID."""
    input:
        pyproject=rules.create_inference_pyproject.output.pyproject,
        image=rules.make_squashfs_image.output.image,
        config=lambda wc: Path(RUN_CONFIGS[wc.run_id]["config"]).resolve(),
        forecasts=lambda wc: (
            [
                OUT_ROOT
                / f"logs/inference_forecaster/{_get_forecaster_run_id(wc.run_id)}-{wc.init_time}.ok"
            ]
            if RUN_CONFIGS[wc.run_id].get("forecaster") is not None
            else []
        ),
    output:
        okfile=touch(OUT_ROOT / "logs/inference_interpolator/{run_id}-{init_time}.ok"),
    params:
        checkpoints_path=parse_input(
            input.pyproject, parse_toml, key="tool.anemoi.checkpoints_path"
        ),
        lead_time=config["lead_time"],
        output_root=(OUT_ROOT / "data").resolve(),
        resources_root=Path("resources/inference").resolve(),
        reftime_to_iso=lambda wc: datetime.strptime(
            wc.init_time, "%Y%m%d%H%M"
        ).strftime("%Y-%m-%dT%H:%M"),
        forecaster_run_id=lambda wc: (
            "null"
            if RUN_CONFIGS[wc.run_id].get("forecaster") is None
            else _get_forecaster_run_id(wc.run_id)
        ),
    log:
        OUT_ROOT / "logs/inference_interpolator/{run_id}-{init_time}.log",
    resources:
        slurm_partition=lambda wc: get_resource(wc, "slurm_partition", "short-shared"),
        cpus_per_task=lambda wc: get_resource(wc, "cpus_per_task", 24),
        mem_mb_per_cpu=lambda wc: get_resource(wc, "mem_mb_per_cpu", 8000),
        runtime=lambda wc: get_resource(wc, "runtime", "20m"),
        gres=lambda wc: f"gpu:{get_resource(wc, 'gpu',1)}",
        slurm_extra=lambda wc, input: f"--uenv={Path(input.image).resolve()}:/user-environment",
        gpus=lambda wc: get_resource(wc, "gpu", 1),
    shell:
        """
        (
        set -euo pipefail
        export TZ=UTC
        source /user-environment/bin/activate
        export ECCODES_DEFINITION_PATH=/user-environment/share/eccodes-cosmo-resources/definitions

        # prepare the working directory
        WORKDIR={params.output_root}/runs/{wildcards.run_id}/{wildcards.init_time}
        mkdir -p $WORKDIR && cd $WORKDIR && mkdir -p grib raw _resources
        cp {input.config} config.yaml && cp -r {params.resources_root}/templates/* _resources/

        # if forecaster_run_id is not "null", link the forecaster grib directory; else, run from files.
        if [ "{params.forecaster_run_id}" != "null" ]; then
            FORECASTER_WORKDIR={params.output_root}/runs/{params.forecaster_run_id}/{wildcards.init_time}
            ln -fns $FORECASTER_WORKDIR/grib forecaster_grib
        else
            echo "Forecaster configuration is null; proceeding with file-based inputs."
        fi

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


rule inference_routing:
    localrule: True
    input:
        _inference_routing_fn,
    output:
        directory(OUT_ROOT / "data/runs/{run_id}/{init_time}/grib"),
        directory(OUT_ROOT / "data/runs/{run_id}/{init_time}/raw"),
