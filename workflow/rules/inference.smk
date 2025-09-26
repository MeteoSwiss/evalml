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
        slurm_partition="short-shared",
        runtime="40m",
        gres="gpu:4",
        slurm_extra=lambda wc, input: f"--uenv={Path(input.image).resolve()}:/user-environment",
        gpus=1,
    shell:
        """
        (
        set -euo pipefail
        echo "Raw SLURM vars: SLURM_JOB_GPUS='${{SLURM_JOB_GPUS:-}}' SLURM_NTASKS='${{SLURM_NTASKS:-}}' SLURM_PROCID='${{SLURM_PROCID:-}}'"

        # Skip if already finished earlier
        if [ -f "{output.okfile}" ]; then
            echo "OK file exists -> skipping."
            exit 0
        fi

        DEBUG=${{DEBUG:-0}}
        [ "$DEBUG" = "1" ] && set -x || true

        # If scheduler spawns multiple tasks, run non-zero ranks later otherwise CUDA unavailable device error as many torchrun instances start simultaneously
        # TODO maybe optimize this or use srun --ntasks=1 --exclusive ...?
        
        PROCID=${{SLURM_PROCID:-0}}
        if [ "$PROCID" != "0" ]; then
            delay=$((60 * PROCID))
            echo "Non-zero SLURM_PROCID ($PROCID) -> delaying start by ${{delay}}s"
            sleep "$delay"
        fi
        if [ -n "${{SLURM_JOB_GPUS:-}}" ]; then
            export CUDA_VISIBLE_DEVICES="${{SLURM_JOB_GPUS}}"
        fi

        export TZ=UTC
        source /user-environment/bin/activate
        export ECCODES_DEFINITION_PATH=/user-environment/share/eccodes-cosmo-resources/definitions
        export OMP_NUM_THREADS=1
        export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
        export NCCL_IB_DISABLE=1
        export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
        export NCCL_DEBUG=WARN
        VISIBLE=${{CUDA_VISIBLE_DEVICES:-}}

        # Recompute number of GPUs
        NGPUS=$(python - <<'PY'
import os
cvd=os.environ.get("CUDA_VISIBLE_DEVICES","").strip()
print(len([x for x in cvd.split(",") if x!=""]))
PY
        )

        echo "================= ENV ================="
        echo "SLURM JOB ID: ${{SLURM_JOB_ID:-unknown}}"
        echo "HOSTNAME: $(hostname)"
        echo "CUDA_VISIBLE_DEVICES: $VISIBLE"
        echo "NGPUS=$NGPUS"
        echo "=========================================================="

        WORKDIR={params.output_root}/runs/{wildcards.run_id}/{wildcards.init_time}
        mkdir -p "$WORKDIR/grib" "$WORKDIR/raw" "$WORKDIR/_resources"

        # Serialize access to WORKDIR (others wait and then re-check OK file)
        exec 9>"$WORKDIR/.lock"
        if ! flock -n 9; then
            echo "Another task active in $WORKDIR; waiting for lock..."
            flock 9
        fi
        if [ -f "{output.okfile}" ]; then
            echo "OK file exists after waiting -> skipping."
            exit 0
        fi

        cd "$WORKDIR"
        cp {input.config} config.yaml
        cp -r {params.resources_root}/templates/* _resources/

        CKPT={params.checkpoints_path}/inference-last.ckpt
        CMD_ARGS=( date={params.reftime_to_iso} checkpoint="$CKPT" lead_time={params.lead_time} )
        echo "Launching torchrun with args: ${{CMD_ARGS[*]}} (nproc_per_node=$NGPUS)"
        torchrun --standalone --nnodes=1 --nproc_per_node="$NGPUS" \
                 --rdzv-backend=c10d --rdzv-endpoint="localhost:0" \
                 /user-environment/bin/anemoi-inference run config.yaml "${{CMD_ARGS[@]}}"
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
        slurm_partition="short-shared",
        cpus_per_task=24,
        mem_mb_per_cpu=8000,
        runtime="30m",
        gres="gpu:1",
        slurm_extra=lambda wc, input: f"--uenv={Path(input.image).resolve()}:/user-environment",
        gpus=1,
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
