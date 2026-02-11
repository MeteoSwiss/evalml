# ----------------------------------------------------- #
# INFERENCE WORKFLOW                                    #
# ----------------------------------------------------- #

import os
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse


def _checkpoint_uri_type(checkpoint_uri: str):
    parsed_url = urlparse(checkpoint_uri)
    if parsed_url.netloc in [
        "mlflow.ecmwf.int",
        "service.meteoswiss.ch",
        "servicedepl.meteoswiss.ch",
    ]:
        return "mlflow"
    elif parsed_url.netloc == "huggingface.co":
        if not parsed_url.path.endswith(".ckpt"):
            raise ValueError(
                f"Expected a .ckpt file for HuggingFace checkpoint URI. Got: {checkpoint_uri}"
            )
        return "huggingface"
    elif parsed_url.netloc == "":
        return "local"
    else:
        raise ValueError(f"Unknown checkpoint URI type: {checkpoint_uri}")


rule prepare_checkpoint:
    localrule: True
    output:
        checkpoint=OUT_ROOT / "data/runs/{run_id}/inference-last.ckpt",
        metadata=OUT_ROOT / "data/runs/{run_id}/anemoi.json",
    params:
        checkpoint=lambda wc: RUN_CONFIGS[wc.run_id]["checkpoint"],
        checkpoint_type=lambda wc: _checkpoint_uri_type(
            RUN_CONFIGS[wc.run_id]["checkpoint"]
        ),
    log:
        OUT_ROOT / "logs/prepare_checkpoint/{run_id}.log",
    shell:
        """(
        mkdir -p $(dirname {output.checkpoint})
        if [ "{params.checkpoint_type}" = "mlflow" ]; then
            ln -s $(python workflow/scripts/inference_get_checkpoint_mlflow.py {params.checkpoint}) {output.checkpoint}
            echo "Located checkpoint from MLFlow log."
            echo "Created symlink: {output.checkpoint} -> $(readlink {output.checkpoint})"
        elif [ "{params.checkpoint_type}" = "huggingface" ]; then
            repo_id=$(python -c "import re; print(re.search(r'huggingface\.co/([^/]+/[^/]+)', '{params.checkpoint}').group(1))")
            file_path=$(python -c "import re; print(re.search(r'huggingface\.co/[^/]+/[^/]+/blob/[^/]+/(.*)', '{params.checkpoint}').group(1))")
            cp $(uvx hf download $repo_id $file_path) {output.checkpoint}
            echo "Copied checkpoint from HuggingFace: {output.checkpoint}"
        elif [ "{params.checkpoint_type}" = "local" ]; then
            ln -s {params.checkpoint} {output.checkpoint}
            echo "Created symlink: {output.checkpoint} -> $(readlink {output.checkpoint})"
        else
            echo "Unknown checkpoint type: {params.checkpoint_type}"
        fi
        anemoi-utils metadata --dump --json {output.checkpoint} > {output.metadata}
        echo "Extracted metadata from checkpoint: {output.metadata}"
        ) > {log} 2>&1
        """


rule create_inference_pyproject:
    """
    Generate a pyproject.toml that contains the information needed
    to set up a virtual environment for inference of a specific checkpoint.
    The list of dependencies is taken from the checkpoint's MLFlow run metadata,
    and additional dependencies can be specified under a run entry in the main
    config file.
    """
    input:
        toml="workflow/envs/anemoi_inference.toml",
        metadata=OUT_ROOT / "data/runs/{run_id}/anemoi.json",
        summary=rules.write_summary.output,
    output:
        requirements=OUT_ROOT / "data/runs/{run_id}/requirements.txt",
        pyproject=OUT_ROOT / "data/runs/{run_id}/pyproject.toml",
    params:
        extra_dependencies=lambda wc: RUN_CONFIGS[wc.run_id].get(
            "extra_dependencies", []
        ),
        mlflow_id=lambda wc: RUN_CONFIGS[wc.run_id].get("mlflow_id"),
    log:
        OUT_ROOT / "logs/create_inference_pyproject/{run_id}.log",
    localrule: True
    shell:
        """(
        python workflow/scripts/inference_get_requirements.py {input.metadata} > {output.requirements}
        echo "Extracted requirements from metadata: {output.requirements}"
        cp {input.toml} {output.pyproject}
        echo "Copied base pyproject.toml to: {output.pyproject}"
        uv --project {output.pyproject} add --no-sync --requirements {output.requirements}
        echo "Added requirements to pyproject.toml: {output.pyproject}"
        ) > {log} 2>&1
        """


rule create_inference_venv:
    """
    Create a virtual environment for inference, using the pyproject.toml created above.
    The virtual environment is managed with uv. The created virtual environment is relocatable,
    so it can be squashed later. Pre-compilation to bytecode is done to speed up imports.
    """
    input:
        pyproject=OUT_ROOT / "data/runs/{run_id}/pyproject.toml",
        metadata=OUT_ROOT / "data/runs/{run_id}/anemoi.json",
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
        PYTHON_VERSION=$(cat {input.metadata} | jq -r ".provenance_training.python")
        PROJECT_ROOT=$(dirname {input.pyproject})
        uv venv --project $PROJECT_ROOT --managed-python --python $PYTHON_VERSION \
            --relocatable --link-mode=copy {output.venv}
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


rule make_squashfs_image:
    """
    Create a squashfs image for the inference virtual environment of
    a specific checkpoint. Find more about this at
    https://docs.cscs.ch/guides/storage/#python-virtual-environments-with-uenv.
    """
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
    """
    Create a zipped directory that, when extracted, can be used as a sandbox
    for running inference jobs for a specific checkpoint. Its main purpose is
    to serve as a development environment for anemoi-inference and to facilitate
    sharing with external collaborators.

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
    if isinstance(rc["inference_resources"], dict):
        return rc["inference_resources"].get(field, default) or default
    else:
        return getattr(rc["inference_resources"], field) or default


def get_leadtime(wc):
    """Get the lead time from the run config."""
    start, end, step = RUN_CONFIGS[wc.run_id]["steps"].split("/")
    return f"{end}h"


rule prepare_inference_forecaster:
    localrule: True
    input:
        pyproject=rules.create_inference_pyproject.output.pyproject,
        config=lambda wc: Path(RUN_CONFIGS[wc.run_id]["config"]).resolve(),
    output:
        config=Path(OUT_ROOT / "data/runs/{run_id}/{init_time}/config.yaml"),
        resources=directory(OUT_ROOT / "data/runs/{run_id}/{init_time}/resources"),
        grib_out_dir=directory(OUT_ROOT / "data/runs/{run_id}/{init_time}/grib"),
        okfile=touch(
            OUT_ROOT / "logs/prepare_inference_forecaster/{run_id}-{init_time}.ok"
        ),
    params:
        checkpoints_path="cio",
        lead_time=lambda wc: get_leadtime(wc),
        output_root=(OUT_ROOT / "data").resolve(),
        resources_root=Path("resources/inference").resolve(),
        reftime_to_iso=lambda wc: datetime.strptime(
            wc.init_time, "%Y%m%d%H%M"
        ).strftime("%Y-%m-%dT%H:%M"),
    log:
        OUT_ROOT / "logs/prepare_inference_forecaster/{run_id}-{init_time}.log",
    script:
        "../scripts/inference_prepare.py"


def _get_forecaster_run_id(run_id):
    """Get the forecaster run ID from the RUN_CONFIGS."""
    return RUN_CONFIGS[run_id]["forecaster"]["run_id"]


rule prepare_inference_interpolator:
    """Run the interpolator for a specific run ID."""
    localrule: True
    input:
        pyproject=rules.create_inference_pyproject.output.pyproject,
        config=lambda wc: Path(RUN_CONFIGS[wc.run_id]["config"]).resolve(),
        forecasts=lambda wc: (
            [
                OUT_ROOT
                / f"logs/execute_inference/{_get_forecaster_run_id(wc.run_id)}-{wc.init_time}.ok"
            ]
            if RUN_CONFIGS[wc.run_id].get("forecaster") is not None
            else []
        ),
    output:
        config=Path(OUT_ROOT / "data/runs/{run_id}/{init_time}/config.yaml"),
        resources=directory(OUT_ROOT / "data/runs/{run_id}/{init_time}/resources"),
        grib_out_dir=directory(OUT_ROOT / "data/runs/{run_id}/{init_time}/grib"),
        forecaster=directory(OUT_ROOT / "data/runs/{run_id}/{init_time}/forecaster"),
        okfile=touch(
            OUT_ROOT / "logs/prepare_inference_interpolator/{run_id}-{init_time}.ok"
        ),
    params:
        checkpoints_path=parse_input(
            input.pyproject, parse_toml, key="tool.anemoi.checkpoints_path"
        ),
        lead_time=lambda wc: get_leadtime(wc),
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
        OUT_ROOT / "logs/prepare_inference_interpolator/{run_id}-{init_time}.log",
    script:
        "../scripts/inference_prepare.py"


def _inference_routing_fn(wc):

    run_config = RUN_CONFIGS[wc.run_id]

    if run_config["model_type"] == "forecaster":
        input_path = f"logs/prepare_inference_forecaster/{wc.run_id}-{wc.init_time}.ok"
    elif run_config["model_type"] == "interpolator":
        input_path = (
            f"logs/prepare_inference_interpolator/{wc.run_id}-{wc.init_time}.ok"
        )
    else:
        raise ValueError(f"Unsupported model type: {run_config['model_type']}")

    return OUT_ROOT / input_path


rule execute_inference:
    localrule: True
    input:
        okfile=_inference_routing_fn,
        image=rules.make_squashfs_image.output.image,
    output:
        okfile=touch(OUT_ROOT / "logs/execute_inference/{run_id}-{init_time}.ok"),
    log:
        OUT_ROOT / "logs/execute_inference/{run_id}-{init_time}.log",
    params:
        image_path=lambda wc, input: f"{Path(input.image).resolve()}",
        workdir=lambda wc: (
            OUT_ROOT / f"data/runs/{wc.run_id}/{wc.init_time}"
        ).resolve(),
        disable_local_definitions=lambda wc: RUN_CONFIGS[wc.run_id].get(
            "disable_local_eccodes_definitions", False
        ),
    resources:
        slurm_partition=lambda wc: get_resource(wc, "slurm_partition", "short-shared"),
        cpus_per_task=lambda wc: get_resource(wc, "cpus_per_task", 24),
        mem_mb_per_cpu=lambda wc: get_resource(wc, "mem_mb_per_cpu", 8000),
        runtime=lambda wc: get_resource(wc, "runtime", "40m"),
        gres=lambda wc: f"gpu:{get_resource(wc, 'gpu',1)}",
        ntasks=lambda wc: get_resource(wc, "tasks", 1),
        gpus=lambda wc: get_resource(wc, "gpu", 1),
    shell:
        """
        (
        set -euo pipefail

        cd {params.workdir}

        squashfs-mount {params.image_path}:/user-environment -- bash -c '
        source /user-environment/bin/activate

        if [ "{params.disable_local_definitions}" = "False" ]; then
            export ECCODES_DEFINITION_PATH=/user-environment/share/eccodes-cosmo-resources/definitions
        fi

        CMD_ARGS=()

        # is GPU > 1, add parallel flag to CMD_ARGS and override automatic cluster detection
        if [ {resources.gpus} -gt 1 ]; then
            CMD_ARGS+=(runner.parallel.cluster=slurm)
        fi

        srun \
            --unbuffered \
            --partition={resources.slurm_partition} \
            --cpus-per-task={resources.cpus_per_task} \
            --mem-per-cpu={resources.mem_mb_per_cpu} \
            --time={resources.runtime} \
            --gres={resources.gres} \
            --ntasks={resources.ntasks} \
            anemoi-inference run config.yaml "${{CMD_ARGS[@]}}"
        '
        ) > {log} 2>&1
        """
