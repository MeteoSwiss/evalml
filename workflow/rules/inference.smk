# ----------------------------------------------------- #
# INFERENCE WORKFLOW                                    #
# ----------------------------------------------------- #

import os
from pathlib import Path
from datetime import datetime


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
    """
    Create a virtual environment for inference, using the pyproject.toml created above.
    The virtual environment is managed with uv. The created virtual environment is relocatable,
    so it can be squashed later. Pre-compilation to bytecode is done to speed up imports.
    """
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
        OUT_ROOT / "logs/prepare_inference_forecaster/{run_id}-{init_time}.log",
    run:
        LOG = setup_logger("prepare_inference_forecaster", log_file=log[0])
        try:
            import yaml
            import shutil

            L(
                "Preparing inference forecaster for run_id=%s, init_time=%s",
                wildcards.run_id,
                wildcards.init_time,
            )

            # prepare working directory
            workdir = (
                Path(params.output_root)
                / "runs"
                / wildcards.run_id
                / wildcards.init_time
            )
            workdir.mkdir(parents=True, exist_ok=True)
            LOG.info("Created working directory at %s", workdir)
            (workdir / "grib").mkdir(parents=True, exist_ok=True)
            LOG.info("Created GRIB output directory at %s", workdir / "grib")
            shutil.copytree(params.resources_root / "templates", output.resources)
            LOG.info("Copied resources to %s", output.resources)
            LOG.info("Resources: \n%s", list(Path(output.resources).rglob("*")))

            # prepare and write config file
            with open(input.config, "r") as f:
                config = yaml.safe_load(f)
            config["checkpoint"] = f"{params.checkpoints_path}/inference-last.ckpt"
            config["date"] = params.reftime_to_iso
            config["lead_time"] = params.lead_time
            with open(output.config, "w") as f:
                yaml.safe_dump(config, f)
            LOG.info("Config: \n%s", config)
            LOG.info("Wrote config file at %s", output.config)
        except Exception as e:
            LOG.error("An error occurred: %s", str(e))
            raise e


def _get_forecaster_run_id(run_id):
    """Get the forecaster run ID from the RUN_CONFIGS."""
    return RUN_CONFIGS[run_id]["forecaster"]["mlflow_id"][0:9]


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
        OUT_ROOT / "logs/prepare_inference_interpolator/{run_id}-{init_time}.log",
    run:
        LOG = setup_logger(
            f"prepare_inference_interpolator_{OUT_ROOT.stem}", log_file=log[0]
        )
        try:
            import yaml
            import shutil

            fct_run_id = params.forecaster_run_id
            init_time = wildcards.init_time

            LOG.info(
                "Preparing inference interpolator for run_id=%s, init_time=%s, forecaster_run_id=%s",
                wildcards.run_id,
                wildcards.init_time,
                fct_run_id,
            )
            # prepare working directory
            workdir = (
                Path(params.output_root)
                / "runs"
                / wildcards.run_id
                / wildcards.init_time
            )
            workdir.mkdir(parents=True, exist_ok=True)
            LOG.info("Created working directory at %s", workdir)
            (workdir / "grib").mkdir(parents=True, exist_ok=True)
            LOG.info("Created GRIB output directory at %s", workdir / "grib")
            shutil.copytree(params.resources_root / "templates", output.resources)
            LOG.info("Copied resources to %s", output.resources)
            LOG.info("Resources: \n%s", list(Path(output.resources).rglob("*")))

            # if forecaster_run_id is not "null", create symbolic link to forecaster grib directory
            if fct_run_id != "null":
                forecaster_workdir = (
                    Path(params.output_root) / "runs" / fct_run_id / init_time
                )
                (workdir / "forecaster").symlink_to(forecaster_workdir / "grib")
                LOG.info(
                    "Created symlink to forecaster GRIB directory at %s",
                    workdir / "forecaster",
                )
            else:
                (workdir / "forecaster").mkdir(parents=True, exist_ok=True)
                (workdir / "forecaster/.dataset").touch()
                LOG.info("No forecaster run ID provided, skipping symlink creation.")

                # prepare and write config file
            with open(input.config, "r") as f:
                config = yaml.safe_load(f)
            config["checkpoint"] = f"{params.checkpoints_path}/inference-last.ckpt"
            config["date"] = params.reftime_to_iso
            config["lead_time"] = params.lead_time
            with open(output.config, "w") as f:
                yaml.safe_dump(config, f)
            LOG.info("Config: \n%s", config)
            LOG.info("Wrote config file at %s", output.config)
        except Exception as e:
            LOG.error("An error occurred: %s", str(e))
            raise e


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
        export ECCODES_DEFINITION_PATH=/user-environment/share/eccodes-cosmo-resources/definitions

        CMD_ARGS=()

        # is GPU > 1, add runner=parallel to CMD_ARGS
        if [ {resources.gpus} -gt 1 ]; then
            CMD_ARGS+=(runner=parallel)
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
