# ----------------------------------------------------- #
# INFERENCE WORKFLOW                                    #
# ----------------------------------------------------- #

import os
from pathlib import Path
from datetime import datetime


rule inference_get_checkpoint:
    output:
        checkpoint=OUT_ROOT / "data/runs/{env_id}/inference-last.ckpt",
        metadata=OUT_ROOT / "data/runs/{env_id}/anemoi.json",
    log:
        OUT_ROOT / "logs/inference_prepare_checkpoint/{env_id}.log",
    localrule: True
    params:
        checkpoint=lambda wc: ENV_CONFIGS[wc.env_id]["checkpoint"],
        checkpoint_type=lambda wc: _checkpoint_uri_type(
            ENV_CONFIGS[wc.env_id]["checkpoint"]
        ),
        checkpoint_is_registry=lambda wc: "/models/"
        in ENV_CONFIGS[wc.env_id]["checkpoint"],
    shell:
        r"""
        (
            mkdir -p $(dirname {output.checkpoint})
            if [ "{params.checkpoint_type}" = "mlflow" ]; then
                if [ "{params.checkpoint_is_registry}" = "True" ]; then
                    python workflow/scripts/inference_get_checkpoint_mlflow.py {params.checkpoint} --output {output.checkpoint}
                    echo "Downloaded checkpoint from MLFlow model registry: {output.checkpoint}"
                else
                    ln -s $(python workflow/scripts/inference_get_checkpoint_mlflow.py {params.checkpoint}) {output.checkpoint}
                    echo "Located checkpoint from MLFlow log."
                    echo "Created symlink: {output.checkpoint} -> $(readlink {output.checkpoint})"
                fi
            elif [ "{params.checkpoint_type}" = "huggingface" ]; then
                repo_id=$(python -c "import re; print(re.search(r'huggingface\.co/([^/]+/[^/]+)', '{params.checkpoint}').group(1))")
                file_path=$(python -c "import re; print(re.search(r'huggingface\.co/[^/]+/[^/]+/(?:blob|resolve)/[^/]+/(.*)', '{params.checkpoint}').group(1))")
                cp $(uvx hf download --quiet $repo_id $file_path) {output.checkpoint}
                echo "Copied checkpoint from HuggingFace: {output.checkpoint}"
            elif [ "{params.checkpoint_type}" = "local" ]; then
                ln -s {params.checkpoint} {output.checkpoint}
                echo "Created symlink: {output.checkpoint} -> $(readlink {output.checkpoint})"
            else
                echo "Unknown checkpoint type: {params.checkpoint_type}"
            fi
            anemoi-utils metadata --dump --json {output.checkpoint} >{output.metadata}
            echo "Extracted metadata from checkpoint: {output.metadata}"
        ) >{log} 2>&1
        """


# Generate a requirements.txt that contains the information needed
# to set up a virtual environment for inference of a specific checkpoint.
# The list of dependencies is taken from the checkpoint's MLFlow run metadata,
# and additional dependencies can be specified under a run entry in the main
# config file.
rule inference_extract_requirements:
    input:
        metadata=OUT_ROOT / "data/runs/{env_id}/anemoi.json",
        script="workflow/scripts/inference_extract_requirements.py",
    output:
        requirements=OUT_ROOT / "data/runs/{env_id}/requirements.txt",
    log:
        OUT_ROOT / "logs/inference_extract_checkpoint_requirements/{env_id}.log",
    localrule: True
    params:
        extra_requirements=lambda wc: ",".join(
            ENV_CONFIGS[wc.env_id].get("extra_requirements", [])
        ),
    shell:
        """
        (
            echo "[$(date)] Starting requirement extraction..."
            python {input.script} {input.metadata} \
                --overrides "{params.extra_requirements}" >{output.requirements}
            echo "[$(date)] Extracted requirements from metadata: {output.requirements}"
            echo $(cat {output.requirements})
        ) >{log} 2>&1
        """


# Prepare the inference environment for a specific checkpoint. The venv is built
# in /dev/shm (RAM) to avoid heavy IO on the parallel filesystem, then squashed
# to a .squashfs image tracked directly as the rule output.
# See https://docs.cscs.ch/guides/storage/#python-virtual-environments-with-uenv.
rule inference_prepare_env:
    input:
        metadata=OUT_ROOT / "data/runs/{env_id}/anemoi.json",
        requirements=OUT_ROOT / "data/runs/{env_id}/requirements.txt",
    output:
        image=OUT_ROOT / "data/runs/{env_id}/venv.squashfs",
    log:
        OUT_ROOT / "logs/inference_prepare_env/{env_id}.log",
    localrule: True
    shell:
        """
        (
            VENV_DIR=$(mktemp -d /dev/shm/evalml_XXXXXXXX)
            trap "rm -rf $VENV_DIR" EXIT
            VENV=$VENV_DIR/.venv

            PYTHON_VERSION=$(cat {input.metadata} | jq -r ".provenance_training.python")
            echo "[$(date)] Creating virtual environment with Python $PYTHON_VERSION in RAM (/dev/shm)..."
            uv venv --managed-python --python $PYTHON_VERSION --relocatable --link-mode=copy $VENV
            source $VENV/bin/activate

            echo "[$(date)] Installing requirements from {input.requirements}..."
            uv pip install -r {input.requirements}

            echo "[$(date)] Compiling Python bytecode..."
            python -m compileall -j 8 -o 0 -o 1 -o 2 $VENV/lib/python*/site-packages
            echo "[$(date)] Testing that eccodes is working..."
            if ! python -c "import eccodes" &>/dev/null; then
                echo "[$(date)] ERROR: eccodes is not installed correctly in the virtual environment."
                exit 1
            fi

            echo "[$(date)] Creating squashfs image from RAM-based venv..."
            mksquashfs $(realpath $VENV) {output.image} -no-recovery -noappend -Xcompression-level 3
            echo "[$(date)] Squashfs image created at {output.image}"
        ) >{log} 2>&1
        """


# Create a zipped directory that, when extracted, can be used as a sandbox
# for running inference jobs for a specific checkpoint. Its main purpose is
# to serve as a development environment for anemoi-inference and to facilitate
# sharing with external collaborators.
rule inference_create_sandbox:
    input:
        script="workflow/scripts/inference_create_sandbox.py",
        checkpoint=lambda wc: OUT_ROOT
        / f"data/runs/{RUN_CONFIGS[wc.run_id]['env_id']}/inference-last.ckpt",
        requirements=lambda wc: OUT_ROOT
        / f"data/runs/{RUN_CONFIGS[wc.run_id]['env_id']}/requirements.txt",
        config=lambda wc: Path(RUN_CONFIGS[wc.run_id]["config"]).resolve(),
        readme_template="resources/inference/sandbox/readme.md.jinja2",
    output:
        sandbox=OUT_ROOT / "data/runs/{run_id}/sandbox.zip",
    log:
        OUT_ROOT / "logs/inference_create_inference_sandbox/{run_id}.log",
    localrule: True
    shell:
        """
        python {input.script} \
            --checkpoint {input.checkpoint} \
            --requirements {input.requirements} \
            --readme-template {input.readme_template} \
            --inference-config {input.config} \
            --output {output.sandbox} \
            >{log} 2>&1
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


rule inference_prepare_forecaster:
    input:
        checkpoint=lambda wc: OUT_ROOT
        / f"data/runs/{RUN_CONFIGS[wc.run_id]['env_id']}/inference-last.ckpt",
        config=lambda wc: Path(RUN_CONFIGS[wc.run_id]["config"]).resolve(),
    output:
        config=Path(OUT_ROOT / "data/runs/{run_id}/{init_time}/config.yaml"),
        resources=directory(OUT_ROOT / "data/runs/{run_id}/{init_time}/resources"),
        grib_out_dir=directory(OUT_ROOT / "data/runs/{run_id}/{init_time}/grib"),
        okfile=OUT_ROOT / "logs/inference_prepare_forecaster/{run_id}-{init_time}.ok",
    log:
        OUT_ROOT / "logs/inference_prepare_forecaster/{run_id}-{init_time}.log",
    localrule: True
    params:
        lead_time=lambda wc: get_leadtime(wc),
        output_root=(OUT_ROOT / "data").resolve(),
        resources_root=Path("resources/inference").resolve(),
        reftime_to_iso=lambda wc: datetime.strptime(
            wc.init_time, "%Y%m%d%H%M"
        ).strftime("%Y-%m-%dT%H:%M"),
    script:
        "../scripts/inference_prepare.py"


def _get_forecaster_run_id(run_id):
    """Get the forecaster run ID from the RUN_CONFIGS."""
    return RUN_CONFIGS[run_id]["forecaster"]["run_id"]


# Prepare temporal downscaler for a specific run ID
rule inference_prepare_temporal_downscaler:
    input:
        checkpoint=lambda wc: OUT_ROOT
        / f"data/runs/{RUN_CONFIGS[wc.run_id]['env_id']}/inference-last.ckpt",
        config=lambda wc: Path(RUN_CONFIGS[wc.run_id]["config"]).resolve(),
        forecasts=lambda wc: (
            [
                OUT_ROOT
                / f"logs/inference_execute/{_get_forecaster_run_id(wc.run_id)}-{wc.init_time}.ok"
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
            OUT_ROOT
            / "logs/inference_prepare_temporal_downscaler/{run_id}-{init_time}.ok"
        ),
    log:
        OUT_ROOT / "logs/inference_prepare_temporal_downscaler/{run_id}-{init_time}.log",
    localrule: True
    params:
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
    script:
        "../scripts/inference_prepare.py"


def _inference_routing_fn(wc):

    run_config = RUN_CONFIGS[wc.run_id]

    if run_config["model_type"] in THIRD_PARTY_MODEL_TYPES:
        # No prepare stage: there's no config.yaml/venv/checkpoint to prepare,
        # the GRIB already exists outside evalml.
        return []
    elif run_config["model_type"] == "forecaster":
        input_path = f"logs/inference_prepare_forecaster/{wc.run_id}-{wc.init_time}.ok"
    elif run_config["model_type"] == "temporal_downscaler":
        input_path = (
            f"logs/inference_prepare_temporal_downscaler/{wc.run_id}-{wc.init_time}.ok"
        )
    else:
        raise ValueError(f"Unsupported model type: {run_config['model_type']}")

    return OUT_ROOT / input_path


def _passthrough_grib_source(wc):
    """Pre-generated GRIB dir for a third-party run (e.g. spatial_downscaler): evalml
    never runs inference for these, so their GRIB always comes from their own `root`,
    regardless of whether FIXTURE_ROOT replay is active. None for run types that need
    real inference (or fixture replay)."""
    rc = RUN_CONFIGS[wc.run_id]
    if rc["model_type"] not in THIRD_PARTY_MODEL_TYPES:
        return None
    return str((Path(rc["root"]) / wc.init_time).resolve())


def _execute_resource(wc, field, default, passthrough_default=None):
    """Resolve a resource value, short-circuiting for third-party (passthrough) runs
    so they never touch RUN_CONFIGS[...]["inference_resources"], which they don't have."""
    if RUN_CONFIGS[wc.run_id]["model_type"] in THIRD_PARTY_MODEL_TYPES:
        return default if passthrough_default is None else passthrough_default
    return get_resource(wc, field, default)


if FIXTURE_ROOT:

    from snakemake.exceptions import WorkflowError

    from evalml.fixtures import fixture_grib_dir, verify_fixture

    _verified_fixtures = set()

    def _staged_grib_source(wc):
        """GRIB to symlink into the run's workdir: a third-party run's own `root`,
        or (otherwise) the frozen fixture for FIXTURE_ROOT replay."""
        passthrough = _passthrough_grib_source(wc)
        p = Path(passthrough) if passthrough else fixture_grib_dir(
            FIXTURE_ROOT, wc.run_id, wc.init_time
        )
        if not p.exists():
            if passthrough:
                raise WorkflowError(
                    f"Spatial-downscaler GRIB not found at {p}. This run reads "
                    f"pre-generated GRIB from its own `root`, produced outside evalml."
                )
            raise WorkflowError(
                f"Fixture GRIB not found at {p}. Capture it once from a real run "
                f"with: evalml capture-fixture <config> {FIXTURE_ROOT}"
            )
        if passthrough:
            # No MANIFEST.yaml/checksum story for externally-owned data.
            return str(p)
        # Validate against the manifest checksum once per dir (input functions
        # can be called repeatedly during DAG building; hashing GBs each time
        # would be wasteful). No-op for fixtures without recorded checksums.
        if str(p) not in _verified_fixtures:
            try:
                verify_fixture(FIXTURE_ROOT, p)
            except ValueError as exc:
                raise WorkflowError(str(exc))
            _verified_fixtures.add(str(p))
        return str(p)

    rule inference_execute:
        input:
            grib=_staged_grib_source,
        output:
            okfile=OUT_ROOT / "logs/inference_execute/{run_id}-{init_time}.ok",
        log:
            OUT_ROOT / "logs/inference_execute/{run_id}-{init_time}.log",
        localrule: True
        params:
            workdir=lambda wc: (
                OUT_ROOT / f"data/runs/{wc.run_id}/{wc.init_time}"
            ).resolve(),
        shell:
            """
            (
                set -euo pipefail
                mkdir -p {params.workdir}
                # Replace a stale fixture symlink, but never delete a real grib
                # directory: that is Snakemake-owned inference output, and a
                # bare `ln -sfn` would otherwise nest the link inside it.
                if [ -L {params.workdir}/grib ]; then
                    rm -f {params.workdir}/grib
                elif [ -e {params.workdir}/grib ]; then
                    echo "ERROR: {params.workdir}/grib is a real directory (Snakemake-owned inference output), not a fixture symlink. Refusing to delete it; move it aside and retry." >&2
                    exit 1
                fi
                ln -sfn {input.grib} {params.workdir}/grib
            ) >{log} 2>&1
            touch {output.okfile}
            """

else:

    rule inference_execute:
        input:
            okfile=_inference_routing_fn,
            image=lambda wc: (
                []
                if _passthrough_grib_source(wc)
                else OUT_ROOT
                / f"data/runs/{RUN_CONFIGS[wc.run_id]['env_id']}/venv.squashfs"
            ),
        output:
            okfile=OUT_ROOT / "logs/inference_execute/{run_id}-{init_time}.ok",
        log:
            OUT_ROOT / "logs/inference_execute/{run_id}-{init_time}.log",
        localrule: True
        resources:
            slurm_partition=lambda wc: _execute_resource(
                wc, "slurm_partition", "short-shared"
            ),
            cpus_per_task=lambda wc: _execute_resource(
                wc, "cpus_per_task", 24, passthrough_default=1
            ),
            mem_mb_per_cpu=lambda wc: _execute_resource(
                wc, "mem_mb_per_cpu", 8000, passthrough_default=500
            ),
            runtime=lambda wc: _execute_resource(
                wc, "runtime", "40m", passthrough_default="5m"
            ),
            gres=lambda wc: f"gpu:{_execute_resource(wc, 'gpu', 1, passthrough_default=0)}",
            ntasks=lambda wc: _execute_resource(wc, "tasks", 1),
            gpus=lambda wc: _execute_resource(wc, "gpu", 1, passthrough_default=0),
        params:
            passthrough_source=lambda wc: _passthrough_grib_source(wc) or "",
            env_path=lambda wc, input: (
                "" if not input.image else f"{Path(input.image).resolve()}"
            ),
            workdir=lambda wc: (
                OUT_ROOT / f"data/runs/{wc.run_id}/{wc.init_time}"
            ).resolve(),
            disable_local_definitions=lambda wc: RUN_CONFIGS[wc.run_id].get(
                "disable_local_eccodes_definitions", False
            ),
        # fmt: off
        shell:
            """
            (
                set -euo pipefail

                mkdir -p {params.workdir}

                if [ -n "{params.passthrough_source}" ]; then
                    # Third-party run (e.g. spatial_downscaler): GRIB already exists
                    # outside evalml, just stage it (mirrors the FIXTURE_ROOT replay
                    # rule's own symlink guard above).
                    if [ -L {params.workdir}/grib ]; then
                        rm -f {params.workdir}/grib
                    elif [ -e {params.workdir}/grib ]; then
                        echo "ERROR: {params.workdir}/grib is a real directory (Snakemake-owned inference output), not a passthrough symlink. Refusing to delete it; move it aside and retry." >&2
                        exit 1
                    fi
                    ln -sfn {params.passthrough_source} {params.workdir}/grib
                else
                    cd {params.workdir}

                    _run_inference() {{
                        local VENV=$1
                        source "$VENV/bin/activate"

                        if [ "{params.disable_local_definitions}" = "False" ]; then
                            export ECCODES_DEFINITION_PATH="$VENV/share/eccodes-cosmo-resources/definitions"
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
                    }}
                    export -f _run_inference

                    squashfs-mount {params.env_path}:/user-environment -- bash -c '_run_inference /user-environment'
                fi
            ) >{log} 2>&1
            touch {output.okfile}
            """
        # fmt: on
