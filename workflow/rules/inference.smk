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
    params:
        fdb_uenv=lambda wc: _get_fdb_uenv_for_env(wc.env_id)[0],
        fdb_view=lambda wc: _get_fdb_uenv_for_env(wc.env_id)[1],
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

            # Bundle FDB native libraries into the venv if this env needs FDB output.
            # The FDB squashfs is mounted at /user-environment (its designed path) so that
            # the absolute symlinks inside env/._<view>/<hash>/ resolve correctly.
            # The linux-zen3/<pkg>/lib64/ actual .so files are copied preserving their path
            # structure, because libfdb5.so's RPATH is hardcoded to those absolute paths.
            # The env/ symlink tree is copied as-is (preserved); after the venv squashfs
            # is mounted at /user-environment at runtime, the full symlink chain resolves.
            if [ -n "{params.fdb_uenv}" ]; then
                echo "[$(date)] Bundling FDB native libraries from uenv '{params.fdb_uenv}' (view: {params.fdb_view})..."
                FDB_SHA=$(uenv image inspect {params.fdb_uenv} | awk '/^sha:/ {{print $2}}')
                FDB_REPO=$(uenv repo status | grep -oP '(?<=the repository at )\S+')
                FDB_SQUASHFS="$FDB_REPO/images/$FDB_SHA/store.squashfs"
                VENV_PATH=$VENV
                FDB_VIEW={params.fdb_view}

                squashfs-mount "$FDB_SQUASHFS:/user-environment" -- bash -c '
                    set -euo pipefail
                    VENV='"$VENV_PATH"'
                    VIEW='"$FDB_VIEW"'

                    echo "[$(date)] Copying env/ view structure (symlinks preserved)..."
                    cp -rp /user-environment/env "$VENV/"

                    echo "[$(date)] Copying linux-zen3 shared libraries (.so files, maintaining path structure)..."
                    find /user-environment/linux-zen3 -name "*.so*" -not -type l | while IFS= read -r f; do
                        rel="${{f#/user-environment/}}"
                        dest_dir="$VENV/${{rel%/*}}"
                        mkdir -p "$dest_dir"
                        cp -p "$f" "$dest_dir/"
                    done
                    echo "[$(date)] Copied $(find "$VENV/linux-zen3" -name "*.so*" 2>/dev/null | wc -l) shared libraries"

                    echo "[$(date)] Copying FDB config and schema files..."
                    mkdir -p "$VENV/meta/recipe/meta/private/fdb_config"
                    cp "/user-environment/meta/recipe/meta/private/fdb_config/${{VIEW}}.yaml" \
                       "$VENV/meta/recipe/meta/private/fdb_config/"
                    cp "/user-environment/meta/recipe/meta/private/fdb_config/${{VIEW}}.schema" \
                       "$VENV/meta/recipe/meta/private/fdb_config/" 2>/dev/null || true
                    echo "[$(date)] FDB config files copied for view: $VIEW"

                    echo "[$(date)] Cloning eccodes-cosmo-mars (varda-ext branch) for MARS definitions..."
                    mkdir -p "$VENV/data/share"
                    git clone --branch varda-ext --depth 1 \
                        git@github.com:meteoswiss/eccodes-cosmo-mars.git \
                        "$VENV/data/share/eccodes-cosmo-mars"
                    echo "[$(date)] Cloned eccodes-cosmo-mars definitions"
                '

                echo "[$(date)] Installing pyfdb (pure-Python FDB binding)..."
                uv pip install pyfdb
            fi

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
    """Fetch a resource field from profile.fdb, or return the default."""
    fdb = (config.get("profile") or {}).get("fdb") or {}
    if isinstance(fdb, dict):
        return fdb.get(field, default) or default
    else:
        return getattr(fdb, field, default) or default


def _get_fdb_uenv_for_env(env_id):
    """Return (fdb_uenv, fdb_view) from profile.fdb."""
    fdb = (config.get("profile") or {}).get("fdb") or {}
    if isinstance(fdb, dict):
        uenv = fdb.get("srun_uenv", "") or ""
        view = fdb.get("srun_view", "") or "realtime"
    else:
        uenv = getattr(fdb, "srun_uenv", "") or ""
        view = getattr(fdb, "srun_view", "") or "realtime"
    return uenv, view


def _get_fdb_roots(run_id):
    """Return (global_root, local_root) for a run, both keyed by env_id.

    Each checkpoint (env_id) gets its own dedicated FDB instance, so different
    checkpoints never mix data in the same root. global_root is '' if
    profile.fdb.fdb_root_global is not configured.
    """
    env_id = RUN_CONFIGS[run_id]["env_id"]
    global_base = get_resource_by_env(env_id, "fdb_root_global", "")
    global_root = f"{global_base}/{env_id}" if global_base else ""
    local_root = str((OUT_ROOT / f"data/fdb/{env_id}").resolve())
    return global_root, local_root


def get_resource_by_env(env_id, field: str, default):
    """Like get_resource, but usable outside a rule (no wildcards available)."""
    fdb = (config.get("profile") or {}).get("fdb") or {}
    if isinstance(fdb, dict):
        return fdb.get(field, default) or default
    else:
        return getattr(fdb, field, default) or default


def _count_fdb_blocks(run_id):
    """Count distinct 'fdb:' targets in this run's own inference config output.tee.

    A single init_time can write several output blocks (e.g. an ICON-grid target
    and an IFS-grid target) into the same env_id-keyed root; the check needs to
    know how many to expect so a run that silently dropped one entirely doesn't
    read as complete just because the other is.
    """
    with open(RUN_CONFIGS[run_id]["config"]) as f:
        run_cfg = yaml.safe_load(f)
    tee = ((run_cfg.get("output") or {}).get("tee")) or []
    return sum(1 for entry in tee if isinstance(entry, dict) and "fdb" in entry)


checkpoint inference_check_fdb:
    """Check whether (run_id, init_time) output already exists, complete, in FDB.

    Checks the global root first (if profile.fdb.fdb_root_global is set), then the
    local per-checkpoint root, in that order. A 'hit' requires every requested lead
    time to be present in one root -- see inference_check_fdb.py for why no further
    MARS-key disambiguation (class/expver/model) is needed. Writes 'hit\\t<root>' or
    'miss\\t' to the status file. Runs inside the squashfs venv so that pyfdb and the
    native FDB C library are available. Skipped entirely (writes 'miss') when this
    run's environment has no FDB uenv bundled (profile.fdb.srun_uenv unset).
    """
    input:
        image=lambda wc: OUT_ROOT
        / f"data/runs/{RUN_CONFIGS[wc.run_id]['env_id']}/venv.squashfs",
        script="workflow/scripts/inference_check_fdb.py",
    output:
        status=OUT_ROOT / "logs/inference_check_fdb/{run_id}-{init_time}.status",
    log:
        OUT_ROOT / "logs/inference_check_fdb/{run_id}-{init_time}.log",
    localrule: True
    params:
        image_path=lambda wc, input: str(Path(input.image).resolve()),
        fdb_configured=lambda wc: bool(
            _get_fdb_uenv_for_env(RUN_CONFIGS[wc.run_id]["env_id"])[0]
        ),
        fdb_root_global=lambda wc: _get_fdb_roots(wc.run_id)[0],
        fdb_root_local=lambda wc: _get_fdb_roots(wc.run_id)[1],
        fdb_schema=str(Path("resources/fdb/realtime-varda.schema").resolve()),
        fdb_view=lambda wc: get_resource(wc, "srun_view", "realtime") or "realtime",
        date=lambda wc: datetime.strptime(wc.init_time, "%Y%m%d%H%M").strftime("%Y%m%d"),
        time=lambda wc: datetime.strptime(wc.init_time, "%Y%m%d%H%M").strftime("%H%M"),
        steps=lambda wc: RUN_CONFIGS[wc.run_id]["steps"],
        expected_blocks=lambda wc: _count_fdb_blocks(wc.run_id),
    shell:
        """
        (
            set -euo pipefail
            if [ "{params.fdb_configured}" != "True" ]; then
                printf 'miss\t' > {output.status}
                echo "FDB check skipped: FDB not configured for this environment"
            else
                GLOBAL_ARG=""
                if [ -n "{params.fdb_root_global}" ]; then
                    GLOBAL_ARG="--fdb-root {params.fdb_root_global}"
                fi
                squashfs-mount {params.image_path}:/user-environment -- bash -c '
                    source /user-environment/bin/activate
                    export LD_LIBRARY_PATH=/user-environment/env/{params.fdb_view}/lib64:/user-environment/env/{params.fdb_view}/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}
                    python {input.script} '"$GLOBAL_ARG"' \
                        --fdb-root {params.fdb_root_local} \
                        --fdb-schema {params.fdb_schema} \
                        --date {params.date} \
                        --time {params.time} \
                        --steps {params.steps} \
                        --expected-blocks {params.expected_blocks} \
                        --output {output.status}
                '
            fi
        ) >{log} 2>&1
        """


def _get_inference_status(wc):
    """Return (okfile_path, fdb_root_or_None) for (run_id, init_time).

    'hit'  -> data already complete in FDB; okfile is the check's status file
              (inference_execute is never requested), fdb_root points at the
              root holding it.
    'miss' or FDB not configured -> falls through to inference_execute's okfile,
              fdb_root is None (data will be local grib / freshly written FDB).
    """
    env_id = RUN_CONFIGS[wc.run_id]["env_id"]
    if _get_fdb_uenv_for_env(env_id)[0]:
        status_file = checkpoints.inference_check_fdb.get(
            run_id=wc.run_id, init_time=wc.init_time
        ).output.status
        status, _, root = Path(status_file).read_text().strip().partition("\t")
        if status == "hit":
            return str(status_file), root
    return str(OUT_ROOT / f"logs/inference_execute/{wc.run_id}-{wc.init_time}.ok"), None


def _get_inference_okfile(wc):
    """Dependency file marking (run_id, init_time) data as ready, whether freshly
    executed or already complete in FDB."""
    return _get_inference_status(wc)[0]


def _get_inference_fdb_root(wc):
    """FDB root already holding this (run_id, init_time)'s data (a check 'hit'),
    or None if the data is local grib / about to be freshly executed."""
    return _get_inference_status(wc)[1]


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

    if run_config["model_type"] == "forecaster":
        input_path = f"logs/inference_prepare_forecaster/{wc.run_id}-{wc.init_time}.ok"
    elif run_config["model_type"] == "temporal_downscaler":
        input_path = (
            f"logs/inference_prepare_temporal_downscaler/{wc.run_id}-{wc.init_time}.ok"
        )
    else:
        raise ValueError(f"Unsupported model type: {run_config['model_type']}")

    return OUT_ROOT / input_path


rule inference_execute:
    input:
        okfile=_inference_routing_fn,
        image=lambda wc: OUT_ROOT
        / f"data/runs/{RUN_CONFIGS[wc.run_id]['env_id']}/venv.squashfs",
    output:
        okfile=OUT_ROOT / "logs/inference_execute/{run_id}-{init_time}.ok",
    log:
        OUT_ROOT / "logs/inference_execute/{run_id}-{init_time}.log",
    localrule: True
    resources:
        slurm_partition=lambda wc: get_resource(wc, "slurm_partition", "short-shared"),
        cpus_per_task=lambda wc: get_resource(wc, "cpus_per_task", 24),
        mem_mb_per_cpu=lambda wc: get_resource(wc, "mem_mb_per_cpu", 8000),
        runtime=lambda wc: get_resource(wc, "runtime", "40m"),
        gres=lambda wc: f"gpu:{get_resource(wc, 'gpu',1)}",
        ntasks=lambda wc: get_resource(wc, "tasks", 1),
        gpus=lambda wc: get_resource(wc, "gpu", 1),
    params:
        env_path=lambda wc, input: f"{Path(input.image).resolve()}",
        workdir=lambda wc: (
            OUT_ROOT / f"data/runs/{wc.run_id}/{wc.init_time}"
        ).resolve(),
        disable_local_definitions=lambda wc: RUN_CONFIGS[wc.run_id].get(
            "disable_local_eccodes_definitions", False
        ),
        srun_prefix=lambda wc: get_resource(wc, "srun_prefix", "") or "",
        fdb_view=lambda wc: get_resource(wc, "srun_view", "realtime") or "realtime",
        fdb_configured=lambda wc: bool(
            _get_fdb_uenv_for_env(RUN_CONFIGS[wc.run_id]["env_id"])[0]
        ),
        fdb_root=lambda wc: (
            _get_fdb_roots(wc.run_id)[0]
            if get_resource(wc, "write_to_global_fdb", False)
            and _get_fdb_roots(wc.run_id)[0]
            else _get_fdb_roots(wc.run_id)[1]
        ),
        fdb_schema=str(Path("resources/fdb/realtime-varda.schema").resolve()),
    # fmt: off
    shell:
        """
        (
            set -euo pipefail

            cd {params.workdir}

            # Write into this checkpoint's dedicated FDB root (env_id-keyed): the global
            # one if write_to_global_fdb is set, otherwise the local per-checkpoint root.
            # Exported so _run_inference (executed by squashfs-mount in a fresh bash) can see it.
            export FDB5_CONFIG_OVERRIDE=""
            if [ "{params.fdb_configured}" = "True" ]; then
                mkdir -p "{params.fdb_root}"
                # Use a patched schema that strips 'domain' (present in IFS GRIB MARS
                # namespace via mars_labeling.def but absent from MeteoSwiss FDB schema).
                cat > fdb5_config.yaml <<FDBEOF
---
type: local
engine: toc
schema: {params.fdb_schema}
spaces:
- handler: Default
  roots:
  - path: {params.fdb_root}
FDBEOF
                FDB5_CONFIG_OVERRIDE="$(pwd)/fdb5_config.yaml"
            fi

            _run_inference() {{
                local VENV=$1
                source "$VENV/bin/activate"

                if [ "{params.disable_local_definitions}" = "False" ]; then
                    export ECCODES_DEFINITION_PATH="$VENV/share/eccodes-cosmo-resources/definitions"
                fi

                # Set up FDB env vars if native libraries were bundled into this venv.
                # The env/<view>/lib64 symlink chain resolves to linux-zen3/<pkg>/lib64/*.so
                # because both the env/ symlink tree and the actual .so files are in the venv.
                if [ -d "$VENV/linux-zen3" ]; then
                    export LD_LIBRARY_PATH="$VENV/env/{params.fdb_view}/lib64:$VENV/env/{params.fdb_view}/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
                    # FDB5_CONFIG_OVERRIDE is set whenever FDB is configured for this env
                    # (see fdb_configured above); the bundled uenv config is only a fallback.
                    if [ -n "$FDB5_CONFIG_OVERRIDE" ]; then
                        export FDB5_CONFIG_FILE="$FDB5_CONFIG_OVERRIDE"
                    else
                        export FDB5_CONFIG_FILE="$VENV/meta/recipe/meta/private/fdb_config/{params.fdb_view}.yaml"
                    fi
                    # Prepend eccodes-cosmo-mars definitions so MARS namespace concepts
                    # (marsModel, marsStream, marsClass, etc.) resolve correctly for MeteoSwiss GRIBs.
                    # Must be set in ECCODES_DEFINITION_PATH (not just GRIB_DEFINITION_PATH) because
                    # the FDB C library uses eccodes internally to resolve MARS namespace keys.
                    export ECCODES_DEFINITION_PATH="$VENV/data/share/eccodes-cosmo-mars/definitions${{ECCODES_DEFINITION_PATH:+:$ECCODES_DEFINITION_PATH}}"
                fi

                CMD_ARGS=()

                # is GPU > 1, add parallel flag to CMD_ARGS and override automatic cluster detection
                if [ {resources.gpus} -gt 1 ]; then
                    CMD_ARGS+=(runner.parallel.cluster=slurm)
                fi

                {params.srun_prefix} srun \
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
        ) >{log} 2>&1
        touch {output.okfile}
        """
# fmt: on
