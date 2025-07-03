# ----------------------------------------------------- #
# INFERENCE WORKFLOW                                    #
# ----------------------------------------------------- #

import os
from pathlib import Path
from datetime import datetime


configfile: "config/config.yaml"

def get_run_id(experiment):
    """Get the run_id from the experiment configuration."""
    with open(CONFIG_ROOT / f"experiments/{experiment}.yaml", "r") as f:
        return yaml.safe_load(f)["run_id"]

        
rule create_inference_pyproject:
    input:
        toml="workflow/envs/anemoi_inference.toml",
    output:
        pyproject="resources/inference/{experiment}/pyproject.toml",
    params:
        run_id=lambda wc: get_run_id(wc.experiment),
    log:
        "logs/create-inference-pyproject-{experiment}.log",
    conda:
        "../envs/anemoi_inference.yaml"
    localrule: True
    script:
        "../scripts/set_inference_pyproject.py"


rule create_inference_venv:
    input:
        pyproject="resources/inference/{experiment}/pyproject.toml",
    output:
        venv=temp(directory("resources/inference/{experiment}/.venv")),
    params:
        py_version=parse_input(
            input.pyproject, parse_toml, key="project.requires-python"
        ),
    localrule: True
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
        image=Path(os.environ.get("SCRATCH")) / "sqfs-images" / "{experiment}.squashfs",
    log:
        "logs/make-squashfs-image-{experiment}.log",
    localrule: True
    shell:
        "mksquashfs {input.venv} {output.image}"
        " -no-recovery -noappend -Xcompression-level 3"
        " > {log} 2>/dev/null"
        # we can safely ignore the many warnings "Unrecognised xattr prefix..."


rule run_inference_group:
    input:
        pyproject=rules.create_inference_pyproject.output.pyproject,
        image=rules.make_squashfs_image.output.image,
        config=str(Path("config/anemoi_inference.yaml").resolve()),
    output:
        okfile=temp(touch(OUT_ROOT / "experiments/{experiment}/group-{group_index}.ok")),
    params:
        checkpoints_path=parse_input(
            input.pyproject, parse_toml, key="tool.anemoi.checkpoints_path"
        ),
        reftimes=lambda wc: [
            t.strftime("%Y-%m-%dT%H:%M") for t in REFTIMES_GROUPS[int(wc.group_index)]
        ],
        lead_time=config["lead_time"],
        output_root=OUT_ROOT,
    # TODO: we can have named logs for each reftime
    log:
        "logs/inference_run/{experiment}-{group_index}.log",
    resources:
        slurm_partition="short",
        cpus_per_task=32,
        mem_mb_per_cpu=8000,
        runtime="20m",
        gres="gpu:1",  # because we use --exclusive, this will be 1 GPU per run (--ntasks-per-gpus is automatically set to 1)
        # see https://github.com/MeteoSwiss/mch-anemoi-evaluation/pull/3#issuecomment-2998997104
        slurm_extra=lambda wc, input: f"--uenv={input.image}:/user-environment --exclusive",
    shell:
        """
        export TZ=UTC
        source /user-environment/bin/activate
        export ECCODES_DEFINITION_PATH=/user-environment/share/eccodes-cosmo-resources/definitions
        i=0
        for reftime in {params.reftimes}; do
            
            # prepare the working directory
            _reftime_str=$(date -d "$reftime" +%Y%m%d%H%M)
            WORKDIR={params.output_root}/experiments/{wildcards.experiment}/$_reftime_str
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
        lambda wc: OUT_ROOT / f"experiments/{wc.experiment}/group-{REFTIME_TO_GROUP[wc.init_time]}.ok",
    output:
        directory(OUT_ROOT / "experiments/{experiment}/{init_time}/grib"),
        directory(OUT_ROOT / "experiments/{experiment}/{init_time}/raw"),
