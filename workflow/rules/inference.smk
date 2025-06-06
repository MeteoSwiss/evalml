# ----------------------------------------------------- #
# INFERENCE WORKFLOW                                    #
# ----------------------------------------------------- #

from pathlib import Path


rule run_inference:
    input:
        config="config/anemoi_inference.yaml",
        checkpoint=Path(config["locations"]["checkpoint_root"])
        / "{run_id}"
        / "inference-last.ckpt",
    output:
        "resources/inference/output/{run_id}/prediction.nc",
    params:
        leadtime="120",  # lead time in hours
    log:
        "logs/anemoi-inference-run-{run_id}.log",
    conda:
        "../envs/anemoi-inference.yaml"
    resources:
        partition="normal",
        cpus_per_task=8,
        time="20m",
        gres="gpu:4",
    shell:
        "export TZ=UTC;"
        "anemoi-inference run {input.config} "
        " checkpoint={input.checkpoint}"
        " lead_time={params.leadtime}h"
        " > {log} 2>&1"
