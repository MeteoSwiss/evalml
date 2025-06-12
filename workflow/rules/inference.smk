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
        "resources/inference/output/{run_id}/{init_time}",
    log:
        "logs/anemoi-inference-run-{run_id}-{init_time}.log",
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
        " lead_time={config[experiment][lead_time]}"
        " > {log} 2>&1"
