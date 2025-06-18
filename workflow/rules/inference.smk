# ----------------------------------------------------- #
# INFERENCE WORKFLOW                                    #
# ----------------------------------------------------- #

from pathlib import Path



rule run_inference_group:
    input:
        image=rules.make_squashfs_image.output.image,
        config="config/anemoi_inference.yaml",
        checkpoint=Path(config["locations"]["checkpoint_root"])
        / {run_id}
        / "inference-last.ckpt",
    output:
        okfile=temp("resources/inference/{run_id}/output/group-{group_index}.ok"),
    params:
        group_reftimes=lambda wc: REFTIMES_GROUPS[int(wc.group_index)],
        group_size=config["execution"]["run_group_size"],
        leadtime="120h",  # lead time in hours
    log:
        "logs/anemoi-inference-run-{run_id}-{group_index}.log",
    conda:
        "../envs/anemoi_inference.yaml"
    resources:
        slurm_partition="normal",
        cpus_per_task=8,
        runtime="20m",
        gres="gpu:4",
    script:
        "../scripts/run_inference_group.py"


rule map_init_time_to_inference_group:
    input:
        lambda wc: f"resources/inference/{run_id}/output/group-{REFTIME_TO_GROUP[wc.init_time]}.ok"
    output:
        "resources/inference/{run_id}/output/{init_time}/raw",
