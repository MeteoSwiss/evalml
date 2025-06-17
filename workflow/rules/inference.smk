# ----------------------------------------------------- #
# INFERENCE WORKFLOW                                    #
# ----------------------------------------------------- #

from pathlib import Path


include: "common.smk"


rule run_inference_all:
    input:
        expand("resources/{{run_id}}/{reftime}/output.nc", reftime=REFTIMES),
    output:
        touch("resources/{run_id}/run-inference-all.done")

for group_index in range(len(REFTIMES_GROUPS)):

    rule:
        name:
            f"run_inference_group_{group_index}"
        input:
            config="config/anemoi_inference.yaml",
            checkpoint=Path(config["locations"]["checkpoint_root"])
            / "{run_id}"
            / "inference-last.ckpt",
        output:
            expand(
                "resources/{{run_id}}/{reftime}/output.nc",
                reftime=REFTIMES_GROUPS[group_index],
            ),
        params:
            group_reftimes=REFTIMES_GROUPS[int(group_index)],
            group_size=config["execution"]["run_group_size"],
            leadtime="120h",  # lead time in hours
        log:
            f"logs/anemoi-inference-run-{{run_id}}_{group_index}.log",
        conda:
            "../envs/anemoi-inference.yaml"
        resources:
            partition="normal",
            cpus_per_task=8,
            time="20m",
            gres="gpu:4",
        run:
            import subprocess
            import os

            run_id = wildcards.run_id
            reftimes = params.group_reftimes
            group_size = params.group_size
            leadtime = params.leadtime

            for i, reftime in enumerate(reftimes):
                workdir = Path(f"resources/{run_id}/{reftime}")
                workdir.mkdir(parents=True, exist_ok=True)
                config_target = workdir / "config.yaml"
                config_target.write_text(Path(input.config).read_text())

                print(f"Running inference for {reftime} on GPU {i}")

                cmd = [
                    "anemoi-inference",
                    "run",
                    str(input.config),
                    f"date={reftime}",
                    f"checkpoint={input.checkpoint}",
                    f"lead_time={leadtime}h",
                ]

                log_path = workdir / "anemoi-inference-run.log"
                with open(log_path, "w") as log_file:
                    # Uncomment the line below to actually run the inference
                    # subprocess.Popen(cmd, env={"CUDA_VISIBLE_DEVICES": str(i)}, stdout=log_file, stderr=subprocess.STDOUT)
                    log_file.write(f"Simulated run for {reftime} on GPU {i}\n")

                    # Wait for all subprocesses if you use Popen (optional)
                    # for proc in procs:
                    #     proc.wait()

                    # Mark the output
            Path(output[0]).touch()
