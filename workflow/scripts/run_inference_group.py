import subprocess
import os
from pathlib import Path

run_id = snakemake.wildcards.run_id
reftimes = snakemake.params.group_reftimes
group_size = snakemake.params.group_size
leadtime = snakemake.params.leadtime
okfile = snakemake.output.okfile

for i, reftime in enumerate(reftimes):
    workdir = Path(f"resources/{run_id}/{reftime}")
    workdir.mkdir(parents=True, exist_ok=True)
    config_target = workdir / "config.yaml"
    config_target.write_text(Path(input.config).read_text())

    print(f"Running inference for {reftime} on GPU {i}")

    cmd = [
        "source /user-environment/bin/activate;"
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

        Path(okfile).touch()
