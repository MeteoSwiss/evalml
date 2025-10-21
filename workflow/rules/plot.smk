# ----------------------------------------------------- #
# PLOTTING WORKFLOW                                     #
# ----------------------------------------------------- #


include: "common.smk"


import pandas as pd


rule plot_forecast_frame:
    input:
        script="workflow/scripts/plot_forecast_frame.mo.py",
        raw_output=rules.inference_routing.output[0],
    output:
        temp(
            OUT_ROOT
            / "showcases/{run_id}/{init_time}/frames/{init_time}_{leadtime}_{param}_{projection}_{region}.png"
        ),
    resources:
        slurm_partition="postproc",
        cpus_per_task=1,
        runtime="5m",
    shell:
        """
        export ECCODES_DEFINITION_PATH=/user-environment/share/eccodes-cosmo-resources/definitions
        python {input.script} \
            --input {input.raw_output}  --date {wildcards.init_time} --outfn {output[0]} \
            --param {wildcards.param} --leadtime {wildcards.leadtime} \
            --projection {wildcards.projection} --region {wildcards.region} \
        # interactive editing (needs to set localrule: True and use only one core)
        # marimo edit {input.script} -- \
        #     --input {input.raw_output}  --date {wildcards.init_time} --outfn {output[0]}\
        #     --param {wildcards.param} --leadtime {wildcards.leadtime} \
        #     --projection {wildcards.projection} --region {wildcards.region} \
        """


LEADTIME = int(pd.to_timedelta(config["lead_time"]).total_seconds() // 3600)


rule make_forecast_animation:
    input:
        expand(
            OUT_ROOT
            / "showcases/{run_id}/{init_time}/frames/{init_time}_{leadtime}_{param}_{projection}_{region}.png",
            leadtime=[f"{i:03}" for i in range(0, LEADTIME + 6, 6)],
            allow_missing=True,
        ),
    output:
        OUT_ROOT
        / "showcases/{run_id}/{init_time}/{init_time}_{param}_{projection}_{region}.gif",
    localrule: True
    shell:
        """
        convert -delay 80 -loop 0 {input} {output}
        """
