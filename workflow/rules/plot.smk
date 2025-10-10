# ----------------------------------------------------- #
# PLOTTING WORKFLOW                                     #
# ----------------------------------------------------- #


include: "common.smk"

wildcard_constraints:
    with_global="true|false"

rule plot_forecast_frame:
    input:
        script="workflow/scripts/plot_forecast_frame.mo.py",
        raw_output=rules.inference_routing.output[0],
    output:
        temp(
            OUT_ROOT
            / "showcases/{run_id}/{init_time}/frames/{init_time}_{leadtime}_{param}_{projection}_{region}_{with_global}.png"
        ),
    resources:
        slurm_partition="postproc",
        cpus_per_task=1,
        runtime="5m",
    # localrule: True
    shell:
        """
        python workflow/scripts/plot_forecast_frame.py \
            --input {input.raw_output}  --date {wildcards.init_time} --outfn {output[0]} \
            --param {wildcards.param} --leadtime {wildcards.leadtime} \
            --projection {wildcards.projection} --region {wildcards.region} --with_global {wildcards.with_global} \
        # interactive editing (needs to set localrule: True and use only one core)
        # marimo edit workflow/scripts/notebook_plot_map.py -- \
        #     --input {input.raw_output}  --date {wildcards.init_time} --outfn {output[0]}\
        #     --param {wildcards.param} --leadtime {wildcards.leadtime} \
        #     --projection {wildcards.projection} --region {wildcards.region} \
        """


LEADTIME = int(pd.to_timedelta(config["lead_time"]).total_seconds() // 3600)


rule make_forecast_animation:
    input:
        expand(
            OUT_ROOT
            / "showcases/{run_id}/{init_time}/frames/{init_time}_{leadtime}_{param}_{projection}_{region}_{with_global}.png",
            leadtime=[f"{i:03}" for i in range(0, LEADTIME + 6, 6)],
            with_global="true",
            allow_missing=True,
        ),
    output:
        OUT_ROOT
        / "showcases/{run_id}/{init_time}/{init_time}_{param}_{projection}_{region}_{with_global}.gif",
    localrule: True
    shell:
        """
        convert -delay 80 -loop 0 {input} {output}
        """
