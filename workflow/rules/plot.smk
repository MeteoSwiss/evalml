# ----------------------------------------------------- #
# PLOTTING WORKFLOW                                     #
# ----------------------------------------------------- #


include: "common.smk"


import pandas as pd


rule plot_forecast_frame:
    input:
        script="workflow/scripts/plot_forecast_frame.mo.py",
        inference_okfile=rules.execute_inference.output.okfile,
    output:
        temp(
            OUT_ROOT
            / "showcases/{run_id}/{init_time}/frames/{init_time}_{leadtime}_{param}_{region}.png"
        ),
    resources:
        slurm_partition="postproc",
        cpus_per_task=1,
        runtime="10m",
    params:
        grib_out_dir=lambda wc: (
            Path(OUT_ROOT) / f"data/runs/{wc.run_id}/{wc.init_time}/grib"
        ).resolve(),
    shell:
        """
        export ECCODES_DEFINITION_PATH=/user-environment/share/eccodes-cosmo-resources/definitions
        python {input.script} \
            --input {params.grib_out_dir}  --date {wildcards.init_time} --outfn {output[0]} \
            --param {wildcards.param} --leadtime {wildcards.leadtime} --region {wildcards.region} \
        # interactive editing (needs to set localrule: True and use only one core)
        # marimo edit {input.script} -- \
        #     --input {params.grib_out_dir}  --date {wildcards.init_time} --outfn {output[0]}\
        #     --param {wildcards.param} --leadtime {wildcards.leadtime} --region {wildcards.region}\
        """


rule plot_meteogram:
    input:
        script="workflow/scripts/plot_meteogram.mo.py",
        fct_grib=rules.inference_routing.output[0],
        analysis_zarr=config["analysis"].get("analysis_zarr"),
    output:
        OUT_ROOT / "showcases/{run_id}/meteograms/{init_time}_{param}_{sta}.png",
    # localrule: True
    resources:
        slurm_partition="postproc",
        cpus_per_task=1,
        runtime="5m",
    shell:
        """
        export ECCODES_DEFINITION_PATH=/user-environment/share/eccodes-cosmo-resources/definitions
        python {input.script} \
            --forecast {input.fct_grib}  --analysis {input.analysis_zarr} \
            --date {wildcards.init_time} --outfn {output[0]} \
            --param {wildcards.param}  --station {wildcards.sta}
        # interactive editing (needs to set localrule: True and use only one core)
        # marimo edit {input.script} -- \
        #     --forecast {input.fct_grib}  --analysis {input.analysis_zarr} \
        #     --date {wildcards.init_time} --outfn {output[0]} \
        #     --param {wildcards.param}  --station {wildcards.sta}
        """


def get_leadtimes(wc):
    """Get all lead times from the run config."""
    start, end, step = map(int, RUN_CONFIGS[wc.run_id]["steps"].split("/"))
    return [f"{i:03}" for i in range(start, end + 1, step)]


rule make_forecast_animation:
    input:
        expand(
            OUT_ROOT
            / "showcases/{run_id}/{init_time}/frames/{init_time}_{leadtime}_{param}_{region}.png",
            leadtime=lambda wc: get_leadtimes(wc),
            allow_missing=True,
        ),
    output:
        OUT_ROOT / "showcases/{run_id}/{init_time}/{init_time}_{param}_{region}.gif",
    localrule: True
    shell:
        """
        convert -delay 10 -loop 0 {input} {output}
        """
