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
    wildcard_constraints:
        leadtime=r"\d+",  # only digits
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
        export ECCODES_DEFINITION_PATH=$(realpath .venv/share/eccodes-cosmo-resources/definitions)
        python {input.script} \
            --input {params.grib_out_dir}  --date {wildcards.init_time} --outfn {output[0]} \
            --param {wildcards.param} --leadtime {wildcards.leadtime} --region {wildcards.region} \
        # interactive editing (needs to set localrule: True and use only one core)
        # marimo edit {input.script} -- \
        #     --input {params.grib_out_dir}  --date {wildcards.init_time} --outfn {output[0]}\
        #     --param {wildcards.param} --leadtime {wildcards.leadtime} --region {wildcards.region}\
        """


def get_leadtimes(wc):
    """Get all lead times from the run config."""
    start, end, step = map(int, RUN_CONFIGS[wc.run_id]["steps"].split("/"))
    # skip lead time 0 for diagnostic variables
    if wc.param in ["tp", "TOT_PREC"]:  # TODO: make this more general
        start += step
    return [f"{i:03}" for i in range(start, end + 1, step)]


rule make_forecast_animation:
    localrule: True
    input:
        expand(
            OUT_ROOT
            / "showcases/{run_id}/{init_time}/frames/{init_time}_{leadtime}_{param}_{region}.png",
            leadtime=lambda wc: get_leadtimes(wc),
            allow_missing=True,
        ),
    output:
        OUT_ROOT / "showcases/{run_id}/{init_time}/{init_time}_{param}_{region}.gif",
    params:
        delay=lambda wc: 10 * int(RUN_CONFIGS[wc.run_id]["steps"].split("/")[2]),
    shell:
        """
        convert -delay {params.delay} -loop 0 {input} {output}
        """


rule plot_summary_stat_maps:
    localrule: True
    input:
        script="workflow/scripts/plot_summary_stat_maps.mo.py",
        verif_file=OUT_ROOT / "data/runs/{run_id}/verif_aggregated.nc",
    output:
        OUT_ROOT / "results/{experiment}/metrics/spatial/runs/{run_id}/{param}_{metric}_{region}_{season}_{leadtime}.png",
    wildcard_constraints:
        leadtime=r"\d+",  # only digits
    resources:
        slurm_partition="postproc",
        cpus_per_task=1,
        runtime="10m",
    params:
        nc_out_dir=lambda wc: (
            Path(OUT_ROOT) / f"data/runs/{wc.run_id}/verif_aggregated.nc"
            # not sure how to do this, because the baselines are in, e.g., output/data/baselines/COSMO-E/verif_aggregated.nc
            # and the runs are in output/data/runs/runID/verif_aggregated.nc
        ).resolve(),
    shell:
        """
        export ECCODES_DEFINITION_PATH=$(realpath .venv/share/eccodes-cosmo-resources/definitions)
        python {input.script} \
            --input {input.verif_file} --outfn {output[0]} --region {wildcards.region} \
            --param {wildcards.param} --leadtime {wildcards.leadtime} --metric {wildcards.metric} \
            --season {wildcards.season} \
        # interactive editing (needs to set localrule: True and use only one core)
        # marimo edit {input.script} -- \
        #     --input {input.verif_file} --outfn {output[0]} --region {wildcards.region} \
        #     --param {wildcards.param} --leadtime {wildcards.leadtime} --metric {wildcards.metric} \
        #     --season {wildcards.season} \
        """

use rule plot_summary_stat_maps as plot_summary_stat_maps_baseline with:
    input:
        script="workflow/scripts/plot_summary_stat_maps.mo.py",
        verif_file=OUT_ROOT / "data/baselines/{baseline_id}/verif_aggregated.nc",
    output:
        OUT_ROOT / "results/{experiment}/metrics/spatial/baselines/{baseline_id}/{param}_{metric}_{region}_{season}_{leadtime}.png",
    params:
        nc_out_dir=lambda wc: (
            Path(OUT_ROOT) / f"data/baselines/{wc.baseline_id}/verif_aggregated.nc"
            # not sure if this is actually needed. Verification file is already specified above as input. Leave it for the time being. 
        ).resolve()