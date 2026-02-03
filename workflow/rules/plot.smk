# ----------------------------------------------------- #
# PLOTTING WORKFLOW                                     #
# ----------------------------------------------------- #


include: "common.smk"


import pandas as pd


def _use_first_baseline_zarr(wc):
    """Get the first available baseline zarr for the given init time."""
    for baseline_id in BASELINE_CONFIGS:
        root = BASELINE_CONFIGS[baseline_id].get("root")
        steps = BASELINE_CONFIGS[baseline_id].get("steps")
        year = wc.init_time[2:4]
        baseline_zarr = f"{root}/FCST{year}.zarr"
        if Path(baseline_zarr).exists():
            return baseline_zarr, steps
    raise ValueError(f"No baseline zarr found for init time {wc.init_time}")


rule plot_meteogram:
    input:
        script="workflow/scripts/plot_meteogram.mo.py",
        inference_okfile=rules.execute_inference.output.okfile,
        analysis_zarr=config["analysis"].get("analysis_zarr"),
        baseline_zarr=lambda wc: _use_first_baseline_zarr(wc)[0],
        peakweather_dir=rules.download_obs_from_peakweather.output.peakweather,
    output:
        OUT_ROOT
        / "results/{showcase}/{run_id}/{init_time}/{init_time}_{param}_{sta}.png",
    # localrule: True
    resources:
        slurm_partition="postproc",
        cpus_per_task=1,
        runtime="10m",
    params:
        grib_out_dir=lambda wc: (
            Path(OUT_ROOT) / f"data/runs/{wc.run_id}/{wc.init_time}/grib"
        ).resolve(),
        baseline_steps=lambda wc: _use_first_baseline_zarr(wc)[1],
    shell:
        """
        export ECCODES_DEFINITION_PATH=$(realpath .venv/share/eccodes-cosmo-resources/definitions)
        python {input.script} \
            --forecast {params.grib_out_dir}  --analysis {input.analysis_zarr} \
            --baseline {input.baseline_zarr} --baseline_steps {params.baseline_steps} \
            --peakweather {input.peakweather_dir} \
            --date {wildcards.init_time} --outfn {output[0]} \
            --param {wildcards.param}  --station {wildcards.sta}
        # interactive editing (needs to set localrule: True and use only one core)
        # marimo edit {input.script} -- \
        #     --forecast {params.grib_out_dir}  --analysis {input.analysis_zarr} \
        #     --baseline {input.baseline_zarr} --peakweather {input.peakweather_dir} \
        #     --date {wildcards.init_time} --outfn {output[0]} \
        #     --param {wildcards.param}  --station {wildcards.sta}
        """


rule plot_forecast_frame:
    input:
        script="workflow/scripts/plot_forecast_frame.mo.py",
        inference_okfile=rules.execute_inference.output.okfile,
    output:
        OUT_ROOT
        / "data/runs/{run_id}/{init_time}/frames/frame_{leadtime}_{param}_{region}.png",
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
            rules.plot_forecast_frame.output,
            leadtime=lambda wc: get_leadtimes(wc),
            allow_missing=True,
        ),
    output:
        OUT_ROOT
        / "results/{showcase}/{run_id}/{init_time}/{init_time}_{param}_{region}.gif",
    params:
        delay=lambda wc: 10 * int(RUN_CONFIGS[wc.run_id]["steps"].split("/")[2]),
    shell:
        """
        convert -delay {params.delay} -loop 0 {input} {output}
        """
