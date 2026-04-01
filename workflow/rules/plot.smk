# ----------------------------------------------------- #
# PLOTTING WORKFLOW                                     #
# ----------------------------------------------------- #


include: "common.smk"


import pandas as pd


def _get_available_baselines(wc) -> list[dict[str, str]]:
    """Get all available baseline zarr datasets for the given init time."""
    baselines = []
    for baseline_id in BASELINE_CONFIGS:
        root = BASELINE_CONFIGS[baseline_id].get("root")
        steps = BASELINE_CONFIGS[baseline_id].get("steps")
        label = BASELINE_CONFIGS[baseline_id].get("label", baseline_id)
        year = wc.init_time[2:4]
        baseline_zarr = f"{root}/FCST{year}.zarr"
        if Path(baseline_zarr).exists():
            baselines.append({"zarr": baseline_zarr, "steps": steps, "label": label})
    if not baselines:
        raise ValueError(f"No baseline zarr found for init time {wc.init_time}")
    return baselines


rule plot_meteogram:
    input:
        script="workflow/scripts/plot_meteogram.mo.py",
        inference_okfile=rules.execute_inference.output.okfile,
        truth=config["truth"]["root"],
        peakweather_dir=rules.download_obs_from_peakweather.output.root,
    output:
        OUT_ROOT
        / "results/{showcase}/{run_id}/{init_time}/{init_time}_{param}_{sta}.png",
    # localrule: True
    resources:
        slurm_partition="postproc",
        cpus_per_task=1,
        runtime="10m",
    params:
        ana_label=lambda wc: config["truth"]["label"],
        fcst_grib=lambda wc: (
            Path(OUT_ROOT) / f"data/runs/{wc.run_id}/{wc.init_time}/grib"
        ).resolve(),
        fcst_steps=lambda wc: RUN_CONFIGS[wc.run_id]["steps"],
        fcst_label=lambda wc: RUN_CONFIGS[wc.run_id]["label"],
        baseline_zarrs=lambda wc: [x["zarr"] for x in _get_available_baselines(wc)],
        baseline_steps=lambda wc: [x["steps"] for x in _get_available_baselines(wc)],
        baseline_labels=lambda wc: [x["label"] for x in _get_available_baselines(wc)],
    shell:
        """
        set -euo pipefail
        export ECCODES_DEFINITION_PATH=$(realpath .venv/share/eccodes-cosmo-resources/definitions)

        BASELINE_ZARRS=({params.baseline_zarrs:q})
        BASELINE_STEPS=({params.baseline_steps:q})
        BASELINE_LABELS=({params.baseline_labels:q})

        CMD_ARGS=(
            --forecast {params.fcst_grib:q}
            --forecast_steps {params.fcst_steps:q}
            --forecast_label {params.fcst_label:q}
            --analysis {input.truth:q}
            --analysis_label {params.ana_label:q}
            --peakweather {input.peakweather_dir:q}
            --date {wildcards.init_time:q}
            --outfn {output[0]:q}
            --param {wildcards.param:q}
            --station {wildcards.sta:q}
        )

        for i in "${{!BASELINE_ZARRS[@]}}"; do
            CMD_ARGS+=(--baseline "${{BASELINE_ZARRS[$i]}}")
            CMD_ARGS+=(--baseline_steps "${{BASELINE_STEPS[$i]}}")
            CMD_ARGS+=(--baseline_label "${{BASELINE_LABELS[$i]}}")
        done

        python {input.script} "${{CMD_ARGS[@]}}"
        # interactive editing (needs to set localrule: True and use only one core)
        # marimo edit {input.script} -- "${{CMD_ARGS[@]}}"
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
