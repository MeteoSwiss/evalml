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
        inference_okfile=rules.inference_execute.output.okfile,
        truth=config["truth"]["root"],
        peakweather_dir=rules.data_download_obs_from_peakweather.output.root,
    output:
        OUT_ROOT
        / "results/{showcase}/{run_id}/{init_time}/{init_time}_{param}_{sta}.png",
    # localrule: True
    resources:
        slurm_partition="postproc",
        cpus_per_task=1,
        runtime="60m",
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
        inference_okfile=rules.inference_execute.output.okfile,
    output:
        OUT_ROOT
        / "data/runs/{run_id}/{init_time}/frames/frame_{leadtime}_{param}_{region}.png",
    wildcard_constraints:
        leadtime=r"\d+",  # only digits
        region="|".join(map(re.escape, SHOWCASE_REGIONS.keys())),
    resources:
        slurm_partition="postproc",
        cpus_per_task=1,
        runtime="10m",
    params:
        grib_out_dir=lambda wc: (
            Path(OUT_ROOT) / f"data/runs/{wc.run_id}/{wc.init_time}/grib"
        ).resolve(),
        region_extra=lambda wc: (
            "--extent {} --projection {}".format(
                " ".join(map(str, SHOWCASE_REGIONS[wc.region]["extent"])),
                SHOWCASE_REGIONS[wc.region]["projection"],
            )
            if SHOWCASE_REGIONS.get(wc.region, {}).get("extent") is not None
            else ""
        ),
        accu=lambda wc: int(RUN_CONFIGS[wc.run_id]["steps"].split("/")[2]),
    shell:
        """
        export ECCODES_DEFINITION_PATH=$(realpath .venv/share/eccodes-cosmo-resources/definitions)
        python {input.script} \
            --input {params.grib_out_dir} --date {wildcards.init_time} --outfn {output[0]} \
            --param {wildcards.param} --leadtime {wildcards.leadtime} --region {wildcards.region} \
            {params.region_extra} \
            --accu {params.accu}
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
    wildcard_constraints:
        run_id="|".join(map(re.escape, RUN_CONFIGS.keys())),
        param="|".join(map(re.escape, SHOWCASE_PARAMS)),
        region="|".join(map(re.escape, SHOWCASE_REGIONS.keys())),
    input:
        lambda wc: expand(
            rules.plot_forecast_frame.output,
            run_id=wc.run_id,
            init_time=wc.init_time,
            param=wc.param,
            region=wc.region,
            leadtime=get_leadtimes(wc),
        ),
    output:
        OUT_ROOT
        / "results/{showcase}/{run_id}/{init_time}/{init_time}_{param}_{region}.gif",
    wildcard_constraints:
        param="|".join(map(re.escape, SHOWCASE_PARAMS)),
        region="|".join(map(re.escape, SHOWCASE_REGIONS.keys())),
    localrule: True
    params:
        delay=lambda wc: round(
            int(RUN_CONFIGS[wc.run_id]["steps"].split("/")[2])
            / config["showcase"]["animations"].get("speed", 10.0)
            * 100
        ),
    shell:
        """
        convert -delay {params.delay} -loop 0 {input} {output}
        """


def _comparison_by_id(comparison_id: str) -> dict:
    """Look up a SHOWCASE_COMPARISONS entry by its id wildcard."""
    for c in SHOWCASE_COMPARISONS:
        if c["id"] == comparison_id:
            return c
    raise ValueError(f"No comparison with id {comparison_id! r}")


def _side_gif_path(side: dict, wc) -> list:
    """Return the GIF path list for one side of a comparison (run or zarr)."""
    if side["type"] == "run":
        return expand(
            rules.make_forecast_animation.output,
            run_id=side["run_id"],
            init_time=wc.init_time,
            param=wc.param,
            region=wc.region,
            showcase=wc.showcase,
        )
    else:
        return expand(
            rules.make_zarr_animation.output,
            source_id=side["label"],
            init_time=wc.init_time,
            param=wc.param,
            region=wc.region,
            showcase=wc.showcase,
        )


def get_zarr_leadtimes(wc):
    """Get lead times for a zarr source, skipping step 0 for TOT_PREC."""
    cfg = ZARR_SOURCES[wc.source_id]
    step = cfg["step"]
    total = cfg["total_hours"]
    start = step  # always skip lead time 0 (no meaningful accumulation at t=0)
    return [f"{i:03}" for i in range(start, total + 1, step)]


rule plot_zarr_frame:
    input:
        script="workflow/scripts/plot_zarr_frame.py",
    output:
        OUT_ROOT
        / "data/zarr/{source_id}/{init_time}/frames/frame_{leadtime}_{param}_{region}.png",
    wildcard_constraints:
        source_id="|".join(map(re.escape, ZARR_SOURCES.keys())) or "NEVER",
        leadtime=r"\d+",
        region="|".join(map(re.escape, SHOWCASE_REGIONS.keys())),
    resources:
        slurm_partition="postproc",
        cpus_per_task=1,
        runtime="10m",
    params:
        zarr_path=lambda wc: ZARR_SOURCES[wc.source_id]["root"],
        source_type=lambda wc: ZARR_SOURCES[wc.source_id]["source_type"],
        region_extra=lambda wc: (
            "--extent {} --projection {}".format(
                " ".join(map(str, SHOWCASE_REGIONS[wc.region]["extent"])),
                SHOWCASE_REGIONS[wc.region]["projection"],
            )
            if SHOWCASE_REGIONS.get(wc.region, {}).get("extent") is not None
            else ""
        ),
        accu=lambda wc: ZARR_SOURCES[wc.source_id]["step"],
    shell:
        """
        export ECCODES_DEFINITION_PATH=$(realpath .venv/share/eccodes-cosmo-resources/definitions)
        python {input.script} \
            --zarr {params.zarr_path} \
            --source_type {params.source_type} \
            --date {wildcards.init_time} \
            --outfn {output} \
            --param {wildcards.param} \
            --leadtime {wildcards.leadtime} \
            --region {wildcards.region} \
            {params.region_extra} \
            --accu {params.accu}
        """


rule make_zarr_animation:
    localrule: True
    wildcard_constraints:
        source_id="|".join(map(re.escape, ZARR_SOURCES.keys())) or "NEVER",
        param="|".join(map(re.escape, SHOWCASE_PARAMS)),
        region="|".join(map(re.escape, SHOWCASE_REGIONS.keys())),
    input:
        lambda wc: expand(
            rules.plot_zarr_frame.output,
            source_id=wc.source_id,
            init_time=wc.init_time,
            param=wc.param,
            region=wc.region,
            leadtime=get_zarr_leadtimes(wc),
        ),
    output:
        OUT_ROOT
        / "results/{showcase}/zarr/{source_id}/{init_time}/{init_time}_{param}_{region}.gif",
    params:
        delay=lambda wc: round(
            ZARR_SOURCES[wc.source_id]["step"]
            / config["showcase"]["animations"].get("speed", 10.0)
            * 100
        ),
    shell:
        """
        convert -delay {params.delay} -loop 0 {input} {output}
        """


rule make_comparison_animation:
    """Side-by-side two-panel animation comparing two sources, synced in simulated time."""
    localrule: True
    wildcard_constraints:
        param="|".join(map(re.escape, SHOWCASE_PARAMS)),
        region="|".join(map(re.escape, SHOWCASE_REGIONS.keys())),
        comparison_id="|".join(map(re.escape, [c["id"] for c in SHOWCASE_COMPARISONS]))
        or "NEVER",
    input:
        left=lambda wc: _side_gif_path(_comparison_by_id(wc.comparison_id)["left"], wc),
        right=lambda wc: _side_gif_path(
            _comparison_by_id(wc.comparison_id)["right"], wc
        ),
        script="workflow/scripts/plot_combine_animations.py",
    output:
        OUT_ROOT
        / "results/{showcase}/comparisons/{comparison_id}/{init_time}/{init_time}_{param}_{region}.gif",
    params:
        left_step=lambda wc: _comparison_by_id(wc.comparison_id)["left"]["step"],
        right_step=lambda wc: _comparison_by_id(wc.comparison_id)["right"]["step"],
        speed=config["showcase"]["animations"].get("speed", 10.0),
    shell:
        """
        python {input.script} \
            --left {input.left} --left_step {params.left_step} \
            --right {input.right} --right_step {params.right_step} \
            --output {output} \
            --speed {params.speed}
        """
