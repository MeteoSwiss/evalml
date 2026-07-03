# ----------------------------------------------------- #
# PLOTTING WORKFLOW                                     #
# ----------------------------------------------------- #


include: "common.smk"


import pandas as pd


def _get_available_baselines(wc) -> list[dict[str, str]]:
    """Get all available baseline datasets for the given init time."""
    baselines = []
    for baseline_id in BASELINE_CONFIGS:
        root = BASELINE_CONFIGS[baseline_id].get("root")
        steps = BASELINE_CONFIGS[baseline_id].get("steps")
        label = BASELINE_CONFIGS[baseline_id].get("label", baseline_id)
        if root and Path(root).exists():
            baselines.append({"root": root, "steps": steps, "label": label})
    if not baselines:
        raise ValueError(f"No baseline data found for init time {wc.init_time}")
    return baselines


rule plot_meteogram:
    input:
        script="workflow/scripts/plot_meteogram.py",
        inference_okfile=rules.inference_execute.output.okfile,
        truth_dep=truth_file_dep,
    output:
        expand(
            OUT_ROOT
            / "results/{{showcase}}/{{run_id}}/{{init_time}}/{{init_time}}_{{param}}_{sta}.png",
            sta=config["showcase"]["meteograms"]["stations"],
        ),
    log:
        OUT_ROOT / "logs/{showcase}/{run_id}/{init_time}/plot_meteogram_{param}.log",
    resources:
        slurm_partition="postproc",
        cpus_per_task=1,
        runtime="60m",
    params:
        ana_label=lambda wc: config["truth"]["label"],
        truth_root=config["truth"]["root"],
        fcst_grib=lambda wc: (
            Path(OUT_ROOT) / f"data/runs/{wc.run_id}/{wc.init_time}/grib"
        ).resolve(),
        fcst_steps=lambda wc: RUN_CONFIGS[wc.run_id]["steps"],
        fcst_label=lambda wc: RUN_CONFIGS[wc.run_id]["label"],
        baseline_roots=lambda wc: [x["root"] for x in _get_available_baselines(wc)],
        baseline_steps=lambda wc: [x["steps"] for x in _get_available_baselines(wc)],
        baseline_labels=lambda wc: [x["label"] for x in _get_available_baselines(wc)],
        outdir=lambda wc: str(
            (
                Path(OUT_ROOT) / f"results/{wc.showcase}/{wc.run_id}/{wc.init_time}"
            ).resolve()
        ),
        stations=config["showcase"]["meteograms"]["stations"],
    shell:
        """
        set -euo pipefail
        export ECCODES_DEFINITION_PATH=$(realpath .venv/share/eccodes-cosmo-resources/definitions)

        BASELINE_ROOTS=({params.baseline_roots:q})
        BASELINE_STEPS=({params.baseline_steps:q})
        BASELINE_LABELS=({params.baseline_labels:q})

        CMD_ARGS=(
            --forecast {params.fcst_grib:q}
            --forecast_steps {params.fcst_steps:q}
            --forecast_label {params.fcst_label:q}
            --analysis {params.truth_root:q}
            --analysis_label {params.ana_label:q}
            --date {wildcards.init_time:q}
            --outdir {params.outdir:q}
            --param {wildcards.param:q}
            --stations {params.stations:q}
        )

        for i in "${{!BASELINE_ROOTS[@]}}"; do
            CMD_ARGS+=(--baseline "${{BASELINE_ROOTS[$i]}}")
            CMD_ARGS+=(--baseline_steps "${{BASELINE_STEPS[$i]}}")
            CMD_ARGS+=(--baseline_label "${{BASELINE_LABELS[$i]}}")
        done

        python {input.script} "${{CMD_ARGS[@]}}" >{log} 2>&1
        """


rule plot_forecast_frame:
    input:
        script="workflow/scripts/plot_forecast_frame.py",
        inference_okfile=rules.inference_execute.output.okfile,
    output:
        expand(
            OUT_ROOT
            / "data/runs/{{run_id}}/{{init_time}}/frames/frame_{{leadtime}}_{{param}}_{region}.png",
            region=list(SHOWCASE_REGIONS.keys()),
        ),
    log:
        OUT_ROOT
        / "logs/{run_id}/{init_time}/plot_forecast_frame_{leadtime}_{param}.log",
    wildcard_constraints:
        leadtime=r"\d+",  # only digits
    resources:
        slurm_partition="postproc",
        cpus_per_task=4,
        runtime="20m",
    params:
        grib_out_dir=lambda wc: str(
            (Path(OUT_ROOT) / f"data/runs/{wc.run_id}/{wc.init_time}/grib").resolve()
        ),
        regions_json=json.dumps(SHOWCASE_REGIONS),
        outdir=lambda wc: str(
            (Path(OUT_ROOT) / f"data/runs/{wc.run_id}/{wc.init_time}/frames").resolve()
        ),
        accu=lambda wc: int(RUN_CONFIGS[wc.run_id]["steps"].split("/")[2]),
    shell:
        """
        export ECCODES_DEFINITION_PATH=$(realpath .venv/share/eccodes-cosmo-resources/definitions)
        python {input.script} \
            --input {params.grib_out_dir:q} --date {wildcards.init_time:q} \
            --param {wildcards.param:q} --leadtime {wildcards.leadtime:q} \
            --regions_json {params.regions_json:q} \
            --outdir {params.outdir:q} \
            --accu {params.accu} >{log} 2>&1
        """


def get_leadtimes(wc):
    """Get all lead times the run produces (accumulated params skip lead 0)."""
    leadtimes = resolve_leadtimes(RUN_CONFIGS[wc.run_id]["steps"], param=wc.param)
    return [str(lt) for lt in leadtimes]


rule make_forecast_animation:
    input:
        lambda wc: expand(
            OUT_ROOT
            / "data/runs/{run_id}/{init_time}/frames/frame_{leadtime}_{param}_{region}.png",
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
        delay=lambda wc: 10 * int(RUN_CONFIGS[wc.run_id]["steps"].split("/")[2]),
    shell:
        """
        FRAMES=$(for f in {input}; do [ -s "$f" ] && echo "$f"; done | tr '\\n' ' ')
        convert -delay {params.delay} -loop 0 $FRAMES {output}
        """


rule plot_scoremaps:
    # localrule: True
    input:
        script="workflow/scripts/plot_scoremaps.mo.py",
        verif_file=OUT_ROOT
        / f"data/runs/{{run_id}}/scoremaps/{{param}}_{{leadtime}}_{TRUTH_HASH}.nc",
    output:
        OUT_ROOT
        / "results/{experiment}/scoremaps/runs/{run_id}/{param}_{score}_{region}_{season}_{init_hour}_{leadtime}.png",
    log:
        OUT_ROOT
        / "logs/plot_scoremaps/{experiment}/{run_id}-{param}-{score}-{region}-{season}-{init_hour}-{leadtime}.log",
    wildcard_constraints:
        leadtime=r"\d+",  # only digits
        init_hour=r"all|\d{1,2}",
    resources:
        slurm_partition="postproc",
        cpus_per_task=1,
        runtime="10m",
    shell:
        """
        export ECCODES_DEFINITION_PATH=$(realpath .venv/share/eccodes-cosmo-resources/definitions)
        uv run python {input.script} \
            --input {input.verif_file} --outfn {output[0]} --region {wildcards.region} \
            --param {wildcards.param} --leadtime {wildcards.leadtime} --score {wildcards.score} \
            --season {wildcards.season} --init_hour {wildcards.init_hour} >{log} 2>&1
        # interactive editing (needs to set localrule: True and use only one core)
        # marimo edit {input.script} -- \
        #     --input {input.verif_file} --outfn {output[0]} --region {wildcards.region} \
        #     --param {wildcards.param} --leadtime {wildcards.leadtime} --score {wildcards.score} \
        #     --season {wildcards.season} --init_hour {wildcards.init_hour}
        """


use rule plot_scoremaps as plot_scoremaps_baseline with:
    input:
        script="workflow/scripts/plot_scoremaps.mo.py",
        verif_file=OUT_ROOT
        / f"data/baselines/{{baseline_id}}/scoremaps/{{param}}_{{leadtime}}_{TRUTH_HASH}.nc",
    output:
        OUT_ROOT
        / "results/{experiment}/scoremaps/baselines/{baseline_id}/{param}_{score}_{region}_{season}_{init_hour}_{leadtime}.png",
    log:
        OUT_ROOT
        / "logs/plot_scoremaps/{experiment}/{baseline_id}-{param}-{score}-{region}-{season}-{init_hour}-{leadtime}.log",
