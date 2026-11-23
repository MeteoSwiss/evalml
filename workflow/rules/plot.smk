# ----------------------------------------------------- #
# PLOTTING WORKFLOW                                     #
# ----------------------------------------------------- #


include: "common.smk"


import shlex

import pandas as pd


def _get_available_baselines(wc) -> list[dict[str, str]]:
    """Get all available baseline datasets for the given init time."""
    baselines = []
    if not BASELINE_CONFIGS:
        return baselines
    for baseline_id in BASELINE_CONFIGS:
        root = BASELINE_CONFIGS[baseline_id].get("root")
        steps = BASELINE_CONFIGS[baseline_id].get("steps")
        label = BASELINE_CONFIGS[baseline_id].get("label", baseline_id)
        if root and Path(root).exists():
            baselines.append({"root": root, "steps": steps, "label": label})
    if not baselines:
        raise ValueError(f"No baseline data found for init time {wc.init_time}")
    return baselines


def _meteogram_outputs() -> list[str]:
    """One meteogram per param and station, plus the assembled panel if enabled."""
    meteograms = config["showcase"]["meteograms"]
    outdir = OUT_ROOT / "results/{showcase}/{init_time}/meteograms"
    outputs = [
        str(outdir / f"{{init_time}}_{param}_{sta}.png")
        for param in SHOWCASE_CONFIG["params"]
        for sta in meteograms["stations"]
    ]
    if meteograms.get("panel", True):
        outputs.append(str(outdir / "{init_time}_meteogram_panel.png"))
    return outputs


rule plot_meteogram:
    """Meteograms of all candidate runs overlaid, one per param and station.

With ``showcase.meteograms.panel`` the same meteograms are also assembled in one
figure (rows: params, columns: stations).
"""
    input:
        script="workflow/scripts/plot_meteogram.py",
        inference_okfiles=lambda wc: expand(
            rules.inference_execute.output.okfile,
            run_id=CANDIDATES,
            init_time=wc.init_time,
        ),
        truth_dep=truth_file_dep,
    output:
        _meteogram_outputs(),
    log:
        OUT_ROOT / "logs/{showcase}/meteograms/{init_time}/plot_meteogram.log",
    wildcard_constraints:
        showcase=r"[^/]+",
        init_time=r"\d{12}",
    resources:
        slurm_partition="postproc",
        cpus_per_task=1,
        runtime="60m",
    params:
        ana_label=lambda wc: config["truth"]["label"],
        truth_root=config["truth"]["root"],
        # Ensembles: the grib/ root with one sub-directory per member (all plotted).
        fcst_gribs=lambda wc: [
            str(run_grib_root(run_id, wc.init_time).resolve()) for run_id in CANDIDATES
        ],
        fcst_members=lambda wc: [ensemble_members(run_id) or 0 for run_id in CANDIDATES],
        fcst_steps=lambda wc: [RUN_CONFIGS[run_id]["steps"] for run_id in CANDIDATES],
        fcst_labels=lambda wc: [RUN_CONFIGS[run_id]["label"] for run_id in CANDIDATES],
        baseline_roots=lambda wc: [x["root"] for x in _get_available_baselines(wc)],
        baseline_steps=lambda wc: [x["steps"] for x in _get_available_baselines(wc)],
        baseline_labels=lambda wc: [x["label"] for x in _get_available_baselines(wc)],
        outdir=lambda wc: str(
            (
                Path(OUT_ROOT) / f"results/{wc.showcase}/{wc.init_time}/meteograms"
            ).resolve()
        ),
        stations=config["showcase"]["meteograms"]["stations"],
        params=SHOWCASE_CONFIG["params"],
        reference_flag=lambda wc: (
            f"--reference {shlex.quote(ref['root'])} "
            f"--reference_label {shlex.quote(ref['label'])}"
            if (ref := config["showcase"]["meteograms"].get("reference"))
            else ""
        ),
        panel_flag=lambda wc, output: (
            f"--panel {output[-1]}"
            if config["showcase"]["meteograms"].get("panel", True)
            else ""
        ),
        lapse_rate_flag=(
            "--lapse_rate_correction"
            if config.get("lapse_rate_correction", True)
            else ""
        ),
    shell:
        """
        set -euo pipefail
        export ECCODES_DEFINITION_PATH=$(realpath .venv/share/eccodes-cosmo-resources/definitions)

        FCST_GRIBS=({params.fcst_gribs:q})
        FCST_STEPS=({params.fcst_steps:q})
        FCST_LABELS=({params.fcst_labels:q})
        FCST_MEMBERS=({params.fcst_members:q})
        BASELINE_ROOTS=({params.baseline_roots:q})
        BASELINE_STEPS=({params.baseline_steps:q})
        BASELINE_LABELS=({params.baseline_labels:q})

        CMD_ARGS=(
            --analysis {params.truth_root:q}
            --analysis_label {params.ana_label:q}
            --date {wildcards.init_time:q}
            --outdir {params.outdir:q}
            --param {params.params:q}
            --stations {params.stations:q}
            {params.lapse_rate_flag}
            {params.panel_flag}
            {params.reference_flag}
        )

        for i in "${{!FCST_GRIBS[@]}}"; do
            CMD_ARGS+=(--forecast "${{FCST_GRIBS[$i]}}")
            CMD_ARGS+=(--forecast_steps "${{FCST_STEPS[$i]}}")
            CMD_ARGS+=(--forecast_label "${{FCST_LABELS[$i]}}")
            CMD_ARGS+=(--forecast_members "${{FCST_MEMBERS[$i]}}")
        done

        for i in "${{!BASELINE_ROOTS[@]}}"; do
            CMD_ARGS+=(--baseline "${{BASELINE_ROOTS[$i]}}")
            CMD_ARGS+=(--baseline_steps "${{BASELINE_STEPS[$i]}}")
            CMD_ARGS+=(--baseline_label "${{BASELINE_LABELS[$i]}}")
        done

        uv run python {input.script} "${{CMD_ARGS[@]}}" >{log} 2>&1
        """


rule plot_forecast_frame:
    input:
        script="workflow/scripts/plot_forecast_frame.py",
        inference_okfile=rules.inference_execute.output.okfile,
    output:
        expand(
            OUT_ROOT
            / "data/runs/{{run_id}}/{{init_time}}/frames/frame_{{leadtime}}_{{param}}_{region}.png",
            region=list(SHOWCASE_CONFIG["regions"].keys()),
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
        grib_out_dir=lambda wc: str(run_grib_dir(wc.run_id, wc.init_time).resolve()),
        regions_json=json.dumps(SHOWCASE_CONFIG["regions"]),
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
        param="|".join(map(re.escape, SHOWCASE_CONFIG["params"])),
        region="|".join(map(re.escape, SHOWCASE_CONFIG["regions"].keys())),
    localrule: True
    run:
        from PIL import Image

        frames = [
            Image.open(f).convert("RGB").quantize()
            for f in input
            if Path(f).stat().st_size > 0
        ]
        frames[0].save(
            output[0],
            save_all=True,
            append_images=frames[1:],
            duration=1000 / SHOWCASE_CONFIG["fps"],
            loop=0,
        )


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
        runtime="20m",
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
