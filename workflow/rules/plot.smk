# ----------------------------------------------------- #
# PLOTTING WORKFLOW                                     #
# ----------------------------------------------------- #


include: "common.smk"


import pandas as pd


# Region names and modes are finite — constrain to disambiguate
# `{param}_{region}_{mode}` splits in animation output filenames.
wildcard_constraints:
    region=r"(globe|europe|centraleurope|switzerland)",
    mode=r"(forecast|truth|error)",


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


def _frame_inputs(wc):
    """Per-mode inputs for `plot_frame`: forecast needs inference output,
    truth needs the truth zarr, error needs both."""
    inputs = {"script": "workflow/scripts/plot_frame.mo.py"}
    if wc.mode in ("forecast", "error"):
        inputs["inference_okfile"] = expand(
            rules.execute_inference.output.okfile,
            run_id=wc.run_id,
            init_time=wc.init_time,
        )
    if wc.mode in ("truth", "error"):
        inputs["truth"] = config["truth"]["root"]
    return inputs


rule plot_frame:
    input:
        unpack(_frame_inputs),
    output:
        OUT_ROOT
        / "data/runs/{run_id}/{init_time}/{mode}_frames/frame_{leadtime}_{param}_{region}.png",
    wildcard_constraints:
        leadtime=r"\d+",
    resources:
        slurm_partition="postproc",
        cpus_per_task=1,
        runtime="10m",
    params:
        grib_out_dir=lambda wc: (
            Path(OUT_ROOT) / f"data/runs/{wc.run_id}/{wc.init_time}/grib"
        ).resolve(),
        truth_path=lambda wc, input: input.get("truth", ""),
    shell:
        """
        export ECCODES_DEFINITION_PATH=$(realpath .venv/share/eccodes-cosmo-resources/definitions)
        CMD_ARGS=(
            --mode {wildcards.mode}
            --date {wildcards.init_time}
            --outfn {output[0]}
            --param {wildcards.param}
            --leadtime {wildcards.leadtime}
            --region {wildcards.region}
        )
        case "{wildcards.mode}" in
            forecast|error) CMD_ARGS+=(--forecast {params.grib_out_dir}) ;;
        esac
        case "{wildcards.mode}" in
            truth|error) CMD_ARGS+=(--truth {params.truth_path}) ;;
        esac
        python {input.script} "${{CMD_ARGS[@]}}"
        """


def get_leadtimes(wc):
    """Get all lead times from the run config."""
    start, end, step = map(int, RUN_CONFIGS[wc.run_id]["steps"].split("/"))
    # skip lead time 0 for diagnostic variables
    if wc.param in ["tp", "TOT_PREC"]:
        start += step
    return [f"{i:03}" for i in range(start, end + 1, step)]


rule make_animation:
    localrule: True
    input:
        expand(
            rules.plot_frame.output,
            leadtime=lambda wc: get_leadtimes(wc),
            allow_missing=True,
        ),
    output:
        OUT_ROOT
        / "results/{showcase}/{run_id}/{init_time}/{init_time}_{param}_{region}_{mode}.gif",
    params:
        delay=lambda wc: 10 * int(RUN_CONFIGS[wc.run_id]["steps"].split("/")[2]),
    shell:
        """
        convert -delay {params.delay} -loop 0 {input} {output}
        """
