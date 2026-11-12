# ----------------------------------------------------- #
# VERIFICATION WORKFLOW                                 #
# ----------------------------------------------------- #
from datetime import datetime
from types import SimpleNamespace

import pandas as pd


include: "common.smk"


def valid_masks_flag(wildcards, input):
    if not input.valid_masks:
        return ""
    return "--valid_masks " + " ".join(input.valid_masks)


# Where a participant has a value for the experiment.score_on_common_points parameters.
# The verification of every participant uses the masks of all participants.
rule verification_valid_mask_baseline:
    input:
        "src/verification/valid_mask.py",
        "src/data_input/__init__.py",
        script="workflow/scripts/verification_valid_mask.py",
        forecast=lambda wc: BASELINE_CONFIGS[wc.baseline_id]["root"],
        truth_dep=truth_file_dep,
    output:
        OUT_ROOT
        / f"data/baselines/{{baseline_id}}/{{init_time}}/valid_mask_{VALID_MASK_HASH}.nc",
    log:
        OUT_ROOT / "logs/verification_valid_mask_baseline/{baseline_id}-{init_time}.log",
    resources:
        cpus_per_task=balfrin_cpus(40_000, actual_cpus=8),
        mem_mb=40_000,
        runtime="30m",
    params:
        baseline_steps=lambda wc: BASELINE_CONFIGS[wc.baseline_id]["steps"],
        member=lambda wc: BASELINE_CONFIGS[wc.baseline_id].get("member", "000"),
        truth=config["truth"]["root"],
        mask_params=",".join(COMMON_POINTS_PARAMS),
    shell:
        """
        export ECCODES_DEFINITION_PATH=$(realpath .venv/share/eccodes-cosmo-resources/definitions)
        uv run {input.script} \
            --forecast {input.forecast} \
            --truth "{params.truth}" \
            --reftime {wildcards.init_time} \
            --steps "{params.baseline_steps}" \
            --params "{params.mask_params}" \
            --member "{params.member}" \
            --output {output} >{log} 2>&1
        """


rule verification_valid_mask:
    input:
        "src/verification/valid_mask.py",
        "src/data_input/__init__.py",
        script="workflow/scripts/verification_valid_mask.py",
        inference_okfile=rules.inference_execute.output.okfile,
        truth_dep=truth_file_dep,
    output:
        OUT_ROOT / f"data/runs/{{run_id}}/{{init_time}}/valid_mask_{VALID_MASK_HASH}.nc",
    log:
        OUT_ROOT / "logs/verification_valid_mask/{run_id}-{init_time}.log",
    resources:
        cpus_per_task=balfrin_cpus(40_000, actual_cpus=8),
        mem_mb=40_000,
        runtime="30m",
    params:
        fcst_steps=lambda wc: RUN_CONFIGS[wc.run_id]["steps"],
        truth=config["truth"]["root"],
        grib_out_dir=lambda wc: (
            Path(OUT_ROOT) / f"data/runs/{wc.run_id}/{wc.init_time}/grib"
        ).resolve(),
        mask_params=",".join(COMMON_POINTS_PARAMS),
    shell:
        """
        export ECCODES_DEFINITION_PATH=$(realpath .venv/share/eccodes-cosmo-resources/definitions)
        uv run {input.script} \
            --forecast {params.grib_out_dir} \
            --truth "{params.truth}" \
            --reftime {wildcards.init_time} \
            --steps "{params.fcst_steps}" \
            --params "{params.mask_params}" \
            --output {output} >{log} 2>&1
        """


rule verification_metrics_baseline:
    input:
        "src/verification/__init__.py",
        "src/verification/spatial.py",
        "src/data_input/__init__.py",
        script="workflow/scripts/verification_metrics.py",
        forecast=lambda wc: BASELINE_CONFIGS[wc.baseline_id]["root"],
        truth_dep=truth_file_dep,
        holdout=holdout_dep,
        valid_masks=participant_valid_mask_files,
    output:
        OUT_ROOT / f"data/baselines/{{baseline_id}}/{{init_time}}/verif_{VERIF_HASH}.nc",
    log:
        OUT_ROOT / "logs/verification_metrics_baseline/{baseline_id}-{init_time}.log",
    resources:
        cpus_per_task=balfrin_cpus(80_000, actual_cpus=24),
        mem_mb=80_000,
        runtime="120m",
    params:
        baseline_steps=lambda wc: BASELINE_CONFIGS[wc.baseline_id]["steps"],
        member=lambda wc: BASELINE_CONFIGS[wc.baseline_id].get("member", "000"),
        truth=(config.get("truth") or {}).get("root", ""),
        truth_source_id=f"truth-{TRUTH_HASH}",
        regions=REGIONS,
        experiment_params=",".join(EXPERIMENT_PARAMS),
        threshold_dict=EXPERIMENT_CONFIG.get("thresholds", {}),
        holdout_flag=lambda wc, input: (
            f"--holdout_stations_file {input.holdout}" if input.holdout else ""
        ),
        lapse_rate_flag=(
            "--lapse_rate_correction"
            if config.get("lapse_rate_correction", True)
            else ""
        ),
        valid_masks_flag=valid_masks_flag,
    shell:
        """
        export ECCODES_DEFINITION_PATH=$(realpath .venv/share/eccodes-cosmo-resources/definitions)
        uv run {input.script} \
            --forecast {input.forecast} \
            --truth "{params.truth}" \
            --reftime {wildcards.init_time} \
            --steps "{params.baseline_steps}" \
            --source_id "{wildcards.baseline_id}" \
            --truth_source_id "{params.truth_source_id}" \
            --regions '{params.regions}' \
            --params "{params.experiment_params}" \
            --threshold_dict "{params.threshold_dict}" \
            {params.holdout_flag} \
            --member "{params.member}" \
            {params.lapse_rate_flag} \
            {params.valid_masks_flag} \
            --output {output} >{log} 2>&1
        """


def _get_no_none(dict, key, replacement):
    out = dict.get(key, replacement)
    if out is None:
        out = replacement
    return out


rule verification_metrics:
    input:
        "src/verification/__init__.py",
        "src/verification/spatial.py",
        "src/data_input/__init__.py",
        script="workflow/scripts/verification_metrics.py",
        inference_okfile=_get_inference_okfile,
        truth_dep=truth_file_dep,
        holdout=holdout_dep,
        valid_masks=participant_valid_mask_files,
    output:
        OUT_ROOT / f"data/runs/{{run_id}}/{{init_time}}/verif_{VERIF_HASH}.nc",
    log:
        OUT_ROOT / "logs/verification_metrics/{run_id}-{init_time}.log",
    resources:
        cpus_per_task=balfrin_cpus(80_000, actual_cpus=24),
        mem_mb=80_000,
        runtime="60m",
    # wildcard_constraints:
    # run_id="^" # to avoid ambiguitiy with run_baseline_verif
    # TODO: implement logic to use experiment name instead of run_id as wildcard
    params:
        fcst_steps=lambda wc: RUN_CONFIGS[wc.run_id]["steps"],
        truth=(config.get("truth") or {}).get("root", ""),
        truth_source_id=f"truth-{TRUTH_HASH}",
        regions=REGIONS,
        grib_out_dir=lambda wc: (
            Path(OUT_ROOT) / f"data/runs/{wc.run_id}/{wc.init_time}/grib"
        ).resolve(),
        experiment_params=",".join(EXPERIMENT_PARAMS),
        threshold_dict=EXPERIMENT_CONFIG.get("thresholds", {}),
        holdout_flag=lambda wc, input: (
            f"--holdout_stations_file {input.holdout}" if input.holdout else ""
        ),
        lapse_rate_flag=(
            "--lapse_rate_correction"
            if config.get("lapse_rate_correction", True)
            else ""
        ),
        fdb_configured=lambda wc: RUN_CONFIGS[wc.run_id]["output"] == "fdb",
        fdb_forecast=lambda wc: (
            "fdb:" + _resolve_fdb_root_for_run(wc.run_id)
            if RUN_CONFIGS[wc.run_id]["output"] == "fdb"
            else ""
        ),
        valid_masks_flag=valid_masks_flag,
    shell:
        """
        set -euo pipefail

        # See plot_meteogram (plot.smk) for why this always wins over the
        # (nonexistent) local grib dir when the run declares output: fdb, and
        # why no squashfs-mount is needed to read it (pyfdb bundles its own
        # native libs).
        FORECAST_SRC={params.grib_out_dir:q}
        if [ "{params.fdb_configured}" = "True" ]; then
            FORECAST_SRC={params.fdb_forecast:q}
            export ECCODES_DEFINITION_PATH="$(realpath eccodes-cosmo-mars/definitions):$(realpath .venv/share/eccodes-cosmo-resources/definitions)"
        else
            export ECCODES_DEFINITION_PATH=$(realpath .venv/share/eccodes-cosmo-resources/definitions)
        fi

        CMD_ARGS=(
            --forecast "$FORECAST_SRC"
            --truth {params.truth:q}
            --reftime {wildcards.init_time:q}
            --steps {params.fcst_steps:q}
            --source_id {wildcards.run_id:q}
            --truth_source_id {params.truth_source_id:q}
            --regions {params.regions:q}
            --params {params.experiment_params:q}
            --threshold_dict {params.threshold_dict:q}
            {params.holdout_flag}
            {params.lapse_rate_flag}
            {params.valid_masks_flag}
            --output {output:q}
        )

        uv run python {input.script} "${{CMD_ARGS[@]}}" >{log} 2>&1
        """


def _restrict_reftimes_to_hours(reftimes, hours=None):
    """Restrict the reference times to specific hours."""
    if hours is None:
        return [t.strftime("%Y%m%d%H%M") for t in reftimes]
    return [t.strftime("%Y%m%d%H%M") for t in reftimes if t.hour in hours]


rule verification_metrics_aggregation:
    input:
        script="workflow/scripts/verification_aggregation.py",
        verif_nc=lambda wc: expand(
            rules.verification_metrics.output,
            init_time=_restrict_reftimes_to_hours(REFTIMES),
            allow_missing=True,
        ),
    output:
        OUT_ROOT / f"data/runs/{{run_id}}/verif_aggregated_{VERIF_HASH}.nc",
    log:
        OUT_ROOT / "logs/verification_metrics_aggregation/{run_id}.log",
    resources:
        cpus_per_task=balfrin_cpus(250_000, actual_cpus=24),
        mem_mb=250_000,
        runtime="2h",
    shell:
        """
        uv run {input.script} {input.verif_nc} --output {output} >{log} 2>&1
        """


use rule verification_metrics_aggregation as verification_metrics_aggregation_baseline with:
    input:
        script="workflow/scripts/verification_aggregation.py",
        verif_nc=lambda wc: expand(
            rules.verification_metrics_baseline.output,
            init_time=_restrict_reftimes_to_hours(REFTIMES),
            allow_missing=True,
        ),
    output:
        OUT_ROOT / f"data/baselines/{{baseline_id}}/verif_aggregated_{VERIF_HASH}.nc",
    log:
        OUT_ROOT / "logs/verification_metrics_aggregation_baseline/{baseline_id}.log",


rule verification_metrics_plot:
    input:
        "src/verification/__init__.py",
        script="workflow/scripts/verification_plot_metrics.py",
        verif=list(EXPERIMENT_PARTICIPANTS.values()),
    output:
        report(
            directory(OUT_ROOT / "results/{experiment}/plots"),
            patterns=["{name}.png"],
        ),
    log:
        OUT_ROOT / "logs/verification_metrics_plot/{experiment}.log",
    resources:
        cpus_per_task=balfrin_cpus(50_000, actual_cpus=16),
        mem_mb=50_000,
        runtime="1h",
    params:
        label_map=",".join(
            "{}:{}".format(
                sid,
                (
                    BASELINE_CONFIGS[sid].get("label", sid)
                    if sid in BASELINE_CONFIGS
                    else RUN_CONFIGS[sid].get("label", sid)
                ),
            )
            for sid in EXPERIMENT_PARTICIPANTS
        )
        + ",truth-{}:{}".format(
            TRUTH_HASH, (config.get("truth") or {}).get("label", "")
        ),
    shell:
        """
        uv run {input.script} {input.verif} --output_dir {output} --labels "{params.label_map}" >{log} 2>&1
        """


rule verification_scoremaps:
    input:
        "src/verification/__init__.py",
        "src/data_input/__init__.py",
        script="workflow/scripts/verification_scoremaps.py",
        inference_okfiles=lambda wc: [
            _get_inference_okfile(SimpleNamespace(run_id=wc.run_id, init_time=it))
            for it in _restrict_reftimes_to_hours(REFTIMES)
        ],
        truth_dep=truth_file_dep,
    output:
        OUT_ROOT
        / f"data/runs/{{run_id}}/scoremaps/{{param}}_{{leadtime}}_{TRUTH_HASH}.nc",
    log:
        OUT_ROOT
        / f"logs/verification_scoremaps/{{run_id}}-{TRUTH_HASH}-{{param}}-{{leadtime}}.log",
    resources:
        cpus_per_task=balfrin_cpus(50_000, actual_cpus=2),
        mem_mb=50_000,
        runtime=scoremap_runtime,
    # wildcard_constraints:
    # run_id="^" # to avoid ambiguitiy with run_baseline_verif
    # TODO: implement logic to use experiment name instead of run_id as wildcard
    params:
        fcst_label=lambda wc: RUN_CONFIGS[wc.run_id].get("label"),
        fcst_steps=lambda wc: RUN_CONFIGS[wc.run_id]["steps"],
        truth=config["truth"]["root"],
        truth_label=config["truth"]["label"],
        reftimes=[t.strftime("%Y%m%d%H%M") for t in REFTIMES],
        run_root=lambda wc: (Path(OUT_ROOT) / f"data/runs/{wc.run_id}").resolve(),
        fdb_configured=lambda wc: RUN_CONFIGS[wc.run_id]["output"] == "fdb",
        # Bare root (no "fdb:" marker prefix): --fdb_root is consumed by
        # iter_init_dirs, which adds the marker itself when constructing
        # load_forecast_data's root argument for each reftime -- unlike
        # plot.smk/verification_metrics' single --forecast argument, which IS
        # load_forecast_data's root directly. A run's declared `output` is the
        # same for every reftime under its run_root (it's a property of the run
        # config, not of any one reftime), so this is the one root iter_init_dirs
        # uses for all of them -- it never needs to guess per-directory.
        fdb_root=lambda wc: (
            _resolve_fdb_root_for_run(wc.run_id)
            if RUN_CONFIGS[wc.run_id]["output"] == "fdb"
            else ""
        ),
    shell:
        """
        set -euo pipefail

        # See plot_meteogram (plot.smk) for why no squashfs-mount is needed to
        # read from FDB (pyfdb bundles its own native libs).
        if [ "{params.fdb_configured}" = "True" ]; then
            export ECCODES_DEFINITION_PATH="$(realpath eccodes-cosmo-mars/definitions):$(realpath .venv/share/eccodes-cosmo-resources/definitions)"
        else
            export ECCODES_DEFINITION_PATH=$(realpath .venv/share/eccodes-cosmo-resources/definitions)
        fi

        # --reftimes is nargs="+" in argparse: pass each timestamp as its own
        # array element (not one space-joined string) so it fans out correctly.
        REFTIMES_ARR=({params.reftimes:q})

        CMD_ARGS=(
            --run_root {params.run_root:q}
            --reftimes "${{REFTIMES_ARR[@]}}"
            --truth {params.truth:q}
            --step {wildcards.leadtime:q}
            --steps {params.fcst_steps:q}
            --param {wildcards.param:q}
            --output {output:q}
        )
        if [ -n "{params.fdb_root}" ]; then
            CMD_ARGS+=(--fdb_root {params.fdb_root:q})
        fi

        uv run python {input.script} "${{CMD_ARGS[@]}}" >{log} 2>&1
        """


rule verification_scoremaps_baseline:
    input:
        "src/verification/__init__.py",
        "src/data_input/__init__.py",
        script="workflow/scripts/verification_scoremaps.py",
        forecast=lambda wc: BASELINE_CONFIGS[wc.baseline_id]["root"],
        truth_dep=truth_file_dep,
    output:
        OUT_ROOT
        / f"data/baselines/{{baseline_id}}/scoremaps/{{param}}_{{leadtime}}_{TRUTH_HASH}.nc",
    log:
        OUT_ROOT
        / f"logs/verification_scoremaps_baseline/{{baseline_id}}-{TRUTH_HASH}-{{param}}-{{leadtime}}.log",
    resources:
        cpus_per_task=balfrin_cpus(50_000, actual_cpus=24),
        mem_mb=50_000,
        runtime=scoremap_runtime,
    params:
        baseline_steps=lambda wc: BASELINE_CONFIGS[wc.baseline_id]["steps"],
        member=lambda wc: BASELINE_CONFIGS[wc.baseline_id].get("member", "000"),
        reftimes=" ".join(t.strftime("%Y%m%d%H%M") for t in REFTIMES),
        truth=config["truth"]["root"],
    shell:
        """
        export ECCODES_DEFINITION_PATH=$(realpath .venv/share/eccodes-cosmo-resources/definitions)
        uv run {input.script} \
            --baseline_root {input.forecast} \
            --reftimes {params.reftimes} \
            --truth "{params.truth}" \
            --step {wildcards.leadtime} \
            --steps "{params.baseline_steps}" \
            --param {wildcards.param} \
            --member "{params.member}" \
            --output {output} >{log} 2>&1
        """


# The SAL scoring raster is experiment-level (not per-participant); pass it from
# the validated config, which fills in the SalConfig defaults. Absent when there
# is no sal block at all, in which case these rules never fire and the script's
# own defaults would apply anyway.
_SAL_CONFIG = EXPERIMENT_CONFIG.get("sal") or {}
_SAL_ARGS = (
    f"--grid-extent {' '.join(str(x) for x in _SAL_CONFIG['grid_extent'])} "
    f"--grid-step-lat {_SAL_CONFIG['grid_step_lat']} "
    f"--grid-step-lon {_SAL_CONFIG['grid_step_lon']}"
    if _SAL_CONFIG
    else ""
)


rule verification_sal:
    input:
        "src/verification/__init__.py",
        "src/verification/spatial.py",
        "src/data_input/__init__.py",
        script="workflow/scripts/verification_sal.py",
        inference_okfiles=lambda wc: expand(
            rules.inference_execute.output.okfile,
            init_time=_restrict_reftimes_to_hours(REFTIMES),
            allow_missing=True,
        ),
        truth=config["truth"]["root"],
    output:
        OUT_ROOT / f"data/runs/{{run_id}}/sal/{{param}}_{{leadtime}}_{TRUTH_HASH}.csv",
    log:
        OUT_ROOT
        / f"logs/verification_sal/{{run_id}}-{TRUTH_HASH}-{{param}}-{{leadtime}}.log",
    resources:
        cpus_per_task=2,
        mem_mb=4_000,
        runtime="60m",
    params:
        reftimes=" ".join(t.strftime("%Y%m%d%H%M") for t in REFTIMES),
        run_root=lambda wc: (Path(OUT_ROOT) / f"data/runs/{wc.run_id}").resolve(),
        sal_args=_SAL_ARGS,
    shell:
        """
        export ECCODES_DEFINITION_PATH=$(realpath .venv/share/eccodes-cosmo-resources/definitions)
        uv run {input.script} \
            --run_root {params.run_root} \
            --reftimes {params.reftimes} \
            --truth {input.truth} \
            --step {wildcards.leadtime} \
            --param {wildcards.param} \
            {params.sal_args} \
            --output {output} >{log} 2>&1
        """


rule verification_sal_baseline:
    input:
        "src/verification/__init__.py",
        "src/verification/spatial.py",
        "src/data_input/__init__.py",
        script="workflow/scripts/verification_sal.py",
        forecast=lambda wc: BASELINE_CONFIGS[wc.baseline_id]["root"],
        truth=config["truth"]["root"],
    output:
        OUT_ROOT
        / f"data/baselines/{{baseline_id}}/sal/{{param}}_{{leadtime}}_{TRUTH_HASH}.csv",
    log:
        OUT_ROOT
        / f"logs/verification_sal_baseline/{{baseline_id}}-{TRUTH_HASH}-{{param}}-{{leadtime}}.log",
    resources:
        cpus_per_task=2,
        mem_mb=4_000,
        runtime="60m",
    params:
        member=lambda wc: BASELINE_CONFIGS[wc.baseline_id].get("member", "000"),
        reftimes=" ".join(t.strftime("%Y%m%d%H%M") for t in REFTIMES),
        sal_args=_SAL_ARGS,
    shell:
        """
        export ECCODES_DEFINITION_PATH=$(realpath .venv/share/eccodes-cosmo-resources/definitions)
        uv run {input.script} \
            --baseline_root {input.forecast} \
            --reftimes {params.reftimes} \
            --truth {input.truth} \
            --step {wildcards.leadtime} \
            --param {wildcards.param} \
            --member "{params.member}" \
            {params.sal_args} \
            --output {output} >{log} 2>&1
        """
