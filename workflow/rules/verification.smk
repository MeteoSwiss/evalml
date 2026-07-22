# ----------------------------------------------------- #
# VERIFICATION WORKFLOW                                 #
# ----------------------------------------------------- #
from datetime import datetime

import pandas as pd


include: "common.smk"


rule verification_metrics_baseline:
    input:
        "src/verification/__init__.py",
        "src/data_input/__init__.py",
        script="workflow/scripts/verification_metrics.py",
        forecast=lambda wc: BASELINE_CONFIGS[wc.baseline_id]["root"],
        truth_dep=truth_file_dep,
    output:
        OUT_ROOT / f"data/baselines/{{baseline_id}}/{{init_time}}/verif_{VERIF_HASH}.nc",
    log:
        OUT_ROOT / "logs/verification_metrics_baseline/{baseline_id}-{init_time}.log",
    resources:
        cpus_per_task=24,
        mem_mb=80_000,
        runtime="120m",
    params:
        baseline_steps=lambda wc: BASELINE_CONFIGS[wc.baseline_id]["steps"],
        member=lambda wc: BASELINE_CONFIGS[wc.baseline_id].get("member", "000"),
        truth=config["truth"]["root"],
        truth_source_id=f"truth-{TRUTH_HASH}",
        regions=REGIONS,
        experiment_params=",".join(EXPERIMENT_PARAMS),
        threshold_dict=config["experiment"]["thresholds"],
        lapse_rate_flag=(
            "--lapse_rate_correction"
            if config.get("lapse_rate_correction", True)
            else ""
        ),
    shell:
        """
        export ECCODES_DEFINITION_PATH=$(realpath .venv/share/eccodes-cosmo-resources/definitions)
        uv run {input.script} \
            --forecast {input.forecast} \
            --truth {params.truth} \
            --reftime {wildcards.init_time} \
            --steps "{params.baseline_steps}" \
            --source_id "{wildcards.baseline_id}" \
            --truth_source_id "{params.truth_source_id}" \
            --regions "{params.regions}" \
            --params "{params.experiment_params}" \
            --threshold_dict "{params.threshold_dict}" \
            --member "{params.member}" \
            {params.lapse_rate_flag} \
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
        "src/data_input/__init__.py",
        script="workflow/scripts/verification_metrics.py",
        inference_okfile=rules.inference_execute.output.okfile,
        truth_dep=truth_file_dep,
    output:
        OUT_ROOT / f"data/runs/{{run_id}}/{{init_time}}/verif_{VERIF_HASH}.nc",
    log:
        OUT_ROOT / "logs/verification_metrics/{run_id}-{init_time}.log",
    resources:
        cpus_per_task=24,
        mem_mb=80_000,
        runtime="60m",
    # wildcard_constraints:
    # run_id="^" # to avoid ambiguitiy with run_baseline_verif
    # TODO: implement logic to use experiment name instead of run_id as wildcard
    params:
        fcst_steps=lambda wc: RUN_CONFIGS[wc.run_id]["steps"],
        truth=config["truth"]["root"],
        truth_source_id=f"truth-{TRUTH_HASH}",
        regions=REGIONS,
        grib_out_dir=lambda wc: (
            Path(OUT_ROOT) / f"data/runs/{wc.run_id}/{wc.init_time}/grib"
        ).resolve(),
        experiment_params=",".join(EXPERIMENT_PARAMS),
        threshold_dict=config["experiment"]["thresholds"],
        lapse_rate_flag=(
            "--lapse_rate_correction"
            if config.get("lapse_rate_correction", True)
            else ""
        ),
    shell:
        """
        export ECCODES_DEFINITION_PATH=$(realpath .venv/share/eccodes-cosmo-resources/definitions)
        uv run {input.script} \
            --forecast {params.grib_out_dir} \
            --truth {params.truth} \
            --reftime {wildcards.init_time} \
            --steps "{params.fcst_steps}" \
            --source_id "{wildcards.run_id}" \
            --truth_source_id "{params.truth_source_id}" \
            --regions "{params.regions}" \
            --params "{params.experiment_params}" \
            --threshold_dict "{params.threshold_dict}" \
            {params.lapse_rate_flag} \
            --output {output} >{log} 2>&1
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
        cpus_per_task=24,
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
        cpus_per_task=16,
        mem_mb=50_000,
        runtime="20m",
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
        + ",truth-{}:{}".format(TRUTH_HASH, config["truth"]["label"]),
    shell:
        """
        uv run {input.script} {input.verif} --output_dir {output} --labels "{params.label_map}" >{log} 2>&1
        """


rule verification_scoremaps:
    input:
        "src/verification/__init__.py",
        "src/data_input/__init__.py",
        script="workflow/scripts/verification_scoremaps.py",
        inference_okfiles=lambda wc: expand(
            rules.inference_execute.output.okfile,
            init_time=_restrict_reftimes_to_hours(REFTIMES),
            allow_missing=True,
        ),
        truth=config["truth"]["root"],
    output:
        OUT_ROOT
        / f"data/runs/{{run_id}}/scoremaps/{{param}}_{{leadtime}}_{TRUTH_HASH}.nc",
    log:
        OUT_ROOT
        / f"logs/verification_scoremaps/{{run_id}}-{TRUTH_HASH}-{{param}}-{{leadtime}}.log",
    resources:
        cpus_per_task=2,
        mem_mb=50_000,
        runtime="60m",
    # wildcard_constraints:
    # run_id="^" # to avoid ambiguitiy with run_baseline_verif
    # TODO: implement logic to use experiment name instead of run_id as wildcard
    params:
        fcst_label=lambda wc: RUN_CONFIGS[wc.run_id].get("label"),
        fcst_steps=lambda wc: RUN_CONFIGS[wc.run_id]["steps"],
        truth_label=config["truth"]["label"],
        reftimes=" ".join(t.strftime("%Y%m%d%H%M") for t in REFTIMES),
        run_root=lambda wc: (Path(OUT_ROOT) / f"data/runs/{wc.run_id}").resolve(),
    shell:
        """
        export ECCODES_DEFINITION_PATH=$(realpath .venv/share/eccodes-cosmo-resources/definitions)
        uv run {input.script} \
            --run_root {params.run_root} \
            --reftimes {params.reftimes} \
            --truth {input.truth} \
            --step {wildcards.leadtime} \
            --steps "{params.fcst_steps}" \
            --param {wildcards.param} \
            --output {output} >{log} 2>&1
        """


rule verification_scoremaps_baseline:
    input:
        "src/verification/__init__.py",
        "src/data_input/__init__.py",
        script="workflow/scripts/verification_scoremaps.py",
        forecast=lambda wc: BASELINE_CONFIGS[wc.baseline_id]["root"],
        truth=config["truth"]["root"],
    output:
        OUT_ROOT
        / f"data/baselines/{{baseline_id}}/scoremaps/{{param}}_{{leadtime}}_{TRUTH_HASH}.nc",
    log:
        OUT_ROOT
        / f"logs/verification_scoremaps_baseline/{{baseline_id}}-{TRUTH_HASH}-{{param}}-{{leadtime}}.log",
    resources:
        cpus_per_task=24,
        mem_mb=50_000,
        runtime="60m",
    params:
        baseline_steps=lambda wc: BASELINE_CONFIGS[wc.baseline_id]["steps"],
        member=lambda wc: BASELINE_CONFIGS[wc.baseline_id].get("member", "000"),
        reftimes=" ".join(t.strftime("%Y%m%d%H%M") for t in REFTIMES),
    shell:
        """
        export ECCODES_DEFINITION_PATH=$(realpath .venv/share/eccodes-cosmo-resources/definitions)
        uv run {input.script} \
            --baseline_root {input.forecast} \
            --reftimes {params.reftimes} \
            --truth {input.truth} \
            --step {wildcards.leadtime} \
            --steps "{params.baseline_steps}" \
            --param {wildcards.param} \
            --member "{params.member}" \
            --output {output} >{log} 2>&1
        """


# ----------------------------------------------------- #
# SAL precipitation verification (Structure–Amplitude–Location)            #
# Compute-only: one CSV per (participant, param, lead time), keyed by    #
# truth hash. Plotting lives in the publication-figures workflow.           #
# ----------------------------------------------------- #

# SAL settings are experiment-level (not per-participant); read them from the
# validated config, falling back to the SalConfig defaults so a rule invoked
# directly (evalml make ...) still works. These rules only fire when
# experiment.sal.enabled is true (gated in the Snakefile target rule).
_SAL_CONFIG = config.get("experiment", {}).get("sal") or {}
_SAL_GRID_EXTENT = _SAL_CONFIG.get("grid_extent", [-1.0, 18.0, 42.0, 50.5])
_SAL_ARGS = (
    f"--thr-factor {_SAL_CONFIG.get('thr_factor', 0.067)} "
    f"--thr-quantile {_SAL_CONFIG.get('thr_quantile', 0.95)} "
    f"--grid-extent {' '.join(str(x) for x in _SAL_GRID_EXTENT)} "
    f"--grid-step-lat {_SAL_CONFIG.get('grid_step_lat', 0.01)} "
    f"--grid-step-lon {_SAL_CONFIG.get('grid_step_lon', 0.0145)}"
)


rule verification_sal:
    input:
        "src/verification/__init__.py",
        "src/verification/spatial.py",
        "src/verification/sal.py",
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
        "src/verification/sal.py",
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
