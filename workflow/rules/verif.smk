# ----------------------------------------------------- #
# VERIFICATION WORKFLOW                                 #
# ----------------------------------------------------- #
from datetime import datetime

import pandas as pd


include: "common.smk"


# TODO: make sure the boundaries aren't used
rule verif_metrics_baseline:
    localrule: True
    input:
        script="workflow/scripts/verif_baseline.py",
        module="workflow/scripts/src/verification.py",
        baseline_zarr=lambda wc: expand(
            "{root}/FCST{year}.zarr",
            root=BASELINE_CONFIGS[wc.baseline_id].get("root"),
            year=wc.init_time[2:4],
        ),
        analysis_zarr=config["analysis"].get("analysis_zarr"),
    params:
        baseline_label=lambda wc: BASELINE_CONFIGS[wc.baseline_id].get("label"),
        baseline_steps=lambda wc: BASELINE_CONFIGS[wc.baseline_id].get("steps"),
        analysis_label=config["analysis"].get("label"),
    output:
        OUT_ROOT / "data/baselines/{baseline_id}/{init_time}/verif.nc",
    log:
        OUT_ROOT / "logs/verif_metrics_baseline/{baseline_id}-{init_time}.log",
    shell:
        """
        uv run {input.script} \
            --analysis_zarr {input.analysis_zarr} \
            --baseline_zarr {input.baseline_zarr} \
            --reftime {wildcards.init_time} \
            --lead_time "{params.baseline_steps}" \
            --baseline_label "{params.baseline_label}" \
            --analysis_label "{params.analysis_label}" \
            --output {output} > {log} 2>&1
        """


# TODO: not have analysis_zarr hardcoded
rule verif_metrics:
    localrule: True
    input:
        script="workflow/scripts/verif_from_grib.py",
        inference_okfile=_inference_routing_fn,
        grib_output=rules.inference_routing.output[0],
        analysis_zarr=config["analysis"].get("analysis_zarr"),
    output:
        OUT_ROOT / "data/runs/{run_id}/{init_time}/verif.nc",
    # wildcard_constraints:
    # run_id="^" # to avoid ambiguitiy with run_baseline_verif
    # TODO: implement logic to use experiment name instead of run_id as wildcard
    params:
        fcst_label=lambda wc: RUN_CONFIGS[wc.run_id].get("label"),
        analysis_label=config["analysis"].get("label"),
    log:
        OUT_ROOT / "logs/verif_metrics/{run_id}-{init_time}.log",
    shell:
        """
        uv run {input.script} \
            --grib_output_dir {input.grib_output} \
            --analysis_zarr {input.analysis_zarr} \
            --fcst_label "{params.fcst_label}" \
            --analysis_label "{params.analysis_label}" \
            --output {output} > {log} 2>&1
        """


def _restrict_reftimes_to_hours(reftimes, hours=None):
    """Restrict the reference times to specific hours."""
    if hours is None:
        return [t.strftime("%Y%m%d%H%M") for t in reftimes]
    return [t.strftime("%Y%m%d%H%M") for t in reftimes if t.hour in hours]


rule verif_metrics_aggregation:
    localrule: True
    input:
        script="workflow/scripts/verif_aggregation.py",
        verif_nc=lambda wc: expand(
            rules.verif_metrics.output,
            init_time=_restrict_reftimes_to_hours(REFTIMES),
            allow_missing=True,
        ),
    output:
        OUT_ROOT / "data/runs/{run_id}/verif_aggregated.nc",
    log:
        OUT_ROOT / "logs/verif_metrics_aggregation/{run_id}.log",
    shell:
        """
        uv run {input.script} {input.verif_nc} \
            --output {output} > {log} 2>&1
        """


rule verif_metrics_aggregation_baseline:
    localrule: True
    input:
        script="workflow/scripts/verif_aggregation.py",
        verif_nc=lambda wc: expand(
            rules.verif_metrics_baseline.output,
            init_time=_restrict_reftimes_to_hours(REFTIMES),
            allow_missing=True,
        ),
    output:
        OUT_ROOT / "data/baselines/{baseline_id}/verif_aggregated.nc",
    log:
        OUT_ROOT / "logs/verif_metrics_aggregation_baseline/{baseline_id}.log",
    shell:
        """
        uv run {input.script} {input.verif_nc} \
            --output {output} > {log} 2>&1
        """


rule verif_metrics_plot:
    localrule: True
    input:
        script="workflow/scripts/verif_plot_metrics.py",
        verif=list(EXPERIMENT_PARTICIPANTS.values()),
    output:
        report(
            directory(OUT_ROOT / "results/{experiment}/metrics/plots"),
            patterns=["{name}.png"],
        ),
    params:
        labels=",".join(list(EXPERIMENT_PARTICIPANTS.keys())),
    log:
        OUT_ROOT / "logs/verif_metrics_plot/{experiment}.log",
    shell:
        """
        uv run {input.script} {input.verif} \
            --output_dir {output} > {log} 2>&1
        """
