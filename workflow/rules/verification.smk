# ----------------------------------------------------- #
# VERIFICATION WORKFLOW                                 #
# ----------------------------------------------------- #
from datetime import datetime

import pandas as pd


include: "common.smk"


# TODO: make sure the boundaries aren't used
def _get_baseline_forecast_path(wc):
    """Return the FCST<year> directory for a baseline in the ICON GRIB archive."""
    root = BASELINE_CONFIGS[wc.baseline_id].get("root")
    year = wc.init_time[2:4]
    return f"{root}/FCST{year}"


rule verification_metrics_baseline:
    input:
        "src/verification/__init__.py",
        "src/data_input/__init__.py",
        script="workflow/scripts/verification_metrics.py",
        forecast=_get_baseline_forecast_path,
        truth=config["truth"]["root"],
    params:
        baseline_label=lambda wc: BASELINE_CONFIGS[wc.baseline_id].get("label"),
        baseline_steps=lambda wc: BASELINE_CONFIGS[wc.baseline_id]["steps"],
        truth_label=config["truth"]["label"],
        regions=REGIONS,
        threshold_dict=config["thresholds"],
    output:
        OUT_ROOT / "data/baselines/{baseline_id}/{init_time}/verif.nc",
    log:
        OUT_ROOT / "logs/verification_metrics_baseline/{baseline_id}-{init_time}.log",
    resources:
        cpus_per_task=24,
        mem_mb=50_000,
        runtime="60m",
    shell:
        """
        uv run {input.script} \
            --forecast {input.forecast} \
            --truth {input.truth} \
            --reftime {wildcards.init_time} \
            --steps "{params.baseline_steps}" \
            --label "{params.baseline_label}" \
            --truth_label "{params.truth_label}" \
            --regions "{params.regions}" \
            --threshold_dict "{params.threshold_dict}" \
            --output {output} > {log} 2>&1
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
        truth=config["truth"]["root"],
    output:
        OUT_ROOT / "data/runs/{run_id}/{init_time}/verif.nc",
    # wildcard_constraints:
    # run_id="^" # to avoid ambiguitiy with run_baseline_verif
    # TODO: implement logic to use experiment name instead of run_id as wildcard
    params:
        fcst_label=lambda wc: RUN_CONFIGS[wc.run_id].get("label"),
        fcst_steps=lambda wc: RUN_CONFIGS[wc.run_id]["steps"],
        truth_label=config["truth"]["label"],
        regions=REGIONS,
        grib_out_dir=lambda wc: (
            Path(OUT_ROOT) / f"data/runs/{wc.run_id}/{wc.init_time}/grib"
        ).resolve(),
        threshold_dict=config["thresholds"],
    log:
        OUT_ROOT / "logs/verification_metrics/{run_id}-{init_time}.log",
    resources:
        cpus_per_task=24,
        mem_mb=50_000,
        runtime="60m",
    shell:
        """
        uv run {input.script} \
            --forecast {params.grib_out_dir} \
            --truth {input.truth} \
            --reftime {wildcards.init_time} \
            --steps "{params.fcst_steps}" \
            --label "{params.fcst_label}" \
            --truth_label "{params.truth_label}" \
            --regions "{params.regions}" \
            --threshold_dict "{params.threshold_dict}" \
            --output {output} > {log} 2>&1
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
        OUT_ROOT / "data/runs/{run_id}/verif_aggregated.nc",
    log:
        OUT_ROOT / "logs/verification_metrics_aggregation/{run_id}.log",
    resources:
        cpus_per_task=24,
        mem_mb=250_000,
        runtime="2h",
    shell:
        """
        uv run {input.script} {input.verif_nc} --output {output} > {log} 2>&1
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
        OUT_ROOT / "data/baselines/{baseline_id}/verif_aggregated.nc",
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
    params:
        labels=",".join(list(EXPERIMENT_PARTICIPANTS.keys())),
    log:
        OUT_ROOT / "logs/verification_metrics_plot/{experiment}.log",
    resources:
        cpus_per_task=16,
        mem_mb=50_000,
        runtime="20m",
    shell:
        """
        uv run {input.script} {input.verif} --output_dir {output} > {log} 2>&1
        """
