# ----------------------------------------------------- #
# VERIFICATION WORKFLOW                                 #
# ----------------------------------------------------- #
from datetime import datetime

import pandas as pd


include: "common.smk"


# TODO: make sure the boundaries aren't used
rule verif_metrics_cosmoe:
    localrule: True
    input:
        script="workflow/scripts/verif_cosmoe_fct.py",
        module="workflow/scripts/src/verification.py",
        # cosmoe_zarr=lambda wc: expand(rules.extract_cosmoe_fcts.output, year=wc.init_time[2:4]),
        cosmoe_zarr=lambda wc: expand(
            "/store_new/mch/msopr/ml/COSMO-E/FCST{year}.zarr",
            year=wc.init_time[2:4],
        ),
        zarr_dataset="/scratch/mch/fzanetta/data/anemoi/datasets/mch-co2-an-archive-0p02-2015-2020-6h-v3-pl13.zarr",
    params:
        cosmoe_label="COSMO-E",
        analysis_label="COSMO-E analysis",
    output:
        OUT_ROOT / "data/baselines/COSMO-E/{init_time}/verif.csv",
    log:
        OUT_ROOT / "logs/verif_metrics_cosmoe/{init_time}.log",
    shell:
        """
        uv run {input.script} \
            --zarr_dataset {input.zarr_dataset} \
            --cosmoe_zarr {input.cosmoe_zarr} \
            --reftime {wildcards.init_time} \
            --cosmoe_label "{params.cosmoe_label}" \
            --analysis_label "{params.analysis_label}" \
            --output {output} > {log} 2>&1
        """


# TODO: not have zarr_dataset hardcoded
rule verif_metrics:
    localrule: True
    input:
        script="workflow/scripts/verif_from_grib.py",
        grib_output=rules.inference_routing.output[0],
        zarr_dataset="/scratch/mch/fzanetta/data/anemoi/datasets/mch-co2-an-archive-0p02-2015-2020-6h-v3-pl13.zarr",
    output:
        OUT_ROOT / "data/runs/{run_id}/{init_time}/verif.csv",
    # wildcard_constraints:
    # run_id="^" # to avoid ambiguitiy with run_cosmoe_verif
    # TODO: implement logic to use experiment name instead of run_id as wildcard
    params:
        fcst_label=lambda wc: RUN_CONFIGS[wc.run_id].get("label"),
        analysis_label="COSMO-E analysis",
    log:
        OUT_ROOT / "logs/verif_metrics/{run_id}-{init_time}.log",
    shell:
        """
        uv run {input.script} \
            --grib_output_dir {input.grib_output} \
            --zarr_dataset {input.zarr_dataset} \
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
        verif_csv=lambda wc: expand(
            rules.verif_metrics.output,
            init_time=_restrict_reftimes_to_hours(REFTIMES),
            allow_missing=True,
        ),
    output:
        OUT_ROOT / "data/runs/{run_id}/verif_aggregated.csv",
    params:
        valid_every=config.get("verification", {}).get("valid_every"),
    log:
        OUT_ROOT / "logs/verif_metrics_aggregation/{run_id}.log",
    shell:
        """
        uv run {input.script} {input.verif_csv} \
            --valid_every '{params.valid_every}' \
            --output {output} > {log} 2>&1
        """


rule verif_metrics_aggregation_cosmoe:
    localrule: True
    input:
        script="workflow/scripts/verif_aggregation.py",
        verif_csv=lambda wc: expand(
            rules.verif_metrics_cosmoe.output,
            init_time=_restrict_reftimes_to_hours(REFTIMES, [0, 12]),
            allow_missing=True,
        ),
    output:
        OUT_ROOT / "data/baselines/COSMO-E/verif_aggregated.csv",
    params:
        valid_every=config.get("verification", {}).get("valid_every"),
    log:
        OUT_ROOT / "logs/verif_metrics_aggregation_cosmoe/COSMO-E.log",
    shell:
        """
        uv run {input.script} {input.verif_csv} \
            --valid_every '{params.valid_every}' \
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
