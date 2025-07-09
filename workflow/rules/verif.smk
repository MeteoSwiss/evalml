# ----------------------------------------------------- #
# VERIFICATION WORKFLOW                                 #
# ----------------------------------------------------- #
from datetime import datetime

import pandas as pd


include: "common.smk"


# TODO: make sure the boundaries aren't used
rule run_verif_cosmoe:
    localrule: True
    input:
        script="workflow/scripts/verif_cosmoe_fct.py",
        cosmoe_zarr=lambda wc: expand(rules.extract_cosmoe_fcts.output, year=wc.init_time[2:4]),
        zarr_dataset="/scratch/mch/fzanetta/data/anemoi/datasets/mch-co2-an-archive-0p02-2015-2020-6h-v3-pl13.zarr",
    output:
        OUT_ROOT / "baselines/COSMO-E/{init_time}/verif.csv",
    log:
        "logs/verif_cosmoe/{init_time}.log",
    shell:
        """
        uv run {input.script} \
            --zarr_dataset {input.zarr_dataset} \
            --cosmoe_zarr {input.cosmoe_zarr} \
            --reftime {wildcards.init_time} \
            --output {output} > {log} 2>&1
        """


# TODO: not have zarr_dataset hardcoded
rule run_verif_fct:
    localrule: True
    input:
        script="workflow/scripts/verif_from_grib.py",
        grib_output=rules.map_init_time_to_inference_group.output[0],
        zarr_dataset="/scratch/mch/fzanetta/data/anemoi/datasets/mch-co2-an-archive-0p02-2015-2020-6h-v3-pl13.zarr",
    output:
        OUT_ROOT / "runs/{run_id}/{init_time}/verif.csv",
    # wildcard_constraints:
        # run_id="^" # to avoid ambiguitiy with run_cosmoe_verif
        # TODO: implement logic to use experiment name instead of run_id as wildcard
    log:
        "logs/verif_from_grib/{run_id}-{init_time}.log",
    shell:
        """
        uv run {input.script} \
            --grib_output_dir {input.grib_output} \
            --zarr_dataset {input.zarr_dataset} \
            --reftime {wildcards.init_time} \
            --output {output} > {log} 2>&1
        """


def _restrict_reftimes_to_hours(reftimes, hours=None):
    """Restrict the reference times to specific hours."""
    if hours is None:
        return [t.strftime("%Y%m%d%H%M") for t in reftimes]
    return [t.strftime("%Y%m%d%H%M") for t in reftimes if t.hour in hours]


rule run_verif_aggregation:
    localrule: True
    input:
        script="workflow/scripts/verif_aggregation.py",
        verif_csv=lambda wc: expand(
            rules.run_verif_fct.output, init_time=_restrict_reftimes_to_hours(REFTIMES), allow_missing=True,
        ),
    output:
        OUT_ROOT / "runs/{run_id}/verif_aggregated.csv",
    params:
        verif_files_glob=lambda wc: OUT_ROOT / f"runs/{wc.run_id}/*/verif.csv",
    log:
        "logs/verif_aggregation/{run_id}.log",
    shell:
        """
        uv run {input.script} {params.verif_files_glob} \
            --output {output} > {log} 2>&1
        """


rule run_verif_aggregation_cosmoe:
    localrule: True
    input:
        script="workflow/scripts/verif_aggregation.py",
        verif_csv=lambda wc: expand(
            rules.run_verif_cosmoe.output,
            init_time=_restrict_reftimes_to_hours(REFTIMES, [0, 12]),
            allow_missing=True,
        ),
    output:
        OUT_ROOT / "baselines/COSMO-E/verif_aggregated.csv",
    params:
        verif_files_glob=lambda wc: OUT_ROOT / "baselines/COSMO-E/*/verif.csv",
    log:
        "logs/verif_aggregation/COSMO-E.log",
    shell:
        """
        uv run {input.script} {params.verif_files_glob} \
            --output {output} > {log} 2>&1
        """



EXPERIMENT_PARTICIPANTS = collect_experiment_participants()

rule run_verif_plot_metrics:
    localrule: True
    input:
        script="workflow/scripts/verif_plot_metrics.py",
        verif=list(EXPERIMENT_PARTICIPANTS.values()),
    output:
        directory("results/{experiment}/plot_metrics"),
    params:
        labels=",".join(list(EXPERIMENT_PARTICIPANTS.keys())),
    log:
        "logs/verif_plot_metrics/{experiment}.log",
    shell:
        """
        uv run {input.script} {input.verif} \
            --labels '{params.labels}' \
            --output_dir {output} > {log} 2>&1
        """
