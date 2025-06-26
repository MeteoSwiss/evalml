# ----------------------------------------------------- #
# VERIFICATION WORKFLOW                                 #
# ----------------------------------------------------- #
from datetime import datetime

import pandas as pd


include: "common.smk"


# TODO: make sure the boundaries aren't used
rule run_cosmoe_verif:
    input:
        script="workflow/scripts/verif_cosmoe_fct.py",
        zarr_dataset="/scratch/mch/fzanetta/data/anemoi/datasets/mch-co2-an-archive-0p02-2015-2020-6h-v3-pl13.zarr",
    output:
        OUT_ROOT / "COSMO-E/{init_time}/verif.csv"
    log:
        "logs/verif_cosmoe/{init_time}.log",
    shell:
        """
        uv run {input.script} \
            --zarr_dataset {input.zarr_dataset} \
            --reftime {wildcards.init_time} \
            --output {output} > {log} 2>&1
        """

rule run_cosmoe_verif_all:
    input:
        expand(
            rules.run_cosmoe_verif.output,
            init_time=[t.strftime("%Y%m%d%H%M") for t in REFTIMES if t.hour in [0, 12]],
        ),

# TODO: not have zarr_dataset hardcoded
rule run_verif_from_grib:
    input:
        script="workflow/scripts/verif_from_grib.py",
        grib_output=rules.map_init_time_to_inference_group.output[0],
        zarr_dataset="/scratch/mch/fzanetta/data/anemoi/datasets/mch-co2-an-archive-0p02-2015-2020-6h-v3-pl13.zarr",
    output:
        OUT_ROOT / "{run_id}/{init_time}/verif.csv",
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

rule run_verif_aggregation:
    localrule: True
    input:
        script="workflow/scripts/verif_aggregation.py",
        verif_csv=expand(
            rules.run_verif_from_grib.output,
            init_time=[t.strftime("%Y%m%d%H%M") for t in REFTIMES],
            allow_missing=True,
        ),
    params:
        verif_files_glob=lambda wc: OUT_ROOT / f"{wc.run_id}/*/verif.csv",
    output:
        OUT_ROOT / "{run_id}/verif_aggregated.csv",
    log:
        "logs/verif_aggregation/{run_id}.log",
    shell:
        """
        uv run {input.script} {params.verif_files_glob} \
            --output {output} > {log} 2>&1
        """
