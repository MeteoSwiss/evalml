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
        cosmoe_zarr=expand(rules.extract_cosmoe_fcts.output, year=20),
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


# TODO: not have zarr_dataset hardcoded
rule run_verif_fct:
    input:
        script="workflow/scripts/verif_from_grib.py",
        grib_output=rules.map_init_time_to_inference_group.output[0],
        zarr_dataset="/scratch/mch/fzanetta/data/anemoi/datasets/mch-co2-an-archive-0p02-2015-2020-6h-v3-pl13.zarr",
    output:
        OUT_ROOT / "{run_id}/{init_time}/verif.csv"

    wildcard_constraints:
        run_id="[a-zA-Z0-9]{32}" # to avoid ambiguitiy with run_cosmoe_verif
        # run_id="^(?!.*COSMO-E).*" # to avoid ambiguitiy with run_cosmoe_veri
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
            rules.run_verif_fct.output,
            init_time=_restrict_reftimes_to_hours(REFTIMES, [0, 12] if wc.run_id == "COSMO-E" else None),
            allow_missing=True,
        ),
    output:
        OUT_ROOT / "{run_id}/verif_aggregated.csv",
    params:
        verif_files_glob=lambda wc: OUT_ROOT / f"{wc.run_id}/*/verif.csv",
    log:
        "logs/verif_aggregation/{run_id}.log",
    shell:
        """
        uv run {input.script} {params.verif_files_glob} \
            --output {output} > {log} 2>&1
        """