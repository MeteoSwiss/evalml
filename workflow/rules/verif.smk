# ----------------------------------------------------- #
# VERIFICATION WORKFLOW                                 #
# ----------------------------------------------------- #


import pandas as pd


include: "common.smk"


rule run_verif:
    input:
        expand(
            rules.map_init_time_to_inference_group.output,
            init_time=[t.strftime("%Y%m%d%H%M") for t in REFTIMES],
            allow_missing=True,
        ),
    output:
        "results/eval_report_{run_id}.html",
    shell:
        ""
