# ----------------------------------------------------- #
# VERIFICATION WORKFLOW                                 #
# ----------------------------------------------------- #


import pandas as pd


include: "common.smk"


rule run_verif:
    input:
        expand(
            "resources/inference/{run_id}/output/{init_time}/raw",
            init_time=REFTIMES,
            allow_missing=True,
        ),
    output:
        "results/eval_report_{run_id}.html",
    shell:
        ""
