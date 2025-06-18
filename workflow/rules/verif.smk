# ----------------------------------------------------- #
# VERIFICATION WORKFLOW                                 #
# ----------------------------------------------------- #


import pandas as pd


include: "common.smk"

rule run_verif:
    input:
        expand(
<<<<<<< MRB-324-pipeline-for-batch-inference
            "resources/inference/output/{run_id}/{init_time}",
            init_time=REFTIMES,
=======
            "resources/inference/{run_id}/output/{init_time}/raw",
            init_time=INIT_TIMES,
>>>>>>> main
            allow_missing=True,
        ),
    output:
        "results/eval_report_{run_id}.html",
    shell:
        ""
