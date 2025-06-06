

rule run_verif:
    input:
        "resources/inference/output/{run_id}/prediction.nc",
    output:
        "results/eval_report_{run_id}.html",
    shell:
        ""
