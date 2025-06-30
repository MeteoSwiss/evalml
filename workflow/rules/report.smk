# ----------------------------------------------------- #
# REPORTING WORKFLOW                                    #
# ----------------------------------------------------- #
from datetime import datetime

include: "common.smk"


# TODO: not have zarr_dataset hardcoded
rule report_study_dashboard:
    input:
        verif=lambda wc: expand(rules.run_verif_aggregation.output, experiment=[e for e in config["studies"][wc.study]["experiments"]])
    output:
        touch("results/eval_report_{study}.html")
    log:
        "logs/report_study_dashboard/{study}.log",

