# ----------------------------------------------------- #
# REPORTING WORKFLOW                                    #
# ----------------------------------------------------- #
from datetime import datetime

include: "common.smk"


rule report_experiment_dashboard:
    input:
        verif=collect_experiment_participants
    output:
        touch("results/{experiment}/report.html")
    log:
        "logs/report_experiment_dashboard/{experiment}.log",

