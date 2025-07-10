# ----------------------------------------------------- #
# REPORTING WORKFLOW                                    #
# ----------------------------------------------------- #
from datetime import datetime

include: "common.smk"


rule report_experiment_dashboard:
    input:
        verif=EXPERIMENT_PARTICIPANTS.values()
    output:
        touch("results/{experiment}/report.html")
    log:
        "logs/report_experiment_dashboard/{experiment}.log",

