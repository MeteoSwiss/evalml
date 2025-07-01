# ----------------------------------------------------- #
# REPORTING WORKFLOW                                    #
# ----------------------------------------------------- #
from datetime import datetime

include: "common.smk"


rule report_study:
    input:
        verif=collect_study_participants
    output:
        touch("results/studies/{study}/eval_report.html")
    log:
        "logs/report_study_dashboard/{study}.log",

