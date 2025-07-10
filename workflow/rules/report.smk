# ----------------------------------------------------- #
# REPORTING WORKFLOW                                    #
# ----------------------------------------------------- #
from datetime import datetime

include: "common.smk"


rule report_experiment_dashboard:
    input:
        verif=EXPERIMENT_PARTICIPANTS.values()
    output:
        touch(OUT_ROOT / "results/{experiment}/report.html")
    log:
        OUT_ROOT / "logs/report_experiment_dashboard/{experiment}.log",

