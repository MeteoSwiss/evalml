# ----------------------------------------------------- #
# REPORTING WORKFLOW                                    #
# ----------------------------------------------------- #
from datetime import datetime

include: "common.smk"


rule report_experiment_dashboard:
    localrule: True
    input:
        script="workflow/scripts/report_experiment_dashboard.py",
        verif=EXPERIMENT_PARTICIPANTS.values(),
        template="resources/report/dashboard/template.html.jinja2",
        js_script="resources/report/dashboard/script.js",
    output:
        OUT_ROOT / "results/{experiment}/dashboard.html"
    log:
        OUT_ROOT / "logs/report_experiment_dashboard/{experiment}.log",
    shell:
        """
        python {input.script} \
            --verif_files {input.verif} \
            --template {input.template} \
            --script {input.js_script} \
            --output {output} > {log} 2>&1
        """