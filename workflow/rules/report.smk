# ----------------------------------------------------- #
# REPORTING WORKFLOW                                    #
# ----------------------------------------------------- #
from datetime import datetime


include: "common.smk"


def make_header_text():
    dates = config["dates"]
    if isinstance(dates, list):
        return f"Explicit initializations from {len(dates)} runs have been used."
    return f"Initializations from {dates.get('start')} to {dates.get('end')} by {dates.get('frequency')} have been used."


rule report_experiment_dashboard:
    localrule: True
    input:
        script="workflow/scripts/report_experiment_dashboard.py",
        verif=EXPERIMENT_PARTICIPANTS.values(),
        template="resources/report/dashboard/template.html.jinja2",
        js_script="resources/report/dashboard/script.js",
    output:
        report(
            directory(OUT_ROOT / "results/{experiment}/metrics/dashboard"),
            htmlindex="dashboard.html",
        ),
    params:
        sources=",".join(list(EXPERIMENT_PARTICIPANTS.keys())),
        header_text=make_header_text(),
    log:
        OUT_ROOT / "logs/report_experiment_dashboard/{experiment}.log",
    shell:
        """
        python {input.script} \
            --verif_files {input.verif} \
            --template {input.template} \
            --script {input.js_script} \
            --header_text "{params.header_text}" \
            --output {output} > {log} 2>&1
        """
